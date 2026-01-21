import asyncio
import hashlib
import json
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Dict, Optional, Tuple

import httpx
from openai import AsyncOpenAI


# Browser-like headers to reduce the chance of being blocked by Cloudflare.
# Note: Intentionally do NOT set Accept-Encoding to let httpx handle decoding.
DEFAULT_BROWSER_HEADERS: Dict[str, str] = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7",
    "Connection": "keep-alive",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}


@dataclass
class OpenAIChatResult:
    response: Any  # ChatCompletion | None
    text: Optional[str]
    usage: Any  # response.usage | stream usage | None
    request_id: Optional[str]
    finish_reason: Optional[str]
    stream: bool
    send_ts: str
    recv_ts: str
    duration_ms: float
    first_byte_ms: Optional[float]
    status: str  # ok / timeout / rate_limit / content_filter / error


# A lightweight cache for (event-loop, base_url, api_key, headers_fp, worker_id) -> client.
# NOTE: worker_id is optional; when provided each worker uses an independent connection pool.
_CLIENT_CACHE: Dict[Tuple[int, str, str, str, Optional[int]], AsyncOpenAI] = {}
_HTTPX_CACHE: Dict[Tuple[int, str, str, str, Optional[int]], httpx.AsyncClient] = {}
_CACHE_LOCK = threading.Lock()


# Shared RPM limiter across OpenAI translators.
# Keyed by (base_url, model) to avoid mixing different providers/endpoints.
_LAST_REQUEST_TS: Dict[Tuple[str, str], float] = {}
_RPM_LOCK = threading.Lock()


def _now_iso_ms() -> str:
    return datetime.now().isoformat(timespec="milliseconds")


def _normalize_base_url(base_url: str) -> str:
    return (base_url or "").strip().rstrip("/")


def _headers_fingerprint(headers: Dict[str, str]) -> str:
    try:
        payload = json.dumps(dict(sorted(headers.items())), ensure_ascii=False, separators=(",", ":"))
    except Exception:
        payload = str(headers)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _default_httpx_limits() -> httpx.Limits:
    # Keep it reasonably high for concurrent batches while still bounded.
    # keepalive_expiry=60 避免连接过早超时
    return httpx.Limits(
        max_connections=256,
        max_keepalive_connections=64,
        keepalive_expiry=60.0  # 从30秒增加到60秒
    )


def _default_httpx_timeout() -> httpx.Timeout:
    # 分别设置各阶段超时，避免无限等待
    # connect: 建立连接的超时
    # read: 等待服务器响应的超时（最关键）
    # write: 发送数据的超时
    # pool: 等待连接池空闲连接的超时
    return httpx.Timeout(
        timeout=300.0,    # 总超时
        connect=30.0,     # 连接超时（从60降到30）
        read=120.0,       # ✅ 新增：读取超时120秒（避免无限等待）
        write=60.0,       # ✅ 新增：写入超时
        pool=30.0         # ✅ 新增：连接池等待超时
    )


async def get_openai_client(
    *,
    api_key: str,
    base_url: str,
    headers: Optional[Dict[str, str]] = None,
    worker_id: Optional[int] = None,
) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI client with a shared httpx connection pool.

    IMPORTANT: The returned client must be treated as shared; callers should NOT close it.
    
    Args:
        worker_id: 可选的 worker ID，传入时每个 worker 使用独立连接池
                   避免 HTTP/2 多路复用导致的互相阻塞
    """

    api_key = api_key or ""
    base_url_n = _normalize_base_url(base_url)
    merged_headers = dict(DEFAULT_BROWSER_HEADERS)
    if headers:
        merged_headers.update(headers)

    loop = asyncio.get_running_loop()
    loop_key = id(loop)
    fp = _headers_fingerprint(merged_headers)
    
    # ✅ 加入 worker_id，不同 worker 使用独立的客户端和连接池
    key = (loop_key, base_url_n, api_key, fp, worker_id)

    if key in _CLIENT_CACHE:
        return _CLIENT_CACHE[key]

    with _CACHE_LOCK:
        if key in _CLIENT_CACHE:
            return _CLIENT_CACHE[key]

        # ✅ 默认禁用 HTTP/2（可通过环境变量 OPENAI_HTTP2=true 启用）
        # HTTP/2 多路复用可能导致一个慢请求阻塞同连接的其他请求
        import os
        use_http2 = os.getenv('OPENAI_HTTP2', 'false').lower() == 'true'
        
        http_client = httpx.AsyncClient(
            headers=merged_headers,
            http2=use_http2,
            limits=_default_httpx_limits(),
            timeout=_default_httpx_timeout(),
        )

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url_n,
            default_headers=merged_headers,
            http_client=http_client,
        )

        _HTTPX_CACHE[key] = http_client
        _CLIENT_CACHE[key] = client
        return client


def invalidate_client_cache(*, base_url: str = None, api_key: str = None):
    """
    清理客户端缓存（连接错误时调用）
    
    参数为空时清理所有缓存，否则只清理匹配的缓存
    """
    with _CACHE_LOCK:
        if base_url is None and api_key is None:
            # 清理所有
            _CLIENT_CACHE.clear()
            _HTTPX_CACHE.clear()
            return
        
        # 清理匹配的缓存
        base_url_n = _normalize_base_url(base_url) if base_url else None
        keys_to_remove = []
        for key in _CLIENT_CACHE.keys():
            # key may be (loop_key, base_url, api_key, headers_fp) or
            # (loop_key, base_url, api_key, headers_fp, worker_id)
            cached_url = key[1] if len(key) > 1 else None
            cached_key = key[2] if len(key) > 2 else None
            if (base_url_n is None or cached_url == base_url_n) and \
               (api_key is None or cached_key == api_key):
                keys_to_remove.append(key)
        
        for key in keys_to_remove:
            _CLIENT_CACHE.pop(key, None)
            _HTTPX_CACHE.pop(key, None)


async def _maybe_rpm_sleep(*, base_url: str, model: str, max_requests_per_minute: int, logger=None) -> None:
    if not max_requests_per_minute or max_requests_per_minute <= 0:
        return

    delay = 60.0 / float(max_requests_per_minute)
    key = (_normalize_base_url(base_url), str(model))

    sleep_time = 0.0
    with _RPM_LOCK:
        last_ts = _LAST_REQUEST_TS.get(key, 0.0)
        now = time.time()
        elapsed = now - last_ts
        if elapsed < delay:
            sleep_time = delay - elapsed

    if sleep_time > 0:
        if logger:
            try:
                logger.info(f"Ratelimit sleep: {sleep_time:.2f}s")
            except Exception:
                pass
        await asyncio.sleep(sleep_time)


async def _mark_rpm_request(*, base_url: str, model: str, max_requests_per_minute: int) -> None:
    if not max_requests_per_minute or max_requests_per_minute <= 0:
        return

    key = (_normalize_base_url(base_url), str(model))
    with _RPM_LOCK:
        _LAST_REQUEST_TS[key] = time.time()


def _classify_error(exc: Exception) -> str:
    msg = str(exc).lower()
    et = type(exc).__name__.lower()

    if isinstance(exc, (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException)):
        return "timeout"
    if "rate limit" in msg or "ratelimit" in msg or "429" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg or "readtimeout" in et:
        return "timeout"

    return "error"


async def chat_completions(
    *,
    api_key: str,
    base_url: str,
    model: str,
    messages: Any,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    stream: bool = False,
    stream_options: Optional[Dict[str, Any]] = None,
    timeout: Optional[float | httpx.Timeout] = None,
    max_requests_per_minute: int = 0,
    headers: Optional[Dict[str, str]] = None,
    metrics_logger: Optional[Callable[..., None]] = None,
    logger: Any = None,
    extra_metrics: Optional[Dict[str, Any]] = None,
    worker_id: Optional[int] = None,
) -> OpenAIChatResult:
    """Unified OpenAI chat.completions call.

    This function:
    - Reuses a shared httpx connection pool per event loop
    - Applies a simple RPM limiter (optional)
    - Supports both stream and non-stream calls
    - Collects basic timing/usage metrics and optionally logs them via `metrics_logger`

    The function does NOT implement semantic retries; callers should handle retries at a higher level.
    """

    client = await get_openai_client(api_key=api_key, base_url=base_url, headers=headers, worker_id=worker_id)

    await _maybe_rpm_sleep(
        base_url=base_url,
        model=str(model),
        max_requests_per_minute=max_requests_per_minute,
        logger=logger,
    )

    send_ts = _now_iso_ms()
    start_perf = time.perf_counter()

    response = None
    usage = None
    request_id = None
    finish_reason = None
    first_byte_ms = None
    status = "error"
    text: Optional[str] = None

    try:
        base_params: Dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        if temperature is not None:
            base_params["temperature"] = temperature
        if max_tokens is not None:
            base_params["max_tokens"] = max_tokens

        if stream:
            base_params["stream"] = True
            if stream_options is not None:
                base_params["stream_options"] = stream_options

            stream_obj = await client.chat.completions.create(**base_params, timeout=timeout)

            chunks: list[str] = []
            first_chunk = True
            async for chunk in stream_obj:
                if first_chunk:
                    first_byte_ms = (time.perf_counter() - start_perf) * 1000
                    first_chunk = False

                if request_id is None:
                    request_id = getattr(chunk, "id", None)

                if getattr(chunk, "choices", None):
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None)
                    delta_content = getattr(delta, "content", None) if delta is not None else None
                    if delta_content:
                        chunks.append(delta_content)

                    fr = getattr(choice0, "finish_reason", None)
                    if fr:
                        finish_reason = fr

                u = getattr(chunk, "usage", None)
                if u:
                    usage = u

            text = "".join(chunks).strip()
            status = "ok" if text else "error"

        else:
            response = await client.chat.completions.create(**base_params, timeout=timeout)
            request_id = getattr(response, "id", None)
            usage = getattr(response, "usage", None)

            try:
                if response and getattr(response, "choices", None):
                    finish_reason = response.choices[0].finish_reason
            except Exception:
                finish_reason = None

            try:
                if response and getattr(response, "choices", None):
                    msg = response.choices[0].message
                    content = getattr(msg, "content", None)
                    if content:
                        text = str(content).strip()
            except Exception:
                text = None

            if finish_reason == "content_filter":
                status = "content_filter"
            else:
                status = "ok"

        await _mark_rpm_request(
            base_url=base_url,
            model=str(model),
            max_requests_per_minute=max_requests_per_minute,
        )

        return OpenAIChatResult(
            response=response,
            text=text,
            usage=usage,
            request_id=request_id,
            finish_reason=finish_reason,
            stream=stream,
            send_ts=send_ts,
            recv_ts=_now_iso_ms(),
            duration_ms=(time.perf_counter() - start_perf) * 1000,
            first_byte_ms=first_byte_ms,
            status=status,
        )

    except asyncio.CancelledError:
        # Preserve cancellation semantics.
        status = "error"
        raise

    except Exception as exc:
        status = _classify_error(exc)
        
        # ✅ 连接错误时清理缓存，下次请求将重建连接
        error_msg = str(exc).lower()
        is_connection_error = any(kw in error_msg for kw in [
            'connection', 'reset', 'closed', 'eof', 'broken',
            'remotedisconnected', 'remoteprotocolerror'
        ]) or isinstance(exc, (ConnectionError, httpx.ConnectError, httpx.RemoteProtocolError))
        
        if is_connection_error:
            invalidate_client_cache(base_url=base_url)
        
        raise

    finally:
        recv_ts = _now_iso_ms()
        duration_ms = (time.perf_counter() - start_perf) * 1000
        extra = dict(extra_metrics or {})
        if request_id is not None:
            extra.setdefault("request_id", request_id)

        if metrics_logger:
            try:
                metrics_logger(
                    model_name=str(model),
                    status=status,
                    send_ts=send_ts,
                    recv_ts=recv_ts,
                    duration_ms=duration_ms,
                    usage=usage,
                    stream=stream,
                    first_byte_ms=first_byte_ms,
                    extra=extra if extra else None,
                )
            except Exception:
                # Metrics should never break the main translation flow.
                pass
