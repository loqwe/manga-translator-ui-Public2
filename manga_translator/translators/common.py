import re
import time
import asyncio
import contextlib
import inspect
import textwrap
import json
import os
import sys
import shutil
import sqlite3
import logging
from typing import List, Tuple, Dict, Any, Optional, Callable, Awaitable
from abc import abstractmethod
import numpy as np
import cv2

from ..utils import InfererModule, ModelWrapper, repeating_sequence, is_valuable_text

# Module-level logger for glossary operations
logger = logging.getLogger('manga_translator')

try:
    import readline
except Exception:
    readline = None


def contains_han_chars(text: str) -> bool:
    """Return True if text contains at least one Han (CJK ideograph) character."""
    if not text:
        return False

    for ch in str(text):
        cp = ord(ch)
        # CJK Unified Ideographs + Extension A + Compatibility Ideographs + Extensions (supplementary planes)
        if (
            (0x3400 <= cp <= 0x4DBF)
            or (0x4E00 <= cp <= 0x9FFF)
            or (0xF900 <= cp <= 0xFAFF)
            or (0x20000 <= cp <= 0x2EBEF)
        ):
            return True

    return False

VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'POL': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
    'ARA': 'Arabic',
    'CNR': 'Montenegrin',
    'SRP': 'Serbian',
    'HRV': 'Croatian',
    'THA': 'Thai',
    'IND': 'Indonesian',
    'FIL': 'Filipino (Tagalog)'
}

ISO_639_1_TO_VALID_LANGUAGES = {
    'zh': 'CHS',
    'ja': 'JPN',
    'en': 'ENG',
    'ko': 'KOR',
    'vi': 'VIN',
    'cs': 'CSY',
    'nl': 'NLD',
    'fr': 'FRA',
    'de': 'DEU',
    'hu': 'HUN',
    'it': 'ITA',
    'pl': 'POL',
    'pt': 'PTB',
    'ro': 'ROM',
    'ru': 'RUS',
    'es': 'ESP',
    'tr': 'TRK',
    'uk': 'UKR',
    'vi': 'VIN',
    'ar': 'ARA',
    'cnr': 'CNR',
    'sr': 'SRP',
    'hr': 'HRV',
    'th': 'THA',
    'id': 'IND',
    'tl': 'FIL'
}

class InvalidServerResponse(Exception):
    pass

class MissingAPIKeyException(Exception):
    pass

class LanguageUnsupportedException(Exception):
    def __init__(self, language_code: str, translator: str = None, supported_languages: List[str] = None):
        error = 'Language not supported for %s: "%s"' % (translator if translator else 'chosen translator', language_code)
        if supported_languages:
            error += '. Supported languages: "%s"' % ','.join(supported_languages)
        super().__init__(error)

class BRMarkersValidationException(Exception):
    """AI断句检查失败异常"""
    def __init__(self, missing_count: int, total_count: int, tolerance: int):
        self.missing_count = missing_count
        self.total_count = total_count
        self.tolerance = tolerance
        super().__init__(
            f"AI断句检查失败：{missing_count}/{total_count} 条翻译缺失[BR]标记（容忍度：{tolerance}）"
        )

class MultimodalUnsupportedException(Exception):
    """模型不支持多模态输入异常"""
    def __init__(self, model_name: str, translator: str):
        self.model_name = model_name
        self.translator = translator
        super().__init__(
            f"模型 {model_name} 不支持多模态输入（图片+文本）"
        )

def validate_openai_response(response, logger=None) -> bool:
    """
    验证OpenAI API响应对象的有效性
    
    Args:
        response: API返回的响应对象
        logger: 日志记录器（可选）
    
    Returns:
        bool: 响应是否有效
    
    Raises:
        Exception: 如果响应对象无效
    """
    # 检查响应对象是否有choices属性
    if not hasattr(response, 'choices'):
        error_msg = f"API返回了无效的响应对象: {type(response).__name__}, 内容: {str(response)[:200]}"
        if logger:
            logger.error(error_msg)
        raise Exception(f"API返回了无效的响应对象，类型: {type(response).__name__}")
    
    return True

def validate_gemini_response(response, logger=None) -> bool:
    """
    验证Gemini API响应对象的有效性
    
    Args:
        response: API返回的响应对象
        logger: 日志记录器（可选）
    
    Returns:
        bool: 响应是否有效
    
    Raises:
        Exception: 如果响应对象无效
    """
    # 检查响应对象是否有candidates属性
    if not hasattr(response, 'candidates'):
        error_msg = f"Gemini API返回了无效的响应对象: {type(response).__name__}, 内容: {str(response)[:200]}"
        if logger:
            logger.error(error_msg)
        raise Exception(f"Gemini API返回了无效的响应对象，类型: {type(response).__name__}")
    
    # 检查是否有text属性（某些错误响应可能没有）
    if not hasattr(response, 'text'):
        error_msg = f"Gemini API响应缺少text属性: {type(response).__name__}"
        if logger:
            logger.error(error_msg)
        raise Exception("Gemini API响应缺少text属性")
    
    return True

def draw_text_boxes_on_image(image, text_regions: List[Any], text_order: List[int], 
                             upscaled_size: Tuple[int, int] = None):
    """
    在图片上绘制带编号的文本框
    
    Args:
        image: 原始图片 (numpy array 或 PIL Image)
        text_regions: 文本区域列表，每个区域应该有 xyxy 或 min_rect 属性
        text_order: 文本顺序列表，对应每个文本框的编号
        upscaled_size: 超分后的图片尺寸 (height, width)，用于坐标转换。如果为None则不转换
    
    Returns:
        绘制了文本框的图片（与输入类型相同）
    """
    if image is None or len(text_regions) == 0:
        return image
    
    # 检查是否为PIL Image，如果是则转换为numpy数组
    from PIL import Image as PILImage
    is_pil = isinstance(image, PILImage.Image)
    if is_pil:
        # 处理各种图片模式，统一转换为RGB
        pil_image = image
        if pil_image.mode == "P":
            pil_image = pil_image.convert("RGBA" if "transparency" in pil_image.info else "RGB")
        if pil_image.mode == "RGBA":
            background = PILImage.new('RGB', pil_image.size, (255, 255, 255))
            background.paste(pil_image, mask=pil_image.split()[-1])
            pil_image = background
        elif pil_image.mode in ("LA", "L", "1", "CMYK"):
            if pil_image.mode == "LA":
                pil_image = pil_image.convert("RGBA")
                background = PILImage.new('RGB', pil_image.size, (255, 255, 255))
                background.paste(pil_image, mask=pil_image.split()[-1])
                pil_image = background
            else:
                pil_image = pil_image.convert("RGB")
        elif pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        canvas = np.array(pil_image)
    else:
        canvas = image.copy()
    
    h, w = canvas.shape[:2]
    
    # 计算坐标缩放比例（超分坐标 -> 原图坐标）
    scale_x, scale_y = 1.0, 1.0
    if upscaled_size is not None:
        upscaled_h, upscaled_w = upscaled_size
        if upscaled_w > 0 and upscaled_h > 0:
            scale_x = w / upscaled_w
            scale_y = h / upscaled_h
    
    # 计算线宽
    lw = max(round(sum(canvas.shape[:2]) / 2 * 0.003), 2)
    
    # 定义多种颜色（RGB格式）
    colors = [
        (255, 0, 0),     # 红
        (0, 255, 0),     # 绿
        (0, 0, 255),     # 蓝
        (255, 165, 0),   # 橙
        (128, 0, 128),   # 紫
        (0, 255, 255),   # 青
        (255, 0, 255),   # 品红
        (255, 255, 0),   # 黄
        (0, 128, 0),     # 深绿
        (128, 0, 0),     # 深红
    ]
    
    # 先收集所有框的边界信息
    all_boxes = []
    for region in text_regions:
        if hasattr(region, 'xyxy'):
            x1, y1, x2, y2 = region.xyxy
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            all_boxes.append((int(x1), int(y1), int(x2), int(y2)))
        elif hasattr(region, 'min_rect'):
            pts = region.min_rect.astype(np.float64)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            bx1, by1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            bx2, by2 = int(pts[:, 0].max()), int(pts[:, 1].max())
            all_boxes.append((bx1, by1, bx2, by2))
    
    def check_overlap(lx, ly, lw_size, lh_size, exclude_idx):
        """检查标签区域是否与其他框重叠"""
        label_rect = (lx, ly - lh_size, lx + lw_size, ly)
        for i, (bx1, by1, bx2, by2) in enumerate(all_boxes):
            if i == exclude_idx:
                continue
            # 检查矩形是否重叠
            if not (label_rect[2] < bx1 or label_rect[0] > bx2 or label_rect[3] < by1 or label_rect[1] > by2):
                return True
        return False
    
    # 遍历每个文本区域并绘制
    for idx, region in enumerate(text_regions):
        if idx >= len(text_order):
            break
            
        order_num = text_order[idx]
        color = colors[idx % len(colors)]
        
        # 获取文本框坐标并转换
        # 边框向外扩展，避免粗边框覆盖文字内容
        expand = lw  # 向外扩展的像素数（等于线宽）
        
        if hasattr(region, 'xyxy'):
            x1, y1, x2, y2 = region.xyxy
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            # 向外扩展边框
            box_x1, box_y1 = int(x1) - expand, int(y1) - expand
            box_x2, box_y2 = int(x2) + expand, int(y2) + expand
            cv2.rectangle(canvas, (box_x1, box_y1), (box_x2, box_y2), color, lw)
        elif hasattr(region, 'min_rect'):
            pts = region.min_rect.astype(np.float64)
            pts[:, 0] *= scale_x
            pts[:, 1] *= scale_y
            # 计算中心点，向外扩展多边形
            center_x = pts[:, 0].mean()
            center_y = pts[:, 1].mean()
            for i in range(len(pts)):
                dx = pts[i, 0] - center_x
                dy = pts[i, 1] - center_y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist > 0:
                    pts[i, 0] += (dx / dist) * expand
                    pts[i, 1] += (dy / dist) * expand
            pts = pts.astype(np.int32)
            cv2.polylines(canvas, [pts], True, color, lw)
            box_x1, box_y1 = int(pts[:, 0].min()), int(pts[:, 1].min())
            box_x2, box_y2 = int(pts[:, 0].max()), int(pts[:, 1].max())
        else:
            continue
        
        # 绘制编号标签
        label_text = str(order_num)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(lw / 2, 0.6)
        font_thickness = max(lw, 2)
        
        (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        margin = 3
        
        # 四个候选位置：上、下、左、右
        candidates = [
            (box_x1, box_y1 - margin),                          # 上
            (box_x1, box_y2 + text_height + margin),            # 下
            (box_x1 - text_width - margin, box_y1 + text_height), # 左
            (box_x2 + margin, box_y1 + text_height),            # 右
        ]
        
        # 选择不重叠且在图片范围内的位置
        label_x, label_y = candidates[0]  # 默认上方
        for cx, cy in candidates:
            # 检查是否在图片范围内
            if cx < 0 or cy - text_height < 0 or cx + text_width > w or cy > h:
                continue
            # 检查是否与其他框重叠
            if not check_overlap(cx, cy, text_width, text_height, idx):
                label_x, label_y = cx, cy
                break
        
        # 最终边界检查
        label_x = max(0, min(label_x, w - text_width))
        label_y = max(text_height, min(label_y, h))
        
        # 绘制编号文本（带黑色描边）
        cv2.putText(canvas, label_text, (label_x, label_y), font, font_scale, (0, 0, 0), font_thickness + 2, cv2.LINE_AA)
        cv2.putText(canvas, label_text, (label_x, label_y), font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    # 如果输入是PIL Image，转换回PIL格式
    if is_pil:
        return PILImage.fromarray(canvas)
    return canvas


# ============================================================================
# AsyncOpenAI 客户端包装器 - 使用 curl_cffi 绕过 TLS 指纹检测
# ============================================================================

class AsyncOpenAICurlCffi:
    """
    异步 OpenAI 客户端包装器，使用 curl_cffi 绕过 TLS 指纹检测
    完全兼容 AsyncOpenAI 的接口，可直接替换使用

    用法:
        client = AsyncOpenAICurlCffi(
            api_key="your-api-key",
            base_url="https://api.openai.com/v1",
            default_headers={"User-Agent": "..."},
            impersonate="chrome110"
        )

        response = await client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}]
        )
    """

    class ChatCompletions:
        """聊天完成接口"""

        def __init__(self, parent):
            self.parent = parent

        async def create(self, model, messages, temperature=None, max_tokens=None, **kwargs):
            """创建聊天完成请求"""
            url = f"{self.parent.base_url}/chat/completions"

            headers = {
                "Authorization": f"Bearer {self.parent.api_key}",
                "Content-Type": "application/json"
            }

            # 合并默认请求头
            if self.parent.default_headers:
                headers.update(self.parent.default_headers)

            # 构建请求数据
            data = {
                "model": model,
                "messages": messages
            }

            if temperature is not None:
                data["temperature"] = temperature
            if max_tokens is not None:
                data["max_tokens"] = max_tokens

            # 添加其他参数
            data.update(kwargs)

            stream_mode = bool(data.get("stream"))
            if stream_mode:
                return self._create_stream(url, data, headers)

            # 发送异步请求
            response = await self.parent.session.post(
                url,
                json=data,
                headers=headers,
                timeout=self.parent.timeout
            )

            if response.status_code != 200:
                print(f"[AsyncOpenAICurlCffi] Error - URL: {url}")
                print(f"[AsyncOpenAICurlCffi] Error - Status: {response.status_code}")
                print(f"[AsyncOpenAICurlCffi] Error - Response: {response.text[:500] if response.text else '(empty)'}")
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', '')}"
                except:
                    error_msg = f"{error_msg}: {response.text[:500] if response.text else '(empty)'}"
                raise Exception(error_msg)

            result = response.json()

            # 转换为类似 OpenAI SDK 的响应对象
            return _OpenAIResponse(result)

        def _create_stream(self, url, data, headers):
            """SSE 流式请求，返回异步可迭代对象。"""

            async def _gen():
                async with self.parent.session.stream(
                    "POST",
                    url,
                    json=data,
                    headers=headers,
                    timeout=self.parent.stream_timeout
                ) as response:
                    if response.status_code != 200:
                        text = await response.atext()
                        raise Exception(f"API request failed with status {response.status_code}: {text[:500] if text else '(empty)'}")

                    async for raw_line in response.aiter_lines():
                        if isinstance(raw_line, (bytes, bytearray)):
                            raw_line = raw_line.decode("utf-8", errors="ignore")
                        line = str(raw_line or "").strip()
                        if not line or not line.startswith("data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == "[DONE]":
                            break
                        try:
                            chunk_data = json.loads(payload)
                        except Exception:
                            continue
                        yield _OpenAIStreamChunk(chunk_data)

            return _gen()

    class Chat:
        """聊天接口"""

        def __init__(self, parent):
            self.completions = AsyncOpenAICurlCffi.ChatCompletions(parent)

    class Models:
        """模型列表接口"""

        def __init__(self, parent):
            self.parent = parent

        async def list(self):
            """获取可用模型列表"""
            url = f"{self.parent.base_url}/models"

            headers = {
                "Authorization": f"Bearer {self.parent.api_key}",
                "Content-Type": "application/json"
            }

            # 合并默认请求头
            if self.parent.default_headers:
                headers.update(self.parent.default_headers)

            # 发送异步请求
            response = await self.parent.session.get(
                url,
                headers=headers,
                timeout=self.parent.timeout
            )

            if response.status_code != 200:
                print(f"[AsyncOpenAICurlCffi] List Error - URL: {url}")
                print(f"[AsyncOpenAICurlCffi] List Error - Status: {response.status_code}")
                print(f"[AsyncOpenAICurlCffi] List Error - Response: {response.text[:500] if response.text else '(empty)'}")
                error_msg = f"API request failed with status {response.status_code}"
                try:
                    error_data = response.json()
                    if "error" in error_data:
                        error_msg = f"{error_msg}: {error_data['error'].get('message', '')}"
                except:
                    error_msg = f"{error_msg}: {response.text[:200] if response.text else '(empty)'}"
                raise Exception(error_msg)

            # 检查响应内容类型
            content_type = response.headers.get('content-type', '')
            if 'application/json' not in content_type and 'text/json' not in content_type:
                raise Exception("API 不支持获取模型列表（返回了非 JSON 响应）。请手动输入模型名称。")

            try:
                result = response.json()
            except Exception as e:
                raise Exception(f"无法解析 API 响应: {str(e)}。请手动输入模型名称。")

            # 转换为类似 OpenAI SDK 的响应对象
            return _ModelsResponse(result)

    def __init__(self, api_key, base_url="https://api.openai.com/v1",
                 default_headers=None, http_client=None, impersonate="chrome110",
                 timeout=600, stream_timeout=300):
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.default_headers = default_headers or {}
        self.timeout = timeout
        self.stream_timeout = stream_timeout
        self.impersonate = impersonate

        # 检测是否是本地地址（本地地址不需要 impersonate，且可能导致超时）
        local_indicators = ['localhost', '127.0.0.1', '0.0.0.0', '192.168.', '10.', '172.16.', '172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
        is_local = any(indicator in base_url.lower() for indicator in local_indicators)

        # 延迟导入 curl_cffi，避免在不需要时导入
        try:
            from curl_cffi.requests import AsyncSession
            if is_local:
                self.session = AsyncSession()
                print(f"[AsyncOpenAICurlCffi] Local address detected, disabled impersonate for: {base_url}")
            else:
                self.session = AsyncSession(impersonate=impersonate)
        except ImportError:
            raise ImportError(
                "curl_cffi is required for TLS fingerprint bypass. "
                "Install it with: pip install curl_cffi"
            )

        # 创建聊天接口
        self.chat = self.Chat(self)
        # 创建模型列表接口
        self.models = self.Models(self)

    async def close(self):
        """关闭 session"""
        if hasattr(self.session, 'close'):
            await self.session.close()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


class _OpenAIResponse:
    """模拟 OpenAI SDK 的响应对象"""

    class Choice:
        class Message:
            def __init__(self, content):
                self.content = content

        def __init__(self, message_content, finish_reason):
            self.message = self.Message(message_content)
            self.finish_reason = finish_reason

    class Usage:
        def __init__(self, total_tokens, prompt_tokens=0, completion_tokens=0):
            self.total_tokens = total_tokens
            self.prompt_tokens = prompt_tokens
            self.completion_tokens = completion_tokens

    def __init__(self, data):
        self.model = data.get('model', '')
        self.choices = [
            self.Choice(
                choice['message']['content'],
                choice.get('finish_reason', 'stop')
            )
            for choice in data.get('choices', [])
        ]
        usage_data = data.get('usage', {})
        self.usage = self.Usage(
            usage_data.get('total_tokens', 0),
            usage_data.get('prompt_tokens', 0),
            usage_data.get('completion_tokens', 0)
        )


class _OpenAIStreamChunk:
    """模拟 OpenAI SDK stream chunk 对象（最小字段集）"""

    class Choice:
        class Delta:
            def __init__(self, content):
                self.content = content

        def __init__(self, delta_content, finish_reason):
            self.delta = self.Delta(delta_content)
            self.finish_reason = finish_reason

    def __init__(self, data):
        choices = data.get("choices", []) if isinstance(data, dict) else []
        self.choices = []
        for c in choices:
            delta = c.get("delta", {}) if isinstance(c, dict) else {}
            delta_content = delta.get("content", "") if isinstance(delta, dict) else ""
            finish_reason = c.get("finish_reason") if isinstance(c, dict) else None
            self.choices.append(self.Choice(delta_content, finish_reason))


class _ModelsResponse:
    """模拟 OpenAI SDK 的模型列表响应对象"""

    class Model:
        def __init__(self, model_data):
            self.id = model_data.get('id', '')
            self.object = model_data.get('object', 'model')
            self.created = model_data.get('created', 0)
            self.owned_by = model_data.get('owned_by', '')

    def __init__(self, data):
        self.data = [
            self.Model(model_data)
            for model_data in data.get('data', [])
        ]
        self.object = data.get('object', 'list')


class MTPEAdapter():
    async def dispatch(self, queries: List[str], translations: List[str]) -> List[str]:
        # TODO: Make it work in windows (e.g. through os.startfile)
        if not readline:
            print('MTPE is currently only supported on linux')
            return translations
        new_translations = []
        print('Running Machine Translation Post Editing (MTPE)')
        for i, (query, translation) in enumerate(zip(queries, translations)):
            print(f'\n[{i + 1}/{len(queries)}] {query}:')
            readline.set_startup_hook(lambda: readline.insert_text(translation.replace('\n', '\\n')))
            new_translation = ''
            try:
                new_translation = input(' -> ').replace('\\n', '\n')
            finally:
                readline.set_startup_hook()
            new_translations.append(new_translation)
        print()
        return new_translations

class CommonTranslator(InfererModule):
    # Translator has to support all languages listed in here. The language codes will be resolved into
    # _LANGUAGE_CODE_MAP[lang_code] automatically if _LANGUAGE_CODE_MAP is a dict.
    # If it is a list it will simply return the language code as is.
    _LANGUAGE_CODE_MAP = {}

    # The amount of repeats upon detecting an invalid translation.
    # Use with _is_translation_invalid and _modify_invalid_translation_query.
    _INVALID_REPEAT_COUNT = 0

    # Will sleep for the rest of the minute if the request count is over this number.
    _MAX_REQUESTS_PER_MINUTE = -1

    def __init__(self):
        super().__init__()
        self.mtpe_adapter = MTPEAdapter()
        self._last_request_ts = 0
        self.enable_post_translation_check = False
        self.post_check_repetition_threshold = 5
        self.post_check_max_retry_attempts = 2
        self.attempts = -1
        self._MAX_SPLIT_ATTEMPTS = 3  # 最大分割层级
        self._SPLIT_THRESHOLD = 2  # 重试N次后触发分割
        self._global_attempt_count = 0  # 全局尝试计数器
        self._max_total_attempts = -1  # 全局最大尝试次数
        self._cancel_check_callback = None  # 取消检查回调
        self._custom_api_params = {}  # 存储自定义API参数
        self._stream_inline_last_len = 0
        self._stream_inline_buffer = ""
        self._stream_json_seen: Dict[int, str] = {}
        self._stream_term_seen: Dict[Tuple[str, str], str] = {}
        self._stream_result_header_printed = False
        self._stream_result_pairs_printed = False
    
    def _normalize_retry_attempts(self, attempts: Any) -> int:
        """
        规范化重试次数配置：
        -1: 无限重试
         0: 不重试（仅首次请求）
        >=1: 重试 N 次
        其他负值会回退为 0。
        """
        try:
            value = int(attempts)
        except (TypeError, ValueError):
            self.logger.warning(f"Invalid attempts value '{attempts}', fallback to -1 (infinite)")
            return -1

        if value == -1:
            return -1
        if value < -1:
            self.logger.warning(f"Invalid attempts value '{value}', fallback to 0 (no retry)")
            return 0
        return value

    def _resolve_max_total_attempts(self) -> int:
        """
        将"重试次数"转换为"总尝试次数"：
        -1 -> -1（无限）
         0 -> 1（仅首次请求）
         N -> N+1（首次 + N 次重试）
        """
        if self.attempts == -1:
            return -1
        return self.attempts + 1
    
    def _load_custom_api_params(self):
        """从固定目录加载自定义API参数配置文件"""
        try:
            from ..utils import BASE_PATH
            import os
            import json
            
            config_path = os.path.join(BASE_PATH, 'examples', 'custom_api_params.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    self._custom_api_params = json.load(f)
                self.logger.info(f"已加载自定义API参数配置: {self._custom_api_params}")
            else:
                self.logger.warning(f"自定义API参数配置文件不存在: {config_path}")
                self._custom_api_params = {}
        except Exception as e:
            self.logger.error(f"加载自定义API参数配置失败: {e}")
            self._custom_api_params = {}

    def _configure_custom_api_params(self, args) -> bool:
        """
        根据配置决定是否加载自定义 API 参数，并统一日志输出格式。
        返回值表示是否启用。
        """
        use_custom_params = getattr(args, 'use_custom_api_params', False)
        if not use_custom_params:
            self._custom_api_params = {}
            return False

        self._load_custom_api_params()
        if self._custom_api_params:
            self.logger.info(f"已启用自定义API参数: {self._custom_api_params}")
        return True

    def set_cancel_check_callback(self, callback):
        """设置取消检查回调"""
        self._cancel_check_callback = callback
    
    def _check_cancelled(self):
        """检查任务是否被取消"""
        if self._cancel_check_callback and self._cancel_check_callback():
            raise asyncio.CancelledError("Translation cancelled by user")

    async def _await_with_cancel_polling(
        self,
        awaitable: Awaitable,
        poll_interval: float = 0.2,
        on_cancel: Optional[Callable[[], Awaitable[None] | None]] = None,
    ):
        """
        Wait for a long-running awaitable while polling for cancellation.
        On cancel, attempt to cancel the inner task and run the on_cancel cleanup callback.
        """
        task = asyncio.create_task(awaitable)
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=poll_interval)
                if done:
                    return task.result()
                self._check_cancelled()
        except asyncio.CancelledError:
            if not task.done():
                task.cancel()
                with contextlib.suppress(Exception):
                    done, _ = await asyncio.wait({task}, timeout=max(poll_interval, 0.2))
                    if not done:
                        self.logger.debug("取消请求任务超时，直接退出等待")
            if on_cancel:
                try:
                    cleanup_result = on_cancel()
                    if asyncio.iscoroutine(cleanup_result):
                        await asyncio.wait_for(cleanup_result, timeout=max(poll_interval, 0.3))
                except asyncio.TimeoutError:
                    self.logger.debug("取消时清理请求超时，直接退出等待")
                except Exception as cleanup_error:
                    self.logger.debug(f"取消时清理请求失败（可忽略）: {cleanup_error}")
            raise

    async def _sleep_with_cancel_polling(self, seconds: float, poll_interval: float = 0.2):
        """Cancellable sleep that polls for stop signals during the wait."""
        if seconds <= 0:
            self._check_cancelled()
            return
        await self._await_with_cancel_polling(
            asyncio.sleep(seconds),
            poll_interval=min(poll_interval, max(seconds, 0.05)),
        )

    async def _run_unified_stream_transport(
        self,
        *,
        create_stream: Callable[[], Any],
        extract_text: Callable[[Any], Any],
        extract_finish_reason: Optional[Callable[[Any], Any]] = None,
        on_chunk: Optional[Callable[[str, str], None]] = None,
        on_cancel: Optional[Callable[[], Awaitable[None] | None]] = None,
        poll_interval: float = 0.2,
        sync_iter_in_thread: bool = False,
        first_chunk_timeout: float = 300.0,
        idle_timeout: float = 300.0,
    ) -> Tuple[str, Any]:
        """
        通用流式传输层：
        - OpenAI async stream（异步迭代）
        - Gemini stream（同步迭代，放入 to_thread 消费）
        返回：(聚合后的完整文本, 最后一次 finish_reason)
        """
        def _normalize_stream_piece(piece_text: str, current_text: str) -> str:
            """
            兼容“增量块/累计块/重复块”三种常见流格式，尽量只返回新增部分。
            """
            if not piece_text:
                return ""
            if not current_text:
                return piece_text

            # 标准累计块：piece = current + delta
            if piece_text.startswith(current_text):
                return piece_text[len(current_text):]

            # 回退/截断块：piece 只是 current 的前缀（且不是极短 token），忽略
            if len(piece_text) >= 16 and current_text.startswith(piece_text):
                return ""

            # 某些服务会把 current 放在 piece 中间，取最后一次出现后的尾部
            pos = piece_text.rfind(current_text)
            if pos != -1:
                return piece_text[pos + len(current_text):]

            # 明确重发：较长片段且 current 已以该片段结尾，忽略
            if len(piece_text) >= 16 and current_text.endswith(piece_text):
                return ""

            # 处理部分重叠：current 尾部 + piece 头部
            max_overlap = min(len(piece_text), len(current_text))
            for overlap in range(max_overlap, 0, -1):
                if current_text.endswith(piece_text[:overlap]):
                    return piece_text[overlap:]

            # 无法判断关系时按增量处理（保守）
            return piece_text

        text_parts: List[str] = []
        last_finish_reason = None

        if sync_iter_in_thread:
            def _consume_sync_stream():
                local_parts: List[str] = []
                local_finish = None
                stream_obj = create_stream()
                for chunk in stream_obj:
                    piece = extract_text(chunk)
                    if piece:
                        piece_text = str(piece)
                        current_text = ''.join(local_parts)
                        normalized_piece = _normalize_stream_piece(piece_text, current_text)
                        if normalized_piece:
                            local_parts.append(normalized_piece)
                        if on_chunk:
                            on_chunk(normalized_piece, ''.join(local_parts))
                    if extract_finish_reason:
                        finish = extract_finish_reason(chunk)
                        if finish is not None:
                            local_finish = finish
                return ''.join(local_parts), local_finish

            return await self._await_with_cancel_polling(
                asyncio.to_thread(_consume_sync_stream),
                poll_interval=poll_interval,
                on_cancel=on_cancel,
            )

        stream_obj = create_stream()
        while inspect.isawaitable(stream_obj):
            stream_obj = await self._await_with_cancel_polling(
                stream_obj,
                poll_interval=poll_interval,
                on_cancel=on_cancel,
            )

        try:
            aiter = stream_obj.__aiter__()
            got_first_chunk = False
            while True:
                chunk_timeout = first_chunk_timeout if not got_first_chunk else idle_timeout
                try:
                    chunk = await self._await_with_cancel_polling(
                        asyncio.wait_for(aiter.__anext__(), timeout=chunk_timeout),
                        poll_interval=poll_interval,
                        on_cancel=on_cancel,
                    )
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as timeout_error:
                    timeout_type = "首包" if not got_first_chunk else "流空闲"
                    raise TimeoutError(f"流式{timeout_type}超时（{chunk_timeout:.0f}s）") from timeout_error

                got_first_chunk = True
                self._check_cancelled()
                piece = extract_text(chunk)
                if piece:
                    piece_text = str(piece)
                    current_text = ''.join(text_parts)
                    normalized_piece = _normalize_stream_piece(piece_text, current_text)
                    if normalized_piece:
                        text_parts.append(normalized_piece)
                    if on_chunk:
                        on_chunk(normalized_piece, ''.join(text_parts))
                if extract_finish_reason:
                    finish = extract_finish_reason(chunk)
                    if finish is not None:
                        last_finish_reason = finish
        except asyncio.CancelledError:
            if on_cancel:
                cleanup_result = on_cancel()
                if asyncio.iscoroutine(cleanup_result):
                    with contextlib.suppress(Exception):
                        await cleanup_result
            raise
        finally:
            close_fn = getattr(stream_obj, "aclose", None)
            if callable(close_fn):
                with contextlib.suppress(Exception):
                    close_ret = close_fn()
                    if asyncio.iscoroutine(close_ret):
                        await close_ret

        return ''.join(text_parts), last_finish_reason

    def _emit_stream_lines(self, prefix: str, text: str, width: int = 100) -> None:
        """将流式增量按固定宽度换行输出，避免命令行单行过长被截断。"""
        content = (text or "").strip()
        if not content:
            return
        for line in textwrap.wrap(content, width=width, break_long_words=True, break_on_hyphens=False):
            self.logger.info(f"{prefix} {line}")

    def _reset_stream_json_preview(self) -> None:
        self._stream_json_seen = {}
        self._stream_term_seen = {}
        self._stream_result_header_printed = False
        self._stream_result_pairs_printed = False

    def _has_stream_result_pairs(self) -> bool:
        return bool(self._stream_result_pairs_printed)

    def _emit_terms_from_list(self, new_terms: List[Dict[str, str]]) -> None:
        """统一输出术语提取结果；按(原文,译文)去重并优先保留带分类版本。"""
        if not new_terms:
            return
        for term in new_terms:
            if not isinstance(term, dict):
                continue
            term_o = str(term.get("original") or term.get("src") or "").strip()
            term_t = str(term.get("translation") or term.get("dst") or "").strip()
            term_c = str(term.get("category") or "").strip()
            if not term_o or not term_t:
                continue

            key = (term_o, term_t)
            prev_category = self._stream_term_seen.get(key)
            if prev_category is not None:
                if prev_category == term_c:
                    continue
                if prev_category and not term_c:
                    continue
            if not term_c and prev_category is None:
                self._stream_term_seen[key] = ""
                continue
            self._stream_term_seen[key] = term_c

            if term_c:
                self.logger.info(f"[TERM] {term_o} -> {term_t} ({term_c})")
            else:
                self.logger.info(f"[TERM] {term_o} -> {term_t}")

    def _emit_stream_json_preview(self, prefix: str, full_text: str, source_texts: Optional[List[str]] = None) -> None:
        """
        从流式累计文本中提取已闭合的 {"id": n, "translation": "..."}，按 id 去重输出。
        """
        if not full_text:
            return
        pattern = r'"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"((?:\\.|[^"\\])*)"'
        for m in re.finditer(pattern, full_text, flags=re.DOTALL):
            try:
                tid = int(m.group(1))
            except Exception:
                continue
            raw_text = m.group(2)
            try:
                decoded_text = json.loads(f'"{ raw_text}"')
            except Exception:
                decoded_text = raw_text.replace('\\"', '"').replace("\\n", "\n")
            if self._stream_json_seen.get(tid) == decoded_text:
                continue
            self._stream_json_seen[tid] = decoded_text
            if not self._stream_result_header_printed:
                self.logger.info("--- Translation Results ---")
                self._stream_result_header_printed = True
            if source_texts and 1 <= tid <= len(source_texts):
                self.logger.info(f"{source_texts[tid - 1]} -> {decoded_text}")
            else:
                self.logger.info(f"{prefix} #{tid}: {decoded_text}")
            self._stream_result_pairs_printed = True

        term_pattern = r'"original"\s*:\s*"((?:\\.|[^"\\])*)"\s*,\s*"translation"\s*:\s*"((?:\\.|[^"\\])*)"(?:\s*,\s*"category"\s*:\s*"((?:\\.|[^"\\])*)")?'
        stream_terms: List[Dict[str, str]] = []
        for tm in re.finditer(term_pattern, full_text, flags=re.DOTALL):
            raw_o, raw_t, raw_c = tm.group(1), tm.group(2), tm.group(3) or ""
            try:
                term_o = json.loads(f'"{raw_o}"')
            except Exception:
                term_o = raw_o
            try:
                term_t = json.loads(f'"{raw_t}"')
            except Exception:
                term_t = raw_t
            try:
                term_c = json.loads(f'"{raw_c}"') if raw_c else ""
            except Exception:
                term_c = raw_c
            stream_terms.append({"original": str(term_o), "translation": str(term_t), "category": str(term_c)})
        if stream_terms:
            if not self._stream_result_header_printed:
                self.logger.info("--- Translation Results ---")
                self._stream_result_header_printed = True
            self._emit_terms_from_list(stream_terms)

    def _update_stream_inline(self, prefix: str, delta_text: str) -> None:
        """按增量流内容刷新；遇到换行符时真正换行输出。"""
        if not delta_text:
            return
        self._stream_inline_buffer += str(delta_text)

        while "\n" in self._stream_inline_buffer:
            line_text, rest = self._stream_inline_buffer.split("\n", 1)
            self._stream_inline_buffer = rest
            line = f"{prefix} {line_text}"
            pad = max(0, self._stream_inline_last_len - len(line))
            try:
                sys.stdout.write("\r" + line + (" " * pad) + "\n")
                sys.stdout.flush()
            except Exception:
                self._emit_stream_lines(prefix, line_text)
            self._stream_inline_last_len = 0

        text = self._stream_inline_buffer
        if not text:
            return
        try:
            term_width = shutil.get_terminal_size(fallback=(120, 24)).columns
        except Exception:
            term_width = 120
        available = max(20, term_width - len(prefix) - 2)
        tail = text[-available:]
        line = f"{prefix} {tail}"
        pad = max(0, self._stream_inline_last_len - len(line))
        try:
            sys.stdout.write("\r" + line + (" " * pad))
            sys.stdout.flush()
            self._stream_inline_last_len = len(line)
        except Exception:
            self._emit_stream_lines(prefix, tail)

    def _finish_stream_inline(self) -> None:
        """结束同一行刷新，补一个换行。"""
        if self._stream_inline_last_len > 0 or self._stream_inline_buffer:
            try:
                sys.stdout.write("\n")
                sys.stdout.flush()
            except Exception:
                pass
            self._stream_inline_last_len = 0
            self._stream_inline_buffer = ""

    def _now_iso(self) -> str:
        """返回当前时间的 ISO 格式字符串（带毫秒）"""
        from datetime import datetime
        return datetime.now().isoformat(timespec='milliseconds')

    def _safe_usage_value(self, usage: Any, key: str) -> Optional[int]:
        """安全获取 usage 对象中的值"""
        if usage is None:
            return None
        if isinstance(usage, dict):
            return usage.get(key)
        return getattr(usage, key, None)

    def log_ai_metrics(
        self,
        model_name: str,
        status: str,
        send_ts: str,
        recv_ts: str,
        duration_ms: float,
        usage: Any = None,
        stream: bool = False,
        first_byte_ms: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
        preset_name: Optional[str] = None,
    ):
        """记录 AI 调用指标到日志和数据库缓存
        
        Args:
            model_name: 模型名称
            status: 状态 (ok / timeout / rate_limit / content_filter / error 等)
            send_ts: 发送时间戳
            recv_ts: 接收时间戳
            duration_ms: 耗时（毫秒）
            usage: API 返回的 usage 对象
            stream: 是否流式
            first_byte_ms: 首字/首包耗时（可选）
            extra: 额外信息（如 request_id）
            preset_name: 预设名称（可选）
        """
        prompt_tokens = self._safe_usage_value(usage, 'prompt_tokens')
        completion_tokens = self._safe_usage_value(usage, 'completion_tokens')
        total_tokens = self._safe_usage_value(usage, 'total_tokens')

        preset = preset_name if preset_name is not None else (os.getenv('MANGA_TRANSLATOR_PRESET') or '')
        try:
            preset_repr = json.dumps(preset, ensure_ascii=False)
        except Exception:
            preset_repr = '""'

        fb_str = 'n/a'
        if first_byte_ms is not None:
            try:
                fb_str = f"{float(first_byte_ms):.1f}"
            except Exception:
                fb_str = 'n/a'

        tokens_str = (
            f"prompt={prompt_tokens if prompt_tokens is not None else 'n/a'}, "
            f"completion={completion_tokens if completion_tokens is not None else 'n/a'}, "
            f"total={total_tokens if total_tokens is not None else 'n/a'}"
        )

        extra_str = f", extra={extra}" if extra else ""
        stream_flag = '1' if stream else '0'

        log_line = (
            f"[AI_METRICS] preset={preset_repr} model={model_name} status={status} stream={stream_flag} "
            f"send={send_ts} recv={recv_ts} duration_ms={duration_ms:.1f} first_byte_ms={fb_str} "
            f"tokens({tokens_str}){extra_str}"
        )
        self.logger.info(log_line)

        # 写入轻量缓存（SQLite），失败不影响主流程
        try:
            self._write_ai_metrics_cache(
                preset=preset,
                model=model_name,
                status=status,
                send_ts=send_ts,
                recv_ts=recv_ts,
                duration_ms=duration_ms,
                first_byte_ms=first_byte_ms,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                extra=extra,
                raw_line=log_line,
            )
        except Exception:
            self.logger.debug("[AI_METRICS] cache write failed", exc_info=True)

    def _write_ai_metrics_cache(
        self,
        preset: str,
        model: str,
        status: str,
        send_ts: str,
        recv_ts: str,
        duration_ms: float,
        first_byte_ms: Any,
        prompt_tokens: Any,
        completion_tokens: Any,
        total_tokens: Any,
        extra: Dict[str, Any],
        raw_line: str,
    ):
        """将 AI_METRICS 追加写入 SQLite 轻量缓存。失败静默降级。"""
        # Prefer an explicit result dir if provided by UI
        result_dir = os.getenv('MANGA_TRANSLATOR_RESULT_DIR')

        if not result_dir:
            try:
                # Keep consistent with desktop_qt_ui/main.py log_dir
                if getattr(sys, 'frozen', False):
                    result_dir = os.path.join(os.path.dirname(sys.executable), '_internal', 'result')
                else:
                    from ..utils import BASE_PATH
                    result_dir = os.path.join(BASE_PATH, 'result')
            except Exception:
                return

        try:
            os.makedirs(result_dir, exist_ok=True)
        except Exception:
            return

        db_path = os.path.join(result_dir, 'ai_metrics.db')

        conn = None
        try:
            conn = sqlite3.connect(db_path, timeout=1.5, isolation_level=None)
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS ai_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT,
                    preset TEXT,
                    model TEXT,
                    status TEXT,
                    send_ts TEXT,
                    recv_ts TEXT,
                    duration_ms REAL,
                    first_byte_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    extra_json TEXT,
                    raw_line TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_metrics_ts ON ai_metrics(ts DESC);"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_ai_metrics_model_status ON ai_metrics(model, status);"
            )

            extra_json = None
            try:
                extra_json = json.dumps(extra, ensure_ascii=False) if extra else None
            except Exception:
                extra_json = None

            cur.execute(
                """
                INSERT INTO ai_metrics (
                    ts, preset, model, status, send_ts, recv_ts, duration_ms, first_byte_ms,
                    prompt_tokens, completion_tokens, total_tokens, extra_json, raw_line
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    recv_ts,
                    preset,
                    model,
                    status,
                    send_ts,
                    recv_ts,
                    float(duration_ms) if duration_ms is not None else None,
                    float(first_byte_ms) if first_byte_ms is not None else None,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                    extra_json,
                    raw_line,
                ),
            )
        finally:
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass

    def _get_retry_temperature(self, base_temperature: float, retry_attempt: int, retry_reason: str = "") -> float:
        """
        根据重试次数和原因动态调整温度，帮助模型跳出错误模式
        
        Args:
            base_temperature: 基础温度（首次请求使用）
            retry_attempt: 当前重试次数（0表示首次请求）
            retry_reason: 重试原因（模型输出问题才提高温度，网络/链接错误不提高）
            
        Returns:
            调整后的温度值
        """
        if retry_attempt <= 0:
            return base_temperature
        
        # 只有模型输出问题才提高温度（数量不匹配、质量检查失败、BR检查失败）
        # 网络错误、链接错误、返回空内容等不需要提高温度
        should_increase_temp = (
            "Translation count mismatch" in retry_reason or
            "Quality check failed" in retry_reason or 
            "BR markers missing" in retry_reason
        )
        
        if not should_increase_temp:
            return base_temperature
        
        # 每次重试增加 0.2，最高不超过 1.0
        adjusted_temp = base_temperature + (retry_attempt * 0.2)
        return min(adjusted_temp, 1.0)

    def _get_retry_hint(self, attempt: int, reason: str = "") -> str:
        """
        生成重试提示信息，用于避免模型服务器缓存导致的重复错误
        
        Args:
            attempt: 当前尝试次数
            reason: 重试原因（可选）
            
        Returns:
            重试提示字符串
        """
        hints = [
            f"[Retry attempt #{attempt}]",
            f"[This is attempt #{attempt}, please provide a different response]",
            f"[Attempt {attempt}: Previous response had issues, please try again]",
            f"[Retry #{attempt}: Please ensure quality this time]",
            f"[Attempt {attempt} - Previous attempts failed quality check]"
        ]
        
        # 根据尝试次数选择不同的提示（循环使用）
        base_hint = hints[(attempt - 1) % len(hints)]
        
        # 如果提供了原因，添加到提示中
        if reason:
            return f"{base_hint} Reason: {reason}\n\n"
        else:
            return f"{base_hint}\n\n"
    
    def _build_user_prompt_for_texts(self, texts: List[str], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """
        统一的用户提示词构建方法（纯文本翻译）
        适用于 openai.py 和 gemini.py

        Args:
            texts: 要翻译的文本列表
            ctx: 上下文对象（可选）
            prev_context: 历史上下文（可选）
            retry_attempt: 重试次数（可选，用于避免缓存）
            retry_reason: 重试原因（可选）

        Returns:
            构建好的用户提示词字符串
        """
        # 检查是否开启AI断句
        enable_ai_break = False
        if ctx and hasattr(ctx, 'config') and ctx.config and hasattr(ctx.config, 'render'):
            enable_ai_break = getattr(ctx.config.render, 'disable_auto_wrap', False)

        prompt = ""

        # 添加重试提示到最前面（如果是重试）
        if retry_attempt > 0:
            prompt += self._get_retry_hint(retry_attempt, retry_reason) + "\n"

        # 添加多页上下文（如果有）
        if prev_context:
            prompt += f"{prev_context}\n\n---\n\n"
            self.logger.info(f"[历史上下文] 长度: {len(prev_context)} 字符")
            self.logger.info(f"[历史上下文内容]\n{prev_context[:500]}...")
        else:
            self.logger.info("[历史上下文] 无历史上下文（可能是第一张图片或context_size=0）")

        prompt += "Please translate the following manga text regions. The input is provided as a JSON array:\n\n"

        input_data = []
        for i, text in enumerate(texts):
            text_to_translate = text.replace('\n', ' ').replace('\ufffd', '')
            item = {
                "id": i + 1,
                "text": text_to_translate
            }
            # 只有开启AI断句时才添加区域信息
            if enable_ai_break and ctx and hasattr(ctx, 'text_regions') and ctx.text_regions and i < len(ctx.text_regions):
                region = ctx.text_regions[i]
                region_count = len(region.lines) if hasattr(region, 'lines') else 1
                item["original_region_count"] = region_count
            
            input_data.append(item)

        prompt += json.dumps(input_data, ensure_ascii=False, indent=2)

        prompt += "\n\nCRITICAL: Provide translations in the exact same order as the input array. Your output must be a JSON array of strings corresponding to the IDs."

        return prompt

    def _build_unified_user_prompt(self, batch_data: List[Dict], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "", is_image_mode: bool = True) -> str:
        """
        统一的用户提示词构建方法（支持多模态和纯文本）
        Unified user prompt builder for both multimodal and text-only modes.

        Args:
            batch_data: List of dicts, each containing 'original_texts' and optional 'text_regions'.
            ctx: Context object.
            prev_context: Previous context string.
            retry_attempt: Retry attempt count.
            retry_reason: Reason for retry.
            is_image_mode: Whether to include image-specific descriptions.

        Returns:
            Constructed user prompt string.
        """
        import json
        
        # 检查是否开启AI断句
        enable_ai_break = False
        if ctx and hasattr(ctx, 'config') and ctx.config and hasattr(ctx.config, 'render'):
            enable_ai_break = getattr(ctx.config.render, 'disable_auto_wrap', False)

        prompt = ""

        # 添加重试提示到最前面（如果是重试）
        if retry_attempt > 0:
            prompt += self._get_retry_hint(retry_attempt, retry_reason) + "\n"

        # 添加多页上下文（如果有）
        if prev_context:
            prompt += f"{prev_context}\n\n---\n\n"
            self.logger.info(f"[历史上下文] 长度: {len(prev_context)} 字符")
        else:
            self.logger.info("[历史上下文] 无")

        if is_image_mode:
            prompt += "Please translate the following manga text regions. I'm providing multiple images with their text regions in reading order:\n\n"
            # 添加图片信息
            for i, data in enumerate(batch_data):
                prompt += f"=== Image {i+1} ===\n"
                prompt += f"Text regions ({len(data['original_texts'])} regions):\n"
                for j, text in enumerate(data['original_texts']):
                    if text is not None:
                        prompt += f"  {j+1}. {text}\n"
                prompt += "\n"
        else:
            prompt += "Please translate the following manga text regions:\n\n"

        prompt += "All texts to translate (JSON Array):\n"
        input_data = []
        text_index = 1
        for img_idx, data in enumerate(batch_data):
            # 获取 text_regions 用于 AI 断句
            text_regions = data.get('text_regions', [])
            
            for region_idx, text in enumerate(data['original_texts']):
                # 跳过 None 值
                if text is None:
                    self.logger.warning(f"跳过 None 文本 (img_idx={img_idx}, region_idx={region_idx})")
                    continue
                
                # 预处理文本：移除换行符
                text_clean = text.replace('\n', ' ').replace('\ufffd', '')
                
                item = {
                    "id": text_index,
                    "text": text_clean
                }
                
                # AI 断句逻辑：获取 original_region_count
                if enable_ai_break:
                    region_count = 1
                    # 尝试从 text_regions 获取
                    if text_regions and region_idx < len(text_regions):
                        region = text_regions[region_idx]
                        if hasattr(region, 'lines') and region.lines is not None:
                            region_count = len(region.lines)
                        elif isinstance(region, dict) and 'lines' in region:
                            region_count = len(region['lines'])
                    
                    # 如果获取失败（比如纯文本模式下 text_regions 为空），回退到数换行符
                    if region_count == 1 and text:
                        newline_count = text.count('\n')
                        if newline_count > 0:
                            region_count = newline_count + 1
                    
                    item["original_region_count"] = region_count
                
                input_data.append(item)
                text_index += 1

        prompt += json.dumps(input_data, ensure_ascii=False, indent=2)
        prompt += "\n\nCRITICAL: Provide translations in the exact same order as the input array. Follow the OUTPUT FORMAT specified in the System Prompt."

        return prompt

    def _build_user_prompt_for_hq(self, batch_data: List, ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """Alias for backward compatibility (HQ mode)"""
        return self._build_unified_user_prompt(batch_data, ctx, prev_context, retry_attempt, retry_reason, is_image_mode=True)

    def _build_user_prompt_for_texts(self, texts: List[str], ctx=None, prev_context: str = "", retry_attempt: int = 0, retry_reason: str = "") -> str:
        """Alias for text mode: wraps texts into batch_data"""
        # 构造伪 batch_data
        batch_data = [{
            'original_texts': texts,
            'text_regions': getattr(ctx, 'text_regions', []) if ctx else []
        }]
        return self._build_unified_user_prompt(batch_data, ctx, prev_context, retry_attempt, retry_reason, is_image_mode=False)

    def _validate_br_markers(self, translations: List[str], queries: List[str] = None, ctx=None, batch_indices: List[int] = None, batch_data: List = None, split_level: int = 0) -> bool:
        """
        检查翻译结果是否包含必要的[BR]标记
        Check if translations contain necessary [BR] markers
        
        Args:
            translations: 翻译结果列表
            queries: 原始查询列表（可选）
            ctx: 上下文（用于获取配置和区域信息）
            batch_indices: 批次索引列表（可选，用于定位text_regions）
            batch_data: 批次数据列表（可选，HQ翻译器使用）
            split_level: 分割级别（可选，用于跳过深度分割时的检查）
            
        Returns:
            True if validation passes, False if BR markers are missing
        """
        import re
        
        # 如果分割级别过深（>=3），跳过BR检查以避免无限重试
        if split_level >= 3:
            self.logger.info(f"[AI断句检查] 分割级别过深 (split_level={split_level})，跳过BR标记检查")
            return True
        
        # 检查是否启用了BR检查
        check_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            check_enabled = getattr(ctx.config.render, 'check_br_and_retry', False)
        
        if not check_enabled:
            return True  # 检查未启用，直接通过
        
        # 检查是否启用了AI断句
        ai_break_enabled = False
        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'render'):
            ai_break_enabled = getattr(ctx.config.render, 'disable_auto_wrap', False)
        
        if not ai_break_enabled:
            return True  # AI断句未启用，不需要检查BR
        
        # 提取每个翻译对应的区域数
        region_counts = []
        if ctx and hasattr(ctx, 'text_regions') and ctx.text_regions:
            for idx in range(len(translations)):
                # 确定实际的region索引
                if batch_indices and idx < len(batch_indices):
                    region_idx = batch_indices[idx]
                else:
                    region_idx = idx
                
                if region_idx < len(ctx.text_regions):
                    region = ctx.text_regions[region_idx]
                    region_count = len(region.lines) if hasattr(region, 'lines') else 1
                    region_counts.append(region_count)
                else:
                    region_counts.append(1)  # 默认为1
        elif batch_data:
            # HQ翻译器使用batch_data
            for idx in range(len(translations)):
                region_idx = idx
                for data in batch_data:
                    if 'text_regions' in data and data['text_regions'] and region_idx < len(data['text_regions']):
                        region = data['text_regions'][region_idx]
                        region_count = len(region.lines) if hasattr(region, 'lines') else 1
                        region_counts.append(region_count)
                        break
                else:
                    region_counts.append(1)
        else:
            region_counts = [1] * len(translations)  # 默认都为1
        
        # 检查每个翻译，统计缺失BR的数量
        needs_check_count = 0
        missing_br_count = 0
        missing_indices = []
        
        for idx, (translation, region_count) in enumerate(zip(translations, region_counts)):
            # 只检查区域数≥2的翻译
            if region_count >= 2:
                needs_check_count += 1
                # 检查是否包含BR标记
                has_br = bool(re.search(r'(\[BR\]|【BR】|<br>)', translation, flags=re.IGNORECASE))
                if not has_br:
                    missing_br_count += 1
                    missing_indices.append(idx + 1)
                    self.logger.warning(
                        f"Translation {idx+1} missing [BR] markers (expected for {region_count} regions): {translation[:50]}..."
                    )
        
        # 计算容忍的错误数量：十分之一，最少1个
        if needs_check_count > 0:
            tolerance = max(1, needs_check_count // 10)
            
            if missing_br_count > tolerance:
                # 超过容忍度，验证失败
                self.logger.warning(
                    f"[AI断句检查] 缺失BR标记的翻译数 ({missing_br_count}/{needs_check_count}) 超过容忍度 ({tolerance})，需要重试"
                )
                return False
            elif missing_br_count > 0:
                # 在容忍度内，警告但通过
                self.logger.warning(
                    f"[AI断句检查] ⚠ {missing_br_count}/{needs_check_count} 条翻译缺失BR标记，但在容忍度内 ({tolerance})，继续执行"
                )
                return True
            else:
                # 全部通过
                self.logger.info(f"[AI断句检查] ✓ 所有多行区域的翻译都包含[BR]标记 (检查了 {needs_check_count}/{len(translations)} 条)")
                return True

        return True  # 没有需要检查的翻译，直接通过

    def _validate_translation_quality(self, queries: List[str], translations: List[str]) -> Tuple[bool, str]:
        """
        验证翻译质量，检测常见问题

        Args:
            queries: 原文列表
            translations: 译文列表

        Returns:
            (is_valid, error_message)
        """
        # 1. 检查数量匹配 (这是必须的，不能跳过)
        if len(translations) != len(queries):
            return False, f"Translation count mismatch: expected {len(queries)}, got {len(translations)}"

        # 2. 检查空翻译（原文不为空但译文为空）- 已禁用
        # empty_translation_errors = []
        # for i, (source, translation) in enumerate(zip(queries, translations)):
        #     if source.strip() and not translation.strip():
        #         empty_translation_errors.append(i + 1)
        # 
        # if empty_translation_errors:
        #     return False, f"Empty translation detected at positions: {empty_translation_errors}"

        # 3. 检查合并翻译（原文是正常文本但译文只有标点）- 已禁用
        # for i, (source, translation) in enumerate(zip(queries, translations)):
        #     is_source_simple = all(char in string.punctuation or char.isspace() for char in source)
        #     is_translation_simple = all(char in string.punctuation or char.isspace() for char in translation)
        #
        #     if is_translation_simple and not is_source_simple:
        #         return False, f"Detected potential merged translation at position {i+1}"

        # 4. 检查可疑符号（模型幻觉）- 已禁用
        # SUSPICIOUS_SYMBOLS = ["ହ", "ି", "ഹ"]
        # for symbol in SUSPICIOUS_SYMBOLS:
        #     for translation in translations:
        #         if symbol in translation:
        #             return False, f"Suspicious symbol '{symbol}' detected in translation"

        return True, ""

    def _reset_global_attempt_count(self):
        """重置全局尝试计数器（每次新的翻译任务开始时调用）"""
        self._global_attempt_count = 0
        self._max_total_attempts = self.attempts

    def _increment_global_attempt(self) -> bool:
        """
        增加全局尝试计数，返回是否还可以继续尝试

        Returns:
            True: 还可以继续尝试
            False: 已达到总次数上限
        """
        self._global_attempt_count += 1

        # 无限重试模式
        if self._max_total_attempts == -1:
            return True

        # 检查是否超过上限
        if self._global_attempt_count >= self._max_total_attempts:
            self.logger.warning(f"Reached max total attempts: {self._global_attempt_count}/{self._max_total_attempts}")
            return False

        return True

    class SplitException(Exception):
        """用于触发分割的特殊异常"""
        def __init__(self, attempt_count, texts):
            self.attempt_count = attempt_count
            self.texts = texts
            super().__init__(f"Split triggered after {attempt_count} attempts")

    async def _translate_with_split(self, translator_func, texts: List[str], split_level: int = 0, **kwargs) -> List[str]:
        """
        带分割重试的翻译包装器（新逻辑）

        Args:
            translator_func: 实际的翻译函数（async callable）
            texts: 要翻译的文本列表
            split_level: 当前分割层级
            **kwargs: 传递给translator_func的其他参数

        Returns:
            翻译结果列表
        """
        # 检查是否超过全局尝试次数
        if self._max_total_attempts != -1 and self._global_attempt_count >= self._max_total_attempts:
            self.logger.error(f"Global attempt limit reached before translation: {self._global_attempt_count}/{self._max_total_attempts}")
            raise Exception(f"Translation failed: reached max total attempts ({self._max_total_attempts})")

        try:
            # 尝试翻译（内部会检查是否需要分割）
            translations = await translator_func(texts, split_level=split_level, **kwargs)
            return translations

        except self.SplitException as split_ex:
            # 触发分割
            if split_level < self._MAX_SPLIT_ATTEMPTS and len(texts) > 1:
                self.logger.warning(
                    f"Splitting after {split_ex.attempt_count} attempts at split_level={split_level}, "
                    f"batch size {len(texts)} → splitting into two halves"
                )

                # 分成两半（只分割texts，不分割batch_data等其他参数）
                mid = len(texts) // 2
                left_texts = texts[:mid]
                right_texts = texts[mid:]

                self.logger.info(f"Split: left={len(left_texts)}, right={len(right_texts)}, global_attempts={self._global_attempt_count}/{self._max_total_attempts}")

                # 并发翻译左右两部分（kwargs保持完整传递）
                try:
                    left_translations, right_translations = await asyncio.gather(
                        self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs),
                        self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs),
                        return_exceptions=False
                    )
                except Exception as split_error:
                    # 如果并发失败，回退到串行处理
                    self.logger.warning(f"Concurrent split failed, falling back to sequential: {split_error}")
                    left_translations = await self._translate_with_split(translator_func, left_texts, split_level + 1, **kwargs)
                    right_translations = await self._translate_with_split(translator_func, right_texts, split_level + 1, **kwargs)

                # 合并结果
                return left_translations + right_translations

            else:
                # 不能再分割了，终止翻译进程
                if len(texts) == 1:
                    self.logger.error(f"Single text translation failed at split_level={split_level}: {texts[0][:50]}...")
                    raise Exception(f"Translation failed for single text after {split_ex.attempt_count} attempts")
                else:
                    self.logger.error(f"Max split level ({self._MAX_SPLIT_ATTEMPTS}) reached, batch size={len(texts)}")
                    raise Exception(f"Translation failed: max split level reached with batch size {len(texts)}")

        except Exception as e:
            # 其他异常（非分割触发的），直接终止
            self.logger.error(f"Translation failed with exception at split_level={split_level}: {e}")
            raise e

    def parse_args(self, config):
        self.enable_post_translation_check = getattr(config, 'enable_post_translation_check', self.enable_post_translation_check)
        self.post_check_repetition_threshold = getattr(config, 'post_check_repetition_threshold', self.post_check_repetition_threshold)
        self.post_check_max_retry_attempts = getattr(config, 'post_check_max_retry_attempts', self.post_check_max_retry_attempts)
        self.attempts = getattr(config, 'attempts', self.attempts)

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        supported_src_languages = ['auto'] + list(self._LANGUAGE_CODE_MAP)
        supported_tgt_languages = list(self._LANGUAGE_CODE_MAP)

        if from_lang not in supported_src_languages:
            if fatal:
                raise LanguageUnsupportedException(from_lang, self.__class__.__name__, supported_src_languages)
            return False
        if to_lang not in supported_tgt_languages:
            if fatal:
                raise LanguageUnsupportedException(to_lang, self.__class__.__name__, supported_tgt_languages)
            return False
        return True

    def parse_language_codes(self, from_lang: str, to_lang: str, fatal: bool = False) -> Tuple[str, str]:
        if not self.supports_languages(from_lang, to_lang, fatal):
            return None, None
        if type(self._LANGUAGE_CODE_MAP) is list:
            return from_lang, to_lang

        _from_lang = self._LANGUAGE_CODE_MAP.get(from_lang) if from_lang != 'auto' else 'auto'
        _to_lang = self._LANGUAGE_CODE_MAP.get(to_lang)
        return _from_lang, _to_lang

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False, ctx=None) -> List[str]:
        """
        Translates list of queries of one language into another.
        """
        if to_lang not in VALID_LANGUAGES:
            raise ValueError('Invalid language code: "%s". Choose from the following: %s' % (to_lang, ', '.join(VALID_LANGUAGES)))
        if from_lang not in VALID_LANGUAGES and from_lang != 'auto':
            raise ValueError('Invalid language code: "%s". Choose from the following: auto, %s' % (from_lang, ', '.join(VALID_LANGUAGES)))
        self.logger.info(f'Translating into {VALID_LANGUAGES[to_lang]}')

        if from_lang == to_lang:
            # 即使源语言和目标语言相同，也应用文本清理（如全角句点替换）
            return [self._clean_translation_output(q, q, to_lang) for q in queries]

        # Dont translate queries without text
        query_indices = []
        final_translations = []
        for i, query in enumerate(queries):
            if not is_valuable_text(query):
                final_translations.append(queries[i])
            else:
                final_translations.append(None)
                query_indices.append(i)

        queries = [queries[i] for i in query_indices]

        translations = [''] * len(queries)
        untranslated_indices = list(range(len(queries)))
        for i in range(1 + self._INVALID_REPEAT_COUNT): # Repeat until all translations are considered valid
            if i > 0:
                self.logger.warning(f'Repeating because of invalid translation. Attempt: {i+1}')
                await asyncio.sleep(0.1)

            # Sleep if speed is over the ratelimit
            await self._ratelimit_sleep()

            # Translate
            _translations = await self._translate(*self.parse_language_codes(from_lang, to_lang, fatal=True), queries, ctx=ctx)

            # Strict validation: translation count must match query count
            if len(_translations) != len(queries):
                error_msg = f"Translation count mismatch: expected {len(queries)}, got {len(_translations)}"
                self.logger.error(error_msg)
                self.logger.error(f"Queries: {queries}")
                self.logger.error(f"Translations: {_translations}")
                raise InvalidServerResponse(error_msg)

            # Only overwrite yet untranslated indices
            for j in untranslated_indices:
                translations[j] = _translations[j]

            if self._INVALID_REPEAT_COUNT == 0:
                break

            new_untranslated_indices = []
            for j in untranslated_indices:
                q, t = queries[j], translations[j]
                # Repeat invalid translations with slightly modified queries
                if self._is_translation_invalid(q, t):
                    new_untranslated_indices.append(j)
                    queries[j] = self._modify_invalid_translation_query(q, t)
            untranslated_indices = new_untranslated_indices

            if not untranslated_indices:
                break

        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]

        if to_lang == 'ARA':
            import arabic_reshaper , bidi.algorithm
            translations = [bidi.algorithm.get_display(arabic_reshaper.reshape(t)) for t in translations]

        if use_mtpe:
            translations = await self.mtpe_adapter.dispatch(queries, translations)

        # Merge with the queries without text
        for i, trans in enumerate(translations):
            final_translations[query_indices[i]] = trans
            self.logger.info(f'{i}: {queries[i]} => {trans}')

        return final_translations

    @abstractmethod
    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        pass

    async def _ratelimit_sleep(self):
        if self._MAX_REQUESTS_PER_MINUTE > 0:
            now = time.time()
            ratelimit_timeout = self._last_request_ts + 60 / self._MAX_REQUESTS_PER_MINUTE
            if ratelimit_timeout > now:
                self.logger.info(f'Ratelimit sleep: {(ratelimit_timeout-now):.2f}s')
                await asyncio.sleep(ratelimit_timeout-now)
            self._last_request_ts = time.time()

    def _is_translation_invalid(self, query: str, trans: str) -> bool:
        if not trans and query:
            return True
        if not query or not trans:
            return False

        query_symbols_count = len(set(query))
        trans_symbols_count = len(set(trans))
        if query_symbols_count > 6 and trans_symbols_count < 6 and trans_symbols_count < 0.25 * len(trans):
            return True
        return False

    def _modify_invalid_translation_query(self, query: str, trans: str) -> str:
        """
        Can be overwritten if _INVALID_REPEAT_COUNT was set. It modifies the query
        for the next translation attempt.
        """
        return query

    def _clean_translation_output(self, query: str, trans: str, to_lang: str) -> str:
        """
        Tries to spot and skim down invalid translations.
        """
        if not query or not trans:
            return ''

        # 移除内部标记：【Original regions: X】或 [Original regions: X]
        # Remove internal markers: 【Original regions: X】 or [Original regions: X]
        trans = re.sub(r'【Original regions:\s*\d+】\s*', '', trans, flags=re.IGNORECASE)
        trans = re.sub(r'\[Original regions:\s*\d+\]\s*', '', trans, flags=re.IGNORECASE)
        
        # 替换全角句点连续出现（．．．或．．）为省略号
        trans = trans.replace('．．．', '…')
        trans = trans.replace('．．', '…')

        # '  ' -> ' '
        trans = re.sub(r'\s+', r' ', trans)
        # 'text.text' -> 'text. text'
        trans = re.sub(r'(?<![.,;!?])([.,;!?])(?=\w)', r'\1 ', trans)
        # ' ! ! . . ' -> ' !!.. '
        trans = re.sub(r'([.,;!?])\s+(?=[.,;!?]|$)', r'\1', trans)

        if to_lang != 'ARA':
            # 'text .' -> 'text.'
            trans = re.sub(r'(?<=[.,;!?\w])\s+([.,;!?])', r'\1', trans)
            # ' ... text' -> ' ...text'
            trans = re.sub(r'((?:\s|^)\.+)\s+(?=\w)', r'\1', trans)

        seq = repeating_sequence(trans.lower())

        # 'aaaaaaaaaaaaa' -> 'aaaaaa'
        if len(trans) < len(query) and len(seq) < 0.5 * len(trans):
            # Shrink sequence to length of original query
            trans = seq * max(len(query) // len(seq), 1)
            # Transfer capitalization of query to translation
            nTrans = ''
            for i in range(min(len(trans), len(query))):
                nTrans += trans[i].upper() if query[i].isupper() else trans[i]
            trans = nTrans

        # words = text.split()
        # elements = list(set(words))
        # if len(elements) / len(words) < 0.1:
        #     words = words[:int(len(words) / 1.75)]
        #     text = ' '.join(words)

        #     # For words that appear more then four times consecutively, remove the excess
        #     for el in elements:
        #         el = re.escape(el)
        #         text = re.sub(r'(?: ' + el + r'){4} (' + el + r' )+', ' ', text)

        return trans

class OfflineTranslator(CommonTranslator, ModelWrapper):
    _MODEL_SUB_DIR = 'translators'

    async def _translate(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, from_lang: str, to_lang: str, queries: List[str]) -> List[str]:
        pass

    async def load(self, from_lang: str, to_lang: str, device: str):
        return await super().load(device, *self.parse_language_codes(from_lang, to_lang))

    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def reload(self, from_lang: str, to_lang: str, device: str):
        return await super().reload(device, from_lang, to_lang)
    
    @abstractmethod
    async def _load(self, from_lang: str, to_lang: str, device: str):
        pass

    async def unload(self, device: str):
        return await super().unload()

def sanitize_text_encoding(text: str) -> str:
    """
    统一的文本编码清理函数，处理各种编码问题
    Unified text encoding sanitization to handle various encoding issues
    
    Args:
        text: 输入文本
        
    Returns:
        清理后的文本
    """
    if not text:
        return text
    
    try:
        # 1. 尝试检测并修复UTF-16-LE编码问题
        # 如果文本包含UTF-16-LE的BOM或特征，尝试重新解码
        if isinstance(text, bytes):
            # 如果是bytes，尝试多种编码
            for encoding in ['utf-8', 'utf-16-le', 'utf-16-be', 'latin-1']:
                try:
                    text = text.decode(encoding, errors='ignore')
                    break
                except (UnicodeDecodeError, AttributeError):
                    continue
        
        # 2. 移除不可见的控制字符和损坏的字符
        # 保留常用的控制字符：换行(\n)、回车(\r)、制表符(\t)
        import unicodedata
        cleaned = []
        for char in text:
            # 跳过控制字符（除了\n, \r, \t）
            if unicodedata.category(char)[0] == 'C' and char not in '\n\r\t':
                continue
            # 跳过私有使用区字符（可能是损坏的编码）
            if '\uE000' <= char <= '\uF8FF':  # 私有使用区
                continue
            # 跳过替换字符（表示解码失败）
            if char == '\ufffd':
                continue
            cleaned.append(char)
        
        text = ''.join(cleaned)
        
        # 3. 修复常见的编码混淆问题
        # UTF-16-LE误识别为UTF-8时会产生的特征模式
        # 例如：每个字符后跟\x00
        if '\x00' in text:
            text = text.replace('\x00', '')
        
        # 4. 确保文本是有效的UTF-8
        # 通过编码再解码来验证和清理
        text = text.encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
        
        return text
        
    except Exception as e:
        import logging
        logger = logging.getLogger('manga_translator')
        logger.warning(f"文本编码清理失败: {e}，返回原文本")
        # 如果清理失败，至少移除明显的问题字符
        if isinstance(text, str):
            return text.replace('\ufffd', '').replace('\x00', '')
        return str(text)

def strip_known_llm_injected_noise(text: str) -> Tuple[str, int]:
    """
    清理代理/网关注入的广告或脏尾巴行，返回(清理后文本, 移除行数)。
    """
    if not text:
        return text, 0

    noise_substrings = (
        'upgrade your plan',
        'proxies cheaper than the market',
        'proxy cheaper than the market',
        'remove this mesage',
        'remove this message',
        'api.airforce',
        'discord.g/airforce',
        'discord.gg/airforce',
        'op.wtf',
        'htps:/',
        'https:/',
    )

    filtered_lines = []
    removed = 0
    for line in str(text).splitlines():
        lowered = line.strip().lower()
        if lowered and any(token in lowered for token in noise_substrings):
            removed += 1
            continue
        filtered_lines.append(line)

    return '\n'.join(filtered_lines).strip(), removed

def extract_json_payload_from_mixed_text(text: str) -> Tuple[str, bool]:
    """
    通用 JSON 提取器：从混杂文本中优先提取最可能的 JSON 负载。
    返回 (payload, extracted)，extracted=True 表示确实抽取到了 JSON 片段。
    """
    import re
    import json

    if text is None:
        return "", False

    raw = str(text).strip()
    if not raw:
        return raw, False

    def _extract_balanced_json_candidates(src: str) -> List[str]:
        candidates = []
        stack = []
        start = -1
        in_str = False
        escape = False
        for i, ch in enumerate(src):
            if in_str:
                if escape:
                    escape = False
                elif ch == '\\':
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
                continue
            if ch in '{[':
                if not stack:
                    start = i
                stack.append(ch)
            elif ch in '}]' and stack:
                top = stack[-1]
                if (top == '{' and ch == '}') or (top == '[' and ch == ']'):
                    stack.pop()
                    if not stack and start >= 0:
                        candidates.append(src[start:i + 1].strip())
                        start = -1
                else:
                    stack = []
                    start = -1
        return candidates

    def _is_json_parseable(candidate: str) -> bool:
        try:
            json.loads(candidate)
            return True
        except Exception:
            try:
                import json5
                json5.loads(candidate)
                return True
            except Exception:
                return False

    candidates: List[str] = []

    # 1) fenced code block candidates
    fenced = re.findall(r'`{3,}\s*(?:json)?\s*([\s\S]*?)`{3,}', raw, flags=re.IGNORECASE)
    for c in fenced:
        c = c.strip()
        if c:
            candidates.append(c)

    # 2) balanced-json candidates from full text
    candidates.extend(_extract_balanced_json_candidates(raw))

    # 3) fallback: from first bracket onward
    first_bracket = raw.find('[')
    first_brace = raw.find('{')
    json_start = -1
    if first_bracket != -1 and first_brace != -1:
        json_start = min(first_bracket, first_brace)
    elif first_bracket != -1:
        json_start = first_bracket
    elif first_brace != -1:
        json_start = first_brace
    if json_start >= 0:
        candidates.append(raw[json_start:].strip())

    # dedup keep-order
    dedup: List[str] = []
    seen = set()
    for c in candidates:
        key = c.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        dedup.append(key)

    if not dedup:
        return raw, False

    preferred_markers = ('"translations"', "'translations'", '"translation"', '"id"', '"new_terms"', '"glossary"')

    best_candidate = None
    best_score = (-1, -1)
    for c in dedup:
        low = c.lower()
        marker_score = sum(1 for m in preferred_markers if m in low)
        parseable_score = 1 if _is_json_parseable(c) else 0
        score = (parseable_score * 10 + marker_score, len(c))
        if score > best_score:
            best_score = score
            best_candidate = c

    if best_candidate is None:
        return raw, False
    return best_candidate, True

def parse_hq_response(result_text: str) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    专门解析HQ翻译器的响应，支持提取翻译和新术语
    Parse HQ translator response, supporting extraction of translations and new terms
    
    Returns:
        (translations, new_terms)
    """
    import json
    import logging
    import re
    
    logger = logging.getLogger('manga_translator')
    
    # 统一的编码清理
    result_text = sanitize_text_encoding(result_text)
    extracted_payload, extracted = extract_json_payload_from_mixed_text(result_text)
    if extracted:
        result_text = extracted_payload
        logger.info("Extracted JSON payload from mixed response text")
    
    _original_text = result_text # Keep for logging
    result_text = result_text.strip()
    if not result_text:
        return [], []

    # 1. 清理Markdown
    if "```" in result_text:
        code_block_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result_text, re.DOTALL)
        if code_block_match:
            result_text = code_block_match.group(1).strip()
        else:
            lines = result_text.split('\n')
            if lines[0].strip().startswith("```"): lines = lines[1:]
            if lines and lines[-1].strip() == "```": lines = lines[:-1]
            result_text = "\n".join(lines).strip()

    # 2. 查找JSON起始 (清理前缀)
    first_bracket = result_text.find('[')
    first_brace = result_text.find('{')
    json_start = -1
    if first_bracket != -1 and first_brace != -1: json_start = min(first_bracket, first_brace)
    elif first_bracket != -1: json_start = first_bracket
    elif first_brace != -1: json_start = first_brace
    
    if json_start > 0:
        result_text = result_text[json_start:].strip()

    translations = []
    new_terms = []
    
    parsed_json = None
    
    # === 策略1: 标准 JSON 解析 ===
    try:
        parsed_json = json.loads(result_text)
    except json.JSONDecodeError:
        # === 策略2: 宽松 JSON5 解析 ===
        try:
            import json5
            parsed_json = json5.loads(result_text)
            logger.info("Using json5 for parsing")
        except (ImportError, Exception):
            parsed_json = None

    # 如果JSON解析成功，提取数据
    if parsed_json is not None:
        try:
            # 情况1: Object format {"translations": [...], "new_terms": [...]}
            if isinstance(parsed_json, dict):
                # 提取翻译
                trans_list = parsed_json.get("translations")
                if not trans_list and "t" in parsed_json: trans_list = parsed_json.get("t") # 兼容简写
                if not trans_list: trans_list = [] # 确保不为None

                if isinstance(trans_list, list):
                    if trans_list and isinstance(trans_list[0], dict):
                        for item in trans_list:
                            text = item.get('translation') or item.get('text') or list(item.values())[0]
                            translations.append(str(text) if text is not None else "")
                    else:
                        translations = [str(x) for x in trans_list]
                
                # 提取术语
                terms_list = parsed_json.get("new_terms") or parsed_json.get("glossary")
                if isinstance(terms_list, list):
                    new_terms = terms_list
            
            # 情况2: Array format [{"id":..., "translation":...}]
            elif isinstance(parsed_json, list):
                if parsed_json:
                    if isinstance(parsed_json[0], dict):
                        for item in parsed_json:
                            text = item.get('translation') or item.get('text') or list(item.values())[0]
                            translations.append(str(text) if text is not None else "")
                    else:
                        translations = [str(x) for x in parsed_json]
            
            return translations, new_terms

        except Exception as e:
             logger.warning(f"JSON structure parsing failed, falling back to regex: {e}")

    # === 策略3: 正则表达式暴力提取 ===
    logger.warning("JSON parsing failed, falling back to Regex extraction")
    
    # 3.1 尝试提取带ID的对象: {"id": 1, "translation": "..."}
    object_pattern = r'\{\s*"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"\s*\}'
    matches = re.findall(object_pattern, result_text)
    
    if matches:
        logger.info(f"Regex extracted {len(matches)} translations with IDs")
        translations = [match[1].replace('\\"', '"').replace('\\n', '\n') for match in matches]
        
        # 尝试提取术语 (简单正则)
        # 假设 new_terms 在后面，格式类似 {"original": "...", ...}
        # 这里的正则很难完美匹配嵌套结构，只能尽力而为
        term_pattern = r'\{\s*"original"\s*:\s*"([^"]+)"\s*,\s*"translation"\s*:\s*"([^"]+)"\s*,\s*"category"\s*:\s*"([^"]+)"\s*\}'
        term_matches = re.findall(term_pattern, result_text)
        for tm in term_matches:
            new_terms.append({"original": tm[0], "translation": tm[1], "category": tm[2]})
            
        return translations, new_terms

    # 3.2 尝试只提取 translation 字段
    translation_pattern = r'"translation"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
    matches = re.findall(translation_pattern, result_text)
    if matches:
         logger.warning(f"Regex extracted {len(matches)} translations (no IDs)")
         translations = [match.replace('\\"', '"').replace('\\n', '\n') for match in matches]
         return translations, []

    # 3.3 最后的兜底：按行分割 (仅当不像JSON时)
    if not result_text.startswith('{') and not result_text.startswith('['):
         for line in result_text.split('\n'):
            line = line.strip()
            if line:
                line = re.sub(r'^\d+\.\s*', '', line)
                line = line.replace('\\n', '\n').replace('↵', '\n')
                translations.append(line)

    return translations, new_terms

def parse_json_or_text_response(result_text: str) -> List[str]:
    """
    解析LLM返回的文本，支持JSON列表格式或按行分割格式
    Wrapper around parse_hq_response for backward compatibility
    """
    translations, _ = parse_hq_response(result_text)
    return translations


def get_custom_prompt_content(file_path: str) -> str:
    """
    读取自定义提示词文件内容
    Read custom prompt file content
    """
    import os
    try:
        if not os.path.exists(file_path):
            return ""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading prompt file {file_path}: {e}")
        return ""

def save_custom_prompt_content(file_path: str, content: str) -> bool:
    """
    保存自定义提示词文件内容
    Save custom prompt file content
    """
    import os
    try:
        # 确保存储目录存在
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"Error saving prompt file {file_path}: {e}")
        return False

def merge_glossary_to_file(file_path: str, new_terms: List[Dict[str, str]]) -> bool:
    """
    将新提取的术语合并到提示词文件中
    Merge newly extracted terms into the prompt file
    
    Args:
        file_path: 提示词文件路径
        new_terms: 新术语列表 [{"original": "...", "translation": "...", "category": "..."}]
    """
    import json
    import os
    
    logger.debug(f"[Glossary] 尝试合并 {len(new_terms) if new_terms else 0} 个术语到: {file_path}")

    # Filter out terms whose translations are not Chinese (must contain at least one Han char).
    filtered_terms: List[Dict[str, str]] = []
    skipped_non_chinese = 0
    for term in (new_terms or []):
        if not isinstance(term, dict):
            continue
        translation = str(term.get("translation") or "").strip()
        if translation and contains_han_chars(translation):
            filtered_terms.append(term)
        else:
            skipped_non_chinese += 1

    if skipped_non_chinese > 0:
        logger.info(f"[Glossary] Skipped {skipped_non_chinese} terms due to non-Chinese translation (no Han chars)")

    new_terms = filtered_terms

    if not new_terms:
        logger.debug(f"[Glossary] No valid terms to merge after filtering")
        return False

    try:
        # 读取现有文件
        data = {}
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    pass # 如果文件损坏或为空，从头开始
        
        # 确保结构完整
        if "glossary" not in data or not isinstance(data["glossary"], dict):
            # 如果旧格式是列表，或者没有 glossary，初始化为新的分类结构
            data["glossary"] = {
                "Person": [], "Location": [], "Org": [], "Item": [], "Skill": [], "Creature": []
            }
        
        glossary = data["glossary"]
        # 确保所有标准分类键都存在
        valid_keys_map = {
            "person": "Person", 
            "location": "Location", 
            "org": "Org", 
            "organization": "Org",
            "item": "Item", 
            "skill": "Skill", 
            "creature": "Creature"
        }
        
        # 确保标准键存在于 glossary 中
        for key in set(valid_keys_map.values()):
            if key not in glossary:
                glossary[key] = []

        # 如果启用了 works 分作品结构，允许按作品写回（term 需包含 work_name 字段）
        works = glossary.get("works") if isinstance(glossary.get("works"), dict) else None

        # 无作品区（仅存储，不参与运行时应用）
        unassigned = glossary.get("unassigned") if isinstance(glossary.get("unassigned"), dict) else None
        if unassigned is None:
            unassigned = {}
            glossary["unassigned"] = unassigned

        def _ensure_list(container: dict, key: str) -> list:
            v = container.get(key)
            if not isinstance(v, list):
                v = []
                container[key] = v
            return v

        def _find_term(term_list: list, original_text: str):
            """按 original 查找术语条目，找不到则返回 None。"""
            for existing_term in term_list:
                if isinstance(existing_term, dict) and existing_term.get("original") == original_text:
                    return existing_term
            return None

        def _ensure_work_entry(work_name: str) -> dict:
            """确保 works[work_name] 存在且包含所有标准分类键。"""
            nonlocal works
            if works is None:
                glossary["works"] = {}
                works = glossary["works"]
            entry = works.get(work_name)
            if not isinstance(entry, dict):
                entry = {}
                works[work_name] = entry
            for k in set(valid_keys_map.values()):
                _ensure_list(entry, k)
            return entry

        def _ensure_unassigned_entry() -> dict:
            """确保 glossary.unassigned 存在且包含所有标准分类键。"""
            entry = unassigned
            if not isinstance(entry, dict):
                entry = {}
                glossary["unassigned"] = entry
            for k in set(valid_keys_map.values()):
                _ensure_list(entry, k)
            return entry

        modified = False

        # ✅ 自动迁移：当 raw_work_name 可映射到熟肉名时，直接写入 works（避免长期留在 unassigned）
        try:
            from ..utils.work_resolver import load_name_mapping, try_map_translated_work_name
            _name_mapping = load_name_mapping()
            _name_mapping_values = set(_name_mapping.values())
        except Exception:
            _name_mapping = {}
            _name_mapping_values = set()

        def _resolve_work_from_raw_name(raw_work: Optional[str]) -> Optional[str]:
            raw = (raw_work or "").strip()
            if not raw:
                return None

            # raw 直接就是作品 key（用户已用熟肉名建 works）
            if works is not None and raw in works:
                return raw

            if _name_mapping:
                mapped = try_map_translated_work_name(raw, _name_mapping)
                if mapped:
                    return mapped
                # raw 本身是熟肉名（出现在映射 value 中）
                if raw in _name_mapping_values:
                    return raw

            return None

        for term in new_terms:
            raw_category = term.get("category", "Item")
            original = term.get("original")
            translation = term.get("translation")

            if not original or not translation:
                continue

            # 作品名（可选）：用于写回 works
            work_name = term.get("work_name") or term.get("work") or term.get("workName")
            if isinstance(work_name, str):
                work_name = work_name.strip() or None
            else:
                work_name = None

            # 生肉名（可选）：用于自动迁移（raw_work_name -> work_name）
            raw_work_name = term.get("raw_work_name") or term.get("rawWorkName")
            if isinstance(raw_work_name, str):
                raw_work_name = raw_work_name.strip() or None
            else:
                raw_work_name = None

            # ✅ 若未提供 work_name，但 raw_work_name 能映射到熟肉名，则直接写入 works
            if not work_name and raw_work_name:
                mapped_work = _resolve_work_from_raw_name(raw_work_name)
                if mapped_work:
                    work_name = mapped_work

            # 映射 Category 到标准 Key
            target_key = "Item"  # Default fallback
            if raw_category:
                normalized_cat = str(raw_category).lower()
                if normalized_cat in valid_keys_map:
                    target_key = valid_keys_map[normalized_cat]
                else:
                    # 尝试模糊匹配或直接使用 Title Case
                    for k in valid_keys_map.values():
                        if k.lower() == normalized_cat:
                            target_key = k
                            break

            # 1) 写回到指定作品（如果 work_name 有效）
            if work_name:
                work_entry = _ensure_work_entry(work_name)
                work_list = _ensure_list(work_entry, target_key)
                existing = _find_term(work_list, original)
                logger.debug(f"[Glossary] 检查术语: work={work_name}, key={target_key}, original={original}, existing={existing is not None}")
                if not existing:
                    work_list.append({
                        "original": original,
                        "translation": translation
                    })
                    modified = True
                    logger.info(f"[Glossary] 添加术语 (work={work_name}): {original} -> {translation} ({target_key})")

                # 同时维护根级合并结构（向后兼容）：仅合并“作品区”术语（不包含 unassigned）
                root_list = _ensure_list(glossary, target_key)
                root_existing = _find_term(root_list, original)
                if not root_existing:
                    root_list.append({
                        "original": original,
                        "translation": translation
                    })
                    modified = True
                    logger.debug(f"[Glossary] 添加到根级: {original} -> {translation} ({target_key})")

            # 2) 无作品区：仅存储到 glossary.unassigned，不写入根级
            else:
                un_entry = _ensure_unassigned_entry()
                un_list = _ensure_list(un_entry, target_key)

                existing = _find_term(un_list, original)

                if existing:
                    # ✅ 如果历史条目缺少 raw_work_name，而新条目携带，则补齐（避免 UI 显示“无法确定”且无法匹配）
                    existing_raw = (existing.get("raw_work_name") or "").strip()
                    existing_translation = (existing.get("translation") or "").strip()
                    if raw_work_name and not existing_raw and existing_translation == translation:
                        existing["raw_work_name"] = raw_work_name
                        modified = True
                else:
                    # 构建术语条目（包含 raw_work_name 用于匹配）
                    term_entry = {
                        "original": original,
                        "translation": translation
                    }
                    # 添加 raw_work_name（如果存在）
                    if raw_work_name:
                        term_entry["raw_work_name"] = raw_work_name
                    un_list.append(term_entry)
                    modified = True
                    raw_info = f", raw_work={raw_work_name}" if raw_work_name else ""
                    logger.info(f"[Glossary] 添加术语 (unassigned{raw_info}): {original} -> {translation} ({target_key})")
        
        # ✅ 额外：把无作品区里 raw_work_name 可映射的条目移动到 works
        moved_from_unassigned = 0
        try:
            if isinstance(unassigned, dict):
                for cat_key in set(valid_keys_map.values()):
                    src_list = _ensure_list(unassigned, cat_key)
                    if not src_list:
                        continue

                    new_src = []
                    for t in src_list:
                        if not isinstance(t, dict):
                            new_src.append(t)
                            continue

                        o = (t.get("original") or "").strip()
                        tr = (t.get("translation") or "").strip()
                        if not o or not tr:
                            new_src.append(t)
                            continue

                        # Avoid migrating (and thus persisting) non-Chinese translations.
                        if not contains_han_chars(tr):
                            new_src.append(t)
                            continue

                        rawn = (t.get("raw_work_name") or "").strip()
                        target_work = _resolve_work_from_raw_name(rawn)
                        if not target_work:
                            new_src.append(t)
                            continue

                        work_entry = _ensure_work_entry(target_work)
                        work_list = _ensure_list(work_entry, cat_key)
                        if not _find_term(work_list, o):
                            entry = {"original": o, "translation": tr}
                            if isinstance(t.get("lang"), str) and t.get("lang").strip():
                                entry["lang"] = t.get("lang").strip()
                            work_list.append(entry)
                            modified = True
                            moved_from_unassigned += 1

                        # 同时维护根级合并结构（向后兼容）
                        root_list = _ensure_list(glossary, cat_key)
                        if not _find_term(root_list, o):
                            root_entry = {"original": o, "translation": tr}
                            if isinstance(t.get("lang"), str) and t.get("lang").strip():
                                root_entry["lang"] = t.get("lang").strip()
                            root_list.append(root_entry)
                            modified = True

                    unassigned[cat_key] = new_src
        except Exception:
            pass

        if moved_from_unassigned > 0:
            logger.info(f"[Glossary] 自动迁移 {moved_from_unassigned} 个 unassigned 术语到 works")

        if modified:
            # 确保存储目录存在
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"[Glossary] 文件已保存: {file_path}")
            return True
            
    except Exception as e:
        logger.error(f"[Glossary] 合并术语失败: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    # 没有修改（所有术语已存在）
    logger.debug(f"[Glossary] 未修改文件（术语已存在或无效）: {file_path}")
    return False

def get_glossary_extraction_prompt(target_lang: str) -> str:
    """
    获取术语提取的追加提示词
    Get the additional prompt for glossary extraction
    """
    from ..utils import BASE_PATH
    import os
    import json
    
    try:
        prompt_path = os.path.join(BASE_PATH, 'dict', 'glossary_extraction_prompt.json')
        if not os.path.exists(prompt_path):
            return ""
            
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        prompt = data.get('glossary_extraction_prompt', '')
        if prompt:
            # Replace placeholder with target language
            # We assume the caller passes the full language name if possible, or we map it here if needed
            # For simplicity, we just use what's passed
            prompt = prompt.replace("{{{target_lang}}}", target_lang)
            return prompt
    except Exception as e:
        print(f"Error loading glossary extraction prompt: {e}")
        return ""
