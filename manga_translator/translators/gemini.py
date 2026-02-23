import os
# import re
import asyncio
# import json
from typing import List, Dict, Any
from google import genai
from google.genai import types

from .common import CommonTranslator, VALID_LANGUAGES, parse_json_or_text_response, parse_hq_response, get_glossary_extraction_prompt, merge_glossary_to_file, validate_gemini_response
from .keys import GEMINI_API_KEY
from ..utils import Context

# 浏览器风格的请求头，避免被 CF 拦截
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7",
    "Origin": "https://aistudio.google.com",
    "Referer": "https://aistudio.google.com/",
}


def _flatten_prompt_data(data: Any, indent: int = 0) -> str:
    """Recursively flattens a dictionary or list into a formatted string."""
    prompt_parts = []
    prefix = "  " * indent

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, (dict, list)):
                prompt_parts.append(f"{prefix}- {key}:")
                prompt_parts.append(_flatten_prompt_data(value, indent + 1))
            else:
                prompt_parts.append(f"{prefix}- {key}: {value}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, (dict, list)):
                prompt_parts.append(_flatten_prompt_data(item, indent + 1))
            else:
                prompt_parts.append(f"{prefix}- {item}")
    
    return "\n".join(prompt_parts)

class GeminiTranslator(CommonTranslator):
    """
    Gemini纯文本翻译器
    支持批量文本翻译，不包含图片处理
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    
    # 类变量: 跨实例共享的RPM限制时间戳
    _GLOBAL_LAST_REQUEST_TS = {}  # {model_name: timestamp}
    
    def __init__(self):
        super().__init__()
        self.client = None
        self.prev_context = ""  # 用于存储多页上下文
        # Initial setup from environment variables
        # 只在非Web环境下重新加载.env文件
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        
        self.api_key = os.getenv('GEMINI_API_KEY', GEMINI_API_KEY)
        self.base_url = os.getenv('GEMINI_API_BASE', 'https://generativelanguage.googleapis.com')
        self.model_name = os.getenv('GEMINI_MODEL', "gemini-1.5-flash")
        self.max_tokens = None  # 不限制，使用模型默认最大值
        self.temperature = 0.1
        self._MAX_REQUESTS_PER_MINUTE = 0  # 默认无限制
        # 使用全局时间戳,跨实例共享
        if self.model_name not in GeminiTranslator._GLOBAL_LAST_REQUEST_TS:
            GeminiTranslator._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
        self._last_request_ts_key = self.model_name
        # 新版 SDK 的安全设置
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
        self._setup_client()
    
    def set_prev_context(self, context: str):
        """设置多页上下文（用于context_size > 0时）"""
        self.prev_context = context if context else ""
    
    def parse_args(self, args):
        """解析配置参数"""
        # 调用父类的 parse_args 来设置通用参数（包括 attempts、post_check 等）
        super().parse_args(args)
        
        # 同步 attempts 到 _max_total_attempts
        self._max_total_attempts = self.attempts
        
        # 从配置中读取RPM限制
        max_rpm = getattr(args, 'max_requests_per_minute', 0)
        if max_rpm > 0:
            self._MAX_REQUESTS_PER_MINUTE = max_rpm
            self.logger.info(f"Setting Gemini max requests per minute to: {max_rpm}")
        
        # 从配置中读取用户级 API Key（优先于环境变量）
        # 这允许 Web 服务器为每个用户使用不同的 API Key
        need_rebuild_client = False
        
        user_api_key = getattr(args, 'user_api_key', None)
        if user_api_key and user_api_key != self.api_key:
            self.api_key = user_api_key
            need_rebuild_client = True
            self.logger.info("[UserAPIKey] Using user-provided API key for Gemini")
        
        user_api_base = getattr(args, 'user_api_base', None)
        if user_api_base and user_api_base != self.base_url:
            self.base_url = user_api_base
            need_rebuild_client = True
            self.logger.info(f"[UserAPIKey] Using user-provided API base: {user_api_base}")
        
        user_api_model = getattr(args, 'user_api_model', None)
        if user_api_model:
            self.model_name = user_api_model
            # 更新全局时间戳的 key
            if self.model_name not in GeminiTranslator._GLOBAL_LAST_REQUEST_TS:
                GeminiTranslator._GLOBAL_LAST_REQUEST_TS[self.model_name] = 0
            self._last_request_ts_key = self.model_name
            self.logger.info(f"[UserAPIKey] Using user-provided model: {user_api_model}")
        
        # 如果 API Key 或 Base URL 变化，重建客户端
        if need_rebuild_client:
            self.client = None
            self._setup_client()
    
    def _setup_client(self, system_instruction=None):
        """设置Gemini客户端"""
        if not self.client and self.api_key:
            # 新版 SDK 使用 genai.Client 初始化
            # 检查是否使用自定义 API Base
            is_custom_api = (
                self.base_url 
                and self.base_url.strip() 
                and self.base_url.strip() not in ["https://generativelanguage.googleapis.com", "https://generativelanguage.googleapis.com/"]
            )
            
            if is_custom_api:
                # 使用自定义 API Base（通过 http_options）
                self.client = genai.Client(
                    api_key=self.api_key,
                    http_options=types.HttpOptions(
                        base_url=self.base_url,
                        headers=BROWSER_HEADERS
                    )
                )
                self.logger.info(f"Gemini客户端初始化完成（自定义API Base）。Base URL: {self.base_url}")
            else:
                # 使用官方 API
                self.client = genai.Client(api_key=self.api_key)
                self.logger.info("Gemini客户端初始化完成。使用官方API")
            
            self.logger.info("安全设置策略：默认发送 OFF，如遇错误自动回退")

    async def _abort_inflight_request(self):
        """Cancel: try to close the current client connection to interrupt blocking requests."""
        if not self.client:
            return

        close_fn = getattr(self.client, "close", None)
        try:
            if callable(close_fn):
                close_result = close_fn()
                if asyncio.iscoroutine(close_result):
                    await close_result
        except Exception as e:
            self.logger.debug(f"中断Gemini请求时关闭客户端失败（可忽略）: {e}")
        finally:
            self.client = None
    
    def _build_system_prompt(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """构建系统提示词"""
        # Map language codes to full names for clarity in the prompt
        lang_map = {
            "CHS": "Simplified Chinese",
            "CHT": "Traditional Chinese",
            "JPN": "Japanese",
            "ENG": "English",
            "KOR": "Korean",
            "VIN": "Vietnamese",
            "FRA": "French",
            "DEU": "German",
            "ITA": "Italian",
        }
        target_lang_full = lang_map.get(target_lang, target_lang)

        custom_prompt_str = ""
        if custom_prompt_json:
            custom_prompt_str = _flatten_prompt_data(custom_prompt_json)

        line_break_prompt_str = ""
        if line_break_prompt_json and line_break_prompt_json.get('line_break_prompt'):
            line_break_prompt_str = line_break_prompt_json['line_break_prompt']

        # 尝试加载 HQ System Prompt，如果失败或不需要（比如既没custom也没glossary），是否回退到简单prompt？
        # 为了统一体验，只要能加载到 system_prompt_hq.json 就用它。
        base_prompt = ""
        try:
            from ..utils import BASE_PATH
            import os
            import json
            prompt_path = os.path.join(BASE_PATH, 'dict', 'system_prompt_hq.json')
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    base_prompt_data = json.load(f)
                base_prompt = base_prompt_data['system_prompt']
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from file: {e}")

        # 如果没加载到 HQ Prompt，使用简单的 fallback
        if not base_prompt:
             base_prompt = f"""You are an expert manga translator. Translate from {source_lang} to {target_lang}. Output only the translation."""

        # Replace placeholder with the full language name
        base_prompt = base_prompt.replace("{{{target_lang}}}", target_lang_full)
        
        # Also replace target_lang placeholder in custom prompt
        if custom_prompt_str:
            custom_prompt_str = custom_prompt_str.replace("{{{target_lang}}}", target_lang_full)

        # Combine prompts
        final_prompt = ""
        
        # 添加重试提示到最前面（如果是重试）
        if retry_attempt > 0:
            final_prompt += self._get_retry_hint(retry_attempt, retry_reason) + "\n"
        
        if line_break_prompt_str:
            final_prompt += f"{line_break_prompt_str}\n\n---\n\n"
        if custom_prompt_str:
            final_prompt += f"{custom_prompt_str}\n\n---\n\n"
        
        final_prompt += base_prompt
        
        # 追加术语提取提示词
        if extract_glossary:
            extraction_prompt = get_glossary_extraction_prompt(target_lang_full)
            if extraction_prompt:
                final_prompt += f"\n\n---\n\n{extraction_prompt}"
                self.logger.info("已启用自动术语提取，提示词已追加。")
        
        return final_prompt

    def _build_user_prompt(self, texts: List[str], ctx: Any, retry_attempt: int = 0, retry_reason: str = "") -> str:
        """构建用户提示词（纯文本版）- 使用 JSON 格式以配合 HQ Prompt"""
        return self._build_user_prompt_for_texts(texts, ctx, self.prev_context, retry_attempt=retry_attempt, retry_reason=retry_reason)
    
    def _get_system_instruction(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """获取完整的系统指令"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """批量翻译方法（纯文本）"""
        if not texts:
            return []
        
        if not self.client:
            self._setup_client()
        
        if not self.client:
            self.logger.error("Gemini客户端初始化失败")
            return texts
        
        # 初始化重试信息
        retry_attempt = 0
        retry_reason = ""
        
        # 保存参数供重试时使用
        _source_lang = source_lang
        _target_lang = target_lang
        _custom_prompt_json = custom_prompt_json
        _line_break_prompt_json = line_break_prompt_json
        
        # 发送请求
        max_retries = self.attempts
        attempt = 0
        is_infinite = max_retries == -1
        last_exception = None
        local_attempt = 0  # 本次批次的尝试次数
        
        # 标记是否需要回退（不发送安全设置）
        should_retry_without_safety = False

        while is_infinite or attempt < max_retries:
            # 检查是否被取消
            self._check_cancelled()
            
            # 检查全局尝试次数
            if not self._increment_global_attempt():
                self.logger.error("Reached global attempt limit. Stopping translation.")
                last_error_msg = str(last_exception) if last_exception else "Unknown error"
                raise Exception(f"达到最大尝试次数 ({self._max_total_attempts})，最后一次错误: {last_error_msg}")

            local_attempt += 1
            attempt += 1

            # 确定是否开启术语提取
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)
            
            extract_glossary = bool(_custom_prompt_json) and config_extract

            # 获取系统指令（不再用于初始化客户端，而是合并到用户消息）
            system_instruction = self._get_system_instruction(_source_lang, _target_lang, custom_prompt_json=_custom_prompt_json, line_break_prompt_json=_line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)
            
            # 初始化客户端（不传入 system_instruction）
            if not self.client:
                self._setup_client(system_instruction=None)
            
            if not self.client:
                self.logger.error("Gemini客户端初始化失败")
                return texts
            
            # 构建用户提示词
            # 如果加载了 HQ Prompt，_build_user_prompt (即 _build_user_prompt_for_texts) 会生成 JSON 格式的输入，与 System Prompt 匹配
            user_prompt = self._build_user_prompt(texts, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)
            
            # 将系统提示词合并到用户消息的开头
            combined_prompt = f"{system_instruction}\n\n{user_prompt}"
            
            # 动态调整温度
            current_temperature = self._get_retry_temperature(self.temperature, retry_attempt, retry_reason)
            
            # 构建生成配置
            generation_config = types.GenerateContentConfig(
                temperature=current_temperature,
                top_p=0.95,
                top_k=64,
                max_output_tokens=self.max_tokens,
                safety_settings=None if should_retry_without_safety else self.safety_settings,
            )

            try:
                # RPM限制
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    now = time.time()
                    delay = 60.0 / self._MAX_REQUESTS_PER_MINUTE
                    elapsed = now - GeminiTranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key]
                    if elapsed < delay:
                        sleep_time = delay - elapsed
                        self.logger.info(f'Ratelimit sleep: {sleep_time:.2f}s')
                        await self._sleep_with_cancel_polling(sleep_time)
                
                if retry_attempt > 0 and current_temperature != self.temperature:
                    self.logger.info(f"[重试] 温度调整: {self.temperature} -> {current_temperature}")
                
                # 使用新版 SDK 的 generate_content 方法
                response = await self._await_with_cancel_polling(
                    asyncio.to_thread(
                        self.client.models.generate_content,
                        model=self.model_name,
                        contents=combined_prompt,
                        config=generation_config
                    ),
                    poll_interval=0.2,
                    on_cancel=self._abort_inflight_request,
                )
                
                if self._MAX_REQUESTS_PER_MINUTE > 0:
                    import time
                    GeminiTranslator._GLOBAL_LAST_REQUEST_TS[self._last_request_ts_key] = time.time()

                # 验证响应对象是否有效
                validate_gemini_response(response, self.logger)

                # 检查finish_reason
                if hasattr(response, 'candidates') and response.candidates:
                    candidate = response.candidates[0]
                    if hasattr(candidate, 'finish_reason'):
                        finish_reason = candidate.finish_reason
                        # 新版 SDK 的 finish_reason 是枚举类型
                        finish_reason_str = str(finish_reason) if finish_reason else ""
                        if "STOP" not in finish_reason_str.upper():  # 不是成功
                            attempt += 1
                            log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                            self.logger.warning(f"Gemini API失败 ({log_attempt}): finish_reason={finish_reason}")
                            if not is_infinite and attempt >= max_retries:
                                break
                            await self._sleep_with_cancel_polling(1)
                            continue

                result_text = response.text.strip()
                
                # 统一的编码清理（处理UTF-16-LE等编码问题）
                from .common import sanitize_text_encoding
                result_text = sanitize_text_encoding(result_text)
                
                self.logger.debug(f"--- Gemini Raw Response ---\n{result_text}\n---------------------------")

                # 使用 parse_hq_response 解析（支持 JSON Object/Array/Text）
                translations, new_terms = parse_hq_response(result_text)
                
                # 处理术语提取（使用缓存，支持并发）
                if extract_glossary:
                    if new_terms:
                        self.logger.info(f"[术语提取] 提取到 {len(new_terms)} 个新术语: {new_terms}")
                        prompt_path = None
                        if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                            prompt_path = getattr(ctx.config.translator, 'high_quality_prompt_path', None)
                        
                        if prompt_path:
                            # ✅ 使用统一函数，自动根据并发数决定写入策略
                            from .glossary_cache import add_glossary_terms
                            added_count = add_glossary_terms(prompt_path, new_terms)
                            if added_count > 0:
                                self.logger.info(f"[术语提取] 已处理 {added_count} 个术语")
                        else:
                            self.logger.warning("[术语提取] 提取到新术语但未找到提示词文件路径")
                    else:
                        self.logger.debug("[术语提取] AI未返回新术语")
                
                # Strict validation: must match input count
                if len(translations) != len(texts):
                    retry_attempt += 1
                    retry_reason = f"Translation count mismatch: expected {len(texts)}, got {len(translations)}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")
                    self.logger.warning(f"Expected texts: {texts}")
                    self.logger.warning(f"Got translations: {translations}")
                    
                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception(f"翻译数量不匹配: 期望 {len(texts)} 条，实际得到 {len(translations)} 条")
                    
                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Translation count mismatch after {max_retries} attempts: expected {len(texts)}, got {len(translations)}")
                    
                    await self._sleep_with_cancel_polling(2)
                    continue

                # 质量验证：检查空翻译、合并翻译、可疑符号等
                is_valid, error_msg = self._validate_translation_quality(texts, translations)
                if not is_valid:
                    retry_attempt += 1
                    retry_reason = f"Quality check failed: {error_msg}"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}. Retrying...")
                    
                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception(f"翻译质量检查失败: {error_msg}")

                    if not is_infinite and attempt >= max_retries:
                        raise Exception(f"Quality check failed after {max_retries} attempts: {error_msg}")

                    await self._sleep_with_cancel_polling(2)
                    continue

                # 打印原文和译文的对应关系
                self.logger.info("--- Translation Results ---")
                for original, translated in zip(texts, translations):
                    self.logger.info(f'{original} -> {translated}')
                self.logger.info("---------------------------")

                # BR检查：检查翻译结果是否包含必要的[BR]标记
                if not self._validate_br_markers(translations, queries=texts, ctx=ctx):
                    retry_attempt += 1
                    retry_reason = "BR markers missing in translations"
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    self.logger.warning(f"[{log_attempt}] {retry_reason}, retrying...")
                    
                    # 记录错误以便在达到最大尝试次数时显示
                    last_exception = Exception("AI断句检查失败: 翻译结果缺少必要的[BR]标记")
                    
                    # 如果达到最大重试次数，抛出友好的异常
                    if not is_infinite and attempt >= max_retries:
                        from .common import BRMarkersValidationException
                        self.logger.error("Gemini翻译在多次重试后仍然失败：AI断句检查失败。")
                        raise BRMarkersValidationException(
                            missing_count=0,  # 具体数字在_validate_br_markers中已记录
                            total_count=len(texts),
                            tolerance=max(1, len(texts) // 10)
                        )
                    await self._sleep_with_cancel_polling(2)
                    continue

                return translations[:len(texts)]

            except Exception as e:
                error_message = str(e)
                last_exception = e  # 保存最后一次错误
                
                # 检查是否是安全设置相关的错误
                is_safety_error = any(keyword in error_message.lower() for keyword in [
                    'safety_settings', 'safetysettings', 'harm', 'block', 'safety'
                ]) or "400" in error_message
                
                # 如果是安全设置错误且还没有尝试回退，则标记回退
                if is_safety_error and not should_retry_without_safety:
                    self.logger.warning(f"检测到安全设置相关错误，将在下次重试时移除安全设置参数: {error_message}")
                    should_retry_without_safety = True
                    # 不增加attempt计数，直接重试
                    await self._sleep_with_cancel_polling(1)
                    continue
                
                attempt += 1
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                self.logger.warning(f"Gemini翻译出错 ({log_attempt}): {e}")

                if "finish_reason: 2" in error_message or "finish_reason is 2" in error_message or "SAFETY" in error_message.upper():
                    self.logger.warning("检测到Gemini安全策略拦截。正在重试...")
                
                # 检查是否达到最大重试次数（注意：attempt已经+1了）
                if not is_infinite and attempt >= max_retries:
                    self.logger.error("Gemini翻译在多次重试后仍然失败。即将终止程序。")
                    raise e
                
                await self._sleep_with_cancel_polling(1)
        
        return texts

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """主翻译方法"""
        if not self.client:
            from .. import manga_translator
            if hasattr(manga_translator, 'config') and hasattr(manga_translator.config, 'translator'):
                self.parse_args(manga_translator.config.translator)

        if not queries:
            return []

        # 重置全局尝试计数器
        self._reset_global_attempt_count()

        self.logger.info(f"使用Gemini纯文本翻译模式处理{len(queries)}个文本，最大尝试次数: {self._max_total_attempts}")
        custom_prompt_json = getattr(ctx, 'custom_prompt_json', None) if ctx else None
        line_break_prompt_json = getattr(ctx, 'line_break_prompt_json', None) if ctx else None

        # 使用分割包装器进行翻译
        translations = await self._translate_with_split(
            self._translate_batch,
            queries,
            split_level=0,
            source_lang=from_lang,
            target_lang=to_lang,
            custom_prompt_json=custom_prompt_json,
            line_break_prompt_json=line_break_prompt_json,
            ctx=ctx
        )

        # 应用文本后处理
        translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]
        return translations
