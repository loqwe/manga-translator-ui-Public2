import os
import re
import asyncio
# import json
from typing import List, Dict, Any
import httpx

from .openai_gateway import DEFAULT_BROWSER_HEADERS, get_openai_client, chat_completions
from .common import CommonTranslator, VALID_LANGUAGES, parse_json_or_text_response, parse_hq_response, get_glossary_extraction_prompt, merge_glossary_to_file, validate_openai_response
from .keys import OPENAI_API_KEY, OPENAI_MODEL
from ..utils import Context

# 浏览器风格的请求头，避免被 CF 拦截
# 注意：移除 Accept-Encoding 让 httpx 自动处理，避免压缩响应导致的 UTF-8 解码错误
BROWSER_HEADERS = DEFAULT_BROWSER_HEADERS


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

class OpenAITranslator(CommonTranslator):
    """
    OpenAI纯文本翻译器
    支持批量文本翻译，不包含图片处理
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    def __init__(self):
        super().__init__()
        self.client = None
        self.prev_context = ""  # 用于存储多页上下文
        # 只在非Web环境下重新加载.env文件
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        
        self.api_key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY)
        self.base_url = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_MODEL', "gpt-4o")
        self.max_tokens = None  # 不限制，使用模型默认最大值
        self.temperature = 0.1
        self._MAX_REQUESTS_PER_MINUTE = 0  # 默认无限制
        # Client is initialized lazily in async context to bind to the correct event loop.
    
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
            self.logger.info(f"Setting OpenAI max requests per minute to: {max_rpm}")
        
        # 从配置中读取用户级 API Key（优先于环境变量）
        need_rebuild_client = False
        
        user_api_key = getattr(args, 'user_api_key', None)
        if user_api_key and user_api_key != self.api_key:
            self.api_key = user_api_key
            need_rebuild_client = True
            self.logger.info("[UserAPIKey] Using user-provided API key")
        
        user_api_base = getattr(args, 'user_api_base', None)
        if user_api_base and user_api_base != self.base_url:
            self.base_url = user_api_base
            need_rebuild_client = True
            self.logger.info(f"[UserAPIKey] Using user-provided API base: {user_api_base}")
        
        user_api_model = getattr(args, 'user_api_model', None)
        if user_api_model:
            self.model = user_api_model
            self.logger.info(f"[UserAPIKey] Using user-provided model: {user_api_model}")
        
        # If API Key / Base URL changes, reset the cached client (re-created lazily in async context)
        if need_rebuild_client:
            self.client = None
    
    async def _ensure_client(self):
        """Ensure a shared OpenAI client is available (do NOT close it in this instance)."""
        if self.client:
            return

        # The client is shared per event loop; do NOT close it from a translator instance.
        self.client = await get_openai_client(
            api_key=self.api_key,
            base_url=self.base_url,
            headers=BROWSER_HEADERS,
        )
    
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

        # 尝试加载按语言分类的提示词
        base_prompt = ""
        try:
            from ..utils import BASE_PATH
            import os
            import json
            
            # 源语言代码到提示词文件的映射
            lang_to_prompt_file = {
                "JPN": "ja.json",
                "KOR": "ko.json",
                "ENG": "en.json",
                "IND": "id.json",
                "ESP": "es.json",
                "VIN": "es.json",  # Vietnamese uses Spanish prompt
            }
            
            # 根据源语言选择提示词文件
            prompt_filename = lang_to_prompt_file.get(source_lang, "default.json")
            prompt_path = os.path.join(BASE_PATH, 'dict', 'prompts', prompt_filename)
            
            # 如果按语言文件不存在，回退到 default.json
            if not os.path.exists(prompt_path):
                prompt_path = os.path.join(BASE_PATH, 'dict', 'prompts', 'default.json')
            
            # 如果 prompts 目录不存在，回退到旧的 system_prompt_hq.json
            if not os.path.exists(prompt_path):
                prompt_path = os.path.join(BASE_PATH, 'dict', 'system_prompt_hq.json')
            
            if os.path.exists(prompt_path):
                with open(prompt_path, 'r', encoding='utf-8') as f:
                    base_prompt_data = json.load(f)
                base_prompt = base_prompt_data.get('system_prompt', '')
                lang_display = {
                    "JPN": "日语",
                    "KOR": "韩语",
                    "ENG": "英语",
                    "IND": "印尼语",
                    "ESP": "西班牙语",
                    "VIN": "越南语",
                    "auto": "自动",
                }.get(source_lang, source_lang)
                self.logger.info(f"加载提示词: {os.path.basename(prompt_path)} (源语言: {lang_display})")
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from file: {e}")

        # Fallback
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
    
    def _get_system_prompt(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """获取完整的系统提示词"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch(self, texts: List[str], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """批量翻译方法（纯文本）"""
        if not texts:
            return []
        
        await self._ensure_client()
        
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

        while is_infinite or attempt < max_retries:
            # 检查是否被取消
            self._check_cancelled()
            
            # 检查全局尝试次数
            if not self._increment_global_attempt():
                self.logger.error("Reached global attempt limit. Stopping translation.")
                # 包含最后一次错误的真正原因
                last_error_msg = str(last_exception) if last_exception else "Unknown error"
                raise Exception(f"达到最大尝试次数 ({self._max_total_attempts})，最后一次错误: {last_error_msg}")

            local_attempt += 1
            attempt += 1
            
            # 确定是否开启术语提取
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)
            
            extract_glossary = bool(_custom_prompt_json) and config_extract

            # 构建系统提示词和用户提示词（包含重试信息以避免缓存）
            system_prompt = self._get_system_prompt(_source_lang, _target_lang, custom_prompt_json=_custom_prompt_json, line_break_prompt_json=_line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)
            user_prompt = self._build_user_prompt(texts, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]

            try:
                # Dynamic temperature adjustment: helps escape failure mode after validation retries
                current_temperature = self._get_retry_temperature(self.temperature, retry_attempt, retry_reason)
                if retry_attempt > 0 and current_temperature != self.temperature:
                    self.logger.info(f"[重试] 温度调整: {self.temperature} -> {current_temperature}")

                result = await chat_completions(
                    api_key=self.api_key,
                    base_url=self.base_url,
                    model=self.model,
                    messages=messages,
                    temperature=current_temperature,
                    max_tokens=self.max_tokens,
                    stream=False,
                    timeout=httpx.Timeout(300.0, connect=60.0),
                    max_requests_per_minute=self._MAX_REQUESTS_PER_MINUTE,
                    headers=BROWSER_HEADERS,
                    metrics_logger=self.log_ai_metrics,
                    logger=self.logger,
                )
                response = result.response

                # 验证响应对象是否有效
                validate_openai_response(response, self.logger)

                # 检查成功条件：有内容就尝试处理，后续会有质量检查
                finish_reason = response.choices[0].finish_reason if (hasattr(response, 'choices') and response.choices) else None
                has_content = response.choices and response.choices[0].message.content
                
                if has_content:
                    result_text = response.choices[0].message.content.strip()
                    
                    # 统一的编码清理（处理UTF-16-LE等编码问题）
                    from .common import sanitize_text_encoding
                    result_text = sanitize_text_encoding(result_text)
                    
                    self.logger.debug(f"--- OpenAI Raw Response ---\n{result_text}\n---------------------------")
                    
                    # 去除 <think>...</think> 标签及内容（LM Studio 等本地模型的思考过程）
                    result_text = re.sub(r'(</think>)?<think>.*?</think>', '', result_text, flags=re.DOTALL)
                    # 提取 <answer>...</answer> 中的内容（如果存在）
                    answer_match = re.search(r'<answer>(.*?)</answer>', result_text, flags=re.DOTALL)
                    if answer_match:
                        result_text = answer_match.group(1).strip()
                    
                    # 增加清理步骤，移除可能的Markdown代码块
                    if result_text.startswith("```") and result_text.endswith("```"):
                         # 这里的正则比简单的切片更安全
                         code_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', result_text, re.DOTALL)
                         if code_match:
                             result_text = code_match.group(1).strip()
                         elif result_text.startswith("```"): # 简单fallback
                             result_text = result_text.strip("`").strip()
                    
                    # 使用通用函数解析响应（支持JSON和纯文本）
                    translations, new_terms = parse_hq_response(result_text)
                    
                    # 处理提取到的术语（使用缓存，支持并发）
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

                        await asyncio.sleep(2)
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

                        await asyncio.sleep(2)
                        continue

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
                            self.logger.error("OpenAI翻译在多次重试后仍然失败：AI断句检查失败。")
                            raise BRMarkersValidationException(
                                missing_count=0,  # 具体数字在_validate_br_markers中已记录
                                total_count=len(texts),
                                tolerance=max(1, len(texts) // 10)
                            )
                        
                        await asyncio.sleep(2)
                        continue

                    return translations[:len(texts)]
                
                # 如果不成功，则记录原因并准备重试
                retry_attempt += 1
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                
                # finish_reason 已在上面获取，根据不同情况处理
                if finish_reason == 'content_filter':
                    retry_reason = "Content filter triggered"
                    self.logger.warning(f"OpenAI内容被安全策略拦截 ({log_attempt})。正在重试...")
                    last_exception = Exception("OpenAI content filter triggered")
                elif finish_reason == 'length':
                    retry_reason = "Response truncated due to length limit"
                    self.logger.warning(f"OpenAI回复被截断（达到token限制） ({log_attempt})。正在重试...")
                    last_exception = Exception("OpenAI response truncated due to length limit")
                elif finish_reason == 'tool_calls':
                    retry_reason = "Tool calls instead of translation"
                    self.logger.warning(f"OpenAI尝试调用工具而非返回翻译 ({log_attempt})。正在重试...")
                    last_exception = Exception("OpenAI attempted tool calls instead of translation")
                elif not has_content:
                    retry_reason = f"Empty content (finish_reason: {finish_reason})"
                    self.logger.warning(f"OpenAI返回空内容 (finish_reason: '{finish_reason}') ({log_attempt})。正在重试...")
                    last_exception = Exception(f"OpenAI returned empty content (finish_reason: {finish_reason})")
                else:
                    retry_reason = f"Unexpected finish_reason: {finish_reason}"
                    self.logger.warning(f"OpenAI返回意外的结束原因 '{finish_reason}' ({log_attempt})。正在重试...")
                    last_exception = Exception(f"OpenAI returned unexpected finish_reason: {finish_reason}")

                if not is_infinite and attempt >= max_retries:
                    self.logger.error("OpenAI翻译在多次重试后仍然失败。即将终止程序。")
                    raise last_exception
                
                await asyncio.sleep(1)

            except Exception as e:
                attempt += 1
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                last_exception = e
                self.logger.warning(f"OpenAI翻译出错 ({log_attempt}): {e}")
                
                if not is_infinite and attempt >= max_retries:
                    self.logger.error("OpenAI翻译在多次重试后仍然失败。即将终止程序。")
                    raise last_exception
                
                await asyncio.sleep(1)

        # 只有在所有重试都失败后才会执行到这里
        raise last_exception if last_exception else Exception("OpenAI translation failed after all retries")

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """主翻译方法"""
        if not queries:
            return []

        # 重置全局尝试计数器
        self._reset_global_attempt_count()

        self.logger.info(f"使用OpenAI纯文本翻译模式处理{len(queries)}个文本，最大尝试次数: {self._max_total_attempts}")
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

