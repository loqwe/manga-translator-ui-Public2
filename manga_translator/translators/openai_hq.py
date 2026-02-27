import os
import re
import asyncio
import time
import base64
import json
import logging
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image

from .common import CommonTranslator, VALID_LANGUAGES, draw_text_boxes_on_image, parse_json_or_text_response, merge_glossary_to_file, get_glossary_extraction_prompt, parse_hq_response, validate_openai_response, AsyncOpenAICurlCffi
from .keys import OPENAI_API_KEY, OPENAI_MODEL
from ..utils import Context

# 禁用openai库的DEBUG日志,避免打印base64图片数据
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

# 浏览器风格的请求头，避免被 CF 拦截
BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8,ja;q=0.7",
    "Connection": "keep-alive",
    "Sec-Ch-Ua": '"Google Chrome";v="131", "Chromium";v="131", "Not_A Brand";v="24"',
    "Sec-Ch-Ua-Mobile": "?0",
    "Sec-Ch-Ua-Platform": '"Windows"',
}


def encode_image_for_openai(image, max_size=1024):
    """将图片编码为base64格式，适合OpenAI API"""
    # 转换图片格式为RGB（处理所有可能的图片模式）
    if image.mode == "P":
        # 调色板模式：转换为RGBA（如果有透明度）或RGB
        image = image.convert("RGBA" if "transparency" in image.info else "RGB")

    if image.mode == "RGBA":
        # RGBA模式：创建白色背景并合并透明通道
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    elif image.mode in ("LA", "L", "1", "CMYK"):
        # LA（灰度+透明）、L（灰度）、1（二值）、CMYK：统一转换为RGB
        if image.mode == "LA":
            # 灰度+透明：先转RGBA再合并到白色背景
            image = image.convert("RGBA")
            background = Image.new('RGB', image.size, (255, 255, 255))
            background.paste(image, mask=image.split()[-1])
            image = background
        else:
            # 其他模式：直接转RGB
            image = image.convert("RGB")
    elif image.mode != "RGB":
        # 其他未知模式：强制转换为RGB
        image = image.convert("RGB")

    # 调整图片大小
    w, h = image.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        image = image.resize((new_w, new_h), Image.LANCZOS)

    # 编码为base64（使用PNG格式确保质量和兼容性）
    buf = BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode('utf-8')


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

class OpenAIHighQualityTranslator(CommonTranslator):
    """
    OpenAI高质量翻译器
    支持多图片批量处理，提供文本框顺序、原文和原图给AI进行更精准的翻译
    """
    _LANGUAGE_CODE_MAP = VALID_LANGUAGES
    def __init__(self):
        super().__init__()
        self.client = None
        self.prev_context = ""  # 用于存储多页上下文
        
        # 只在非Web环境下重新加载.env文件
        # Web环境下不重新加载，避免覆盖用户临时设置的环境变量
        is_web_server = os.getenv('MANGA_TRANSLATOR_WEB_SERVER', 'false').lower() == 'true'
        if not is_web_server:
            from dotenv import load_dotenv
            load_dotenv(override=True)
        
        self.api_key = os.getenv('OPENAI_API_KEY', OPENAI_API_KEY)
        self.base_url = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1')
        self.model = os.getenv('OPENAI_MODEL', "gpt-4o")
        self.max_tokens = None  # 不限制，使用模型默认最大值
        self.temperature = 0.1
        self.use_stream = False  # 流式输出开关，默认关闭
        self._MAX_REQUESTS_PER_MINUTE = 0  # 默认无限制
        self._rate_limit_attempt = 0  # 429限速错误独立计数器
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
            self.logger.info(f"Setting OpenAI HQ max requests per minute to: {max_rpm}")
        
        # 从配置中读取用户级 API Key（优先于环境变量）
        # 这允许 Web 服务器为每个用户使用不同的 API Key
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
        
        # 从配置中读取流式输出开关
        use_stream = getattr(args, 'use_stream', None)
        if use_stream is not None:
            self.use_stream = use_stream
            self.logger.info(f"[流式] 流式模式: {'已启用' if use_stream else '已关闭'}")
        
        # 如果 API Key 或 Base URL 变化，重建客户端
        if need_rebuild_client:
            self.client = None
            self._setup_client()
    
    def _setup_client(self, force_recreate: bool = False):
        """设置OpenAI客户端

        Args:
            force_recreate: 是否强制重建客户端（用于重试时断开旧连接）
        """
        if force_recreate and self.client:
            try:
                import asyncio
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    asyncio.create_task(self.client.close())
                else:
                    loop.run_until_complete(self.client.close())
            except Exception as e:
                self.logger.debug(f"关闭旧客户端时出错（可忽略）: {e}")
            self.client = None

        if not self.client:
            self.client = AsyncOpenAICurlCffi(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=BROWSER_HEADERS,
                impersonate="chrome110",
                timeout=600.0,
                stream_timeout=300.0
            )
            self.logger.debug("已创建新的OpenAI HQ客户端连接（curl_cffi 模式）")

    def _rebuild_client_for_retry(self):
        """并发安全的客户端重建：创建新客户端替换，不关闭旧连接。
        
        并发模式下其他 worker 可能正在使用旧客户端发送请求。
        直接 close() 会断开它们的 in-flight 连接。
        这里只创建新实例赋值给 self.client，旧实例由 in-flight 协程持有引用，
        请求完成后无引用自动 GC 释放。
        """
        self.client = AsyncOpenAICurlCffi(
            api_key=self.api_key,
            base_url=self.base_url,
            default_headers=BROWSER_HEADERS,
            impersonate="chrome110",
            timeout=600.0,
            stream_timeout=300.0
        )
        self.logger.info("重试前重建客户端连接")

    async def _cleanup(self):
        """清理资源"""
        if self.client:
            try:
                await self.client.close()
            except Exception:
                pass

    async def _abort_inflight_request(self):
        """取消时中断当前请求连接，避免长时间阻塞。"""
        if not self.client:
            return
        try:
            await self.client.close()
        except Exception as e:
            self.logger.debug(f"中断请求时关闭客户端失败（可忽略）: {e}")
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
        target_lang_full = lang_map.get(target_lang, target_lang) # Fallback to the code itself

        custom_prompt_str = ""
        if custom_prompt_json:
            custom_prompt_str = _flatten_prompt_data(custom_prompt_json)
            # self.logger.info(f"--- Custom Prompt Content ---\n{custom_prompt_str}\n---------------------------")

        line_break_prompt_str = ""
        if line_break_prompt_json and line_break_prompt_json.get('line_break_prompt'):
            line_break_prompt_str = line_break_prompt_json['line_break_prompt']

        try:
            from ..utils import BASE_PATH
            import os
            import json
            
            # 源语言代码到提示词文件的映射（使用内部语言码）
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

            # 如果目标语言是中文，追加可选的 R18 中文用词提示词
            if target_lang in ("CHS", "CHT"):
                r18_path = os.path.join(BASE_PATH, 'dict', 'prompts', 'target_zh_r18.json')
                if os.path.exists(r18_path):
                    with open(r18_path, 'r', encoding='utf-8') as f:
                        r18_data = json.load(f)
                    r18_prompt = r18_data.get('r18_prompt', '')
                    if r18_prompt:
                        base_prompt += f"\n\n{r18_prompt}"
                        self.logger.debug("已追加 R18+ 中文用词提示词")
        except Exception as e:
            self.logger.warning(f"Failed to load system prompt from file, falling back to hardcoded prompt. Error: {e}")
            base_prompt = f"""You are an expert manga translator. Your task is to accurately translate manga text from the source language into **{{{target_lang}}}**. You will be given the full manga page for context.

**CRITICAL INSTRUCTIONS (FOLLOW STRICTLY):**

1.  **DIRECT TRANSLATION ONLY**: Your output MUST contain ONLY the raw, translated text. Nothing else.
    -   DO NOT include the original text.
    -   DO NOT include any explanations, greetings, apologies, or any conversational text.
    -   DO NOT use Markdown formatting (like ```json or ```).
    -   The output is fed directly to an automated script. Any extra text will cause it to fail.

2.  **MATCH LINE COUNT**: The number of lines in your output MUST EXACTLY match the number of text regions you are asked to translate. Each line in your output corresponds to one numbered text region in the input.

3.  **TRANSLATE EVERYTHING**: Translate all text provided, including sound effects and single characters. Do not leave any line untranslated.

4.  **ACCURACY AND TONE**:
    -   Preserve the original tone, emotion, and character's voice.
    -   Ensure consistent translation of names, places, and special terms.
    -   For onomatopoeia (sound effects), provide the equivalent sound in {{{target_lang}}} or a brief description (e.g., '(rumble)', '(thud)').

---

**EXAMPLE OF CORRECT AND INCORRECT OUTPUT:**

**[ CORRECT OUTPUT EXAMPLE ]**
This is a correct response. Notice it only contains the translated text, with each translation on a new line.

(Imagine the user input was: "1. うるさい！", "2. 黙れ！")
```
吵死了！
闭嘴！
```

**[ ❌ INCORRECT OUTPUT EXAMPLE ]**
This is an incorrect response because it includes extra text and explanations.

(Imagine the user input was: "1. うるさい！", "2. 黙れ！")
```
好的，这是您的翻译：
1. 吵死了！
2. 闭嘴！
```
**REASONING:** The above example is WRONG because it includes "好的，这是您的翻译：" and numbering. Your response must be ONLY the translated text, line by line.

---

**FINAL INSTRUCTION:** Now, perform the translation task. Remember, your response must be clean, containing only the translated text.
"""

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
        
        # self.logger.info(f"--- OpenAI HQ Final System Prompt ---\n{final_prompt}")
        return final_prompt

    def _build_user_prompt(self, batch_data: List[Dict], ctx: Any, retry_attempt: int = 0, retry_reason: str = "") -> str:
        """构建用户提示词（高质量版）- 使用统一方法，只包含上下文和待翻译文本"""
        return self._build_user_prompt_for_hq(batch_data, ctx, self.prev_context, retry_attempt=retry_attempt, retry_reason=retry_reason)
    
    def _get_system_prompt(self, source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, retry_attempt: int = 0, retry_reason: str = "", extract_glossary: bool = False) -> str:
        """获取完整的系统提示词（包含断句提示词、自定义提示词和基础系统提示词）"""
        return self._build_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)

    async def _translate_batch_high_quality(self, texts: List[str], batch_data: List[Dict], source_lang: str, target_lang: str, custom_prompt_json: Dict[str, Any] = None, line_break_prompt_json: Dict[str, Any] = None, ctx: Any = None, split_level: int = 0) -> List[str]:
        """高质量批量翻译方法"""
        if not texts:
            return []
        
        if not self.client:
            self._setup_client()
        
        # 准备图片
        self.logger.info(f"高质量翻译模式：正在打包 {len(batch_data)} 张图片并发送...")

        image_contents = []
        for img_idx, data in enumerate(batch_data):
            image = data['image']
            
            # 在图片上绘制带编号的文本框
            text_regions = data.get('text_regions', [])
            text_order = data.get('text_order', [])
            upscaled_size = data.get('upscaled_size')
            if text_regions and text_order:
                image = draw_text_boxes_on_image(image, text_regions, text_order, upscaled_size)
                self.logger.debug(f"已在图片上绘制 {len(text_regions)} 个带编号的文本框")
            
            base64_img = encode_image_for_openai(image)
            image_contents.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{base64_img}"}
            })

        # INFO级顺序诊断日志：确认发送前的编号与文本顺序
        for img_idx, data in enumerate(batch_data):
            text_order = data.get('text_order', []) or []
            original_texts = data.get('original_texts', []) or []
            if not original_texts:
                continue
            preview_pairs = []
            preview_n = min(6, len(original_texts))
            for i in range(preview_n):
                disp_id = text_order[i] if i < len(text_order) else i + 1
                preview_pairs.append(f"{disp_id}:{str(original_texts[i])[:18]}")
            self.logger.info(
                f"[HQ Input Order] image={img_idx + 1}, regions={len(original_texts)}, "
                f"preview={preview_pairs}"
            )
        # INFO级完整发送文本（按id显示），用于排查"发送顺序"
        send_items = []
        for data in batch_data:
            text_order = data.get('text_order', []) or []
            original_texts = data.get('original_texts', []) or []
            for i, text in enumerate(original_texts):
                disp_id = text_order[i] if i < len(text_order) else i + 1
                send_items.append((int(disp_id), str(text)))
        if send_items:
            send_items.sort(key=lambda x: x[0])
            self.logger.info(f"[HQ Send Payload] total={len(send_items)}")
            for sid, stext in send_items:
                self.logger.info(f"[HQ Send Payload] id={sid} text={stext}")
        
        # 初始化重试信息
        retry_attempt = 0
        retry_reason = ""
        
        # 标记是否发送图片（降级机制）
        send_images = True
        
        # 发送请求
        max_retries = self.attempts
        attempt = 0
        is_infinite = max_retries == -1
        last_exception = None
        local_attempt = 0  # 本次批次的尝试次数
        
        # 数量不匹配/BR缺失/质量检查失败的内部重试固定为2次
        max_quality_retries = 2
        quality_retry_count = 0  # 质量类错误重试计数

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

            # 文本分割逻辑已禁用
            # if local_attempt > self._SPLIT_THRESHOLD and len(texts) > 1 and split_level < self._MAX_SPLIT_ATTEMPTS:
            #     self.logger.warning(f"Triggering split after {local_attempt} local attempts")
            #     raise self.SplitException(local_attempt, texts)
            
            # 确定是否开启术语提取
            # 必须同时满足：1. 有自定义提示词（才有地方存） 2. 配置开启了提取开关
            config_extract = False
            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                config_extract = getattr(ctx.config.translator, 'extract_glossary', False)
            
            extract_glossary = bool(custom_prompt_json) and config_extract

            # 构建系统提示词和用户提示词（包含重试信息以避免缓存）
            system_prompt = self._get_system_prompt(source_lang, target_lang, custom_prompt_json=custom_prompt_json, line_break_prompt_json=line_break_prompt_json, retry_attempt=retry_attempt, retry_reason=retry_reason, extract_glossary=extract_glossary)
            user_prompt = self._build_user_prompt(batch_data, ctx, retry_attempt=retry_attempt, retry_reason=retry_reason)
            user_content = [{"type": "text", "text": user_prompt}]
            
            # 降级检查：如果 send_images 为 True，则发送图片
            if send_images:
                user_content.extend(image_contents)
            elif retry_attempt > 0:
                 self.logger.warning("降级模式：仅发送文本，不发送图片")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ]
            # INFO级完整发送消息日志（过滤图片base64）
            sanitized_messages = []
            for msg in messages:
                role = msg.get("role")
                content = msg.get("content")
                if isinstance(content, list):
                    sanitized_content = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            raw_url = ((part.get("image_url") or {}).get("url") or "")
                            if isinstance(raw_url, str) and "base64," in raw_url:
                                prefix, b64 = raw_url.split("base64,", 1)
                                safe_url = f"{prefix}base64,[FILTERED:{len(b64)} chars]"
                            else:
                                safe_url = "[FILTERED_IMAGE_URL]"
                            sanitized_content.append({
                                "type": "image_url",
                                "image_url": {"url": safe_url}
                            })
                        else:
                            sanitized_content.append(part)
                    sanitized_messages.append({"role": role, "content": sanitized_content})
                else:
                    sanitized_messages.append({"role": role, "content": content})
            self.logger.info(
                "[HQ Send Messages Full]\n%s",
                json.dumps(sanitized_messages, ensure_ascii=False, indent=2)
            )

            _retries_exhausted = False
            try:
                # Dynamic temperature adjustment: helps escape failure mode after validation retries
                current_temperature = self._get_retry_temperature(self.temperature, retry_attempt, retry_reason)
                if retry_attempt > 0 and current_temperature != self.temperature:
                    self.logger.info(f"[重试] 温度调整: {self.temperature} -> {current_temperature}")

                # 主日志：记录流式开关状态（每个批次首轮仅记录一次）
                if retry_attempt == 0 and local_attempt == 1:
                    self.logger.info(f"[Stream] use_stream={self.use_stream}")

                # 构建API参数
                api_params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": current_temperature
                }
                if self.max_tokens is not None:
                    api_params["max_tokens"] = self.max_tokens

                def _extract_openai_stream_text(chunk):
                    if not (hasattr(chunk, 'choices') and chunk.choices):
                        return ""
                    choice = chunk.choices[0]
                    delta = getattr(choice, 'delta', None)
                    return getattr(delta, 'content', '') if delta else ""

                def _extract_openai_stream_finish_reason(chunk):
                    if not (hasattr(chunk, 'choices') and chunk.choices):
                        return None
                    return getattr(chunk.choices[0], 'finish_reason', None)

                def _on_stream_chunk(delta_text, _full_text):
                    if self.logger.isEnabledFor(logging.DEBUG):
                        self._emit_stream_json_preview("[OpenAI HQ Stream]", _full_text, source_texts=texts)

                streamed_text = None
                response = None

                if self.use_stream:
                    # 流式模式：尝试流式传输，失败自动 fallback 到非流式
                    try:
                        self._reset_stream_json_preview()
                        stream_params = dict(api_params)
                        stream_params["stream"] = True
                        streamed_text, streamed_finish_reason = await self._run_unified_stream_transport(
                            create_stream=lambda: self.client.chat.completions.create(**stream_params),
                            extract_text=_extract_openai_stream_text,
                            extract_finish_reason=_extract_openai_stream_finish_reason,
                            on_chunk=_on_stream_chunk,
                            on_cancel=self._abort_inflight_request,
                            poll_interval=0.2,
                            sync_iter_in_thread=False,
                        )
                        self._finish_stream_inline()
                    except Exception as stream_error:
                        self._finish_stream_inline()
                        self.logger.warning(f"流式请求不可用，已回退普通请求: {stream_error}")
                        response = await self._await_with_cancel_polling(
                            self.client.chat.completions.create(**api_params),
                            poll_interval=0.2,
                            on_cancel=self._abort_inflight_request,
                        )
                else:
                    # 非流式模式：直接调用
                    response = await self._await_with_cancel_polling(
                        self.client.chat.completions.create(**api_params),
                        poll_interval=0.2,
                        on_cancel=self._abort_inflight_request,
                    )

                if streamed_text is not None:
                    finish_reason = streamed_finish_reason
                    has_content = bool(streamed_text)
                else:
                    # 验证响应对象是否有效
                    validate_openai_response(response, self.logger)
                    finish_reason = response.choices[0].finish_reason if (hasattr(response, 'choices') and response.choices) else None
                    has_content = response.choices and response.choices[0].message.content

                if has_content:
                    result_text = streamed_text.strip() if streamed_text is not None else response.choices[0].message.content.strip()
                    # 统一的编码清理（处理UTF-16-LE等编码问题）
                    from .common import sanitize_text_encoding
                    result_text = sanitize_text_encoding(result_text)
                    
                    self.logger.debug(f"--- OpenAI Raw Response ---\n{result_text}\n---------------------------")
                    self.logger.info(f"[HQ Raw Response Full]\n{result_text}")
                    
                    # ✅ 检测HTML错误响应
                    if result_text.startswith('<!DOCTYPE') or result_text.startswith('<html') or '<h1>404</h1>' in result_text:
                        raise Exception(f"API_404_ERROR: API返回HTML错误页面 - API地址({self.base_url})或模型({self.model})配置错误")
                    
                    # 去除 <think>...</think> 标签及内容（LM Studio 等本地模型的思考过程）
                    result_text = re.sub(r'(</think>)?<think>.*?</think>', '', result_text, flags=re.DOTALL)
                    # 提取 <answer>...</answer> 中的内容（如果存在）
                    answer_match = re.search(r'<answer>(.*?)</answer>', result_text, flags=re.DOTALL)
                    if answer_match:
                        result_text = answer_match.group(1).strip()
                    
                    # 如果结果为空字符串
                    if not result_text:
                        self.logger.warning("OpenAI API返回空文本，下次重试将不再发送图片")
                        send_images = False
                        raise Exception("OpenAI API returned empty text")
                    
                    # 解析翻译结果（支持提取翻译和术语）
                    translations, new_terms = parse_hq_response(result_text)

                    # INFO级顺序诊断日志：检查原始返回中是否携带id，以及id顺序/完整性
                    id_pattern = r'"id"\s*:\s*(\d+)\s*,\s*"translation"\s*:\s*"((?:\\\\.|[^"\\\\])*)"'
                    id_matches = re.findall(id_pattern, result_text, flags=re.DOTALL)
                    if id_matches:
                        id_seq = [int(m[0]) for m in id_matches]
                        id_count_map = {}
                        for _id in id_seq:
                            id_count_map[_id] = id_count_map.get(_id, 0) + 1
                        duplicate_ids = sorted([k for k, v in id_count_map.items() if v > 1])
                        unique_ids = sorted(set(id_seq))
                        expected_ids = set(range(1, len(texts) + 1))
                        missing_ids = sorted(expected_ids - set(unique_ids))
                        extra_ids = sorted(set(unique_ids) - expected_ids)
                        self.logger.info(
                            f"[HQ Parse IDs] matches={len(id_matches)}, unique={len(unique_ids)}, "
                            f"first_ids={id_seq[:12]}"
                        )
                        if missing_ids or extra_ids:
                            self.logger.info(
                                f"[HQ Parse IDs] id_set_mismatch missing={missing_ids[:20]} extra={extra_ids[:20]}"
                            )
                        if duplicate_ids:
                            self.logger.info(f"[HQ Parse IDs] duplicate_ids={duplicate_ids[:20]}")
                        else:
                            self.logger.info("[HQ Parse IDs] id_set_ok (1..N complete)")
                    else:
                        self.logger.info(
                            "[HQ Parse IDs] no explicit id-translation pairs found in raw response; "
                            "fallback to positional parsing"
                        )

                    self.logger.info(
                        f"[HQ Parse Result] parsed_translations={len(translations)}, expected={len(texts)}"
                    )
                    
                    # 处理提取到的术语（使用缓存，支持并发）
                    if extract_glossary:
                        if new_terms:
                            # 标记所属作品名（用于 works 结构写回）；无法明确识别时不写 work_name（默认落入无作品区）
                            work_name = getattr(ctx, 'glossary_write_work_name', None) if ctx else None
                            # 获取原始生肉名（用于 unassigned 区域匹配 / 自动迁移）
                            raw_work_name = getattr(ctx, 'raw_work_name', None) if ctx else None

                            # ✅ 自动迁移：当 raw_work_name 能命中 name_mapping.json 时，直接写入 works
                            mapped_work_name = None
                            if not work_name and raw_work_name:
                                try:
                                    from ..utils.work_resolver import resolve_translated_work_name_from_raw
                                    mapped_work_name = resolve_translated_work_name_from_raw(raw_work_name)
                                except Exception:
                                    mapped_work_name = None

                            for t in new_terms:
                                if not isinstance(t, dict):
                                    continue

                                # 有确定作品名时写入 work_name
                                if work_name and 'work_name' not in t:
                                    t['work_name'] = work_name

                                # 无确定作品名但可映射到熟肉名：写入 works
                                elif not work_name and mapped_work_name and 'work_name' not in t:
                                    t['work_name'] = mapped_work_name
                                    # 可选保留 raw_work_name 便于追溯（不影响 works 使用）
                                    if raw_work_name and 'raw_work_name' not in t:
                                        t['raw_work_name'] = raw_work_name

                                # 仍无法映射：落入无作品区，并写 raw_work_name 供后续匹配
                                elif not work_name and raw_work_name and 'raw_work_name' not in t:
                                    t['raw_work_name'] = raw_work_name

                            self.logger.info(f"[术语提取] 提取到 {len(new_terms)} 个新术语: {new_terms}")
                            prompt_path = None
                            if ctx and hasattr(ctx, 'config') and hasattr(ctx.config, 'translator'):
                                prompt_path = getattr(ctx.config.translator, 'high_quality_prompt_path', None)
                            
                            if prompt_path:
                                # ✅ 使用统一函数，自动根据并发数决定写入策略
                                # 并发=1：直接写入文件；并发>1：缓存后批量写入
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
                        quality_retry_count += 1
                        retry_reason = f"Translation count mismatch: expected {len(texts)}, got {len(translations)}"
                        log_attempt = f"{quality_retry_count}/{max_quality_retries}"
                        self.logger.warning(f"[数量不匹配 {log_attempt}] {retry_reason}. Retrying...")
                        self.logger.warning(f"Expected texts: {texts}")
                        self.logger.warning(f"Got translations: {translations}")
                        
                        # 记录错误以便在达到最大尝试次数时显示
                        last_exception = Exception(f"翻译数量不匹配: 期望 {len(texts)} 条，实际得到 {len(translations)} 条")

                        if quality_retry_count >= max_quality_retries:
                            raise Exception(f"Translation count mismatch after {max_quality_retries} quality retries: expected {len(texts)}, got {len(translations)}")

                        # 重试前重建客户端（并发安全）
                        self._rebuild_client_for_retry()
                        await self._sleep_with_cancel_polling(2)
                        continue

                    # 质量验证：检查空翻译、合并翻译、可疑符号等
                    is_valid, error_msg = self._validate_translation_quality(texts, translations)
                    if not is_valid:
                        retry_attempt += 1
                        quality_retry_count += 1
                        retry_reason = f"Quality check failed: {error_msg}"
                        log_attempt = f"{quality_retry_count}/{max_quality_retries}"
                        self.logger.warning(f"[质量检查失败 {log_attempt}] {retry_reason}. Retrying...")
                        
                        # 记录错误以便在达到最大尝试次数时显示
                        last_exception = Exception(f"翻译质量检查失败: {error_msg}")

                        if quality_retry_count >= max_quality_retries:
                            raise Exception(f"Quality check failed after {max_quality_retries} quality retries: {error_msg}")

                        # 重试前重建客户端（并发安全）
                        self._rebuild_client_for_retry()
                        await self._sleep_with_cancel_polling(2)
                        continue

                    self.logger.info("--- Translation Results ---")
                    for original, translated in zip(texts, translations):
                        self.logger.info(f'{original} -> {translated}')
                    self.logger.info("---------------------------")

                    # BR检查：检查翻译结果是否包含必要的[BR]标记
                    # BR check: Check if translations contain necessary [BR] markers
                    if not self._validate_br_markers(translations, batch_data=batch_data, ctx=ctx):
                        retry_attempt += 1
                        quality_retry_count += 1
                        retry_reason = "BR markers missing in translations"
                        log_attempt = f"{quality_retry_count}/{max_quality_retries}"
                        self.logger.warning(f"[BR标记缺失 {log_attempt}] {retry_reason}, retrying...")
                        
                        # 记录错误以便在达到最大尝试次数时显示
                        last_exception = Exception("AI断句检查失败: 翻译结果缺少必要的[BR]标记")
                        
                        # 如果达到最大重试次数，抛出友好的异常
                        if quality_retry_count >= max_quality_retries:
                            from .common import BRMarkersValidationException
                            self.logger.error(f"OpenAI高质量翻译在 {max_quality_retries} 次重试后仍然失败：AI断句检查失败。")
                            raise BRMarkersValidationException(
                                missing_count=0,  # 具体数字在_validate_br_markers中已记录
                                total_count=len(texts),
                                tolerance=max(1, len(texts) // 10)
                            )
                        
                        # 重试前重建客户端（并发安全）
                        self._rebuild_client_for_retry()
                        await self._sleep_with_cancel_polling(2)
                        continue

                    self._rate_limit_attempt = 0  # 成功后重置429计数器
                    return translations[:len(texts)]
                
                # 如果不成功，则记录原因并准备重试
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                
                # finish_reason 已在上面获取，根据不同情况处理
                if finish_reason == 'content_filter':
                    self.logger.warning(f"OpenAI内容被安全策略拦截 ({log_attempt})。下次重试将不再发送图片")
                    send_images = False
                    last_exception = Exception("OpenAI content filter triggered")
                elif finish_reason == 'length':
                    self.logger.warning(f"OpenAI回复被截断（达到token限制） ({log_attempt})。下次重试将不再发送图片")
                    send_images = False
                    last_exception = Exception("OpenAI response truncated due to length limit")
                elif finish_reason == 'tool_calls':
                    self.logger.warning(f"OpenAI尝试调用工具而非返回翻译 ({log_attempt})。下次重试将不再发送图片")
                    send_images = False
                    last_exception = Exception("OpenAI attempted tool calls instead of translation")
                elif not has_content:
                    self.logger.warning(f"OpenAI返回空内容 (finish_reason: '{finish_reason}') ({log_attempt})。下次重试将不再发送图片")
                    send_images = False
                    last_exception = Exception(f"OpenAI returned empty content (finish_reason: {finish_reason})")
                else:
                    self.logger.warning(f"OpenAI返回意外的结束原因 '{finish_reason}' ({log_attempt})。下次重试将不再发送图片")
                    send_images = False
                    last_exception = Exception(f"OpenAI returned unexpected finish_reason: {finish_reason}")

                if not is_infinite and attempt >= max_retries:
                    _retries_exhausted = True
                    self.logger.error(f"OpenAI翻译在 {max_retries} 次重试后仍然失败。")
                    raise last_exception
                
                # 重试前重建客户端（并发安全）
                self._rebuild_client_for_retry()
                await self._sleep_with_cancel_polling(1)

            except Exception as e:
                # 如果是 max_retries 耗尽后的 re-raise，直接传播，不再重复计数
                if _retries_exhausted:
                    raise
                error_msg_raw = str(e)
                error_msg_lower = error_msg_raw.lower()

                # 专门处理400错误，检查是否是多模态不支持问题
                is_bad_request = '400' in error_msg_raw or 'bad request' in error_msg_lower
                is_multimodal_unsupported = is_bad_request and any(keyword in error_msg_lower for keyword in [
                    'image_url', 'multimodal', 'vision', 'expected `text`', 'unknown variant'
                ])

                if is_multimodal_unsupported:
                    self.logger.error(f"❌ 模型 {self.model} 不支持多模态输入（图片+文本）")
                    self.logger.error("💡 解决方案：")
                    self.logger.error("   1. 使用支持多模态的模型（如 gpt-4o, gpt-4-vision-preview）")
                    self.logger.error("   2. 或者切换到普通翻译模式（不使用 _hq 高质量翻译器）")
                    self.logger.error("   3. DeepSeek模型不支持多模态，请勿使用 openai_hq 翻译器")
                    raise Exception(f"模型不支持多模态输入: {self.model}") from e
                elif is_bad_request:
                    # 其他400错误，正常重试
                    log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                    last_exception = e
                    self.logger.warning(f"OpenAI高质量翻译出错 ({log_attempt}): {e}")
                    
                    if not is_infinite and attempt >= max_retries:
                        self.logger.error(f"OpenAI翻译在 {max_retries} 次重试后仍然失败。")
                        raise last_exception
                    
                    # 重试前重建客户端（并发安全）
                    self._rebuild_client_for_retry()
                    await self._sleep_with_cancel_polling(1)
                    continue

                # --- 以下为原来的通用 except Exception 逻辑 ---
                error_msg = str(e).lower()
                
                # 429 限速错误：独立计数，不受 max_retries 限制
                if 'rate limit' in error_msg or 'ratelimit' in error_msg or '429' in error_msg or 'too many requests' in error_msg:
                    self._rate_limit_attempt += 1
                    max_rate_limit_attempts = 4  # 总共尝试4次（初始1次 + 重试3次）
                    
                    # 指数退避：2^n 秒，但第1、2次都是2秒
                    if self._rate_limit_attempt <= 2:
                        wait_time = 2
                    else:
                        wait_time = 2 ** (self._rate_limit_attempt - 1)  # 第3次=4秒，第4次=8秒
                    wait_time = min(wait_time, 64)  # 最大64秒
                    
                    self.logger.warning(f"[限速] 触发 429 限速错误 ({self._rate_limit_attempt}/{max_rate_limit_attempts})")
                    self.logger.warning(f"[限速] 等待 {wait_time}s 后重试...")
                    
                    # 通知全局限速器（如果有）
                    if ctx and hasattr(ctx, 'rate_limiter') and ctx.rate_limiter:
                        ctx.rate_limiter.report_rate_limit_error()
                    
                    if self._rate_limit_attempt >= max_rate_limit_attempts:
                        self._rate_limit_attempt = 0  # 重置计数器
                        last_exception = Exception(f"429限速错误: 已重试{max_rate_limit_attempts}次仍失败")
                        raise last_exception
                    
                    await self._sleep_with_cancel_polling(wait_time)
                    continue  # 不计入 max_retries，继续重试
                
                # 其他错误：正常计数
                log_attempt = f"{attempt}/{max_retries}" if not is_infinite else f"Attempt {attempt}"
                last_exception = e
                
                # 降级检查：502错误
                if '502' in str(e):
                     self.logger.warning(f"检测到网络错误(502)，下次重试将不再发送图片。错误信息: {e}")
                     send_images = False
                
                self.logger.warning(f"OpenAI高质量翻译出错 ({log_attempt}): {e}")
                
                if not is_infinite and attempt >= max_retries:
                    self.logger.error(f"OpenAI翻译在 {max_retries} 次重试后仍然失败。")
                    raise last_exception
                
                # 重试前重建客户端（并发安全）
                self._rebuild_client_for_retry()
                await self._sleep_with_cancel_polling(1)

        # 只有在所有重试都失败后才会执行到这里
        raise last_exception if last_exception else Exception("OpenAI translation failed after all retries")

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        """主翻译方法"""
        if not queries:
            return []

        # 重置全局尝试计数器
        self._reset_global_attempt_count()

        # 检查是否为高质量批量翻译模式
        if ctx and hasattr(ctx, 'high_quality_batch_data'):
            batch_data = ctx.high_quality_batch_data
            if batch_data and len(batch_data) > 0:
                self.logger.info(f"使用OpenAI高质量翻译模式处理{len(batch_data)}张图片，最大尝试次数: {self._max_total_attempts}")
                custom_prompt_json = getattr(ctx, 'custom_prompt_json', None)
                line_break_prompt_json = getattr(ctx, 'line_break_prompt_json', None)

                # 使用分割包装器进行翻译
                translations = await self._translate_with_split(
                    self._translate_batch_high_quality,
                    queries,
                    split_level=0,
                    batch_data=batch_data,
                    source_lang=from_lang,
                    target_lang=to_lang,
                    custom_prompt_json=custom_prompt_json,
                    line_break_prompt_json=line_break_prompt_json,
                    ctx=ctx
                )

                # 应用文本后处理（与普通翻译器保持一致）
                translations = [self._clean_translation_output(q, r, to_lang) for q, r in zip(queries, translations)]
                return translations
        
        # 普通单文本翻译（后备方案）
        if not self.client:
            self._setup_client()
        
        try:
            simple_prompt = f"Translate the following {from_lang} text to {to_lang}. Provide only the translation:\n\n" + "\n".join(queries)

            api_params = {
                "model": self.model,
                "messages": [{"role": "user", "content": simple_prompt}],
                "temperature": self.temperature,
            }
            if self.max_tokens is not None:
                api_params["max_tokens"] = self.max_tokens

            response = await self._await_with_cancel_polling(
                self.client.chat.completions.create(**api_params),
                poll_interval=0.2,
                on_cancel=self._abort_inflight_request,
            )

            result_text = response.choices[0].message.content.strip() if (hasattr(response, 'choices') and response.choices) else None
            if result_text:
                # 统一的编码清理（处理UTF-16-LE等编码问题）
                from .common import sanitize_text_encoding
                result_text = sanitize_text_encoding(result_text)

                # 去除 <think>...</think> 标签及内容（LM Studio 等本地模型的思考过程）
                result_text = re.sub(r'(</think>)?<think>.*?</think>', '', result_text, flags=re.DOTALL)
                # 提取 <answer>...</answer> 中的内容（如果存在）
                answer_match = re.search(r'<answer>(.*?)</answer>', result_text, flags=re.DOTALL)
                if answer_match:
                    result_text = answer_match.group(1).strip()

                translations = result_text.split('\\n')
                translations = [t.strip() for t in translations if t.strip()]

                # Strict validation: must match input count
                if len(translations) != len(queries):
                    error_msg = f"Translation count mismatch: expected {len(queries)}, got {len(translations)}"
                    self.logger.error(error_msg)
                    self.logger.error(f"Queries: {queries}")
                    self.logger.error(f"Translations: {translations}")
                    raise Exception(error_msg)

                return translations

        except Exception as e:
            self.logger.error(f"OpenAI翻译出错: {e}")

        return queries
