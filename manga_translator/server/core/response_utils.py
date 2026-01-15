"""
响应工具模块

负责图像转换、JSON转换和临时环境变量管理。
"""

import io
import json
from fastapi import HTTPException

from manga_translator.server.to_json import to_translation
from manga_translator import Config


def transform_to_image(ctx):
    """
    将翻译上下文转换为图像字节
    
    Args:
        ctx: 翻译上下文对象
    
    Returns:
        图像字节数据
    """
    # 检查 ctx.result 是否存在
    if ctx.result is None:
        raise HTTPException(500, detail="Translation failed: no result image generated")
    
    # 检查是否使用占位符（在web模式下final.png保存后会设置此标记）
    if hasattr(ctx, 'use_placeholder') and ctx.use_placeholder:
        # ctx.result已经是1x1占位符图片，快速传输
        img_byte_arr = io.BytesIO()
        ctx.result.save(img_byte_arr, format="PNG")
        return img_byte_arr.getvalue()

    # 返回完整的翻译结果
    img_byte_arr = io.BytesIO()
    ctx.result.save(img_byte_arr, format="PNG")
    return img_byte_arr.getvalue()


def transform_to_json(ctx):
    """
    将翻译上下文转换为JSON字节
    
    Args:
        ctx: 翻译上下文对象
    
    Returns:
        JSON字节数据
    """
    return to_translation(ctx).model_dump_json().encode("utf-8")


def transform_to_bytes(ctx):
    """
    将翻译上下文转换为自定义字节格式
    
    Args:
        ctx: 翻译上下文对象
    
    Returns:
        自定义字节数据
    """
    return to_translation(ctx).to_bytes()


async def apply_user_env_vars(user_env_vars_str: str, config: Config, admin_settings: dict, username: str = None):
    """
    解析用户提供的环境变量，并检查策略
    如果用户没有提供 API Keys，尝试从用户选择的预设中获取
    
    **重要**: 此函数会将用户 API Key 设置到 config.translator 中，
    翻译器会在 parse_args 时读取这些值，实现用户级 API Key 隔离。
    
    Args:
        user_env_vars_str: JSON 字符串，包含用户的 API Keys
        config: 配置对象
        admin_settings: 管理员设置
        username: 用户名（用于获取预设配置）
    
    Returns:
        dict: 用户提供的环境变量字典，如果没有则返回 None
    
    Raises:
        HTTPException: 如果策略不允许
    """
    import logging
    logger = logging.getLogger('manga_translator.server')
    
    logger.info(f"[EnvVars] apply_user_env_vars called for user '{username}'")
    
    policy = admin_settings.get('api_key_policy', {})
    require_user_keys = policy.get('require_user_keys', False)
    allow_server_keys = policy.get('allow_server_keys', True)
    
    # 先尝试解析用户直接提供的 API Keys
    user_env_vars = None
    if user_env_vars_str and user_env_vars_str.strip() not in ('{}', ''):
        try:
            user_env_vars = json.loads(user_env_vars_str)
            user_env_vars = {k: v for k, v in user_env_vars.items() if v and k.isupper()}
            if user_env_vars:
                logger.info(f"[EnvVars] User '{username}' provided direct env vars: {list(user_env_vars.keys())}")
                # 将用户 API Key 设置到 config 中，供翻译器使用
                _apply_env_vars_to_config(user_env_vars, config, logger)
                return user_env_vars
        except json.JSONDecodeError:
            pass
    
    # 用户没有直接提供 API Keys，尝试从预设获取
    logger.info(f"[EnvVars] No direct env vars, trying preset for user '{username}'")
    if username:
        preset_env_vars = await get_user_preset_env_vars(username)
        if preset_env_vars:
            logger.info(f"[EnvVars] Using preset env vars for user '{username}': {list(preset_env_vars.keys())}")
            # 将预设 API Key 设置到 config 中，供翻译器使用
            _apply_env_vars_to_config(preset_env_vars, config, logger)
            return preset_env_vars
    
    # 没有用户 API Keys 也没有预设
    logger.info(f"[EnvVars] No preset env vars for user '{username}', using server defaults")
    if require_user_keys:
        # 强制要求用户提供 API Keys
        raise HTTPException(403, detail="User API keys are required")
    
    if not allow_server_keys:
        # 不允许使用服务器的 API Keys，但用户也没提供
        raise HTTPException(403, detail="Server API keys are not allowed, please provide your own")
    
    # 允许使用服务器的 API Keys（已经在环境变量中）
    # 清除 config 中的用户级 API Key，确保使用服务器默认值
    config.translator.user_api_key = None
    config.translator.user_api_base = None
    config.translator.user_api_model = None
    return None


def _apply_env_vars_to_config(env_vars: dict, config: Config, logger):
    """
    将环境变量映射到 config.translator 的用户级字段
    
    支持的环境变量:
    - OPENAI_API_KEY, GEMINI_API_KEY -> user_api_key
    - OPENAI_API_BASE, GEMINI_API_BASE -> user_api_base
    - OPENAI_MODEL, GEMINI_MODEL -> user_api_model
    
    注意：预设可能使用 OPENAI_* 变量来配置第三方 API（如 Gemini 通过 OpenAI 兼容接口）
    所以我们统一将这些变量映射到 user_api_* 字段，翻译器会根据自己的类型使用这些值
    
    Args:
        env_vars: 环境变量字典
        config: 配置对象
        logger: 日志记录器
    """
    logger.info(f"[EnvVars->Config] Processing env vars: {list(env_vars.keys())}")
    
    # API Key 映射（按优先级）
    api_key_vars = ['OPENAI_API_KEY', 'GEMINI_API_KEY']
    for var in api_key_vars:
        if var in env_vars and env_vars[var]:
            config.translator.user_api_key = env_vars[var]
            # 只显示前10个字符用于调试
            masked = env_vars[var][:10] + '...' if len(env_vars[var]) > 10 else env_vars[var]
            logger.info(f"[EnvVars->Config] Set user_api_key from {var}: {masked}")
            break
    
    # API Base URL 映射
    api_base_vars = ['OPENAI_API_BASE', 'GEMINI_API_BASE']
    for var in api_base_vars:
        if var in env_vars and env_vars[var]:
            config.translator.user_api_base = env_vars[var]
            logger.info(f"[EnvVars->Config] Set user_api_base from {var}: {env_vars[var]}")
            break
    
    # Model 映射
    model_vars = ['OPENAI_MODEL', 'GEMINI_MODEL']
    for var in model_vars:
        if var in env_vars and env_vars[var]:
            config.translator.user_api_model = env_vars[var]
            logger.info(f"[EnvVars->Config] Set user_api_model from {var}: {env_vars[var]}")
            break
    
    # 最终确认
    logger.info(f"[EnvVars->Config] Final config.translator: user_api_key={'SET' if config.translator.user_api_key else 'NOT SET'}, user_api_base={config.translator.user_api_base}, user_api_model={config.translator.user_api_model}")


async def get_user_preset_env_vars(username: str) -> dict:
    """
    获取用户选择的预设中的 API Keys
    
    Args:
        username: 用户名
    
    Returns:
        dict: 预设中的环境变量，如果没有则返回 None
    """
    import logging
    logger = logging.getLogger('manga_translator.server')
    
    try:
        from manga_translator.server.core.config_management_service import ConfigManagementService
        
        config_service = ConfigManagementService()
        
        # 获取用户配置
        user_config = config_service.get_user_config(username)
        logger.info(f"[Preset] User '{username}' config: {user_config}")
        if not user_config:
            logger.info(f"[Preset] No user config found for '{username}'")
            return None
        
        # 获取用户选择的预设ID
        preset_id = user_config.get('selected_preset_id')
        logger.info(f"[Preset] User '{username}' selected preset_id: {preset_id}")
        if not preset_id:
            logger.info(f"[Preset] No preset selected for user '{username}'")
            return None
        
        # 获取预设配置（解密）
        preset = config_service.get_preset(preset_id, decrypt=True)
        logger.info(f"[Preset] Preset '{preset_id}' loaded: {preset is not None}")
        if not preset:
            logger.warning(f"[Preset] Preset '{preset_id}' not found")
            return None
        
        # 从预设配置中提取 API Keys
        preset_config = preset.get('config', {})
        logger.info(f"[Preset] Preset config keys: {list(preset_config.keys())}")
        if not preset_config:
            logger.info(f"[Preset] Preset '{preset_id}' has no config")
            return None
        
        # 提取环境变量（大写的键）
        env_vars = {k: v for k, v in preset_config.items() if v and k.isupper()}
        
        # 调试：检查 API Key 是否成功解密（只显示前10个字符）
        for key in env_vars:
            if 'API_KEY' in key:
                value = env_vars[key]
                if value:
                    masked = value[:10] + '...' if len(value) > 10 else value
                    logger.info(f"[Preset] {key} decrypted (first 10 chars): {masked}")
                else:
                    logger.warning(f"[Preset] {key} is empty after decryption!")
        
        logger.info(f"[Preset] Extracted env vars for user '{username}': {list(env_vars.keys())}")
        return env_vars if env_vars else None
        
    except Exception as e:
        import logging
        logging.getLogger('manga_translator.server').warning(f"Failed to get user preset env vars: {e}")
        import traceback
        traceback.print_exc()
        return None
