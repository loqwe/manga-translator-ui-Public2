"""
身份验证和授权模块

负责管理员令牌管理、密码验证和访问控制。
"""

import secrets
from typing import Optional
from fastapi import Header, HTTPException


# 有效的管理员 tokens（登录后生成）
valid_admin_tokens = set()


def generate_admin_token() -> str:
    """生成管理员令牌"""
    return secrets.token_hex(32)


def validate_admin_token(token: str) -> bool:
    """验证管理员令牌"""
    return token in valid_admin_tokens


def add_admin_token(token: str):
    """添加管理员令牌到有效集合"""
    valid_admin_tokens.add(token)


def remove_admin_token(token: str):
    """从有效集合中移除管理员令牌"""
    valid_admin_tokens.discard(token)


def clear_admin_tokens():
    """清除所有管理员令牌"""
    valid_admin_tokens.clear()


async def require_admin_token(token: str = Header(alias="X-Admin-Token", default=None)) -> str:
    """
    FastAPI 依赖注入函数：要求管理员令牌
    
    Args:
        token: 从请求头获取的令牌
    
    Returns:
        验证通过的令牌
    
    Raises:
        HTTPException: 如果令牌无效或缺失
    """
    if not token or not validate_admin_token(token):
        raise HTTPException(401, detail="Unauthorized")
    return token


def admin_login(password: str, admin_password: str) -> dict:
    """
    管理员登录
    
    Args:
        password: 用户提供的密码
        admin_password: 正确的管理员密码
    
    Returns:
        包含 success 和 token/message 的字典
    """
    if not admin_password:
        return {"success": False, "message": "Admin password not set. Please setup first."}
    
    if password == admin_password:
        token = generate_admin_token()
        add_admin_token(token)
        return {"success": True, "token": token}
    
    return {"success": False, "message": "Invalid password"}


def setup_admin_password(password: str, current_password: Optional[str]) -> dict:
    """
    首次设置管理员密码
    
    Args:
        password: 新密码
        current_password: 当前密码（如果已设置）
    
    Returns:
        包含 success 和 token/message 的字典
    """
    # 只有在没有密码时才允许设置
    if current_password:
        return {"success": False, "message": "Admin password already set"}
    
    if not password or len(password) < 6:
        return {"success": False, "message": "Password must be at least 6 characters"}
    
    # 生成 token
    token = generate_admin_token()
    add_admin_token(token)
    
    return {"success": True, "token": token}


def change_admin_password(old_password: str, new_password: str, admin_password: str) -> dict:
    """
    更改管理员密码
    
    Args:
        old_password: 旧密码
        new_password: 新密码
        admin_password: 当前的管理员密码
    
    Returns:
        包含 success 和 message 的字典
    """
    # 验证旧密码
    if old_password != admin_password:
        return {"success": False, "message": "旧密码错误"}
    
    # 验证新密码
    if not new_password or len(new_password) < 6:
        return {"success": False, "message": "新密码至少需要6位"}
    
    # 清除所有旧的 token（强制重新登录）
    clear_admin_tokens()
    
    return {"success": True, "message": "密码已更改，请重新登录"}


def user_login(password: str, user_access: dict) -> dict:
    """
    用户登录
    
    Args:
        password: 用户提供的密码
        user_access: 用户访问配置
    
    Returns:
        包含 success 和 message 的字典
    """
    # 如果不需要密码，直接允许访问
    if not user_access.get('require_password', False):
        return {"success": True, "message": "No password required"}
    
    # 验证密码
    if password == user_access.get('user_password', ''):
        return {"success": True, "message": "Login successful"}
    
    return {"success": False, "message": "Invalid password"}


def check_user_access(user_access: dict) -> dict:
    """
    检查用户访问是否需要密码
    
    Args:
        user_access: 用户访问配置
    
    Returns:
        包含 require_password 的字典
    """
    return {
        "require_password": user_access.get('require_password', False)
    }
