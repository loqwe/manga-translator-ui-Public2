"""
Admin routes module.

This module contains all /admin/* endpoints for the manga translator server.
Updated to use the new session-based authentication system.
"""

import io
import secrets
from datetime import datetime
from typing import Optional
from fastapi import APIRouter, Header, Form, HTTPException, Depends
from fastapi.responses import StreamingResponse

from manga_translator.server.core.config_manager import (
    admin_settings, save_admin_settings, ADMIN_CONFIG_PATH
)
from manga_translator.server.core.task_manager import server_config, init_semaphore
from manga_translator.server.core.auth import valid_admin_tokens
from manga_translator.server.core.logging_manager import (
    global_log_queue, task_logs, task_logs_lock, add_log
)
from manga_translator.server.core.task_manager import (
    active_tasks, active_tasks_lock
)
from manga_translator.server.core.middleware import require_admin
from manga_translator.server.core.models import Session
import os
import logging

logger = logging.getLogger('manga_translator.server')

router = APIRouter(prefix="/admin", tags=["admin"])


# ============================================================================
# Admin Authentication Endpoints
# ============================================================================

@router.get("/need-setup")
async def check_admin_setup():
    """Check if admin password needs first-time setup"""
    admin_password = admin_settings.get('admin_password')
    return {
        "need_setup": not admin_password or admin_password == ''
    }


@router.post("/setup")
async def setup_admin_password(password: str = Form(...)):
    """First-time admin password setup"""
    # Only allow setup if no password exists
    if admin_settings.get('admin_password'):
        raise HTTPException(403, detail="Admin password already set")
    
    if not password or len(password) < 6:
        raise HTTPException(400, detail="Password must be at least 6 characters")
    
    # Save password to admin_settings
    admin_settings['admin_password'] = password
    save_admin_settings(admin_settings)
    
    # Also save to server_config (runtime use)
    server_config['admin_password'] = password
    
    # Generate token
    token = secrets.token_hex(32)
    valid_admin_tokens.add(token)
    
    logger.info("管理员密码已设置")
    return {"success": True, "token": token}


@router.post("/login")
async def admin_login(password: str = Form(...)):
    """
    Admin login (DEPRECATED - redirects to new auth system)
    
    This endpoint is maintained for backward compatibility.
    New clients should use POST /auth/login instead.
    """
    logger.warning("Deprecated /admin/login endpoint used. Please migrate to /auth/login")
    
    # For backward compatibility, we still support the old token system
    # But we recommend using the new session-based system
    admin_password = admin_settings.get('admin_password')
    
    if not admin_password:
        raise HTTPException(400, detail="Admin password not set. Please setup first.")
    
    if password == admin_password:
        token = secrets.token_hex(32)
        valid_admin_tokens.add(token)
        return {"success": True, "token": token}
    return {"success": False, "message": "Invalid password"}


@router.post("/change-password")
async def change_admin_password(
    old_password: str = Form(...),
    new_password: str = Form(...),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """Change admin password"""
    # Verify token
    if not token or token not in valid_admin_tokens:
        raise HTTPException(401, detail="Unauthorized")
    
    # Verify old password
    admin_password = admin_settings.get('admin_password')
    if old_password != admin_password:
        return {"success": False, "message": "旧密码错误"}
    
    # Verify new password
    if not new_password or len(new_password) < 6:
        return {"success": False, "message": "新密码至少需要6位"}
    
    # Update password
    admin_settings['admin_password'] = new_password
    save_admin_settings(admin_settings)
    
    # Also update runtime config
    server_config['admin_password'] = new_password
    
    # Clear all old tokens (force re-login)
    valid_admin_tokens.clear()
    
    logger.info("管理员密码已更改")
    return {"success": True, "message": "密码已更改，请重新登录"}


# ============================================================================
# Admin Settings Management Endpoints
# ============================================================================

@router.get("/settings")
async def get_admin_settings(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Get admin settings
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    return admin_settings


@router.post("/settings")
@router.put("/settings")
async def update_admin_settings(
    settings: dict,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Update admin settings
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    Supports both POST and PUT methods
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    # Support partial updates (allow new keys)
    for key, value in settings.items():
        if key in admin_settings and isinstance(admin_settings[key], dict) and isinstance(value, dict):
            admin_settings[key].update(value)
        else:
            admin_settings[key] = value
    
    # Save to file
    if save_admin_settings(admin_settings):
        logger.info(f"Admin settings updated by user '{session.username}'")
        return {"success": True, "message": "Settings saved to file"}
    else:
        return {"success": False, "message": "Failed to save settings to file"}


@router.post("/settings/parameter-visibility")
async def update_parameter_visibility(
    data: dict,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Update parameter visibility settings (hide/readonly/default values)
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    if 'hidden_keys' in data:
        admin_settings['hidden_keys'] = data['hidden_keys']
    if 'readonly_keys' in data:
        admin_settings['readonly_keys'] = data['readonly_keys']
    if 'default_values' in data:
        admin_settings['default_values'] = data['default_values']
    
    # Save to file
    if save_admin_settings(admin_settings):
        logger.info(f"Parameter visibility updated by user '{session.username}'")
        return {"success": True, "message": "Settings saved to file"}
    else:
        return {"success": False, "message": "Failed to save settings to file"}


# ============================================================================
# Server Configuration Endpoints
# ============================================================================

@router.get("/server-config")
async def get_server_config(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Get server configuration
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    return {
        "max_concurrent_tasks": server_config.get('max_concurrent_tasks', 3),
        "use_gpu": server_config.get('use_gpu', False),
        "verbose": server_config.get('verbose', False),
        "admin_config_path": ADMIN_CONFIG_PATH,
        "admin_config_exists": os.path.exists(ADMIN_CONFIG_PATH),
    }


@router.post("/server-config")
async def update_server_config(
    config: dict,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Update server configuration
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    if 'max_concurrent_tasks' in config:
        old_value = server_config.get('max_concurrent_tasks', 3)
        new_value = config['max_concurrent_tasks']
        server_config['max_concurrent_tasks'] = new_value
        
        # If concurrency changed, reinitialize semaphore
        if old_value != new_value:
            init_semaphore()
            logger.info(f"并发数已更新: {old_value} -> {new_value} by user '{session.username}'")
        
        # Save to admin_config.json (persist)
        try:
            admin_settings['max_concurrent_tasks'] = new_value
            save_admin_settings(admin_settings)
            logger.info(f"并发数已保存到配置文件: {new_value}")
        except Exception as e:
            logger.error(f"保存并发数到配置文件失败: {e}")
    
    return {"success": True}


# ============================================================================
# Announcement Management Endpoints
# ============================================================================

@router.put("/announcement")
async def update_announcement(
    announcement: dict,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Update announcement (admin)
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    admin_settings['announcement'] = announcement
    save_admin_settings(admin_settings)
    logger.info(f"公告已更新 by user '{session.username}': enabled={announcement.get('enabled')}, type={announcement.get('type')}")
    return {"success": True}


# ============================================================================
# Task Management Endpoints
# ============================================================================

@router.get("/tasks")
async def get_active_tasks_endpoint(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Get all active tasks with user information
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    # Use the task_manager function to get tasks
    from manga_translator.server.core.task_manager import get_active_tasks
    return get_active_tasks()


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(
    task_id: str,
    force: bool = False,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Cancel specified translation task
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    
    Args:
        task_id: Task ID
        force: Whether to force cancel (immediately terminate task, don't wait for checkpoint)
        session: Admin session
        token: Legacy admin token
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    with active_tasks_lock:
        if task_id in active_tasks:
            active_tasks[task_id]["cancel_requested"] = True
            
            if force:
                # Force cancel: directly call asyncio.Task.cancel()
                task = active_tasks[task_id].get("task")
                if task and not task.done():
                    task.cancel()
                    add_log(f"管理员 {session.username} 强制取消任务: {task_id[:8]}", "WARNING")
                    return {"success": True, "message": "任务已强制终止"}
                else:
                    add_log(f"管理员 {session.username} 请求强制取消任务，但任务已完成: {task_id[:8]}", "INFO")
                    return {"success": True, "message": "任务已完成，无需取消"}
            else:
                # Cooperative cancel: set flag, wait for task to respond at checkpoint
                add_log(f"管理员 {session.username} 请求取消任务: {task_id[:8]}", "WARNING")
                return {"success": True, "message": "取消请求已发送（协作式取消）"}
        else:
            raise HTTPException(404, detail="任务不存在或已完成")


# ============================================================================
# Log Management Endpoints
# ============================================================================

@router.get("/logs")
async def get_logs_endpoint(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None),
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    level: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
    offset: int = 0
):
    """
    Get logs with filtering support
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    
    Args:
        task_id: Filter by task ID (optional)
        session_id: Filter by session ID (optional) - 用于显示单个会话的日志
        level: Filter by log level (INFO, WARNING, ERROR, DEBUG, all) (optional)
        start_time: Filter logs after this time (ISO format) (optional)
        end_time: Filter logs before this time (ISO format) (optional)
        limit: Maximum number of logs to return (default: 100, max: 1000)
        offset: Number of logs to skip (for pagination)
    
    Returns:
        JSON object with logs array, total count, limit, and offset
    """
    from datetime import timezone
    
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    # Validate and cap limit
    limit = min(max(1, limit), 1000)
    offset = max(0, offset)
    
    try:
        with task_logs_lock:
            if task_id:
                # Get logs for specific task
                logs = list(task_logs.get(task_id, []))
            else:
                # Get global logs
                logs = list(global_log_queue)
        
        # Filter by session_id if specified
        if session_id:
            logs = [log for log in logs if log.get('session_id') == session_id]
        
        # Filter by level if specified (skip if 'all')
        if level and level.lower() != 'all':
            level_upper = level.upper()
            logs = [log for log in logs if log.get('level', '').upper() == level_upper]
        
        # Filter by time range if specified
        if start_time:
            try:
                start_dt = datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                logs = [log for log in logs 
                       if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) >= start_dt]
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid start_time format: {start_time}, error: {e}")
        
        if end_time:
            try:
                end_dt = datetime.fromisoformat(end_time.replace('Z', '+00:00'))
                logs = [log for log in logs 
                       if datetime.fromisoformat(log.get('timestamp', '').replace('Z', '+00:00')) <= end_dt]
            except (ValueError, AttributeError) as e:
                logger.warning(f"Invalid end_time format: {end_time}, error: {e}")
        
        # Get total before pagination
        total = len(logs)
        
        # Apply pagination (get most recent logs)
        # Reverse to get newest first, then slice, then reverse back
        logs = list(reversed(logs))
        paginated_logs = logs[offset:offset + limit]
        
        # Ensure all log messages are properly escaped and formatted
        for log in paginated_logs:
            if 'message' in log and isinstance(log['message'], str):
                # Ensure message is a string and handle any encoding issues
                log['message'] = log['message']
            if 'timestamp' in log:
                # Ensure timestamp is in ISO format
                try:
                    if isinstance(log['timestamp'], str):
                        # Validate it's a proper ISO format
                        datetime.fromisoformat(log['timestamp'].replace('Z', '+00:00'))
                except (ValueError, AttributeError):
                    # If invalid, use current time
                    log['timestamp'] = datetime.now(timezone.utc).isoformat()
        
        return {
            "logs": paginated_logs,
            "total": total,
            "limit": limit,
            "offset": offset,
            "has_more": (offset + limit) < total
        }
    
    except Exception as e:
        logger.error(f"Error fetching logs: {e}", exc_info=True)
        raise HTTPException(500, detail=f"Failed to fetch logs: {str(e)}")


@router.get("/logs/export")
async def export_logs(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None),
    task_id: Optional[str] = None
):
    """
    Export logs as text file
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    with task_logs_lock:
        if task_id:
            logs = list(task_logs.get(task_id, []))
            filename = f"logs_{task_id[:8]}.txt"
        else:
            logs = list(global_log_queue)
            from datetime import timezone
            filename = f"logs_all_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.txt"
    
    # Generate log text
    log_text = "\n".join([
        f"[{log['timestamp']}] [{log['level']}] {log['message']}"
        for log in logs
    ])
    
    return StreamingResponse(
        io.BytesIO(log_text.encode('utf-8')),
        media_type="text/plain",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ============================================================================
# Environment Variables Management Endpoints
# ============================================================================

@router.get("/env-vars")
async def get_env_vars(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None),
    show_values: bool = False
):
    """
    Get current environment variables (admin only)
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    from dotenv import dotenv_values
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    
    if os.path.exists(env_path):
        env_vars = dotenv_values(env_path)
        if show_values:
            # Admin can see actual values
            return {
                'path': env_path,
                'vars': {key: value for key, value in env_vars.items()}
            }
        else:
            # Only return whether set or not
            return {key: bool(value.strip()) for key, value in env_vars.items()}
    return {'path': env_path, 'vars': {}}


@router.post("/env-vars")
async def save_env_vars(
    env_vars: dict,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    Save environment variables to .env file (admin only)
    
    Supports both new session-based auth (X-Session-Token) and legacy token auth (X-Admin-Token)
    """
    # Legacy token support for backward compatibility
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    from dotenv import load_dotenv
    from manga_translator.server.core.env_service import EnvService
    env_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '.env')
    
    try:
        env_service = EnvService(env_path)
        
        # Save to .env file using EnvService for consistent formatting
        for key, value in env_vars.items():
            if value:  # Only save non-empty values
                env_service.update_env_var(key, value)
            else:
                # If value is empty, remove from .env (if exists)
                env_service.delete_env_var(key)
        
        # Reload .env file to ensure all variables are up to date
        load_dotenv(env_path, override=True)
        
        logger.info(f"Environment variables updated by user '{session.username}': {list(env_vars.keys())}")
        return {"success": True, "message": "环境变量已更新并立即生效"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to save env vars: {str(e)}")


# ============================================================================
# Storage and Cleanup Management Endpoints
# ============================================================================

def get_directory_stats(directory: str) -> dict:
    """获取目录的文件统计信息"""
    total_size = 0
    file_count = 0
    
    if os.path.exists(directory):
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    total_size += os.path.getsize(file_path)
                    file_count += 1
                except (OSError, IOError):
                    pass
    
    return {"size": total_size, "count": file_count}


@router.get("/storage/info")
async def get_storage_info(
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    获取存储使用情况
    """
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    # 获取各目录路径 - 使用 server 模块内的数据目录
    server_dir = os.path.dirname(os.path.dirname(__file__))  # manga_translator/server
    data_dir = os.path.join(server_dir, "data")
    results_dir = os.path.join(data_dir, "results")
    
    # 用户上传的资源目录
    user_resources_dir = os.path.join(server_dir, "user_resources")
    user_fonts_dir = os.path.join(user_resources_dir, "fonts")
    user_prompts_dir = os.path.join(user_resources_dir, "prompts")
    
    # 获取统计信息
    results_stats = get_directory_stats(results_dir)
    user_fonts_stats = get_directory_stats(user_fonts_dir)
    user_prompts_stats = get_directory_stats(user_prompts_dir)
    
    # 用户上传资源合计
    uploads_size = user_fonts_stats["size"] + user_prompts_stats["size"]
    uploads_count = user_fonts_stats["count"] + user_prompts_stats["count"]
    
    total_size = results_stats["size"] + uploads_size
    
    return {
        "uploads_size": uploads_size,
        "uploads_count": uploads_count,
        "results_size": results_stats["size"],
        "results_count": results_stats["count"],
        "cache_size": 0,  # 暂无独立缓存目录
        "cache_count": 0,
        "total_size": total_size,
        # 详细信息
        "user_fonts_size": user_fonts_stats["size"],
        "user_fonts_count": user_fonts_stats["count"],
        "user_prompts_size": user_prompts_stats["size"],
        "user_prompts_count": user_prompts_stats["count"]
    }


@router.post("/cleanup/{target}")
async def cleanup_storage(
    target: str,
    session: Session = Depends(require_admin),
    token: str = Header(alias="X-Admin-Token", default=None)
):
    """
    清理指定目录
    
    Args:
        target: 清理目标 (uploads, results, cache, all)
    """
    import shutil
    
    if token and token in valid_admin_tokens:
        logger.debug("Using legacy admin token authentication")
    
    # 使用 server 模块内的数据目录
    server_dir = os.path.dirname(os.path.dirname(__file__))  # manga_translator/server
    data_dir = os.path.join(server_dir, "data")
    user_resources_dir = os.path.join(server_dir, "user_resources")
    
    targets = {
        "uploads": [
            os.path.join(user_resources_dir, "fonts"),
            os.path.join(user_resources_dir, "prompts")
        ],
        "results": [os.path.join(data_dir, "results")],
        "cache": []  # 暂无独立缓存目录
    }
    
    if target not in targets and target != "all":
        raise HTTPException(400, detail=f"无效的清理目标: {target}")
    
    freed_bytes = 0
    cleaned_dirs = []
    
    if target == "all":
        dirs_to_clean = []
        for dirs in targets.values():
            dirs_to_clean.extend(dirs)
    else:
        dirs_to_clean = targets[target]
    
    for dir_path in dirs_to_clean:
        if os.path.exists(dir_path):
            # 计算清理前的大小
            stats = get_directory_stats(dir_path)
            freed_bytes += stats["size"]
            
            # 清理目录内容（保留目录本身和 index.json）
            for item in os.listdir(dir_path):
                # 保留 index.json 文件
                if item == "index.json":
                    continue
                item_path = os.path.join(dir_path, item)
                try:
                    if os.path.isfile(item_path):
                        os.remove(item_path)
                    elif os.path.isdir(item_path):
                        shutil.rmtree(item_path)
                except Exception as e:
                    logger.warning(f"清理失败: {item_path}, 错误: {e}")
            
            cleaned_dirs.append(dir_path)
    
    logger.info(f"管理员 {session.username} 清理了 {target} 目录，释放 {freed_bytes} 字节")
    
    return {
        "success": True,
        "freed_bytes": freed_bytes,
        "cleaned_dirs": cleaned_dirs
    }
