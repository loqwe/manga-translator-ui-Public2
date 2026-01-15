"""
Routes module for manga_translator server.

This module contains all API route definitions organized by functionality.
"""

from manga_translator.server.routes.translation import router as translation_router
from manga_translator.server.routes.admin import router as admin_router
from manga_translator.server.routes.config import router as config_router
from manga_translator.server.routes.files import router as files_router
from manga_translator.server.routes.web import router as web_router
from manga_translator.server.routes.users import router as users_router
from manga_translator.server.routes.audit import router as audit_router
from manga_translator.server.routes.auth import router as auth_router, init_auth_services
from manga_translator.server.routes.groups import router as groups_router
from manga_translator.server.routes.resources import router as resources_router, init_resource_routes
from manga_translator.server.routes.history import router as history_router, init_history_routes
from manga_translator.server.routes.quota import router as quota_router, init_quota_routes
# Cleanup routes - 延迟导入避免循环依赖
try:
    from manga_translator.server.routes.cleanup import router as cleanup_router, init_cleanup_routes, init_auto_cleanup_scheduler
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Cleanup routes not available: {e}")
    cleanup_router = None
    init_cleanup_routes = None
    init_auto_cleanup_scheduler = None
from manga_translator.server.routes.config_management import router as config_management_router
from manga_translator.server.routes.logs import logs_router
from manga_translator.server.routes.locales import router as locales_router, init_locales_routes

# Import sessions router
from manga_translator.server.routes.sessions import router as sessions_router

__all__ = [
    'translation_router',
    'admin_router',
    'config_router',
    'files_router',
    'web_router',
    'users_router',
    'audit_router',
    'auth_router',
    'init_auth_services',
    'groups_router',
    'resources_router',
    'init_resource_routes',
    'history_router',
    'init_history_routes',
    'quota_router',
    'init_quota_routes',
    'cleanup_router',
    'init_cleanup_routes',
    'init_auto_cleanup_scheduler',
    'config_management_router',
    'logs_router',
    'locales_router',
    'init_locales_routes',
    'sessions_router',
]
