"""
Cleanup management API routes.

Provides endpoints for:
- Creating cleanup rules
- Getting cleanup rules
- Deleting cleanup rules
- Manual cleanup operations
"""

import logging
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from manga_translator.server.core.middleware import require_admin
from manga_translator.server.core.models import Session
from manga_translator.server.core.cleanup_service import CleanupSchedulerService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin/cleanup", tags=["cleanup"])

# Global service instance (will be initialized at server startup)
_cleanup_service: CleanupSchedulerService = None


def init_cleanup_routes(cleanup_service: CleanupSchedulerService) -> None:
    """
    Initialize cleanup routes with service instance.
    
    Args:
        cleanup_service: Cleanup scheduler service
    """
    global _cleanup_service
    _cleanup_service = cleanup_service
    logger.info("Cleanup routes initialized")


def get_cleanup_service() -> CleanupSchedulerService:
    """Get cleanup scheduler service instance."""
    if not _cleanup_service:
        raise RuntimeError("Cleanup service not initialized")
    return _cleanup_service


# ============================================================================
# Request/Response Models
# ============================================================================

class CreateCleanupRuleRequest(BaseModel):
    """Request model for creating cleanup rule."""
    level: str  # global, user_group, user
    retention_days: int
    target_id: Optional[str] = None


class CleanupRuleResponse(BaseModel):
    """Response model for cleanup rule."""
    id: str
    level: str
    target_id: Optional[str]
    retention_days: int
    enabled: bool
    created_at: Optional[str]
    created_by: Optional[str]


class ManualCleanupRequest(BaseModel):
    """Request model for manual cleanup."""
    user_id: Optional[str] = None
    user_group_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    session_tokens: Optional[List[str]] = None


class CleanupReportResponse(BaseModel):
    """Response model for cleanup report."""
    timestamp: str
    deleted_sessions_count: int
    deleted_sessions: List[str]
    deleted_files_count: int
    freed_space_mb: float
    freed_space_bytes: int
    errors: List[str]
    success: bool


class SessionPreview(BaseModel):
    """Preview of a session that would be deleted."""
    session_token: str
    user_id: str
    timestamp: str
    file_count: int
    total_size: int


# ============================================================================
# API Endpoints
# ============================================================================


@router.post("/rules", response_model=Dict)
async def create_cleanup_rule(
    request: CreateCleanupRuleRequest,
    session: Session = Depends(require_admin),
    cleanup_service: CleanupSchedulerService = Depends(get_cleanup_service)
):
    """
    Create a new cleanup rule.
    
    Args:
        request: Cleanup rule creation request
        session: Admin session
        cleanup_service: Cleanup scheduler service
    
    Returns:
        Created cleanup rule
        
    Validates: Requirement 6.1, 6.4
    """
    try:
        # Create cleanup rule
        rule = cleanup_service.configure_auto_cleanup(
            level=request.level,
            retention_days=request.retention_days,
            target_id=request.target_id,
            admin_id=session.user_id
        )
        
        logger.info(f"Admin {session.user_id} created cleanup rule {rule.id}")
        
        return {
            'success': True,
            'rule': rule.to_dict()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error creating cleanup rule: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/rules", response_model=Dict)
async def get_cleanup_rules(
    session: Session = Depends(require_admin),
    cleanup_service: CleanupSchedulerService = Depends(get_cleanup_service)
):
    """
    Get all cleanup rules.
    
    Args:
        session: Admin session
        cleanup_service: Cleanup scheduler service
    
    Returns:
        List of cleanup rules
        
    Validates: Requirement 6.4
    """
    try:
        rules = cleanup_service.get_cleanup_rules()
        
        return {
            'success': True,
            'rules': [rule.to_dict() for rule in rules],
            'count': len(rules)
        }
    
    except Exception as e:
        logger.error(f"Error getting cleanup rules: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.delete("/rules/{rule_id}", response_model=Dict)
async def delete_cleanup_rule(
    rule_id: str,
    session: Session = Depends(require_admin),
    cleanup_service: CleanupSchedulerService = Depends(get_cleanup_service)
):
    """
    Delete a cleanup rule.
    
    Args:
        rule_id: ID of the rule to delete
        session: Admin session
        cleanup_service: Cleanup scheduler service
    
    Returns:
        Success status
        
    Validates: Requirement 6.4
    """
    try:
        success = cleanup_service.delete_cleanup_rule(rule_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Cleanup rule not found")
        
        logger.info(f"Admin {session.user_id} deleted cleanup rule {rule_id}")
        
        return {
            'success': True,
            'message': 'Cleanup rule deleted successfully'
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting cleanup rule: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/manual", response_model=Dict)
async def manual_cleanup(
    request: ManualCleanupRequest,
    session: Session = Depends(require_admin),
    cleanup_service: CleanupSchedulerService = Depends(get_cleanup_service)
):
    """
    Execute manual cleanup with filters.
    
    Args:
        request: Manual cleanup request with filters
        session: Admin session
        cleanup_service: Cleanup scheduler service
    
    Returns:
        Cleanup report
        
    Validates: Requirements 7.1-7.5, 8.1-8.5
    """
    try:
        # Build filters
        filters = {}
        
        if request.user_id:
            filters['user_id'] = request.user_id
        
        if request.user_group_id:
            filters['user_group_id'] = request.user_group_id
        
        if request.start_date:
            filters['start_date'] = request.start_date
        
        if request.end_date:
            filters['end_date'] = request.end_date
        
        if request.session_tokens:
            filters['session_tokens'] = request.session_tokens
        
        # Get user group mapping (if needed)
        user_group_mapping = {}
        if request.user_group_id:
            # Load user group mapping from accounts
            # This would need to be implemented based on your account system
            pass
        
        # Execute manual cleanup
        report = cleanup_service.manual_cleanup(
            filters=filters,
            admin_id=session.user_id,
            user_group_mapping=user_group_mapping
        )
        
        logger.info(
            f"Admin {session.user_id} executed manual cleanup: "
            f"deleted {len(report.deleted_sessions)} sessions"
        )
        
        return {
            'success': True,
            'report': report.to_dict()
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    except Exception as e:
        logger.error(f"Error executing manual cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/preview", response_model=Dict)
async def preview_cleanup(
    request: ManualCleanupRequest,
    session: Session = Depends(require_admin),
    cleanup_service: CleanupSchedulerService = Depends(get_cleanup_service)
):
    """
    Preview what would be deleted by manual cleanup without actually deleting.
    
    Args:
        request: Manual cleanup request with filters
        session: Admin session
        cleanup_service: Cleanup scheduler service
    
    Returns:
        List of sessions that would be deleted
    """
    try:
        # Build filters
        filters = {}
        
        if request.user_id:
            filters['user_id'] = request.user_id
        
        if request.user_group_id:
            filters['user_group_id'] = request.user_group_id
        
        if request.start_date:
            filters['start_date'] = request.start_date
        
        if request.end_date:
            filters['end_date'] = request.end_date
        
        if request.session_tokens:
            filters['session_tokens'] = request.session_tokens
        
        # Get sessions that would be deleted
        from manga_translator.server.repositories.translation_repository import TranslationRepository
        from manga_translator.server.models import TranslationResult
        from datetime import datetime
        
        translation_repo = TranslationRepository(
            'manga_translator/server/data/translation_history.json'
        )
        
        all_sessions_data = translation_repo.get_all_sessions()
        all_sessions = [TranslationResult.from_dict(s) for s in all_sessions_data]
        
        # Apply filters
        matching_sessions = []
        
        for session in all_sessions:
            # Filter by specific session tokens
            if 'session_tokens' in filters:
                if session.session_token not in filters['session_tokens']:
                    continue
            
            # Filter by user_id
            if 'user_id' in filters:
                if session.user_id != filters['user_id']:
                    continue
            
            # Filter by date range
            try:
                session_time = datetime.fromisoformat(session.timestamp.replace('Z', '+00:00'))
                
                if 'start_date' in filters:
                    start_date = datetime.fromisoformat(filters['start_date'])
                    if session_time < start_date:
                        continue
                
                if 'end_date' in filters:
                    end_date = datetime.fromisoformat(filters['end_date'])
                    if session_time > end_date:
                        continue
            
            except (ValueError, AttributeError):
                continue
            
            matching_sessions.append(session)
        
        return {
            'success': True,
            'sessions_count': len(matching_sessions),
            'sessions': [
                {
                    'session_token': s.session_token,
                    'user_id': s.user_id,
                    'timestamp': s.timestamp,
                    'file_count': s.file_count,
                    'total_size': s.total_size
                }
                for s in matching_sessions
            ]
        }
    
    except Exception as e:
        logger.error(f"Error previewing cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


# ============================================================================
# Automatic Cleanup Endpoints
# ============================================================================

# Global scheduler instance (will be initialized at server startup)
_auto_cleanup_scheduler = None


def init_auto_cleanup_scheduler(scheduler) -> None:
    """Initialize the automatic cleanup scheduler."""
    global _auto_cleanup_scheduler
    _auto_cleanup_scheduler = scheduler
    logger.info("Auto cleanup scheduler initialized in routes")


def get_auto_cleanup_scheduler():
    """Get the automatic cleanup scheduler instance."""
    if not _auto_cleanup_scheduler:
        raise RuntimeError("Auto cleanup scheduler not initialized")
    return _auto_cleanup_scheduler


@router.get("/auto/status", response_model=Dict)
async def get_auto_cleanup_status(
    session: Session = Depends(require_admin)
):
    """
    Get automatic cleanup scheduler status.
    
    Args:
        session: Admin session
    
    Returns:
        Scheduler status information
        
    Validates: Requirement 6.3
    """
    try:
        scheduler = get_auto_cleanup_scheduler()
        status = scheduler.get_status()
        
        return {
            'success': True,
            'status': status
        }
    
    except Exception as e:
        logger.error(f"Error getting auto cleanup status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/auto/trigger", response_model=Dict)
async def trigger_auto_cleanup(
    session: Session = Depends(require_admin)
):
    """
    Manually trigger automatic cleanup to run immediately.
    
    Args:
        session: Admin session
    
    Returns:
        Cleanup report
        
    Validates: Requirement 6.3
    """
    try:
        scheduler = get_auto_cleanup_scheduler()
        report = scheduler.run_now()
        
        if not report:
            raise HTTPException(status_code=500, detail="Cleanup failed")
        
        logger.info(f"Admin {session.user_id} triggered automatic cleanup")
        
        return {
            'success': True,
            'report': report
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering auto cleanup: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/auto/history", response_model=Dict)
async def get_auto_cleanup_history(
    limit: int = 10,
    session: Session = Depends(require_admin)
):
    """
    Get automatic cleanup history.
    
    Args:
        limit: Maximum number of reports to return
        session: Admin session
    
    Returns:
        List of cleanup reports
        
    Validates: Requirement 6.3, 6.5
    """
    try:
        scheduler = get_auto_cleanup_scheduler()
        history = scheduler.get_cleanup_history(limit)
        
        return {
            'success': True,
            'history': history,
            'count': len(history)
        }
    
    except Exception as e:
        logger.error(f"Error getting auto cleanup history: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
