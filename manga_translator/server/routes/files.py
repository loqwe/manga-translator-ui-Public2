"""
Files routes module.

This module contains file management endpoints for the manga translator server.
"""

import os
from fastapi import APIRouter, Header, UploadFile, File, HTTPException, Depends

from manga_translator.server.core.config_manager import admin_settings, FONTS_DIR
from manga_translator.server.core.middleware import require_admin
from manga_translator.server.core.models import Session
from manga_translator.utils import BASE_PATH

router = APIRouter(tags=["files"])


# ============================================================================
# Font Management Endpoints
# ============================================================================

@router.post("/upload/font")
async def upload_font(
    file: UploadFile = File(...),
    session: Session = Depends(require_admin)
):
    """Upload a font file to server (admin only)"""
    # Check permissions
    if not admin_settings.get('permissions', {}).get('can_upload_fonts', True):
        raise HTTPException(403, detail="Font upload is disabled")
    
    if not file.filename.lower().endswith(('.ttf', '.otf', '.ttc')):
        raise HTTPException(400, detail="Invalid font file format")
    
    # 防止路径遍历攻击
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        raise HTTPException(400, detail="Invalid filename")
    
    os.makedirs(FONTS_DIR, exist_ok=True)
    
    file_path = os.path.join(FONTS_DIR, file.filename)
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {"success": True, "filename": file.filename}


@router.delete("/fonts/{filename}")
async def delete_font(
    filename: str,
    session: Session = Depends(require_admin)
):
    """Delete a font file (admin only)"""
    # Check permissions
    if not admin_settings.get('permissions', {}).get('can_delete_fonts', True):
        raise HTTPException(403, detail="Font deletion is disabled")
    
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(400, detail="Invalid filename")
    
    # Find in fonts directory
    file_path = os.path.join(FONTS_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Font file not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete file: {str(e)}")


# ============================================================================
# Prompt Management Endpoints
# ============================================================================

@router.post("/upload/prompt")
async def upload_prompt(
    file: UploadFile = File(...),
    session: Session = Depends(require_admin)
):
    """Upload a high-quality translation prompt file to server (admin only)"""
    # Check permissions
    if not admin_settings.get('permissions', {}).get('can_upload_prompts', True):
        raise HTTPException(403, detail="Prompt upload is disabled")
    
    if not file.filename.lower().endswith('.json'):
        raise HTTPException(400, detail="Invalid prompt file format (must be .json)")
    
    # 防止路径遍历攻击
    if '..' in file.filename or '/' in file.filename or '\\' in file.filename:
        raise HTTPException(400, detail="Invalid filename")
    
    # Prohibit uploading system prompt filenames
    if file.filename in ['system_prompt_hq.json', 'system_prompt_line_break.json', 'glossary_extraction_prompt.json']:
        raise HTTPException(403, detail="Cannot overwrite system prompt files")
    
    dict_dir = os.path.join(BASE_PATH, 'dict')
    os.makedirs(dict_dir, exist_ok=True)
    
    file_path = os.path.join(dict_dir, file.filename)
    with open(file_path, 'wb') as f:
        content = await file.read()
        f.write(content)
    
    return {"success": True, "filename": file.filename}


@router.get("/prompts")
async def list_prompts():
    """List available prompt files (excluding system prompts)"""
    try:
        dict_dir = os.path.join(BASE_PATH, 'dict')
        prompts = []
        
        print(f"[DEBUG] Listing prompts from: {dict_dir}")
        print(f"[DEBUG] dict_dir exists: {os.path.exists(dict_dir)}")
        
        # Read from dict directory (directory used by desktop version)
        if os.path.exists(dict_dir):
            files = os.listdir(dict_dir)
            print(f"[DEBUG] Found {len(files)} files in dict_dir")
            for f in files:
                # Filter out system prompt files
                if f.lower().endswith('.json') and f not in [
                    'system_prompt_hq.json',
                    'system_prompt_line_break.json',
                    'glossary_extraction_prompt.json'
                ]:
                    prompts.append(f)
                    print(f"[DEBUG] Added prompt: {f}")
        
        print(f"[DEBUG] Returning {len(prompts)} prompts")
        return sorted(prompts)
    except Exception as e:
        print(f"[ERROR] Failed to list prompts: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(500, detail=f"Failed to list prompts: {str(e)}")


@router.get("/prompts/{filename}")
async def get_prompt(filename: str, token: str = Header(alias="X-Admin-Token", default=None)):
    """Get prompt file content (admin only)"""
    dict_dir = os.path.join(BASE_PATH, 'dict')
    
    # Find in dict directory
    file_path = os.path.join(dict_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Prompt file not found")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    return {"filename": filename, "content": content}


@router.delete("/prompts/{filename}")
async def delete_prompt(
    filename: str,
    session: Session = Depends(require_admin)
):
    """Delete a prompt file (admin only)"""
    # Check permissions
    if not admin_settings.get('permissions', {}).get('can_delete_prompts', True):
        raise HTTPException(403, detail="Prompt deletion is disabled")
    
    # 防止路径遍历攻击
    if '..' in filename or '/' in filename or '\\' in filename:
        raise HTTPException(400, detail="Invalid filename")
    
    dict_dir = os.path.join(BASE_PATH, 'dict')
    
    # Prohibit deleting system prompts
    if filename in ['system_prompt_hq.json', 'system_prompt_line_break.json', 'glossary_extraction_prompt.json']:
        raise HTTPException(403, detail="Cannot delete system prompt files")
    
    # Find in dict directory
    file_path = os.path.join(dict_dir, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(404, detail="Prompt file not found")
    
    try:
        os.remove(file_path)
        return {"success": True, "message": f"Deleted {filename}"}
    except Exception as e:
        raise HTTPException(500, detail=f"Failed to delete file: {str(e)}")
