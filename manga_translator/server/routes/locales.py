"""
Locales API Routes

提供国际化语言文件的API端点。
从 desktop_qt_ui/locales/ 目录读取语言文件。

需求: 38.1, 38.6, 38.9
"""

# import os
import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

router = APIRouter(prefix='/api/locales', tags=['locales'])

# 获取locales目录路径
# 从server目录向上两级到项目根目录，然后进入desktop_qt_ui/locales
LOCALES_DIR = Path(__file__).parent.parent.parent.parent / 'desktop_qt_ui' / 'locales'

# 支持的语言列表
SUPPORTED_LANGUAGES = ['zh_CN', 'zh_TW', 'en_US', 'ja_JP', 'ko_KR', 'es_ES']


@router.get('/{lang}.json')
async def get_locale(lang: str):
    """
    获取指定语言的翻译文件
    
    Args:
        lang: 语言代码 (例如: zh_CN, en_US)
        
    Returns:
        JSON格式的翻译文件
        
    需求: 38.1, 38.6, 38.9
    """
    # 验证语言代码
    if lang not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=404,
            detail={
                'error': 'Language not supported',
                'supported_languages': SUPPORTED_LANGUAGES
            }
        )
    
    # 构建文件路径
    locale_file = LOCALES_DIR / f'{lang}.json'
    
    # 检查文件是否存在
    if not locale_file.exists():
        raise HTTPException(
            status_code=404,
            detail={
                'error': f'Locale file not found: {lang}.json',
                'path': str(locale_file)
            }
        )
    
    try:
        # 读取并返回JSON文件
        with open(locale_file, 'r', encoding='utf-8') as f:
            translations = json.load(f)
        
        return JSONResponse(content=translations)
    
    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Invalid JSON in locale file',
                'details': str(e)
            }
        )
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                'error': 'Error reading locale file',
                'details': str(e)
            }
        )


@router.get('/list')
async def list_locales():
    """
    获取所有可用的语言列表
    
    Returns:
        可用语言的列表，包含代码和名称
        
    需求: 38.1
    """
    languages = [
        {'code': 'zh_CN', 'name': '简体中文', 'nativeName': '简体中文'},
        {'code': 'zh_TW', 'name': '繁体中文', 'nativeName': '繁體中文'},
        {'code': 'en_US', 'name': 'English', 'nativeName': 'English'},
        {'code': 'ja_JP', 'name': 'Japanese', 'nativeName': '日本語'},
        {'code': 'ko_KR', 'name': 'Korean', 'nativeName': '한국어'},
        {'code': 'es_ES', 'name': 'Spanish', 'nativeName': 'Español'}
    ]
    
    # 检查每个语言文件是否存在
    available_languages = []
    for lang in languages:
        locale_file = LOCALES_DIR / f"{lang['code']}.json"
        if locale_file.exists():
            available_languages.append(lang)
    
    return JSONResponse(content={
        'languages': available_languages,
        'default': 'en_US'
    })


@router.get('/check')
async def check_locales():
    """
    检查locales目录和文件的状态
    
    Returns:
        locales目录的状态信息
        
    用于调试和诊断
    """
    status = {
        'locales_dir': str(LOCALES_DIR),
        'dir_exists': LOCALES_DIR.exists(),
        'files': {}
    }
    
    if LOCALES_DIR.exists():
        for lang in SUPPORTED_LANGUAGES:
            locale_file = LOCALES_DIR / f'{lang}.json'
            status['files'][lang] = {
                'exists': locale_file.exists(),
                'path': str(locale_file)
            }
            
            if locale_file.exists():
                try:
                    status['files'][lang]['size'] = locale_file.stat().st_size
                    # 尝试读取以验证JSON格式
                    with open(locale_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        status['files'][lang]['keys_count'] = len(data)
                        status['files'][lang]['valid'] = True
                except Exception as e:
                    status['files'][lang]['valid'] = False
                    status['files'][lang]['error'] = str(e)
    
    return JSONResponse(content=status)


def init_locales_routes(app):
    """
    初始化locales路由
    
    Args:
        app: FastAPI应用实例
    """
    app.include_router(router)
    print(f"Locales routes initialized. Locales directory: {LOCALES_DIR}")
    
    # 检查locales目录是否存在
    if not LOCALES_DIR.exists():
        print(f"WARNING: Locales directory not found: {LOCALES_DIR}")
    else:
        # 列出可用的语言文件
        available = []
        for lang in SUPPORTED_LANGUAGES:
            if (LOCALES_DIR / f'{lang}.json').exists():
                available.append(lang)
        print(f"Available languages: {', '.join(available)}")
