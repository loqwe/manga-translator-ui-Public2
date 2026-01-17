# 文本过滤工具 - 统一从 watermark_filter.json 加载规则
import json
import os
import sys
import re
from typing import List, Optional, Tuple, Pattern

from . import get_logger

logger = get_logger('TextFilter')

# Filter list cache: (contains, exact, regex)
_filter_lists: Optional[Tuple[List[str], List[str], List[Pattern]]] = None


def _get_watermark_filter_path() -> str:
    """
    获取水印过滤配置文件路径
    
    打包环境：_internal/examples/config/watermark_filter.json
    开发环境：项目根目录/examples/config/watermark_filter.json
    """
    if getattr(sys, 'frozen', False):
        # 打包环境
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(sys._MEIPASS, 'examples', 'config', 'watermark_filter.json')
        else:
            return os.path.join(os.path.dirname(sys.executable), 'examples', 'config', 'watermark_filter.json')
    else:
        # 开发环境
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) 
        return os.path.join(project_root, 'examples', 'config', 'watermark_filter.json')


def load_filter_list(force_reload: bool = False) -> Tuple[List[str], List[str], List[Pattern]]:
    """
    从 watermark_filter.json 加载过滤规则
    
    规则映射：
    - partial_match_patterns (regex=false) -> contains_list (包含匹配)
    - exact_match_patterns (regex=false) -> exact_list (精确匹配)
    - 任何 regex=true 的规则 -> regex_list (正则匹配)
    
    Args:
        force_reload: 是否强制重新加载
    
    Returns:
        (contains_list, exact_list, regex_list)
    """
    global _filter_lists
    
    if _filter_lists is not None and not force_reload:
        return _filter_lists
    
    contains_list = []
    exact_list = []
    regex_list = []
    config_path = _get_watermark_filter_path()
    
    if not os.path.exists(config_path):
        logger.debug(f"过滤配置文件不存在: {config_path}")
        _filter_lists = (contains_list, exact_list, regex_list)
        return _filter_lists
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 支持 groups 格式
        if isinstance(config, dict) and isinstance(config.get('groups'), list):
            for group in config['groups']:
                if not isinstance(group, dict):
                    continue
                
                group_name = group.get('name', '未命名')
                
                # 处理 partial_match_patterns -> contains 或 regex
                for pattern in group.get('partial_match_patterns', []) or []:
                    _process_pattern(pattern, 'partial', contains_list, regex_list, group_name)
                
                # 处理 exact_match_patterns -> exact 或 regex
                for pattern in group.get('exact_match_patterns', []) or []:
                    _process_pattern(pattern, 'exact', exact_list, regex_list, group_name)
        
        # 输出统一的加载日志
        total_rules = len(contains_list) + len(exact_list) + len(regex_list)
        if total_rules > 0:
            logger.info(f"已加载过滤规则: 包含过滤 {len(contains_list)} 条, 精确过滤 {len(exact_list)} 条, 正则过滤 {len(regex_list)} 条 (来源: watermark_filter.json)")
        
        _filter_lists = (contains_list, exact_list, regex_list)
    except json.JSONDecodeError as e:
        logger.error(f"解析 watermark_filter.json 失败: {e}")
        _filter_lists = ([], [], [])
    except Exception as e:
        logger.error(f"加载过滤规则失败: {e}")
        _filter_lists = ([], [], [])
    
    return _filter_lists


def _process_pattern(pattern, mode: str, text_list: List[str], regex_list: List[Pattern], group_name: str):
    """
    处理单个规则，根据类型添加到对应列表
    
    Args:
        pattern: 规则（字符串或字典）
        mode: 'partial' 或 'exact'
        text_list: 文本匹配列表 (contains 或 exact)
        regex_list: 正则匹配列表
        group_name: 组名（用于日志）
    """
    if isinstance(pattern, str):
        # 简单字符串规则
        if pattern.strip():
            text_list.append(pattern.lower())
    elif isinstance(pattern, dict):
        pat = str(pattern.get('pattern', '')).strip()
        if not pat:
            return
        
        is_regex = bool(pattern.get('regex', False))
        case_sensitive = bool(pattern.get('case_sensitive', False))
        
        if is_regex:
            # 正则规则
            try:
                flags = 0 if case_sensitive else re.IGNORECASE
                compiled = re.compile(pat, flags)
                regex_list.append(compiled)
            except re.error as e:
                logger.warning(f"[组:{group_name}] 正则编译失败: {pat} - {e}")
        else:
            # 普通文本规则
            text_list.append(pat.lower() if not case_sensitive else pat)


def match_filter(text: str) -> Optional[Tuple[str, str]]:
    """
    检查文本是否匹配过滤列表
    
    Args:
        text: 要检查的文本
    
    Returns:
        (匹配的过滤词, 匹配类型)，如果没有匹配返回 None
        匹配类型: "包含" 或 "精确" 或 "正则"
    """
    if not text:
        return None
    
    contains_list, exact_list, regex_list = load_filter_list()
    text_lower = text.lower()
    
    # 先检查精确匹配
    for filter_word in exact_list:
        if text_lower == filter_word:
            return (filter_word, "精确")
    
    # 再检查包含匹配
    for filter_word in contains_list:
        if filter_word in text_lower:
            return (filter_word, "包含")
    
    # 最后检查正则匹配
    for pattern in regex_list:
        if pattern.search(text):
            return (pattern.pattern, "正则")
    
    return None


def should_filter(text: str) -> bool:
    """
    检查文本是否应该被过滤
    
    Args:
        text: 要检查的文本
    
    Returns:
        True 如果应该过滤，False 否则
    """
    return match_filter(text) is not None


def ensure_filter_list_exists() -> None:
    """
    确保过滤配置文件存在，如果不存在则创建默认配置
    
    在应用启动时调用，保证 watermark_filter.json 存在
    """
    config_path = _get_watermark_filter_path()
    
    if os.path.exists(config_path):
        return
    
    # Create default config structure
    default_config = {
        "groups": [
            {
                "name": "示例组",
                "partial_match_patterns": [],
                "exact_match_patterns": []
            }
        ]
    }
    
    # Ensure directory exists
    config_dir = os.path.dirname(config_path)
    if config_dir and not os.path.exists(config_dir):
        try:
            os.makedirs(config_dir, exist_ok=True)
        except OSError as e:
            logger.warning(f"无法创建过滤配置目录: {config_dir} - {e}")
            return
    
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        logger.info(f"已创建默认过滤配置文件: {config_path}")
    except OSError as e:
        logger.warning(f"无法创建过滤配置文件: {config_path} - {e}")
