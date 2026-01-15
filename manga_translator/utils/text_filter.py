# 文本过滤工具
import os
import sys
import re
from typing import List, Optional, Tuple, Pattern

from . import get_logger

logger = get_logger('TextFilter')

# Filter list cache: (contains, exact, regex)
_filter_lists: Optional[Tuple[List[str], List[str], List[Pattern]]] = None

# 默认过滤列表内容
_DEFAULT_FILTER_LIST_CONTENT = """# 过滤文本列表
# 一行一个，不区分大小写
# 以 # 开头的行为注释
# 匹配的文本区域会被跳过（不翻译、不擦除、不渲染）

[包含过滤]
# 原文「包含」这些文本就过滤
# 示例：
# 广告
# 水印

[精确过滤]
# 原文必须「完全等于」这些文本才过滤
# 示例：
# v.com
# ©

[正则过滤]
# 正则表达式匹配（不区分大小写）
# 示例：
# https?://\S+
"""


def _get_filter_list_path() -> str:
    """
    获取过滤列表文件路径
    
    打包环境：_internal/examples/filter_list.txt
    开发环境：项目根目录/examples/filter_list.txt
    """
    if getattr(sys, 'frozen', False):
        # 打包环境
        if hasattr(sys, '_MEIPASS'):
            # PyInstaller 打包：_MEIPASS 指向 _internal 目录
            return os.path.join(sys._MEIPASS, 'examples', 'filter_list.txt')
        else:
            # 其他打包方式
            return os.path.join(os.path.dirname(sys.executable), 'examples', 'filter_list.txt')
    else:
        # 开发环境：从当前文件向上找到项目根目录
        # text_filter.py -> utils -> manga_translator -> 项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return os.path.join(project_root, 'examples', 'filter_list.txt')


def ensure_filter_list_exists() -> str:
    """
    确保过滤列表文件存在，如果不存在则创建默认文件
    
    Returns:
        过滤列表文件路径
    """
    filter_path = _get_filter_list_path()
    
    if not os.path.exists(filter_path):
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(filter_path), exist_ok=True)
            # 创建默认文件
            with open(filter_path, 'w', encoding='utf-8') as f:
                f.write(_DEFAULT_FILTER_LIST_CONTENT)
            logger.info(f"已创建过滤列表文件: {filter_path}")
        except Exception as e:
            logger.error(f"创建过滤列表文件失败: {e}")
    
    return filter_path


def load_filter_list(force_reload: bool = False) -> Tuple[List[str], List[str], List[Pattern]]:
    """
    加载过滤列表
    
    Args:
        force_reload: 是否强制重新加载
    
    Returns:
        (contains, exact, regex)
        contains/exact are lowercase, regex are compiled patterns.
    """
    global _filter_lists
    
    if _filter_lists is not None and not force_reload:
        return _filter_lists
    
    contains_list = []
    exact_list = []
    regex_list = []
    filter_path = _get_filter_list_path()
    
    if not os.path.exists(filter_path):
        _filter_lists = (contains_list, exact_list, regex_list)
        return _filter_lists
    
    try:
        current_section = None
        with open(filter_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                # 跳过空行和注释
                if not line or line.startswith('#'):
                    continue
                
                # 检查区域标记
                if line == '[包含过滤]':
                    current_section = 'contains'
                    continue
                elif line == '[精确过滤]':
                    current_section = 'exact'
                    continue
                elif line == '[正则过滤]':
                    current_section = 'regex'
                    continue
                
                # 添加到对应列表
                if current_section == 'contains':
                    contains_list.append(line.lower())
                elif current_section == 'exact':
                    exact_list.append(line.lower())
                elif current_section == 'regex':
                    try:
                        pattern = re.compile(line, re.IGNORECASE)
                        regex_list.append(pattern)
                    except re.error as e:
                        logger.warning(f"正则表达式编译失败 (第{line_num}行): {line} - {e}")
        
        if contains_list or exact_list or regex_list:
            logger.info(f"已加载过滤规则: 包含过滤 {len(contains_list)} 条, 精确过滤 {len(exact_list)} 条, 正则过滤 {len(regex_list)} 条")
        
        _filter_lists = (contains_list, exact_list, regex_list)
    except Exception as e:
        logger.error(f"加载过滤列表失败: {e}")
        _filter_lists = ([], [], [])
    
    return _filter_lists


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
