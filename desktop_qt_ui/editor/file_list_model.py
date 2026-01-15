"""
文件列表模型 - 统一处理原图和翻译后的图
"""
import os
import json
from typing import List, Optional, Dict, Tuple
from enum import Enum
from dataclasses import dataclass


class FileType(Enum):
    """文件类型枚举"""
    SOURCE = "source"           # 原图（有JSON）
    TRANSLATED = "translated"   # 翻译后的图（有translation_map.json）
    UNTRANSLATED = "untranslated"  # 未翻译的图（既没有JSON也没有map）


@dataclass
class FileItem:
    """文件项数据类"""
    path: str                    # 文件路径
    file_type: FileType          # 文件类型
    json_path: Optional[str] = None      # JSON路径（如果是原图）
    map_path: Optional[str] = None       # translation_map.json路径（如果是翻译后的图）
    source_path: Optional[str] = None    # 对应的源文件路径（如果是翻译后的图）
    translated_path: Optional[str] = None  # 对应的翻译后文件路径（如果是原图）


class FileListModel:
    """
    文件列表模型 - 统一处理原图和翻译后的图
    
    核心逻辑：
    1. 检查目录中是否有 JSON 文件 → 原图
    2. 检查目录中是否有 translation_map.json → 翻译后的图
    3. 都没有 → 未翻译的图
    """
    
    def __init__(self):
        self.files: List[FileItem] = []
        self._map_cache: Dict[str, dict] = {}  # 缓存 translation_map.json
    
    def clear(self):
        """清空文件列表"""
        self.files.clear()
        self._map_cache.clear()
    
    def add_files(self, file_paths: List[str]) -> List[FileItem]:
        """
        添加文件到列表
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            添加的文件项列表
        """
        added_items = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue
            
            # 检查是否已存在
            norm_path = os.path.normpath(file_path)
            if any(os.path.normpath(item.path) == norm_path for item in self.files):
                continue
            
            # 识别文件类型
            file_item = self._identify_file(file_path)
            self.files.append(file_item)
            added_items.append(file_item)
        
        return added_items
    
    def remove_file(self, file_path: str) -> bool:
        """
        移除文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否成功移除
        """
        norm_path = os.path.normpath(file_path)
        for i, item in enumerate(self.files):
            if os.path.normpath(item.path) == norm_path:
                self.files.pop(i)
                return True
        return False
    
    def get_file_item(self, file_path: str) -> Optional[FileItem]:
        """
        获取文件项
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件项，如果不存在返回 None
        """
        norm_path = os.path.normpath(file_path)
        for item in self.files:
            if os.path.normpath(item.path) == norm_path:
                return item
        return None
    
    def get_all_files(self) -> List[FileItem]:
        """获取所有文件项"""
        return self.files.copy()
    
    def get_files_by_type(self, file_type: FileType) -> List[FileItem]:
        """
        获取指定类型的文件
        
        Args:
            file_type: 文件类型
            
        Returns:
            文件项列表
        """
        return [item for item in self.files if item.file_type == file_type]
    
    def _identify_file(self, file_path: str) -> FileItem:
        """
        识别文件类型
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件项
        """
        norm_path = os.path.normpath(file_path)
        file_dir = os.path.dirname(norm_path)
        file_name = os.path.basename(norm_path)
        file_name_no_ext = os.path.splitext(file_name)[0]
        
        # 1. 检查是否有对应的 JSON 文件（原图）
        json_path = self._find_json_file(file_dir, file_name_no_ext)
        if json_path:
            # 这是原图，尝试找到对应的翻译后文件
            translated_path = self._find_translated_file(norm_path)
            return FileItem(
                path=norm_path,
                file_type=FileType.SOURCE,
                json_path=json_path,
                translated_path=translated_path
            )
        
        # 2. 检查是否在 translation_map.json 中（翻译后的图）
        map_path = os.path.join(file_dir, 'translation_map.json')
        if os.path.exists(map_path):
            source_path = self._find_source_from_map(norm_path, map_path)
            if source_path:
                return FileItem(
                    path=norm_path,
                    file_type=FileType.TRANSLATED,
                    map_path=map_path,
                    source_path=source_path
                )
        
        # 3. 都没有，未翻译的图
        return FileItem(
            path=norm_path,
            file_type=FileType.UNTRANSLATED
        )
    
    def _find_json_file(self, file_dir: str, file_name_no_ext: str) -> Optional[str]:
        """
        查找对应的 JSON 文件
        
        优先从新目录结构查找，支持向后兼容
        """
        # 新目录结构：manga_translator_work/json/xxx_translations.json
        new_json_path = os.path.join(
            file_dir, 
            'manga_translator_work', 
            'json', 
            f'{file_name_no_ext}_translations.json'
        )
        if os.path.exists(new_json_path):
            return new_json_path
        
        # 旧目录结构：同目录下的 xxx_translations.json
        old_json_path = os.path.join(file_dir, f'{file_name_no_ext}_translations.json')
        if os.path.exists(old_json_path):
            return old_json_path
        
        return None
    
    def _find_translated_file(self, source_path: str) -> Optional[str]:
        """
        根据源文件路径查找翻译后的文件
        
        遍历所有可能的输出目录，查找 translation_map.json
        """
        # 这里简化处理，实际使用时可能需要从配置中获取输出目录
        # 暂时返回 None，由调用方处理
        return None
    
    def _find_source_from_map(self, translated_path: str, map_path: str) -> Optional[str]:
        """
        从 translation_map.json 中查找源文件路径
        
        Args:
            translated_path: 翻译后的文件路径
            map_path: translation_map.json 路径
            
        Returns:
            源文件路径，如果不存在返回 None
        """
        try:
            # 使用缓存
            if map_path not in self._map_cache:
                with open(map_path, 'r', encoding='utf-8') as f:
                    self._map_cache[map_path] = json.load(f)
            
            translation_map = self._map_cache[map_path]
            norm_translated = os.path.normpath(translated_path)
            
            # translation_map 的格式：{translated_path: source_path}
            source_path = translation_map.get(norm_translated)
            if source_path and os.path.exists(source_path):
                return source_path
        except Exception:
            pass
        
        return None
    
    def refresh_file(self, file_path: str) -> Optional[FileItem]:
        """
        刷新文件项（重新识别文件类型）
        
        Args:
            file_path: 文件路径
            
        Returns:
            更新后的文件项，如果不存在返回 None
        """
        norm_path = os.path.normpath(file_path)
        for i, item in enumerate(self.files):
            if os.path.normpath(item.path) == norm_path:
                # 重新识别
                new_item = self._identify_file(file_path)
                self.files[i] = new_item
                return new_item
        return None
