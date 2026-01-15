#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
漫画名称替换器模块
功能：管理生肉和熟肉漫画名称的映射关系，批量替换文件夹名称
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


class NameReplacer:
    """漫画名称替换器"""
    
    def __init__(self, mapping_file: str = "name_mapping.json"):
        """
        初始化名称替换器
        
        Args:
            mapping_file: 名称映射配置文件路径
        """
        self.mapping_file = Path(mapping_file)
        self.mapping = self._load_mapping()
    
    def _load_mapping(self) -> Dict[str, str]:
        """
        从JSON文件加载名称映射
        
        Returns:
            映射字典 {生肉名: 熟肉名}
        """
        if self.mapping_file.exists():
            try:
                with open(self.mapping_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"加载映射文件失败: {e}")
                return {}
        return {}
    
    def _save_mapping(self):
        """保存名称映射到JSON文件"""
        try:
            with open(self.mapping_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存映射文件失败: {e}")
    
    def add_mapping(self, raw_name: str, translated_name: str):
        """
        添加或更新名称映射
        
        Args:
            raw_name: 生肉漫画名
            translated_name: 熟肉漫画名
        """
        self.mapping[raw_name] = translated_name
        self._save_mapping()
    
    def remove_mapping(self, raw_name: str) -> bool:
        """
        删除名称映射
        
        Args:
            raw_name: 要删除的生肉漫画名
            
        Returns:
            是否成功删除
        """
        if raw_name in self.mapping:
            del self.mapping[raw_name]
            self._save_mapping()
            return True
        return False
    
    def get_translated_name(self, raw_name: str) -> str:
        """
        获取翻译后的名称
        支持多个生肉名用 | 分隔：韩文名|英文名|日文名 → 中文名
        
        Args:
            raw_name: 生肉漫画名
            
        Returns:
            熟肉漫画名，如果没有映射则返回原名
        """
        # 首先检查直接匹配
        if raw_name in self.mapping:
            return self.mapping[raw_name]
        
        # 检查是否匹配任何包含 | 的映射键
        for key, value in self.mapping.items():
            if '|' in key:
                # 分割多个名称
                variants = [v.strip() for v in key.split('|')]
                if raw_name in variants:
                    return value
        
        return raw_name
    
    def replace_folder_names(self, base_folder: Path) -> List[Tuple[str, str]]:
        """
        批量替换文件夹名称（只替换第一层子文件夹）
        
        Args:
            base_folder: 要处理的基础文件夹
            
        Returns:
            替换记录列表 [(旧名称, 新名称), ...]
        """
        replaced = []
        
        if not base_folder.exists():
            return replaced
        
        for folder in base_folder.iterdir():
            if not folder.is_dir():
                continue
            
            old_name = folder.name
            new_name = self.get_translated_name(old_name)
            
            # 如果名称相同，跳过
            if old_name == new_name:
                continue
            
            new_path = folder.parent / new_name
            
            # 如果目标路径已存在，跳过（避免冲突）
            if new_path.exists():
                print(f"跳过 {old_name}: 目标文件夹 {new_name} 已存在")
                continue
            
            try:
                folder.rename(new_path)
                replaced.append((old_name, new_name))
                print(f"✓ 重命名: {old_name} -> {new_name}")
            except Exception as e:
                print(f"✗ 重命名失败 {old_name}: {e}")
        
        return replaced
    
    def get_all_mappings(self) -> Dict[str, str]:
        """
        获取所有名称映射
        
        Returns:
            完整的映射字典
        """
        return self.mapping.copy()
    
    def import_mappings(self, mappings: Dict[str, str], merge: bool = True):
        """
        导入名称映射
        
        Args:
            mappings: 要导入的映射字典
            merge: 是否与现有映射合并，False则覆盖
        """
        if merge:
            self.mapping.update(mappings)
        else:
            self.mapping = mappings.copy()
        self._save_mapping()
    
    def export_mappings(self, export_file: str):
        """
        导出名称映射到文件
        
        Args:
            export_file: 导出文件路径
        """
        try:
            with open(export_file, 'w', encoding='utf-8') as f:
                json.dump(self.mapping, f, ensure_ascii=False, indent=2)
            print(f"映射已导出到: {export_file}")
        except Exception as e:
            print(f"导出失败: {e}")
