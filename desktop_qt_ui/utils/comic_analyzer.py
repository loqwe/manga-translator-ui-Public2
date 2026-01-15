#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
漫画章节分析器模块
功能：分析漫画文件夹结构，识别章节更新情况
"""

import re
from pathlib import Path
from typing import Dict, List


class ComicAnalyzer:
    """漫画章节分析器"""
    
    def __init__(self, download_folder: str, translated_folder: str, storage_folder: str):
        """
        初始化分析器
        
        Args:
            download_folder: 下载文件夹路径（第一层：网站名，第二层：漫画名，第三层：章节）
            translated_folder: 翻译输出文件夹路径（第一层：漫画名，第二层：章节）
            storage_folder: 存储文件夹路径（第一层：漫画名，包含CBZ文件）
        """
        self.download_folder = Path(download_folder)
        self.translated_folder = Path(translated_folder)
        self.storage_folder = Path(storage_folder)
        
    def analyze_updates(self) -> Dict[str, Dict]:
        """
        分析所有漫画的更新情况
        
        Returns:
            字典，键为漫画名，值为包含各来源章节列表的字典
            {
                '漫画名': {
                    'raw_chapters': [...],      # 生肉章节列表
                    'translated_chapters': [...],  # 已翻译章节列表
                    'stored_chapters': [...]    # 已存储章节列表（CBZ）
                }
            }
        """
        results = {}
        
        # 分析下载文件夹（生肉）
        if self.download_folder.exists():
            for site_folder in self.download_folder.iterdir():
                if not site_folder.is_dir():
                    continue
                    
                for comic_folder in site_folder.iterdir():
                    if not comic_folder.is_dir():
                        continue
                        
                    comic_name = comic_folder.name
                    chapters = self._get_chapters(comic_folder)
                    
                    if comic_name not in results:
                        results[comic_name] = {
                            'raw_chapters': [],
                            'translated_chapters': [],
                            'stored_chapters': []
                        }
                    results[comic_name]['raw_chapters'].extend(chapters)
        
        # 分析翻译输出文件夹
        if self.translated_folder.exists():
            for comic_folder in self.translated_folder.iterdir():
                if not comic_folder.is_dir():
                    continue
                    
                comic_name = comic_folder.name
                chapters = self._get_chapters(comic_folder)
                
                if comic_name not in results:
                    results[comic_name] = {
                        'raw_chapters': [],
                        'translated_chapters': [],
                        'stored_chapters': []
                    }
                results[comic_name]['translated_chapters'].extend(chapters)
        
        # 分析存储文件夹（CBZ文件）
        if self.storage_folder.exists():
            for comic_folder in self.storage_folder.iterdir():
                if not comic_folder.is_dir():
                    continue
                    
                comic_name = comic_folder.name
                # 检查 CBZ 文件
                cbz_files = list(comic_folder.glob('*.cbz'))
                chapters = [f.stem for f in cbz_files]
                
                if comic_name not in results:
                    results[comic_name] = {
                        'raw_chapters': [],
                        'translated_chapters': [],
                        'stored_chapters': []
                    }
                results[comic_name]['stored_chapters'].extend(chapters)
        
        return results
    
    def _get_chapters(self, comic_folder: Path, max_depth: int = 3) -> List[str]:
        """
        获取漫画文件夹中的所有章节（递归查找）
        
        Args:
            comic_folder: 漫画文件夹路径
            max_depth: 最大递归深度
            
        Returns:
            章节名称列表，已排序
        """
        chapters = []
        self._find_chapters_recursive(comic_folder, chapters, current_depth=0, max_depth=max_depth)
        return sorted(list(set(chapters)), key=self._extract_chapter_number)
    
    def _find_chapters_recursive(self, folder: Path, chapters: List[str], current_depth: int, max_depth: int):
        """
        递归查找章节文件夹
        
        Args:
            folder: 当前文件夹
            chapters: 章节列表（会被修改）
            current_depth: 当前深度
            max_depth: 最大深度
        """
        if current_depth > max_depth:
            return
        
        for item in folder.iterdir():
            if not item.is_dir():
                continue
            
            # 检查文件夹名是否包含章节号
            if self._is_chapter_folder(item):
                chapters.append(item.name)
            else:
                # 继续向下查找
                self._find_chapters_recursive(item, chapters, current_depth + 1, max_depth)
    
    def _is_chapter_folder(self, folder: Path) -> bool:
        """
        判断是否为章节文件夹（包含章节号）
        
        Args:
            folder: 文件夹路径
            
        Returns:
            是否为章节文件夹
        """
        folder_name = folder.name.lower()
        
        # 常见的章节关键词
        chapter_keywords = ['chapter', 'ch', '第', '话', 'episode', 'ep', 'vol', 'volume']
        
        # 检查是否包含章节关键词和数字
        has_keyword = any(keyword in folder_name for keyword in chapter_keywords)
        has_number = bool(re.search(r'\d+', folder_name))
        
        # 如果包含章节关键词和数字，或者纯数字文件夹名，则认为是章节文件夹
        if has_keyword and has_number:
            return True
        
        # 检查是否为纯数字或数字开头的文件夹
        if re.match(r'^\d+', folder_name):
            return True
        
        return False
    
    def _extract_chapter_number(self, chapter_name: str) -> float:
        """
        从章节名中提取章节号用于排序
        
        Args:
            chapter_name: 章节名称（如 "Chapter 1", "第3话"等）
            
        Returns:
            章节号（浮点数）
        """
        # 尝试匹配数字（包括小数）
        match = re.search(r'(\d+\.?\d*)', chapter_name)
        if match:
            return float(match.group(1))
        return 0
    
    def get_latest_chapter(self, comic_name: str, results: Dict) -> str:
        """
        获取指定漫画的最新章节
        
        Args:
            comic_name: 漫画名称
            results: analyze_updates()返回的结果字典
            
        Returns:
            最新章节名称
        """
        if comic_name not in results:
            return "未找到漫画"
            
        # 合并所有来源的章节
        all_chapters = (results[comic_name]['raw_chapters'] + 
                       results[comic_name]['translated_chapters'] + 
                       results[comic_name]['stored_chapters'])
        
        if not all_chapters:
            return "未找到章节"
        
        # 去重并排序
        sorted_chapters = sorted(set(all_chapters), key=self._extract_chapter_number)
        return sorted_chapters[-1] if sorted_chapters else "未找到章节"
    
    def get_summary(self, results: Dict) -> str:
        """
        生成分析摘要报告
        
        Args:
            results: analyze_updates()返回的结果字典
            
        Returns:
            格式化的摘要文本
        """
        from datetime import datetime
        
        output = []
        output.append("=" * 60)
        output.append("漫画更新分析报告")
        output.append(f"分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        output.append("=" * 60)
        output.append("")
        
        for comic_name in sorted(results.keys()):
            data = results[comic_name]
            output.append(f"【{comic_name}】")
            
            latest = self.get_latest_chapter(comic_name, results)
            output.append(f"  最新章节: {latest}")
            
            if data['raw_chapters']:
                output.append(f"  生肉章节: {len(data['raw_chapters'])} 话")
            if data['translated_chapters']:
                output.append(f"  已翻译: {len(data['translated_chapters'])} 话")
            if data['stored_chapters']:
                output.append(f"  已存储: {len(data['stored_chapters'])} 话 (CBZ)")
            
            output.append("")
        
        if not results:
            output.append("未找到任何漫画")
        else:
            output.append("=" * 60)
            output.append(f"总计: {len(results)} 部漫画")
        
        return "\n".join(output)
