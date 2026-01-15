#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CBZ压缩器模块
功能：将漫画章节文件夹压缩为CBZ格式，避免重复打包
"""

from pathlib import Path
from zipfile import ZipFile, ZIP_STORED
from typing import List, Tuple, Set


class CBZCompressor:
    """CBZ 压缩器"""
    
    def __init__(self, processed_file: str = "processed_folders.txt"):
        """
        初始化压缩器
        
        Args:
            processed_file: 记录已处理文件夹的文件路径
        """
        self.processed_file = Path(processed_file)
        self.processed = self._load_processed()
    
    def _load_processed(self) -> Set[str]:
        """
        加载已处理的文件夹列表
        
        Returns:
            已处理文件夹路径集合
        """
        if self.processed_file.exists():
            try:
                with open(self.processed_file, 'r', encoding='utf-8') as f:
                    return set(line.strip() for line in f if line.strip())
            except Exception as e:
                print(f"加载已处理列表失败: {e}")
                return set()
        return set()
    
    def _save_processed(self):
        """保存已处理的文件夹列表"""
        try:
            with open(self.processed_file, 'w', encoding='utf-8') as f:
                for item in sorted(self.processed):
                    f.write(f"{item}\n")
        except Exception as e:
            print(f"保存已处理列表失败: {e}")
    
    def compress_folders(self, base_folder: Path, recursive: bool = True) -> List[Tuple[str, str]]:
        """
        压缩文件夹为 CBZ 格式
        
        结构：base_folder / 漫画名 / 章节名 / 图片文件
        将每个章节文件夹压缩为 CBZ
        
        Args:
            base_folder: 基础文件夹路径
            recursive: 是否递归处理子文件夹
            
        Returns:
            压缩记录列表 [(源文件夹, CBZ文件), ...]
        """
        compressed = []
        
        if not base_folder.exists():
            print(f"文件夹不存在: {base_folder}")
            return compressed
        
        # 遍历漫画文件夹（第一层）
        for comic_folder in base_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            # 遍历章节文件夹（第二层）
            for chapter_folder in comic_folder.iterdir():
                if not chapter_folder.is_dir():
                    continue
                
                # 生成唯一标识符（相对路径）
                folder_id = str(chapter_folder.relative_to(base_folder))
                
                # 检查是否已处理
                if folder_id in self.processed:
                    continue
                
                # 检查是否有图片文件
                image_files = self._get_image_files(chapter_folder)
                if not image_files:
                    print(f"跳过（无图片）: {chapter_folder.name}")
                    continue
                
                # 生成 CBZ 文件路径（与章节文件夹同级）
                cbz_path = chapter_folder.parent / f"{chapter_folder.name}.cbz"
                
                # 如果 CBZ 已存在，跳过
                if cbz_path.exists():
                    print(f"跳过（已存在）: {cbz_path.name}")
                    self.processed.add(folder_id)
                    self._save_processed()
                    continue
                
                # 执行压缩
                try:
                    success = self._create_cbz(chapter_folder, cbz_path, image_files)
                    if success:
                        compressed.append((str(chapter_folder), str(cbz_path)))
                        self.processed.add(folder_id)
                        self._save_processed()
                        print(f"✓ 已压缩: {chapter_folder.name} -> {cbz_path.name}")
                except Exception as e:
                    print(f"✗ 压缩失败 {chapter_folder.name}: {e}")
        
        return compressed
    
    def _create_cbz(self, source_folder: Path, cbz_path: Path, image_files: List[Path]) -> bool:
        """
        创建 CBZ 文件
        
        Args:
            source_folder: 源文件夹
            cbz_path: 目标 CBZ 文件路径
            image_files: 图片文件列表
            
        Returns:
            是否成功创建
        """
        try:
            with ZipFile(cbz_path, 'w', ZIP_STORED) as cbz:
                for img_file in sorted(image_files):
                    # 使用相对路径作为压缩包内的文件名
                    arcname = img_file.name
                    cbz.write(img_file, arcname)
            return True
        except Exception as e:
            print(f"创建CBZ失败: {e}")
            # 删除不完整的文件
            if cbz_path.exists():
                cbz_path.unlink()
            return False
    
    def compress_folder(self, chapter_folder: Path) -> str:
        """
        压缩单个章节文件夹为 CBZ 格式
        
        Args:
            chapter_folder: 章节文件夹路径
            
        Returns:
            CBZ文件路径，失败返回None
        """
        if not chapter_folder.is_dir():
            return None
        
        # 检查是否有图片文件
        image_files = self._get_image_files(chapter_folder)
        if not image_files:
            return None
        
        # 生成 CBZ 文件路径（与章节文件夹同级）
        cbz_path = chapter_folder.parent / f"{chapter_folder.name}.cbz"
        
        # 如果 CBZ 已存在，跳过
        if cbz_path.exists():
            return str(cbz_path)
        
        # 执行压缩
        try:
            success = self._create_cbz(chapter_folder, cbz_path, image_files)
            if success:
                return str(cbz_path)
        except Exception as e:
            print(f"压缩失败 {chapter_folder.name}: {e}")
        
        return None
    
    def _get_image_files(self, folder: Path) -> List[Path]:
        """
        获取文件夹中的所有图片文件
        
        Args:
            folder: 文件夹路径
            
        Returns:
            图片文件路径列表
        """
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
        image_files = []
        
        for file in folder.iterdir():
            if file.is_file() and file.suffix.lower() in image_extensions:
                image_files.append(file)
        
        return image_files
    
    def clear_processed_history(self):
        """清空已处理记录（慎用）"""
        self.processed.clear()
        self._save_processed()
        print("已清空处理记录")
    
    def get_processed_count(self) -> int:
        """获取已处理的文件夹数量"""
        return len(self.processed)
    
    def is_processed(self, folder_path: Path, base_folder: Path) -> bool:
        """
        检查文件夹是否已处理
        
        Args:
            folder_path: 要检查的文件夹路径
            base_folder: 基础文件夹路径
            
        Returns:
            是否已处理
        """
        folder_id = str(folder_path.relative_to(base_folder))
        return folder_id in self.processed
