#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CBZ文件转移器模块
功能：自动将翻译完成的CBZ文件转移到存储文件夹
"""

import shutil
from pathlib import Path
from typing import List, Tuple


class CBZTransfer:
    """CBZ 文件转移器"""
    
    def __init__(self, source_folder: Path, target_folder: Path):
        """
        初始化转移器
        
        Args:
            source_folder: 源文件夹路径（翻译输出文件夹）
            target_folder: 目标文件夹路径（存储文件夹）
        """
        self.source_folder = Path(source_folder)
        self.target_folder = Path(target_folder)
    
    def transfer_cbz_files(self, move: bool = True) -> List[Tuple[str, str]]:
        """
        转移 CBZ 文件到目标文件夹
        
        结构：
        - 源: source_folder / 漫画名 / 章节.cbz
        - 目标: target_folder / 漫画名 / 章节.cbz
        
        Args:
            move: True=移动文件，False=复制文件
            
        Returns:
            转移记录列表 [(源路径, 目标路径), ...]
        """
        transferred = []
        
        if not self.source_folder.exists():
            print(f"源文件夹不存在: {self.source_folder}")
            return transferred
        
        # 确保目标文件夹存在
        self.target_folder.mkdir(parents=True, exist_ok=True)
        
        # 遍历源文件夹中的漫画文件夹
        for comic_folder in self.source_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            # 查找该漫画文件夹下的所有 CBZ 文件
            cbz_files = list(comic_folder.glob('*.cbz'))
            
            if not cbz_files:
                continue
            
            # 创建目标漫画文件夹
            target_comic_folder = self.target_folder / comic_folder.name
            target_comic_folder.mkdir(parents=True, exist_ok=True)
            
            # 转移每个 CBZ 文件
            for cbz_file in cbz_files:
                target_path = target_comic_folder / cbz_file.name
                
                # 检查目标文件是否已存在
                if target_path.exists():
                    # 比较文件大小，决定是否跳过
                    if cbz_file.stat().st_size == target_path.stat().st_size:
                        print(f"跳过（已存在且大小相同）: {cbz_file.name}")
                        continue
                    else:
                        print(f"警告: 目标文件已存在但大小不同: {cbz_file.name}")
                        # 为避免数据丢失，跳过
                        continue
                
                try:
                    if move:
                        shutil.move(str(cbz_file), str(target_path))
                        action = "移动"
                    else:
                        shutil.copy2(str(cbz_file), str(target_path))
                        action = "复制"
                    
                    transferred.append((str(cbz_file), str(target_path)))
                    print(f"✓ {action}: {comic_folder.name}/{cbz_file.name}")
                except Exception as e:
                    print(f"✗ 转移失败 {cbz_file.name}: {e}")
        
        return transferred
    
    def scan_cbz_files(self) -> dict:
        """
        扫描源文件夹中的所有 CBZ 文件
        
        Returns:
            字典 {漫画名: [CBZ文件列表]}
        """
        cbz_dict = {}
        
        if not self.source_folder.exists():
            return cbz_dict
        
        for comic_folder in self.source_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            cbz_files = list(comic_folder.glob('*.cbz'))
            if cbz_files:
                cbz_dict[comic_folder.name] = [f.name for f in cbz_files]
        
        return cbz_dict
    
    def get_transfer_summary(self, cbz_dict: dict) -> str:
        """
        生成转移前的摘要报告
        
        Args:
            cbz_dict: scan_cbz_files()返回的字典
            
        Returns:
            格式化的摘要文本
        """
        output = []
        output.append("=" * 60)
        output.append("待转移 CBZ 文件汇总")
        output.append("=" * 60)
        output.append("")
        
        total_files = 0
        for comic_name in sorted(cbz_dict.keys()):
            cbz_files = cbz_dict[comic_name]
            output.append(f"【{comic_name}】 - {len(cbz_files)} 个文件")
            for cbz_file in sorted(cbz_files):
                output.append(f"  • {cbz_file}")
            output.append("")
            total_files += len(cbz_files)
        
        if not cbz_dict:
            output.append("未找到任何 CBZ 文件")
        else:
            output.append("=" * 60)
            output.append(f"总计: {len(cbz_dict)} 部漫画, {total_files} 个文件")
        
        return "\n".join(output)
    
    def clean_empty_folders(self):
        """清理源文件夹中的空文件夹"""
        if not self.source_folder.exists():
            return
        
        removed = []
        for comic_folder in self.source_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            # 检查文件夹是否为空
            if not any(comic_folder.iterdir()):
                try:
                    comic_folder.rmdir()
                    removed.append(comic_folder.name)
                    print(f"✓ 已删除空文件夹: {comic_folder.name}")
                except Exception as e:
                    print(f"✗ 删除失败 {comic_folder.name}: {e}")
        
        return removed
