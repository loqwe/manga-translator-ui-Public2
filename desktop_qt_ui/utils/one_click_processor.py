#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
一键处理模块
功能：整合名称替换、CBZ压缩、文件转移三个步骤
"""

from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional

from name_replacer import NameReplacer
from cbz_compressor import CBZCompressor
from cbz_transfer import CBZTransfer


class OneClickProcessor:
    """一键处理器"""
    
    def __init__(self, 
                 input_folder: str, 
                 output_folder: str,
                 progress_callback: Optional[Callable[[str], None]] = None):
        """
        初始化一键处理器
        
        Args:
            input_folder: 输入文件夹路径（翻译输出文件夹）
            output_folder: 输出文件夹路径（存储文件夹）
            progress_callback: 进度回调函数
        """
        self.input_folder = Path(input_folder)
        self.output_folder = Path(output_folder)
        self.progress_callback = progress_callback
        
        # 初始化各个处理器
        self.replacer = NameReplacer()
        self.compressor = CBZCompressor()
        self.transferer = None  # 在处理时创建
        
        # 处理结果
        self.results = {
            'compressed': [],
            'organized': [],
            'transferred': [],
            'errors': []
        }
    
    def _report_progress(self, message: str):
        """报告进度"""
        if self.progress_callback:
            self.progress_callback(message)
    
    def process_all(self) -> Dict[str, List]:
        """
        执行完整的一键处理流程
        新流程：压缩 → 提取组织 → 转移
        
        Returns:
            处理结果字典
        """
        self.results = {
            'compressed': [],
            'organized': [],
            'transferred': [],
            'errors': []
        }
        
        if not self.input_folder.exists():
            error_msg = f"输入文件夹不存在: {self.input_folder}"
            self.results['errors'].append(error_msg)
            self._report_progress(f"❌ {error_msg}")
            return self.results
        
        # 步骤1：提取漫画名并组织
        self._report_progress("=" * 60)
        self._report_progress("步骤 1/3: 提取漫画名并组织文件夹")
        self._report_progress("=" * 60)
        try:
            organized = self.step_organize()
            self.results['organized'] = organized
            if organized:
                self._report_progress(f"✓ 组织了 {len(organized)} 个文件夹")
            else:
                self._report_progress("• 没有需要组织的文件夹")
        except Exception as e:
            error_msg = f"文件组织失败: {str(e)}"
            self.results['errors'].append(error_msg)
            self._report_progress(f"❌ {error_msg}")
        
        # 步骤2：CBZ压缩
        self._report_progress("")
        self._report_progress("=" * 60)
        self._report_progress("步骤 2/3: CBZ 压缩")
        self._report_progress("=" * 60)
        try:
            compressed = self.step_compress()
            self.results['compressed'] = compressed
            if compressed:
                self._report_progress(f"✓ 压缩了 {len(compressed)} 个章节")
            else:
                self._report_progress("• 没有需要压缩的章节")
        except Exception as e:
            error_msg = f"CBZ压缩失败: {str(e)}"
            self.results['errors'].append(error_msg)
            self._report_progress(f"❌ {error_msg}")
        
        # 步骤3：文件转移
        self._report_progress("")
        self._report_progress("=" * 60)
        self._report_progress("步骤 3/3: 文件转移")
        self._report_progress("=" * 60)
        try:
            transferred = self.step_transfer()
            self.results['transferred'] = transferred
            if transferred:
                self._report_progress(f"✓ 转移了 {len(transferred)} 个文件")
            else:
                self._report_progress("• 没有需要转移的文件")
        except Exception as e:
            error_msg = f"文件转移失败: {str(e)}"
            self.results['errors'].append(error_msg)
            self._report_progress(f"❌ {error_msg}")
        
        # 总结
        self._report_progress("")
        self._report_progress("=" * 60)
        self._report_progress("处理完成！")
        self._report_progress("=" * 60)
        self._report_progress(f"CBZ压缩: {len(self.results['compressed'])} 个")
        self._report_progress(f"文件组织: {len(self.results['organized'])} 个")
        self._report_progress(f"文件转移: {len(self.results['transferred'])} 个")
        if self.results['errors']:
            self._report_progress(f"错误: {len(self.results['errors'])} 个")
        
        return self.results
    
    def step_organize(self) -> List[Tuple[str, str]]:
        """
        步骤1：提取漫画名并组织章节文件夹
        从 _source_path.txt 提取漫画名，使用映射表替换，按漫画名组织文件夹
        
        Returns:
            组织记录列表 [(章节文件夹, 目标路径), ...]
        """
        import shutil
        
        organized = []
        
        # 扫描所有章节文件夹
        for chapter_folder in self.input_folder.iterdir():
            if not chapter_folder.is_dir():
                continue
            
            # 检查是否已经是漫画文件夹的（包含子文件夹）
            has_subdirs = any(item.is_dir() for item in chapter_folder.iterdir())
            if has_subdirs:
                # 已有子目录结构，检查主目录名是否需要替换
                old_name = chapter_folder.name
                new_name = self.replacer.get_translated_name(old_name)
                if old_name != new_name:
                    new_path = chapter_folder.parent / new_name
                    if not new_path.exists():
                        try:
                            chapter_folder.rename(new_path)
                            organized.append((str(chapter_folder), str(new_path)))
                            self._report_progress(f"  • 重命名: {old_name} → {new_name}")
                        except Exception as e:
                            self._report_progress(f"  ⚠ 重命名失败 {old_name}: {e}")
                    else:
                        self._report_progress(f"  ⚠ 目标文件夹已存在: {new_name}")
                continue
            
            try:
                # 读取源路径
                source_path_file = chapter_folder / "_source_path.txt"
                if not source_path_file.exists():
                    self._report_progress(f"  ⚠ 未找到 _source_path.txt: {chapter_folder.name}")
                    continue
                
                with open(source_path_file, 'r', encoding='utf-8') as f:
                    source_path = f.read().strip()
                
                # 提取漫画名（倒数第二级目录）
                path_parts = Path(source_path).parts
                if len(path_parts) < 2:
                    self._report_progress(f"  ⚠ 路径格式错误: {source_path}")
                    continue
                
                raw_comic_name = path_parts[-2]  # 倒数第二级
                
                # 使用映射表替换漫画名
                comic_name = self.replacer.get_translated_name(raw_comic_name)
                
                # 创建漫画文件夹
                comic_folder = self.input_folder / comic_name
                comic_folder.mkdir(exist_ok=True)
                
                # 移动章节文件夹到漫画文件夹
                target_path = comic_folder / chapter_folder.name
                if not target_path.exists():
                    shutil.move(str(chapter_folder), str(target_path))
                    organized.append((str(chapter_folder), str(target_path)))
                    
                    # 显示映射信息
                    if raw_comic_name != comic_name:
                        self._report_progress(f"  • {chapter_folder.name} → {comic_name}/ (映射: {raw_comic_name})")
                    else:
                        self._report_progress(f"  • {chapter_folder.name} → {comic_name}/")
                
            except Exception as e:
                error_msg = f"组织文件夹失败 {chapter_folder.name}: {str(e)}"
                self.results['errors'].append(error_msg)
                self._report_progress(f"  ❌ {error_msg}")
        
        return organized
    
    def step_compress(self) -> List[Tuple[str, str]]:
        """
        步骤2：CBZ压缩
        压缩已组织的漫画文件夹中的章节文件夹为CBZ文件
        
        Returns:
            压缩记录列表 [(源文件夹, CBZ文件), ...]
        """
        compressed = []
        
        # 扫描漫画文件夹
        for comic_folder in self.input_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            # 扫描漫画文件夹中的章节文件夹
            for chapter_folder in comic_folder.iterdir():
                if not chapter_folder.is_dir():
                    continue
                
                try:
                    # 压缩章节文件夹
                    cbz_path = self.compressor.compress_folder(chapter_folder)
                    if cbz_path:
                        compressed.append((str(chapter_folder), str(cbz_path)))
                        self._report_progress(f"  • {comic_folder.name}/{chapter_folder.name} → {Path(cbz_path).name}")
                except Exception as e:
                    error_msg = f"压缩失败 {comic_folder.name}/{chapter_folder.name}: {str(e)}"
                    self.results['errors'].append(error_msg)
                    self._report_progress(f"  ❌ {error_msg}")
        
        return compressed
    
    def step_transfer(self) -> List[Tuple[str, str]]:
        """
        步骤3：文件转移
        
        Returns:
            转移记录列表 [(源路径, 目标路径), ...]
        """
        # 创建转移器
        self.transferer = CBZTransfer(self.input_folder, self.output_folder)
        transferred = self.transferer.transfer_cbz_files(move=True)
        
        for src, dst in transferred:
            src_name = Path(src).name
            dst_folder = Path(dst).parent.name
            self._report_progress(f"  • {src_name} → {dst_folder}/")
        
        return transferred
    
    def get_summary(self) -> str:
        """
        获取处理摘要
        
        Returns:
            格式化的摘要文本
        """
        output = []
        output.append("=" * 60)
        output.append("一键处理摘要报告")
        output.append("=" * 60)
        output.append("")
        output.append(f"输入文件夹: {self.input_folder}")
        output.append(f"输出文件夹: {self.output_folder}")
        output.append("")
        output.append("处理结果:")
        output.append("-" * 60)
        output.append(f"CBZ压缩:  {len(self.results['compressed'])} 个章节")
        output.append(f"文件组织: {len(self.results['organized'])} 个文件")
        output.append(f"文件转移: {len(self.results['transferred'])} 个文件")
        
        if self.results['errors']:
            output.append("")
            output.append("错误信息:")
            for error in self.results['errors']:
                output.append(f"  ❌ {error}")
        
        output.append("")
        output.append("=" * 60)
        
        return "\n".join(output)
    
    def preview(self) -> Dict[str, any]:
        """
        预览将要处理的内容（不执行实际操作）
        
        Returns:
            预览信息字典
        """
        preview_info = {
            'input_folder': str(self.input_folder),
            'output_folder': str(self.output_folder),
            'comic_folders': [],
            'cbz_files': [],
            'estimated_operations': 0
        }
        
        if not self.input_folder.exists():
            return preview_info
        
        # 扫描漫画文件夹
        for comic_folder in self.input_folder.iterdir():
            if not comic_folder.is_dir():
                continue
            
            comic_info = {
                'name': comic_folder.name,
                'chapters': [],
                'cbz_count': 0
            }
            
            # 扫描章节
            for item in comic_folder.iterdir():
                if item.is_dir():
                    comic_info['chapters'].append(item.name)
                elif item.suffix.lower() == '.cbz':
                    comic_info['cbz_count'] += 1
            
            if comic_info['chapters'] or comic_info['cbz_count']:
                preview_info['comic_folders'].append(comic_info)
                preview_info['estimated_operations'] += len(comic_info['chapters']) + comic_info['cbz_count']
        
        return preview_info
