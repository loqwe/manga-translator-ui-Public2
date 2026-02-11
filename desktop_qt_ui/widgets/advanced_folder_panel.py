#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
高级文件夹管理面板 - 使用现代化UI
支持三层目录结构（Downloads/Source/Title/Chapters）
智能聚合、名称映射、章节详情、多选等功能
"""

import os
import sys
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
from PyQt6.QtCore import Qt, QSettings, QSize, QTimer, QThread, QObject
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QPushButton,
    QLineEdit, QTextEdit, QLabel, QHeaderView, QAbstractItemView,
    QFileDialog, QMessageBox, QComboBox, QTreeWidget, QTreeWidgetItem,
    QDialog, QSplitter, QScrollArea, QToolButton, QStyle, QMenu,
    QCheckBox, QSpinBox, QListWidget, QListWidgetItem, QFormLayout,
    QProgressBar, QApplication
)
from PyQt6.QtGui import QColor, QPalette, QDragEnterEvent, QDropEvent
from PyQt6.QtCore import pyqtSignal

# 导入名称映射
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from utils.name_replacer import NameReplacer
except ImportError:
    NameReplacer = None

try:
    from utils.long_image_stitcher import LongImageStitcher
except ImportError:
    LongImageStitcher = None

try:
    from utils.long_image_splitter import LocalSplitter, split_image_sync, SplitResult
except ImportError:
    LocalSplitter = None
    split_image_sync = None
    SplitResult = None


class ScanWorker(QObject):
    """后台扫描工作线程"""
    finished = pyqtSignal(dict)  # 扫描完成，发送 folder_data
    progress = pyqtSignal(str)   # 扫描进度日志
    error = pyqtSignal(str)      # 错误信息
    
    def __init__(self, root_paths: List[str], name_replacer=None):
        super().__init__()
        self.root_paths = root_paths
        self.name_replacer = name_replacer
        self._is_running = True
    
    def stop(self):
        """请求停止扫描"""
        self._is_running = False
    
    def run(self):
        """执行扫描（在后台线程中运行）"""
        import re
        
        folder_data = {}
        
        try:
            scanned_sources = 0
            scanned_titles = 0
            
            # 系统文件夹和隐藏文件夹列表
            system_folders = {'$RECYCLE.BIN', 'System Volume Information', 'Config.Msi',
                            'ProgramData', 'Windows', 'Program Files', 'Program Files (x86)',
                            'PerfLogs', 'Recovery', '$Windows.~BT', '$Windows.~WS'}
            
            # 遍历所有根目录
            for root_path in self.root_paths:
                if not self._is_running:
                    break
                    
                root = Path(root_path)
                self.progress.emit(f"  扫描: {root_path}")
                
                try:
                    first_level_dirs = list(root.iterdir())
                except PermissionError:
                    self.progress.emit(f"⚠️ 无权限访问: {root_path}")
                    continue
                except Exception as e:
                    self.progress.emit(f"⚠️ 扫描错误: {root_path} - {e}")
                    continue
                
                for first_level_dir in first_level_dirs:
                    if not self._is_running:
                        break
                    
                    # 跳过系统文件夹、隐藏文件夹和以.开头的文件夹
                    if (not first_level_dir.is_dir() or
                        first_level_dir.name in system_folders or
                        first_level_dir.name.startswith('.') or
                        first_level_dir.name.startswith('$')):
                        continue
                    
                    try:
                        # 智能识别目录结构：检查子目录是否为章节（Chapter开头或纯数字）
                        sub_dirs = [d for d in first_level_dir.iterdir() if d.is_dir()]
                        is_title_dir = self._is_chapter_folder_list(sub_dirs)
                        
                        if is_title_dir:
                            # 两层结构：根目录/作品/章节
                            title_name = first_level_dir.name
                            source_name = os.path.basename(root_path)
                            scanned_titles += 1
                            
                            mapped_name = self._get_mapped_name(title_name)
                            
                            if mapped_name not in folder_data:
                                folder_data[mapped_name] = {
                                    'original_names': set(),
                                    'sources': set(),
                                    'chapters': {}
                                }
                            
                            folder_data[mapped_name]['original_names'].add(title_name)
                            folder_data[mapped_name]['sources'].add(source_name)
                            
                            # 扫描章节
                            chapters = []
                            for chapter in sub_dirs:
                                if not self._is_running:
                                    break
                                chapters.append({
                                    'name': chapter.name,
                                    'path': str(chapter),
                                    'source': source_name
                                })
                            
                            if source_name in folder_data[mapped_name]['chapters']:
                                folder_data[mapped_name]['chapters'][source_name].extend(chapters)
                            else:
                                folder_data[mapped_name]['chapters'][source_name] = chapters
                        else:
                            # 三层结构：根目录/来源/作品/章节
                            source_name = first_level_dir.name
                            scanned_sources += 1
                            
                            for title_dir in first_level_dir.iterdir():
                                if not self._is_running:
                                    break
                                if not title_dir.is_dir():
                                    continue
                                
                                title_name = title_dir.name
                                scanned_titles += 1
                                
                                mapped_name = self._get_mapped_name(title_name)
                                
                                if mapped_name not in folder_data:
                                    folder_data[mapped_name] = {
                                        'original_names': set(),
                                        'sources': set(),
                                        'chapters': {}
                                    }
                                
                                folder_data[mapped_name]['original_names'].add(title_name)
                                folder_data[mapped_name]['sources'].add(source_name)
                                
                                # 扫描章节
                                chapters = []
                                for chapter in title_dir.iterdir():
                                    if not self._is_running:
                                        break
                                    if chapter.is_dir():
                                        chapters.append({
                                            'name': chapter.name,
                                            'path': str(chapter),
                                            'source': source_name
                                        })
                                
                                if source_name in folder_data[mapped_name]['chapters']:
                                    folder_data[mapped_name]['chapters'][source_name].extend(chapters)
                                else:
                                    folder_data[mapped_name]['chapters'][source_name] = chapters
                                
                    except PermissionError:
                        self.progress.emit(f"⚠️ 跳过无权限访问的文件夹: {first_level_dir.name}")
                        continue
                    except Exception as e:
                        self.progress.emit(f"⚠️ 跳过错误文件夹 {first_level_dir.name}: {e}")
                        continue
            
            if self._is_running:
                self.progress.emit(f"✓ 扫描完成！发现 {scanned_sources} 个来源，{len(folder_data)} 个作品")
            else:
                self.progress.emit(f"⚠️ 扫描已取消")
            
            self.finished.emit(folder_data)
            
        except Exception as e:
            self.error.emit(f"扫描失败: {e}")
            self.finished.emit({})
    
    def _is_chapter_folder_list(self, dirs: List[Path]) -> bool:
        """判断目录列表是否为章节目录"""
        import re
        
        if not dirs:
            return False
        
        chapter_patterns = [
            r'^[Cc]hapter[_\s\-]?\d+',
            r'^\d+$',
            r'^第\d+[话章集卷]',
            r'^[Ee][Pp][_\s\-]?\d+',
            r'^[Vv]ol[._\s]?\d+',
            r'^[Cc]h[._\s]?\d+',
        ]
        
        chapter_count = 0
        for d in dirs:
            name = d.name
            for pattern in chapter_patterns:
                if re.match(pattern, name):
                    chapter_count += 1
                    break
        
        return chapter_count > len(dirs) * 0.5
    
    def _get_mapped_name(self, original_name: str) -> str:
        """获取映射后的名称"""
        if not self.name_replacer:
            return original_name
        
        for raw_name, translated_name in self.name_replacer.mapping.items():
            raw_names = [n.strip() for n in raw_name.split('|')]
            if original_name in raw_names:
                return translated_name
        
        return original_name


class StitchWorker(QObject):
    """后台拼接工作线程"""
    finished = pyqtSignal(list)  # 拼接完成，发送结果文件列表
    progress = pyqtSignal(str)   # 进度日志
    chapter_progress = pyqtSignal(int, int)  # 当前章节进度 (current, total)
    error = pyqtSignal(str)      # 错误信息
    
    def __init__(self, chapters: List[str], max_height: int, margin: int):
        super().__init__()
        self.chapters = chapters
        self.max_height = max_height
        self.margin = margin
        self._is_running = True
    
    def stop(self):
        """请求停止拼接"""
        self._is_running = False
    
    def run(self):
        """执行拼接（在后台线程中运行）"""
        from PIL import Image
        import re
        
        all_result_files = []
        total_chapters = len(self.chapters)
        
        try:
            self.progress.emit(f"[长图拼接] 启用智能长图拼接功能")
            self.progress.emit(f"[长图拼接] 配置: 最大高度={self.max_height}px, 边界检测={self.margin}px")
            self.progress.emit(f"[长图拼接] 检测到 {total_chapters} 个章节，分别处理避免跨章节拼接")
            
            for idx, chapter_path in enumerate(self.chapters):
                if not self._is_running:
                    self.progress.emit(f"[长图拼接] ⚠️ 拼接已取消")
                    break
                
                chapter_name = os.path.basename(chapter_path)
                self.chapter_progress.emit(idx + 1, total_chapters)
                
                try:
                    result_files = self._stitch_chapter_images(chapter_path)
                    all_result_files.extend(result_files)
                    if result_files:
                        self.progress.emit(f"[长图拼接] ✓ 完成: {chapter_name} 生成 {len(result_files)} 张拼接图")
                except Exception as e:
                    self.progress.emit(f"[长图拼接] ⚠️ 失败: {chapter_name} - {e}")
            
            if self._is_running:
                self.progress.emit(f"[长图拼接] ✓ 全部完成，共生成 {len(all_result_files)} 张拼接图")
            
            self.finished.emit(all_result_files)
            
        except Exception as e:
            self.error.emit(f"拼接失败: {e}")
            self.finished.emit([])
    
    def _stitch_chapter_images(self, chapter_path: str) -> List[str]:
        """拼接章节内的图片（优化版：使用PIL处理，解决中文路径问题）"""
        from PIL import Image
        import numpy as np
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        image_files = []
        
        # 收集所有图片文件
        for f in sorted(os.listdir(chapter_path)):
            if not self._is_running:
                return []
            ext = os.path.splitext(f)[1].lower()
            if ext not in image_extensions:
                continue
            if f.lower().startswith('seg'):
                return []  # 已拼接过
            image_files.append(os.path.join(chapter_path, f))
        
        if len(image_files) <= 1:
            return []
        
        # 提取图片序号
        def extract_number(filepath: str) -> int:
            match = re.search(r'(\d+)', os.path.splitext(os.path.basename(filepath))[0])
            return int(match.group(1)) if match else 0
        
        # === 第一阶段：快速扫描尺寸（使用PIL，支持中文路径） ===
        def get_image_size(path: str) -> dict:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                    return {'path': path, 'width': w, 'height': h, 
                            'number': extract_number(path), 'filename': os.path.basename(path)}
            except Exception as e:
                self.progress.emit(f"[长图拼接] ⚠️ 无法读取图片: {os.path.basename(path)} - {e}")
                pass
            return None
        
        # 并行扫描尺寸
        image_info = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(get_image_size, p): p for p in image_files}
            for future in as_completed(futures):
                if not self._is_running:
                    return []
                result = future.result()
                if result:
                    image_info.append(result)
        
        # 按文件名排序
        image_info.sort(key=lambda x: x['number'])
        
        if len(image_info) <= 1:
            return []
        
        # === 第二阶段：根据尺寸计算分组 ===
        groups = []
        current_group = []
        current_height = 0
        
        for info in image_info:
            if current_height + info['height'] > self.max_height and current_group:
                groups.append(current_group)
                current_group = [info]
                current_height = info['height']
            else:
                current_group.append(info)
                current_height += info['height']
        if current_group:
            groups.append(current_group)
        
        # === 第三阶段：并行拼接各组 ===
        def stitch_group(args) -> tuple:
            batch_index, group = args
            try:
                # 读取图片
                imgs = []
                for info in group:
                    try:
                        with Image.open(info['path']) as img:
                            imgs.append((img.copy(), info))
                    except Exception as e:
                        self.progress.emit(f"[长图拼接] ⚠️ 无法读取图片: {info['filename']} - {e}")
                
                if not imgs:
                    return None, []
                
                # 计算画布尺寸
                max_width = max(img.width for img, _ in imgs)
                total_height = sum(img.height for img, _ in imgs) + self.margin * (len(imgs) - 1)
                
                # 创建白色画布并拼接
                canvas = Image.new('RGB', (max_width, total_height), (255, 255, 255))
                y_offset = 0
                for img, info in imgs:
                    # 居中放置
                    x_offset = (max_width - img.width) // 2
                    canvas.paste(img.convert('RGB'), (x_offset, y_offset))
                    y_offset += img.height + self.margin
                
                # 生成文件名
                start_num = group[0]['number']
                end_num = group[-1]['number']
                filename = f"seg{batch_index:02d}_img{start_num:03d}-{end_num:03d}_{len(group)}p.jpg"
                result_path = os.path.join(chapter_path, filename)
                
                # 保存为JPEG（比PNG快很多）
                canvas.save(result_path, 'JPEG', quality=95, optimize=True)
                
                return result_path, [info['path'] for info in group]
            except Exception as e:
                self.progress.emit(f"[长图拼接] ⚠️ 拼接失败: {e}")
                return None, []
        
        result_files = []
        files_to_delete = []
        
        # 并行拼接
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(stitch_group, (i+1, g)) for i, g in enumerate(groups)]
            for future in as_completed(futures):
                if not self._is_running:
                    break
                result_path, paths = future.result()
                if result_path:
                    result_files.append(result_path)
                    files_to_delete.extend(paths)
        
        # 删除原图
        for path in files_to_delete:
            try:
                os.remove(path)
            except Exception:
                pass
        
        return result_files
    
    def _save_stitched_image(self, chapter_path: str, batch: List[dict], batch_index: int) -> Optional[str]:
        """保存拼接后的图片"""
        from PIL import Image
        
        if not batch:
            return None
        
        try:
            # 计算总高度和最大宽度
            total_height = sum(item['image'].height for item in batch) + self.margin * (len(batch) - 1)
            max_width = max(item['image'].width for item in batch)
            
            # 创建新图片
            stitched = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            
            # 拼接
            y_offset = 0
            for item in batch:
                img = item['image']
                x_offset = (max_width - img.width) // 2
                stitched.paste(img.convert('RGB'), (x_offset, y_offset))
                y_offset += img.height + self.margin
            
            # 生成文件名
            start_num = batch[0]['number']
            end_num = batch[-1]['number']
            count = len(batch)
            
            filename = f"seg{batch_index:02d}_img{start_num:03d}-{end_num:03d}_{count}p.png"
            result_path = os.path.join(chapter_path, filename)
            
            # 保存为PNG格式
            stitched.save(result_path, 'PNG', optimize=True)
            stitched.close()
            
            return result_path
            
        except Exception as e:
            self.progress.emit(f"[长图拼接] ⚠️ 保存失败: {e}")
            return None


class SplitWorker(QObject):
    """后台拆分工作线程（使用 LocalSplitter + YOLO 智能检测）"""
    finished = pyqtSignal(list)  # 拆分完成，发送结果文件列表
    progress = pyqtSignal(str)   # 进度日志
    chapter_progress = pyqtSignal(int, int)  # 当前章节进度 (current, total)
    error = pyqtSignal(str)      # 错误信息
    
    def __init__(self, chapters: List[str], skip_threshold: int, target_height: int, buffer_range: int, min_segment_height: int, naming_pattern: str = "{index}_{source}", index_digits: int = 1, index_start: int = 0, reset_index_per_chapter: bool = True):
        super().__init__()
        self.chapters = chapters
        self.skip_threshold = skip_threshold  # 短图阈值（低于此高度不拆分）
        self.target_height = target_height
        self.buffer_range = buffer_range
        self.min_segment_height = min_segment_height
        self.naming_pattern = naming_pattern
        self.index_digits = index_digits
        self.index_start = index_start
        self.reset_index_per_chapter = reset_index_per_chapter
        self._global_index = index_start
        self._is_running = True
        self._splitter: Optional[LocalSplitter] = None
    
    def stop(self):
        """请求停止拆分"""
        self._is_running = False
    
    def run(self):
        """执行拆分（在后台线程中运行）"""
        import asyncio
        
        # 创建新的事件循环（后台线程需要独立的 loop）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(self._run_async())
        finally:
            loop.close()
    
    async def _run_async(self):
        """异步执行拆分"""
        all_result_files = []
        total_chapters = len(self.chapters)
        
        try:
            self.progress.emit(f"[长图拆分] 启用智能长图拆分功能 (YOLO 模式)")
            self.progress.emit(f"[长图拆分] 配置: 短图阈值={self.skip_threshold}px, 目标高度={self.target_height}px, 缓冲范围=±{self.buffer_range}px")
            self.progress.emit(f"[长图拆分] 命名: 模式={self.naming_pattern}, 位数={self.index_digits}, 起始序号={self.index_start}")
            self.progress.emit(f"[长图拆分] 检测到 {total_chapters} 个章节，分别处理")
            
            # 初始化 LocalSplitter
            self._splitter = LocalSplitter(
                target_height=self.target_height,
                buffer_range=self.buffer_range,
                min_segment_height=self.min_segment_height,
                naming_pattern=self.naming_pattern,
                index_digits=self.index_digits,
                index_start=self.index_start
            )
            
            # 初始化调试目录（整个批次共用一个时间戳目录）
            self._splitter._init_debug_dir()
            self.progress.emit(f"[长图拆分] 调试输出目录: {self._splitter.debug_dir}")
            
            # 加载 YOLO 模型
            self.progress.emit(f"[长图拆分] 正在加载 YOLO 模型...")
            await self._splitter.load_yolo()
            if self._splitter._yolo_loaded:
                self.progress.emit(f"[长图拆分] ✓ YOLO 模型加载成功，将检测气泡禁切区")
            else:
                self.progress.emit(f"[长图拆分] ⚠️ YOLO 加载失败，使用纯图像处理模式")
            
            for idx, chapter_path in enumerate(self.chapters):
                if not self._is_running:
                    self.progress.emit(f"[长图拆分] ⚠️ 拆分已取消")
                    break
                
                if self.reset_index_per_chapter:
                    self._global_index = self.index_start
                
                chapter_name = os.path.basename(chapter_path)
                self.chapter_progress.emit(idx + 1, total_chapters)
                
                try:
                    result_files = await self._split_chapter_images_async(chapter_path)
                    all_result_files.extend(result_files)
                    if result_files:
                        self.progress.emit(f"[长图拆分] ✓ 完成: {chapter_name} 生成 {len(result_files)} 张拆分图")
                except Exception as e:
                    self.progress.emit(f"[长图拆分] ⚠️ 失败: {chapter_name} - {e}")
            
            if self._is_running:
                self.progress.emit(f"[长图拆分] ✓ 全部完成，共生成 {len(all_result_files)} 张拆分图")
            
            self.finished.emit(all_result_files)
            
        except Exception as e:
            import traceback
            self.error.emit(f"拆分失败: {e}\n{traceback.format_exc()}")
            self.finished.emit([])
    
    async def _split_chapter_images_async(self, chapter_path: str) -> List[str]:
        """异步拆分章节内的图片"""
        import cv2
        import numpy as np
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        image_files = []
        
        # 收集所有图片文件
        for f in sorted(os.listdir(chapter_path)):
            if not self._is_running:
                return []
            ext = os.path.splitext(f)[1].lower()
            if ext not in image_extensions:
                continue
            # 跳过已拆分的文件
            if f.lower().startswith('split'):
                self.progress.emit(f"[长图拆分] ⚠️ 检测到已拆分文件: {f}，跳过此章节")
                return []
            image_files.append(os.path.join(chapter_path, f))
        
        if not image_files:
            return []
        
        all_result_files = []
        
        for image_path in image_files:
            if not self._is_running:
                return all_result_files
            
            filename = os.path.basename(image_path)
            
            # 读取图片获取尺寸（支持中文路径）
            data = np.fromfile(image_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR) if data is not None else None
            if img is None:
                self.progress.emit(f"[长图拆分] ⚠️ 无法读取: {filename}")
                continue
            
            height, width = img.shape[:2]
            
            # 检查是否需要拆分（低于短图阈值不拆分）
            if height <= self.skip_threshold:
                continue  # 短图，不需要拆分
            
            self.progress.emit(f"[长图拆分] 处理: {filename} ({width}x{height})")
            
            try:
                # 使用 LocalSplitter 异步拆分（包含 YOLO 检测）
                start_index = self._global_index
                result = await self._splitter.split(
                    image_path,
                    output_dir=chapter_path,
                    quality=95,
                    delete_original=True,
                    start_index=start_index
                )
                
                # 更新全局序号（只统计真正生成的文件）
                created_count = 0
                for p in result.output_files:
                    if os.path.abspath(p) != os.path.abspath(image_path):
                        created_count += 1
                if created_count > 0:
                    self._global_index += created_count
                
                if result.cuts:
                    self.progress.emit(f"[长图拆分]   → 拆分为 {result.segment_count} 个片段，切点: {result.cuts}")
                    all_result_files.extend(result.output_files)
                    
            except Exception as e:
                self.progress.emit(f"[长图拆分] ⚠️ 拆分失败 {filename}: {e}")
                # 回退到同步版本
                try:
                    if split_image_sync:
                        start_index = self._global_index
                        result = split_image_sync(
                            image_path,
                            target_height=self.target_height,
                            buffer_range=self.buffer_range,
                            min_segment_height=self.min_segment_height,
                            output_dir=chapter_path,
                            quality=95,
                            delete_original=True,
                            naming_pattern=self.naming_pattern,
                            index_digits=self.index_digits,
                            index_start=start_index
                        )
                        # 更新全局序号
                        created_count = 0
                        for p in result.output_files:
                            if os.path.abspath(p) != os.path.abspath(image_path):
                                created_count += 1
                        if created_count > 0:
                            self._global_index += created_count
                        
                        if result.cuts:
                            self.progress.emit(f"[长图拆分]   → (回退) 拆分为 {result.segment_count} 个片段")
                            all_result_files.extend(result.output_files)
                except Exception as e2:
                    self.progress.emit(f"[长图拆分] ⚠️ 回退也失败: {e2}")
        
        return all_result_files


class DroppableListWidget(QListWidget):
    """支持拖放文件夹的列表控件"""
    folders_dropped = pyqtSignal(list)  # 发送拖放的文件夹路径列表
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
    
    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()
    
    def dropEvent(self, event: QDropEvent):
        if event.mimeData().hasUrls():
            folders = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                print(f"[拖放] 检测到路径: {path}")
                if path and os.path.isdir(path):
                    folders.append(path)
                    print(f"[拖放] 添加文件夹: {path}")
            if folders:
                print(f"[拖放] 发送信号: {len(folders)} 个文件夹")
                self.folders_dropped.emit(folders)
            event.acceptProposedAction()
        else:
            event.ignore()


class AdvancedFolderDialog(QDialog):
    """高级文件夹选择对话框 - 现代化UI风格"""
    
    # 日志信号：发送日志到主程序
    log_to_main = pyqtSignal(str, str)  # (message, level)
    
    def __init__(self, parent=None, start_dir: str = ""):
        super().__init__(parent)
        self.setWindowTitle("高级文件夹 - 批量选择章节")
        self.setMinimumSize(1200, 750)
        self.resize(1200, 750)
        
        self.name_replacer = NameReplacer() if NameReplacer else None
        self.folder_data = {}  # {title: {'sources': [source1, source2], 'chapters': {source: [chapters]}}}
        self.selected_chapters = []
        
        # 递归锁：防止在处理选择改变时被递归调用
        self._processing_selection = False
        
        # 后台扫描线程
        self._scan_thread = None
        self._scan_worker = None
        self._is_scanning = False
        
        # 后台拼接线程
        self._stitch_thread = None
        self._stitch_worker = None
        self._is_stitching = False
        self._stitch_accept_after = False  # 拼接完成后是否自动接受对话框
        
        # 后台拆分线程
        self._split_thread = None
        self._split_worker = None
        self._is_splitting = False
        self._split_accept_after = False  # 拆分完成后是否自动接受对话框
        
        # 撤回功能：记录最近一次操作的备份数据
        self._undo_data: Optional[dict] = None  # {章节路径: {"backup_dir": 备份目录, "generated_files": [生成的文件]}}
        self._undo_type: str = ""  # "split" 或 "stitch"
        
        # 支持多个根目录
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        saved_roots = settings.value("root_paths", [])
        if saved_roots and isinstance(saved_roots, list):
            self.root_paths = [p for p in saved_roots if os.path.isdir(p)]
        else:
            # 兼容旧版本单个根目录
            saved_root = settings.value("last_dir", "")
            if saved_root and os.path.isdir(saved_root):
                self.root_paths = [saved_root]
            elif start_dir and os.path.isdir(start_dir):
                self.root_paths = [start_dir]
            else:
                self.root_paths = []
        
        # 兼容旧代码：第一个根目录作为默认
        self.root_path = self.root_paths[0] if self.root_paths else ""
        
        self._init_ui()
        self._apply_modern_style()
        
        # 使用延迟初始化，避免阻塞UI
        QTimer.singleShot(0, self._delayed_init)
    
    def _delayed_init(self):
        """延迟初始化：在UI显示后执行耗时操作"""
        # 自动刷新名称映射
        self.refresh_mapping_names()
        
        # 尝试加载上次扫描结果
        self._load_scan_cache()
        
        # 如果有有效的根目录且没有缓存数据，自动扫描（异步）
        if self.root_path and os.path.isdir(self.root_path) and not self.folder_data:
            # 使用短延迟确保窗口已完全显示
            QTimer.singleShot(100, self.scan_folders_async)
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # 工具栏区域
        toolbar = self._create_toolbar()
        layout.addWidget(toolbar)
        
        # 地址栏区域
        address_bar = self._create_address_bar()
        layout.addWidget(address_bar)
        
        # 主内容区域（使用分割器）
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：快捷访问和搜索
        left_panel = self._create_left_panel()
        splitter.addWidget(left_panel)
        
        # 右侧：作品和章节列表
        right_panel = self._create_right_panel()
        splitter.addWidget(right_panel)
        
        # 设置分割比例：30% : 70%
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        
        layout.addWidget(splitter, 1)
        
        # 底部信息栏
        info_bar = self._create_info_bar()
        layout.addWidget(info_bar)
        
        # 底部按钮
        button_bar = self._create_button_bar()
        layout.addWidget(button_bar)
    
    def _create_toolbar(self) -> QWidget:
        """创建工具栏"""
        toolbar = QWidget()
        toolbar.setFixedHeight(45)
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(8, 4, 8, 4)
        toolbar_layout.setSpacing(4)
        
        toolbar_layout.addStretch()
        
        # 统计标签
        self.stats_label = QLabel("作品: 0 | 章节: 0 | 已选: 0")
        toolbar_layout.addWidget(self.stats_label)
        
        return toolbar
    
    def _create_address_bar(self) -> QWidget:
        """创建地址栏（简洁版：根目录管理+扫描按钮）"""
        address_widget = QWidget()
        address_layout = QHBoxLayout(address_widget)
        address_layout.setContentsMargins(8, 8, 8, 8)
        address_layout.setSpacing(8)
        
        # 根目录数量显示
        self.roots_count_label = QLabel(f"根目录: {len(self.root_paths)} 个")
        address_layout.addWidget(self.roots_count_label)
        
        # 管理根目录按钮（弹窗）
        manage_roots_btn = QPushButton("📁 管理根目录...")
        manage_roots_btn.setFixedWidth(120)
        manage_roots_btn.clicked.connect(self._show_roots_dialog)
        address_layout.addWidget(manage_roots_btn)
        
        # 来源筛选下拉框
        address_layout.addWidget(QLabel("来源:"))
        self.source_filter_combo = QComboBox()
        self.source_filter_combo.setMinimumWidth(120)
        self.source_filter_combo.addItem("全部", None)
        self.source_filter_combo.currentIndexChanged.connect(self._on_source_filter_changed)
        address_layout.addWidget(self.source_filter_combo)
        
        address_layout.addStretch()
        
        # 扫描作品按钮
        self.scan_btn = QPushButton("🔍 扫描作品")
        self.scan_btn.setFixedWidth(100)
        self.scan_btn.clicked.connect(self.scan_folders_async)
        address_layout.addWidget(self.scan_btn)
        
        # 扫描进度条（初始隐藏）
        self.scan_progress = QProgressBar()
        self.scan_progress.setFixedWidth(150)
        self.scan_progress.setRange(0, 0)  # 不确定进度模式
        self.scan_progress.setVisible(False)
        address_layout.addWidget(self.scan_progress)
        
        # 兼容旧代码（隐藏）
        self.root_path_edit = QLineEdit()
        self.root_path_edit.setVisible(False)
        self.root_paths_list = QListWidget()
        self.root_paths_list.setVisible(False)
        
        return address_widget
    
    def _create_left_panel(self) -> QWidget:
        """创建左侧面板"""
        panel = QWidget()
        panel.setMinimumWidth(200)
        panel.setMaximumWidth(300)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 搜索框
        layout.addWidget(QLabel("搜索作品:"))
        self.search_combo = QComboBox()
        self.search_combo.setEditable(True)
        self.search_combo.setPlaceholderText("输入或选择...")
        self.search_combo.currentTextChanged.connect(self.apply_filter)
        self.search_combo.lineEdit().textChanged.connect(self.apply_filter)
        layout.addWidget(self.search_combo)
        
        # 刷新映射名称按钮
        refresh_mapping_btn = QPushButton()
        refresh_mapping_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload))
        refresh_mapping_btn.setText("  刷新映射名称")
        refresh_mapping_btn.setToolTip("从名称映射管理同步最新配置")
        refresh_mapping_btn.clicked.connect(self.refresh_mapping_names)
        layout.addWidget(refresh_mapping_btn)
        
        # 已选章节显示框（支持拖放导入）
        chapters_header = QHBoxLayout()
        chapters_header.setSpacing(4)
        chapters_header.addWidget(QLabel("已选章节:"))
        chapters_header.addStretch()
        
        # 清空按钮（使用图标）
        clear_btn = QPushButton()
        clear_btn.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DialogResetButton))
        clear_btn.setFixedSize(24, 24)
        clear_btn.setToolTip("清空已选章节列表")
        clear_btn.clicked.connect(self.clear_selected_chapters)
        chapters_header.addWidget(clear_btn)
        
        layout.addLayout(chapters_header)
        
        self.selected_chapters_list = DroppableListWidget(self)
        self.selected_chapters_list.setMinimumHeight(120)
        self.selected_chapters_list.setMaximumHeight(180)
        self.selected_chapters_list.setAlternatingRowColors(True)
        self.selected_chapters_list.folders_dropped.connect(self._on_folders_dropped)
        self.selected_chapters_list.itemDoubleClicked.connect(self._on_selected_chapter_double_clicked)
        layout.addWidget(self.selected_chapters_list)
        
        # 最近操作的作品
        layout.addWidget(QLabel("最近操作:"))
        
        # 创建最近作品列表（单行显示）
        self.recent_works_list = QTreeWidget()
        self.recent_works_list.setHeaderHidden(True)
        self.recent_works_list.setMinimumHeight(100)  # 显示4个操作（单行）
        self.recent_works_list.setMaximumHeight(100)
        self.recent_works_list.setRootIsDecorated(False)
        self.recent_works_list.setWordWrap(True)  # 启用文本换行
        self.recent_works_list.itemDoubleClicked.connect(self.on_recent_work_double_clicked)
        layout.addWidget(self.recent_works_list)
        
        # 加载最近操作的作品
        self._load_recent_works()
        
        # 快速操作
        layout.addWidget(QLabel("快速操作:"))
        
        # 智能选择按钮（带下拉菜单）
        self.smart_select_btn = QToolButton()
        self.smart_select_btn.setText("⚡ 智能选择")
        self.smart_select_btn.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.smart_select_btn.setMinimumSize(120, 30)  # 设置最小宽度和高度
        self.smart_select_btn.clicked.connect(self.execute_smart_select)
        
        # 创建可勾选的菜单
        smart_menu = QMenu()
        
        # 加载上次选择的模式
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        self.smart_select_mode = settings.value("smart_select_mode", "latest")  # latest 或 recent_hour
        
        # 选项1：选择所有最新章节
        self.action_latest = smart_menu.addAction("选择所有最新章节")
        self.action_latest.setCheckable(True)
        self.action_latest.setChecked(self.smart_select_mode == "latest")
        self.action_latest.triggered.connect(lambda: self.set_smart_select_mode("latest", execute=False))  # 只打勾不执行
        
        # 选项2：选择1小时内下载的章节
        self.action_recent_hour = smart_menu.addAction("选择1小时内下载的章节")
        self.action_recent_hour.setCheckable(True)
        self.action_recent_hour.setChecked(self.smart_select_mode == "recent_hour")
        self.action_recent_hour.triggered.connect(lambda: self.set_smart_select_mode("recent_hour", execute=False))  # 只打勾不执行
        
        self.smart_select_btn.setMenu(smart_menu)
        layout.addWidget(self.smart_select_btn)
        
        select_all_btn = QPushButton("全选章节")
        select_all_btn.setMinimumSize(120, 30)  # 与智能选择按钮同宽高
        select_all_btn.clicked.connect(self.select_all_chapters)
        layout.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("取消全选")
        deselect_all_btn.setMinimumSize(120, 30)  # 与智能选择按钮同宽高
        deselect_all_btn.clicked.connect(self.deselect_all_chapters)
        layout.addWidget(deselect_all_btn)
        
        layout.addStretch()
        
        return panel
    
    def _create_right_panel(self) -> QWidget:
        """创建右侧面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 8, 8, 8)
        layout.setSpacing(8)
        
        # 作品列表标题栏（带排序下拉框）
        title_bar = QWidget()
        title_bar_layout = QHBoxLayout(title_bar)
        title_bar_layout.setContentsMargins(0, 0, 0, 0)
        title_bar_layout.setSpacing(8)
        
        title_label = QLabel("作品与章节列表")
        title_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        title_bar_layout.addWidget(title_label)
        
        title_bar_layout.addStretch()
        
        # 排序方式下拉框
        sort_label = QLabel("排序:")
        title_bar_layout.addWidget(sort_label)
        
        self.sort_combo = QComboBox()
        self.sort_combo.addItem("智能排序", "natural")
        self.sort_combo.addItem("名称", "name")
        self.sort_combo.addItem("修改日期", "date")
        self.sort_combo.setMinimumWidth(100)
        self.sort_combo.currentIndexChanged.connect(self.on_sort_changed)
        title_bar_layout.addWidget(self.sort_combo)
        
        # 升序/降序切换按钮
        self.sort_order_btn = QPushButton("↑ 升序")
        self.sort_order_btn.setMaximumWidth(80)
        self.sort_order_btn.clicked.connect(self.toggle_sort_order)
        title_bar_layout.addWidget(self.sort_order_btn)
        
        layout.addWidget(title_bar)
        
        # 树形视图
        self.title_tree = QTreeWidget()
        self.title_tree.setHeaderLabels(["作品/章节", "来源", "最新话/数量", "状态"])
        self.title_tree.setRootIsDecorated(True)
        self.title_tree.setAlternatingRowColors(True)
        # 启用扩展多选模式（支持Shift/Ctrl拉取多选）
        self.title_tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.title_tree.itemExpanded.connect(self.on_item_expanded)
        self.title_tree.itemChanged.connect(self.on_item_changed)
        # 连接选择改变信号，同步到复选框
        self.title_tree.itemSelectionChanged.connect(self.on_selection_changed)
        
        # 追踪上一次的选择状态，用于支持多重拖拽
        self.last_selected_paths = set()
        
        # 当前排序方式 - 从设置中加载，默认为智能排序+升序
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        self.sort_by = settings.value("sort_by", "natural")  # natural, name, date
        self.sort_ascending = settings.value("sort_ascending", True, type=bool)  # True=升序, False=降序
        
        # 应用加载的排序设置到UI
        index = self.sort_combo.findData(self.sort_by)
        if index >= 0:
            self.sort_combo.setCurrentIndex(index)
        
        if self.sort_ascending:
            self.sort_order_btn.setText("↑ 升序")
        else:
            self.sort_order_btn.setText("↓ 降序")
        
        # 设置列宽
        header = self.title_tree.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.resizeSection(0, 350)
        header.resizeSection(1, 200)
        header.resizeSection(2, 150)
        header.resizeSection(3, 100)
        
        # 加载保存的列宽
        self._load_column_widths()
        
        layout.addWidget(self.title_tree)
        
        # 日志区域标题栏（带折叠按钮）
        log_title_bar = QWidget()
        log_title_layout = QHBoxLayout(log_title_bar)
        log_title_layout.setContentsMargins(0, 0, 0, 0)
        log_title_layout.setSpacing(4)
        
        log_label = QLabel("操作日志:")
        log_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        log_title_layout.addWidget(log_label)
        
        self.log_toggle_btn = QPushButton("收起 ▼")
        self.log_toggle_btn.setMaximumWidth(80)
        self.log_toggle_btn.clicked.connect(self.toggle_log)
        log_title_layout.addWidget(self.log_toggle_btn)
        
        log_title_layout.addStretch()
        
        layout.addWidget(log_title_bar)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(120)
        
        # 底部区域：日志 + 设置面板（水平布局）
        bottom_area = QWidget()
        bottom_area.setFixedHeight(120)
        bottom_layout = QHBoxLayout(bottom_area)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(8)
        
        # 左侧日志
        bottom_layout.addWidget(self.log_text, 1)
        
        # 加载设置
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        
        # 右侧设置面板（包含长图拼接和自动清理）
        settings_group = QGroupBox("")
        settings_group.setFixedWidth(220)
        settings_group.setFixedHeight(120)
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(6)
        
        # 启用长图拼接 + 设置按钮（同一行）
        stitch_row = QHBoxLayout()
        stitch_row.setSpacing(4)
        
        self.stitch_enabled_cb = QCheckBox("启用长图拼接")
        self.stitch_enabled_cb.setChecked(settings.value("stitch_enabled", False, type=bool))
        self.stitch_enabled_cb.stateChanged.connect(self._save_stitch_settings)
        stitch_row.addWidget(self.stitch_enabled_cb)
        
        stitch_row.addStretch()
        
        # 设置按钮（⚙，更轻量）
        self.stitch_settings_btn = QToolButton()
        self.stitch_settings_btn.setText("⚙")
        self.stitch_settings_btn.setAutoRaise(True)
        self.stitch_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.stitch_settings_btn.setToolTip("长图拼接参数设置")
        self.stitch_settings_btn.setFixedSize(22, 22)
        self.stitch_settings_btn.setStyleSheet(
            "QToolButton{border:none;padding:0px;}"
            "QToolButton:hover{background:rgba(0,0,0,0.06);border-radius:3px;}"
        )
        self.stitch_settings_btn.clicked.connect(self._show_stitch_settings_popup)
        stitch_row.addWidget(self.stitch_settings_btn)
        
        settings_layout.addLayout(stitch_row)
        
        # 长图拼接参数（隐藏的 SpinBox，用于保存值）
        self.stitch_max_height_spin = QSpinBox()
        self.stitch_max_height_spin.setRange(1000, 50000)
        self.stitch_max_height_spin.setSingleStep(1000)
        self.stitch_max_height_spin.setValue(settings.value("stitch_max_height", 10000, type=int))
        self.stitch_max_height_spin.setVisible(False)
        
        self.stitch_margin_spin = QSpinBox()
        self.stitch_margin_spin.setRange(0, 500)
        self.stitch_margin_spin.setSingleStep(10)
        self.stitch_margin_spin.setValue(settings.value("stitch_margin", 100, type=int))
        self.stitch_margin_spin.setVisible(False)
        
        # 启用长图拆分 + 设置按钮（同一行）
        split_row = QHBoxLayout()
        split_row.setSpacing(4)
        
        self.split_enabled_cb = QCheckBox("启用长图拆分")
        self.split_enabled_cb.setChecked(settings.value("split_enabled", False, type=bool))
        self.split_enabled_cb.stateChanged.connect(self._save_split_settings)
        self.split_enabled_cb.setToolTip("将长图拆分为多个短图，便于阅读")
        split_row.addWidget(self.split_enabled_cb)
        
        split_row.addStretch()
        
        # 拆分设置按钮
        self.split_settings_btn = QToolButton()
        self.split_settings_btn.setText("⚙")
        self.split_settings_btn.setAutoRaise(True)
        self.split_settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.split_settings_btn.setToolTip("长图拆分参数设置")
        self.split_settings_btn.setFixedSize(22, 22)
        self.split_settings_btn.setStyleSheet(
            "QToolButton{border:none;padding:0px;}"
            "QToolButton:hover{background:rgba(0,0,0,0.06);border-radius:3px;}"
        )
        self.split_settings_btn.clicked.connect(self._show_split_settings_popup)
        split_row.addWidget(self.split_settings_btn)
        
        settings_layout.addLayout(split_row)
        
        # 长图拆分参数（隐藏的 SpinBox，用于保存值）
        self.split_target_height_spin = QSpinBox()
        self.split_target_height_spin.setRange(1000, 10000)
        self.split_target_height_spin.setSingleStep(100)
        self.split_target_height_spin.setValue(settings.value("split_target_height", 2600, type=int))
        self.split_target_height_spin.setVisible(False)
        
        self.split_buffer_range_spin = QSpinBox()
        self.split_buffer_range_spin.setRange(50, 1000)
        self.split_buffer_range_spin.setSingleStep(50)
        self.split_buffer_range_spin.setValue(settings.value("split_buffer_range", 300, type=int))
        self.split_buffer_range_spin.setVisible(False)
        
        self.split_min_segment_spin = QSpinBox()
        self.split_min_segment_spin.setRange(500, 5000)
        self.split_min_segment_spin.setSingleStep(100)
        self.split_min_segment_spin.setValue(settings.value("split_min_segment", 1000, type=int))
        self.split_min_segment_spin.setVisible(False)
        
        # 短图阈值（低于此高度不拆分）
        self.split_skip_threshold_spin = QSpinBox()
        self.split_skip_threshold_spin.setRange(1000, 10000)
        self.split_skip_threshold_spin.setSingleStep(100)
        self.split_skip_threshold_spin.setValue(settings.value("split_skip_threshold", 3500, type=int))
        self.split_skip_threshold_spin.setVisible(False)
        
        # 命名模式（隐藏的 ComboBox）
        self.split_naming_combo = QComboBox()
        self.split_naming_combo.addItems(["序号_源文件名", "序号", "源文件名_序号", "序号_宽x高"])
        saved_naming = settings.value("split_naming", "序号_源文件名", type=str)
        idx = self.split_naming_combo.findText(saved_naming)
        if idx >= 0:
            self.split_naming_combo.setCurrentIndex(idx)
        self.split_naming_combo.setVisible(False)
        
        # 序号位数（隐藏的 SpinBox）
        self.split_index_digits_spin = QSpinBox()
        self.split_index_digits_spin.setRange(1, 6)
        self.split_index_digits_spin.setSingleStep(1)
        self.split_index_digits_spin.setValue(settings.value("split_index_digits", 1, type=int))
        self.split_index_digits_spin.setVisible(False)
        
        # 起始序号（隐藏的 SpinBox）
        self.split_index_start_spin = QSpinBox()
        self.split_index_start_spin.setRange(0, 9999)
        self.split_index_start_spin.setSingleStep(1)
        self.split_index_start_spin.setValue(settings.value("split_index_start", 0, type=int))
        self.split_index_start_spin.setVisible(False)
        
        # Reset index per chapter (hidden CheckBox, shown in settings popup)
        self.split_reset_index_per_chapter_cb = QCheckBox("序号按章节重置")
        self.split_reset_index_per_chapter_cb.setChecked(settings.value("split_reset_index_per_chapter", True, type=bool))
        self.split_reset_index_per_chapter_cb.setToolTip("批量拆分多个章节时，每个章节的序号从起始序号重新开始")
        self.split_reset_index_per_chapter_cb.setVisible(False)
        
        # 启用自动清理哈希后缀
        self.auto_clean_hash_cb = QCheckBox("导入时自动清理哈希后缀")
        self.auto_clean_hash_cb.setChecked(settings.value("auto_clean_hash", False, type=bool))
        self.auto_clean_hash_cb.setToolTip("导入章节时自动去除文件夹名中的哈希后缀\n如 Chapter 61_563e24 → Chapter 61")
        self.auto_clean_hash_cb.stateChanged.connect(self._save_auto_clean_setting)
        settings_layout.addWidget(self.auto_clean_hash_cb)
        
        settings_layout.addStretch()
        
        bottom_layout.addWidget(settings_group)
        
        layout.addWidget(bottom_area)
        
        # 加载日志展开状态
        log_expanded = settings.value("log_expanded", True, type=bool)
        if not log_expanded:
            self.log_text.hide()
            self.log_toggle_btn.setText("展开 ▲")
        
        return panel
    
    def _create_info_bar(self) -> QWidget:
        """创建信息栏"""
        info_bar = QWidget()
        info_layout = QHBoxLayout(info_bar)
        info_layout.setContentsMargins(8, 4, 8, 4)
        
        tip_label = QLabel("💡 提示：展开作品后勾选需要的章节，支持 Ctrl/Shift 批量选择")
        info_layout.addWidget(tip_label)
        info_layout.addStretch()
        
        return info_bar
    
    def _create_button_bar(self) -> QWidget:
        """创建按钮栏"""
        button_bar = QWidget()
        button_layout = QHBoxLayout(button_bar)
        button_layout.setContentsMargins(8, 8, 8, 8)
        
        # 拼接进度条（初始隐藏）
        self.stitch_progress_label = QLabel("")
        self.stitch_progress_label.setVisible(False)
        button_layout.addWidget(self.stitch_progress_label)
        
        self.stitch_progress = QProgressBar()
        self.stitch_progress.setFixedWidth(200)
        self.stitch_progress.setVisible(False)
        button_layout.addWidget(self.stitch_progress)
        
        button_layout.addStretch()
        
        # 拼接按钮（对所选章节执行长图拼接）
        self.stitch_button = QPushButton("拼接")
        self.stitch_button.setMinimumWidth(80)
        self.stitch_button.setMinimumHeight(32)
        self.stitch_button.setEnabled(False)
        self.stitch_button.setToolTip("对所选章节执行长图拼接（替换原图）")
        self.stitch_button.clicked.connect(self._on_stitch_clicked)
        button_layout.addWidget(self.stitch_button)
        
        # 拆分按钮（对所选章节执行长图拆分）
        self.split_button = QPushButton("拆分")
        self.split_button.setMinimumWidth(80)
        self.split_button.setMinimumHeight(32)
        self.split_button.setEnabled(False)
        self.split_button.setToolTip("对所选章节执行长图拆分（替换原图）")
        self.split_button.clicked.connect(self._on_split_clicked)
        button_layout.addWidget(self.split_button)
        
        # 撤回按钮（撤销最近的拆分/拼接操作）
        self.undo_button = QPushButton("撤回")
        self.undo_button.setMinimumWidth(80)
        self.undo_button.setMinimumHeight(32)
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("撤销最近的拆分/拼接操作")
        self.undo_button.clicked.connect(self._on_undo_clicked)
        button_layout.addWidget(self.undo_button)
        
        self.ok_button = QPushButton("确定")
        self.ok_button.setMinimumWidth(100)
        self.ok_button.setMinimumHeight(32)
        self.ok_button.setEnabled(False)
        self.ok_button.setToolTip("如启用拼接则先拼接再导入翻译器队列")
        self.ok_button.clicked.connect(self._on_ok_clicked)
        button_layout.addWidget(self.ok_button)
        
        cancel_button = QPushButton("取消")
        cancel_button.setMinimumWidth(100)
        cancel_button.setMinimumHeight(32)
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        return button_bar
    
    def _determine_root_path(self, start_dir: str) -> str:
        """智能判断根目录路径"""
        # 如果没有提供路径，使用Downloads
        if not start_dir:
            return str(Path.home() / "Downloads")
        
        # 如果路径不存在，使用Downloads
        if not os.path.isdir(start_dir):
            return str(Path.home() / "Downloads")
        
        # 检查是否是三层结构中的章节路径（最底层）
        # 路径模式：Downloads/Source/Title/Chapter
        path = Path(start_dir)
        
        # 尝试向上查找可能的根目录
        # 如果当前路径看起来像章节文件夹（包含图片文件）
        if self._looks_like_chapter_folder(path):
            # 向上3层到根目录
            if path.parent and path.parent.parent and path.parent.parent.parent:
                root_candidate = path.parent.parent.parent
                if os.path.isdir(root_candidate):
                    return str(root_candidate)
        
        # 如果当前路径看起来像作品文件夹（包含多个子文件夹）
        elif self._looks_like_title_folder(path):
            # 向上2层到根目录
            if path.parent and path.parent.parent:
                root_candidate = path.parent.parent
                if os.path.isdir(root_candidate):
                    return str(root_candidate)
        
        # 如果当前路径看起来像来源文件夹
        elif self._looks_like_source_folder(path):
            # 向上1层到根目录
            if path.parent:
                root_candidate = path.parent
                if os.path.isdir(root_candidate):
                    return str(root_candidate)
        
        # 默认使用提供的路径
        return start_dir
    
    def _looks_like_chapter_folder(self, path: Path) -> bool:
        """检查是否看起来像章节文件夹"""
        try:
            # 检查是否包含图片文件
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
            for item in path.iterdir():
                if item.is_file() and item.suffix.lower() in image_extensions:
                    return True
        except:
            pass
        return False
    
    def _looks_like_title_folder(self, path: Path) -> bool:
        """检查是否看起来像作品文件夹"""
        try:
            # 检查是否包含多个子文件夹（可能是章节）
            subfolders = [item for item in path.iterdir() if item.is_dir()]
            return len(subfolders) >= 2
        except:
            pass
        return False
    
    def _looks_like_source_folder(self, path: Path) -> bool:
        """检查是否看起来像来源文件夹"""
        try:
            # 检查是否包含多个子文件夹（可能是作品）
            subfolders = [item for item in path.iterdir() if item.is_dir()]
            # 来源文件夹通常包含多个作品文件夹
            return len(subfolders) >= 1
        except:
            pass
        return False
    
    def _apply_modern_style(self):
        """应用现代化样式"""
        palette = self.palette()
        
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {palette.color(QPalette.ColorRole.Window).name()};
            }}
            QWidget {{
                font-size: 13px;
            }}
            QLineEdit, QComboBox {{
                padding: 6px;
                border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                border-radius: 3px;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border: 2px solid #0078d4;
            }}
            QPushButton {{
                padding: 6px 16px;
                border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                border-radius: 3px;
                background-color: {palette.color(QPalette.ColorRole.Button).name()};
            }}
            QPushButton:hover {{
                background-color: {palette.color(QPalette.ColorRole.Light).name()};
            }}
            QPushButton:pressed {{
                background-color: {palette.color(QPalette.ColorRole.Midlight).name()};
            }}
            QPushButton#ok_button {{
                background-color: #0078d4;
                color: white;
                border: none;
            }}
            QPushButton#ok_button:hover {{
                background-color: #106ebe;
            }}
            QPushButton#ok_button:disabled {{
                background-color: #cccccc;
                color: #888888;
            }}
            QTreeWidget {{
                border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                selection-background-color: #0078d4;
            }}
            QTextEdit {{
                border: 1px solid {palette.color(QPalette.ColorRole.Mid).name()};
                border-radius: 3px;
            }}
        """)
        
        self.ok_button.setObjectName("ok_button")
    
    def _show_roots_dialog(self):
        """显示根目录管理弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("管理根目录")
        dialog.setMinimumSize(500, 300)
        
        layout = QVBoxLayout(dialog)
        
        # 根目录列表
        layout.addWidget(QLabel("根目录列表:"))
        roots_list = QListWidget()
        roots_list.setAlternatingRowColors(True)
        for path in self.root_paths:
            roots_list.addItem(path)
        layout.addWidget(roots_list)
        
        # 按钮区域
        btn_layout = QHBoxLayout()
        
        add_btn = QPushButton("+ 添加")
        def on_add():
            folder = QFileDialog.getExistingDirectory(
                dialog, "选择根目录",
                self.root_paths[0] if self.root_paths else str(Path.home())
            )
            if folder and folder not in self.root_paths:
                self.root_paths.append(folder)
                roots_list.addItem(folder)
                self._save_root_paths()
                self._update_roots_count()
        add_btn.clicked.connect(on_add)
        btn_layout.addWidget(add_btn)
        
        remove_btn = QPushButton("- 移除选中")
        def on_remove():
            current = roots_list.currentItem()
            if current:
                path = current.text()
                if path in self.root_paths:
                    self.root_paths.remove(path)
                roots_list.takeItem(roots_list.row(current))
                self._save_root_paths()
                self._update_roots_count()
        remove_btn.clicked.connect(on_remove)
        btn_layout.addWidget(remove_btn)
        
        btn_layout.addStretch()
        
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(close_btn)
        
        layout.addLayout(btn_layout)
        
        dialog.exec()
    
    def _update_roots_count(self):
        """更新根目录数量显示"""
        self.roots_count_label.setText(f"根目录: {len(self.root_paths)} 个")
    
    def add_root_folder(self):
        """添加根目录（兼容旧代码）"""
        self._show_roots_dialog()
    
    def remove_root_folder(self):
        """移除根目录（兼容旧代码）"""
        self._show_roots_dialog()
    
    def _save_root_paths(self):
        """保存根目录列表"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("root_paths", self.root_paths)
        # 兼容旧代码
        if self.root_paths:
            self.root_path = self.root_paths[0]
            settings.setValue("last_dir", self.root_path)
    
    def browse_folder(self):
        """浏览选择文件夹（兼容旧代码）"""
        self.add_root_folder()
    
    def scan_folders_async(self):
        """异步扫描所有根目录下的文件夹结构"""
        if not self.root_paths:
            QMessageBox.warning(self, "路径错误", "请先添加根目录")
            return
        
        if self._is_scanning:
            self._log("⚠️ 扫描正在进行中，请稍候...")
            return
        
        self._log(f"开始扫描 {len(self.root_paths)} 个根目录...")
        self._is_scanning = True
        
        # 显示扫描进度条，禁用扫描按钮
        self.scan_btn.setEnabled(False)
        self.scan_btn.setText("扫描中...")
        self.scan_progress.setVisible(True)
        
        # 清空旧数据
        self.folder_data.clear()
        
        # 创建后台线程和工作器
        self._scan_thread = QThread()
        self._scan_worker = ScanWorker(self.root_paths, self.name_replacer)
        self._scan_worker.moveToThread(self._scan_thread)
        
        # 连接信号
        self._scan_thread.started.connect(self._scan_worker.run)
        self._scan_worker.finished.connect(self._on_scan_finished)
        self._scan_worker.progress.connect(self._log)
        self._scan_worker.error.connect(self._on_scan_error)
        self._scan_worker.finished.connect(self._scan_thread.quit)
        self._scan_worker.finished.connect(self._scan_worker.deleteLater)
        self._scan_thread.finished.connect(self._scan_thread.deleteLater)
        
        # 启动扫描
        self._scan_thread.start()
    
    def _on_scan_finished(self, folder_data: dict):
        """扫描完成回调"""
        self._is_scanning = False
        
        # 恢复UI状态
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("🔍 扫描作品")
        self.scan_progress.setVisible(False)
        
        # 保存扫描结果
        self.folder_data = folder_data
        
        if folder_data:
            self.refresh_title_list()
            self.refresh_mapping_names()
            
            # 扫描成功后保存根目录
            self._save_root_path()
            
            # 保存扫描结果到缓存
            self._save_scan_cache()
    
    def _on_scan_error(self, error_msg: str):
        """扫描错误回调"""
        self._is_scanning = False
        
        # 恢复UI状态
        self.scan_btn.setEnabled(True)
        self.scan_btn.setText("🔍 扫描作品")
        self.scan_progress.setVisible(False)
        
        self._log(f"❌ {error_msg}")
        QMessageBox.critical(self, "扫描错误", f"扫描文件夹时出错:\n{error_msg}")
    
    def _stop_scan_if_running(self):
        """如果扫描正在进行，停止它"""
        if self._is_scanning and self._scan_worker:
            self._scan_worker.stop()
            if self._scan_thread and self._scan_thread.isRunning():
                self._scan_thread.quit()
                self._scan_thread.wait(1000)  # 最多等待1秒
            self._is_scanning = False
    
    def scan_folders(self):
        """同步扫描（兼容旧代码，内部调用异步版本）"""
        self.scan_folders_async()
    
    def _is_chapter_folder_list(self, dirs: List[Path]) -> bool:
        """
        判断目录列表是否为章节目录
        
        规则：如果超过50%的子目录符合章节命名模式，则认为是章节目录
        章节命名模式：
        - Chapter开头（Chapter_01, Chapter 1, Chapter1等）
        - 纯数字（001, 01, 1等）
        - 第X话/第X章（第1话, 第01章等）
        - EP/Vol开头（EP01, Vol.1等）
        """
        import re
        
        if not dirs:
            return False
        
        chapter_patterns = [
            r'^[Cc]hapter[_\s\-]?\d+',  # Chapter_01, Chapter 1, Chapter-1
            r'^\d+$',                    # 纯数字 001, 01, 1
            r'^第\d+[话章集卷]',          # 第1话, 第01章
            r'^[Ee][Pp][_\s\-]?\d+',     # EP01, EP_01
            r'^[Vv]ol[._\s]?\d+',        # Vol.1, Vol 1
            r'^[Cc]h[._\s]?\d+',         # Ch.1, Ch 1
        ]
        
        chapter_count = 0
        for d in dirs:
            name = d.name
            for pattern in chapter_patterns:
                if re.match(pattern, name):
                    chapter_count += 1
                    break
        
        # 超过50%符合章节模式
        return chapter_count > len(dirs) * 0.5
    
    def _get_mapped_name(self, original_name: str) -> str:
        """获取映射后的名称"""
        if not self.name_replacer:
            return original_name
        
        for raw_name, translated_name in self.name_replacer.mapping.items():
            raw_names = [n.strip() for n in raw_name.split('|')]
            if original_name in raw_names:
                return translated_name
        
        return original_name
    
    def refresh_title_list(self):
        """刷新作品列表"""
        self.title_tree.clear()
        
        total_chapters = 0
        
        for title, data in sorted(self.folder_data.items()):
            sources_str = ", ".join(sorted(data['sources']))
            chapter_count = sum(len(chapters) for chapters in data['chapters'].values())
            total_chapters += chapter_count
            
            parent_item = QTreeWidgetItem(self.title_tree)
            parent_item.setText(0, title)
            parent_item.setText(1, sources_str)
            parent_item.setText(2, f"{chapter_count} 章节")
            parent_item.setText(3, "未展开")
            parent_item.setData(0, Qt.ItemDataRole.UserRole, title)
            
            # 多来源高亮
            if len(data['sources']) > 1:
                parent_item.setBackground(1, QColor(255, 243, 205))
        
        # 暂时断开itemChanged信号，避免在批量加载时触发
        self.title_tree.itemChanged.disconnect(self.on_item_changed)
        
        try:
            # 不使用 expandAll，改为手动展开每个作品并加载章节
            for i in range(self.title_tree.topLevelItemCount()):
                parent = self.title_tree.topLevelItem(i)
                # 手动加载章节（不等待用户点击）
                self._load_chapters_for_item(parent)
                # 展开作品
                parent.setExpanded(True)
        finally:
            # 重新连接信号
            self.title_tree.itemChanged.connect(self.on_item_changed)
        
        self._update_stats(len(self.folder_data), total_chapters, 0)
        
        # 自动填充搜索框（优先使用映射名称）
        self._populate_search_combo_with_mapping()
        
        # 更新来源筛选下拉框
        self._update_source_filter_combo()
    
    def _load_chapters_for_item(self, item: QTreeWidgetItem):
        """为指定作品项加载章节（三层结构：作品 → 来源 → 章节）"""
        # 如果已经加载过章节，不重复加载
        if item.childCount() > 0:
            return
        
        title = item.data(0, Qt.ItemDataRole.UserRole)
        if not title or title not in self.folder_data:
            return
        
        data = self.folder_data[title]
        sources = sorted(data['sources'])
        
        # 如果有多个来源，创建三层结构：作品 → 来源 → 章节
        if len(sources) > 1:
            # 获取每个来源的最新章节信息（用于排序）
            source_info = []
            for source_name in sources:
                chapters = data['chapters'].get(source_name, [])
                if chapters:
                    # 排序章节获取最新的
                    sorted_chapters = self._sort_chapters(chapters)
                    # 注意：_sort_chapters 已根据 sort_ascending 处理过升/降序
                    # 升序 -> 最新在最后一个；降序 -> 最新在第一个
                    latest_chapter = sorted_chapters[-1] if self.sort_ascending else sorted_chapters[0]
                    # 获取最新话的修改时间
                    try:
                        latest_time = os.path.getmtime(latest_chapter['path'])
                    except:
                        latest_time = 0
                    
                    source_info.append({
                        'name': source_name,
                        'chapters': chapters,
                        'latest_chapter': latest_chapter,
                        'latest_time': latest_time,
                        'count': len(chapters)
                    })
            
            # 按最新话时间排序来源（最新的在前）
            source_info.sort(key=lambda x: x['latest_time'], reverse=True)
            
            # 创建来源节点
            for idx, info in enumerate(source_info):
                source_item = QTreeWidgetItem(item)
                source_item.setText(0, info['name'])
                source_item.setText(1, "来源")
                # 显示：最新话 / 总数（提取章节号）
                import re
                latest_name = info['latest_chapter']['name']
                match = re.search(r'(\d+)', latest_name)
                if match:
                    chapter_num = match.group(1)
                    source_item.setText(2, f"{chapter_num}话 / {info['count']}章")
                else:
                    source_item.setText(2, f"{latest_name} / {info['count']}章")
                
                # 排序章节
                sorted_chapters = self._sort_chapters(info['chapters'])
                
                # 在来源下添加章节
                for chapter in sorted_chapters:
                    chapter_item = QTreeWidgetItem(source_item)
                    chapter_item.setText(0, chapter['name'])
                    chapter_item.setText(1, info['name'])
                    # 显示修改时间而不是路径（智能格式）
                    try:
                        mtime = os.path.getmtime(chapter['path'])
                        time_str = self._format_time_smart(mtime)
                        chapter_item.setText(2, time_str)
                    except:
                        chapter_item.setText(2, "未知")
                    chapter_item.setData(0, Qt.ItemDataRole.UserRole, chapter['path'])
                    # 设置复选框
                    chapter_item.setCheckState(0, Qt.CheckState.Unchecked)
                    chapter_item.setFlags(chapter_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                
                # 自动展开第一个来源（最新的）
                if idx == 0:
                    source_item.setExpanded(True)
        else:
            # 只有一个来源，直接显示章节
            source_name = sources[0]
            chapters = data['chapters'].get(source_name, [])
            
            # 排序章节
            sorted_chapters = self._sort_chapters(chapters)
            
            # 更新作品的"最新话/数量"列：显示最新章节号 / 总数
            if sorted_chapters:
                # 升序 -> 最新在最后一个；降序 -> 最新在第一个
                latest_chapter = sorted_chapters[-1] if self.sort_ascending else sorted_chapters[0]
                import re
                latest_name = latest_chapter['name']
                match = re.search(r'(\d+)', latest_name)
                if match:
                    chapter_num = match.group(1)
                    item.setText(2, f"{chapter_num}话 / {len(sorted_chapters)}章")
                else:
                    item.setText(2, f"{latest_name} / {len(sorted_chapters)}章")
            
            for chapter in sorted_chapters:
                child_item = QTreeWidgetItem(item)
                child_item.setText(0, chapter['name'])
                child_item.setText(1, source_name)
                # 显示修改时间而不是路径（智能格式）
                try:
                    mtime = os.path.getmtime(chapter['path'])
                    time_str = self._format_time_smart(mtime)
                    child_item.setText(2, time_str)
                except:
                    child_item.setText(2, "未知")
                child_item.setData(0, Qt.ItemDataRole.UserRole, chapter['path'])
                # 设置复选框
                child_item.setCheckState(0, Qt.CheckState.Unchecked)
                child_item.setFlags(child_item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        
        item.setText(3, "已展开")
    
    def on_item_expanded(self, item: QTreeWidgetItem):
        """当项目展开时，加载章节（用户手动展开时）"""
        self._load_chapters_for_item(item)
    
    def on_item_changed(self, item: QTreeWidgetItem, column: int):
        """项目状态改变时更新统计"""
        if column == 0:
            self._update_selection_count()
    
    def on_selection_changed(self):
        """选择改变时同步到复选框（正向累加，反向取消）"""
        # 防止递归调用
        if self._processing_selection:
            return
        
        self._processing_selection = True
        
        # 暂时断开itemChanged信号，避免递归
        try:
            self.title_tree.itemChanged.disconnect(self.on_item_changed)
        except TypeError:
            # 信号未连接时忽略
            pass
        
        try:
            # 获取当前选中的项目
            selected_items = self.title_tree.selectedItems()
            current_selected_paths = set()
            
            # 收集所有选中的章节路径
            for item in selected_items:
                path = item.data(0, Qt.ItemDataRole.UserRole)
                if path and isinstance(path, str) and os.path.isdir(path):  # 是章节项
                    current_selected_paths.add(path)
            
            # 获取所有已勾选的章节路径
            root = self.title_tree.invisibleRootItem()
            checked_paths = self._get_all_checked_paths(root)
            
            # 上一次的选择集合（用于判断是扩展、收缩还是新的选择）
            last_paths = getattr(self, "last_selected_paths", set())
            
            if last_paths and current_selected_paths and current_selected_paths.issuperset(last_paths):
                # 情况1：当前选择是上次选择的扩展（正向拖拽）→ 只勾选新增部分
                newly_selected = current_selected_paths - checked_paths
                if newly_selected:
                    self._toggle_checkboxes_for_paths(root, newly_selected, Qt.CheckState.Checked)
            
            elif last_paths and current_selected_paths and current_selected_paths.issubset(last_paths):
                # 情况2：当前选择是上次选择的子集（反向拖拽）→ 只取消本次被“缩出去”的部分
                to_uncheck = (last_paths - current_selected_paths) & checked_paths
                if to_uncheck:
                    self._toggle_checkboxes_for_paths(root, to_uncheck, Qt.CheckState.Unchecked)
            
            else:
                # 情况3：与上次选择无关（新的区域或复杂形状）→ 默认为累加选择
                newly_selected = current_selected_paths - checked_paths
                if newly_selected:
                    self._toggle_checkboxes_for_paths(root, newly_selected, Qt.CheckState.Checked)
            
            # 更新上一次的选择状态
            self.last_selected_paths = current_selected_paths.copy()
            
        finally:
            # 重新连接信号
            try:
                self.title_tree.itemChanged.connect(self.on_item_changed)
            except Exception:
                # 忽略重复连接错误
                pass
            # 更新统计
            self._update_selection_count()
            # 释放递归锁
            self._processing_selection = False
    
    def _get_all_checked_paths(self, parent: QTreeWidgetItem) -> set:
        """获取所有已勾选的章节路径（递归）"""
        checked_paths = set()
        for i in range(parent.childCount()):
            item = parent.child(i)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            
            if path and isinstance(path, str) and os.path.isdir(path):  # 是章节项
                if item.checkState(0) == Qt.CheckState.Checked:
                    checked_paths.add(path)
            
            # 递归处理子项
            if item.childCount() > 0:
                checked_paths.update(self._get_all_checked_paths(item))
        
        return checked_paths
    
    def _toggle_checkboxes_for_paths(self, parent: QTreeWidgetItem, paths: set, new_state: Qt.CheckState):
        """为指定路径的项目切换复选框状态（递归）"""
        for i in range(parent.childCount()):
            item = parent.child(i)
            path = item.data(0, Qt.ItemDataRole.UserRole)
            
            if path and isinstance(path, str) and os.path.isdir(path):  # 是章节项
                if path in paths:
                    # 切换到新状态
                    item.setCheckState(0, new_state)
            
            # 递归处理子项
            if item.childCount() > 0:
                self._toggle_checkboxes_for_paths(item, paths, new_state)
    
    def on_sort_changed(self):
        """排序方式改变"""
        self.sort_by = self.sort_combo.currentData()
        
        # 保存排序设置
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("sort_by", self.sort_by)
        
        sort_names = {'name': '名称', 'natural': '智能排序', 'date': '修改日期'}
        self._log(f"✓ 排序方式: {sort_names.get(self.sort_by, self.sort_by)}")
        self._reload_all_chapters()
    
    def toggle_sort_order(self):
        """切换升序/降序"""
        self.sort_ascending = not self.sort_ascending
        
        # 保存排序设置
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("sort_ascending", self.sort_ascending)
        
        if self.sort_ascending:
            self.sort_order_btn.setText("↑ 升序")
            self._log("✓ 排序顺序: 升序")
        else:
            self.sort_order_btn.setText("↓ 降序")
            self._log("✓ 排序顺序: 降序")
        
        self._reload_all_chapters()
    
    def _reload_all_chapters(self):
        """重新加载所有展开的章节"""
        for i in range(self.title_tree.topLevelItemCount()):
            item = self.title_tree.topLevelItem(i)
            if item.childCount() > 0:
                # 清除现有子项
                item.takeChildren()
                # 重新加载
                self._load_chapters_for_item(item)
                item.setExpanded(True)
    
    def _sort_chapters(self, chapters: List[Dict]) -> List[Dict]:
        """根据当前排序方式排序章节"""
        import re
        
        if self.sort_by == "name":
            # 按名称排序
            sorted_list = sorted(chapters, key=lambda x: x['name'])
        elif self.sort_by == "natural":
            # 智能排序（自然排序，处理数字）
            def natural_key(text):
                return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', text)]
            sorted_list = sorted(chapters, key=lambda x: natural_key(x['name']))
        elif self.sort_by == "date":
            # 按修改日期排序
            def get_mtime(chapter):
                try:
                    return os.path.getmtime(chapter['path'])
                except:
                    return 0
            sorted_list = sorted(chapters, key=get_mtime)
        else:
            sorted_list = chapters
        
        # 应用升序/降序
        if not self.sort_ascending:
            sorted_list = list(reversed(sorted_list))
        
        return sorted_list
    
    def apply_filter(self):
        """应用搜索过滤（同时考虑关键词和来源筛选）"""
        keyword = self.search_combo.currentText().strip().lower()
        selected_source = self.source_filter_combo.currentData()
        
        for i in range(self.title_tree.topLevelItemCount()):
            item = self.title_tree.topLevelItem(i)
            title = item.text(0).lower()
            sources = item.text(1).lower()
            
            # Check keyword filter
            keyword_match = not keyword or keyword in title or keyword in sources
            
            # Check source filter
            source_match = selected_source is None or selected_source.lower() in sources
            
            item.setHidden(not (keyword_match and source_match))
    
    def _on_source_filter_changed(self):
        """来源筛选改变时触发"""
        selected_source = self.source_filter_combo.currentText()
        if selected_source == "全部":
            self._log("✓ 显示全部来源")
        else:
            self._log(f"✓ 筛选来源: {selected_source}")
        self.apply_filter()
    
    def _update_source_filter_combo(self):
        """更新来源筛选下拉框"""
        # Collect all sources from folder_data
        all_sources = set()
        for data in self.folder_data.values():
            all_sources.update(data.get('sources', set()))
        
        # Remember current selection
        current_selection = self.source_filter_combo.currentData()
        
        # Update combo box
        self.source_filter_combo.blockSignals(True)
        self.source_filter_combo.clear()
        self.source_filter_combo.addItem("全部", None)
        for source in sorted(all_sources):
            self.source_filter_combo.addItem(source, source)
        
        # Restore selection if still valid
        if current_selection:
            index = self.source_filter_combo.findData(current_selection)
            if index >= 0:
                self.source_filter_combo.setCurrentIndex(index)
        
        self.source_filter_combo.blockSignals(False)
    
    def _populate_search_combo(self):
        """填充搜索框（使用当前扫描到的作品名称）"""
        self.search_combo.clear()
        
        # 添加所有作品名称
        if self.folder_data:
            titles = sorted(self.folder_data.keys())
            self.search_combo.addItems(titles)
    
    def _populate_search_combo_with_mapping(self):
        """填充搜索框（优先使用名称映射管理中的熟肉名称）"""
        self.search_combo.clear()
        
        # 优先尝试从名称映射管理获取
        if self.name_replacer and self.name_replacer.mapping:
            translated_names = set(self.name_replacer.mapping.values())
            if translated_names:
                self.search_combo.addItems(sorted(translated_names))
                self._log(f"✓ 已从名称映射加载 {len(translated_names)} 个作品名称")
                return
        
        # 如果没有映射，使用扫描到的作品名称
        if self.folder_data:
            titles = sorted(self.folder_data.keys())
            self.search_combo.addItems(titles)
            self._log(f"✓ 已加载 {len(titles)} 个作品名称（未使用映射）")
    
    def refresh_mapping_names(self):
        """刷新映射名称到搜索框（从名称映射管理同步）"""
        if not self.name_replacer:
            self._log("⚠️ 名称映射功能未加载")
            return
        
        # 重新加载映射
        try:
            self.name_replacer.reload()
            self._log("✓ 已重新加载名称映射配置")
        except:
            pass
        
        # 使用统一的填充方法
        self._populate_search_combo_with_mapping()
    
    def select_all_chapters(self):
        """全选所有章节"""
        count = 0
        for i in range(self.title_tree.topLevelItemCount()):
            parent = self.title_tree.topLevelItem(i)
            if not parent.isHidden():
                count += self._check_all_chapters_recursive(parent, True)
        
        self._log(f"✓ 已全选 {count} 个章节")
    
    def deselect_all_chapters(self):
        """取消全选"""
        count = 0
        for i in range(self.title_tree.topLevelItemCount()):
            parent = self.title_tree.topLevelItem(i)
            count += self._check_all_chapters_recursive(parent, False)
        
        self._log(f"✓ 已取消全选")
    
    def clear_selected_chapters(self):
        """清空已选章节列表（包括拖放导入的章节）"""
        # 先取消树形控件中的选择
        self.deselect_all_chapters()
        
        # 清空拖放导入的章节
        self.selected_chapters = []
        self.selected_chapters_list.clear()
        
        # 更新统计
        self._update_selection_count()
        
        self._log(f"✓ 已清空已选章节列表")
    
    def _clean_hash_suffixes(self):
        """清理文件夹名的哈希后缀
        
        将类似 "Chapter 61_563e24" 重命名为 "Chapter 61"
        支持的后缀格式：
        - _[0-9a-f]{4,8}  如 _563e24, _a1b2c3d4
        - -[0-9a-f]{4,8}  如 -563e24
        """
        import re
        
        # 获取已选章节
        selected = self.get_selected_chapters()
        if not selected:
            QMessageBox.information(self, "提示", "请先选择要清理的章节文件夹")
            return
        
        # 哈希后缀模式：下划线或连字符 + 4-8位十六进制字符
        hash_pattern = re.compile(r'[_-][0-9a-fA-F]{4,8}$')
        
        # 统计可重命名的文件夹
        rename_list = []  # [(old_path, new_path, old_name, new_name), ...]
        
        for chapter_path in selected:
            folder_name = os.path.basename(chapter_path)
            parent_dir = os.path.dirname(chapter_path)
            
            # 检查是否匹配哈希后缀
            match = hash_pattern.search(folder_name)
            if match:
                new_name = folder_name[:match.start()].strip()
                if new_name:  # 确保新名称不为空
                    new_path = os.path.join(parent_dir, new_name)
                    # 检查目标路径是否已存在
                    if not os.path.exists(new_path):
                        rename_list.append((chapter_path, new_path, folder_name, new_name))
                    else:
                        self._log(f"⚠️ 跳过: {folder_name} → {new_name}（目标已存在）")
        
        if not rename_list:
            QMessageBox.information(self, "提示", "所选文件夹中没有发现哈希后缀")
            return
        
        # 确认对话框
        preview = "\n".join([f"{old} → {new}" for _, _, old, new in rename_list[:10]])
        if len(rename_list) > 10:
            preview += f"\n... 等 {len(rename_list)} 个文件夹"
        
        reply = QMessageBox.question(
            self,
            "确认重命名",
            f"将重命名 {len(rename_list)} 个文件夹：\n\n{preview}\n\n确认执行？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 执行重命名
        success_count = 0
        error_count = 0
        
        for old_path, new_path, old_name, new_name in rename_list:
            try:
                os.rename(old_path, new_path)
                self._log(f"✓ 重命名: {old_name} → {new_name}")
                success_count += 1
            except Exception as e:
                self._log(f"✗ 失败: {old_name} - {e}", "ERROR")
                error_count += 1
        
        # 显示结果
        self._log(f"✓ 清理完成: 成功 {success_count} 个，失败 {error_count} 个")
        
        # 如果有成功重命名的，重新扫描
        if success_count > 0:
            QMessageBox.information(
                self,
                "清理完成",
                f"成功重命名 {success_count} 个文件夹\n将重新扫描文件夹..."
            )
            # 清空选择并重新扫描
            self.clear_selected_chapters()
            self.scan_folders_async()
    
    def _auto_clean_hash_on_import(self, chapter_paths: List[str]) -> List[str]:
        """导入时自动清理哈希后缀（静默执行，不弹窗确认）
        
        Args:
            chapter_paths: 章节路径列表
            
        Returns:
            更新后的路径列表（重命名成功的路径会更新）
        """
        import re
        
        # 哈希后缀模式：下划线或连字符 + 4-8位十六进制字符
        hash_pattern = re.compile(r'[_-][0-9a-fA-F]{4,8}$')
        
        updated_paths = []
        renamed_count = 0
        
        for chapter_path in chapter_paths:
            folder_name = os.path.basename(chapter_path)
            parent_dir = os.path.dirname(chapter_path)
            
            # 检查是否匹配哈希后缀
            match = hash_pattern.search(folder_name)
            if match:
                new_name = folder_name[:match.start()].strip()
                if new_name:  # 确保新名称不为空
                    new_path = os.path.join(parent_dir, new_name)
                    # 检查目标路径是否已存在
                    if not os.path.exists(new_path):
                        try:
                            os.rename(chapter_path, new_path)
                            self._log(f"✓ 自动清理: {folder_name} → {new_name}")
                            updated_paths.append(new_path)
                            renamed_count += 1
                            continue
                        except Exception as e:
                            self._log(f"⚠️ 自动清理失败: {folder_name} - {e}")
                    else:
                        self._log(f"⚠️ 跳过: {folder_name}（目标 {new_name} 已存在）")
            
            # 保持原路径
            updated_paths.append(chapter_path)
        
        if renamed_count > 0:
            self._log(f"✓ 自动清理哈希后缀完成: {renamed_count} 个")
        
        return updated_paths
    
    def get_selected_chapters(self) -> List[str]:
        """获取选中的章节路径（包括树形控件选中的和拖放导入的）"""
        selected = []
        
        # 从树形控件收集选中的章节
        for i in range(self.title_tree.topLevelItemCount()):
            parent = self.title_tree.topLevelItem(i)
            self._collect_checked_chapters_recursive(parent, selected)
        
        # 合并拖放导入的章节
        for chapter in self.selected_chapters:
            if isinstance(chapter, dict):
                path = chapter.get('path', '')
            else:
                path = chapter
            if path and path not in selected:
                selected.append(path)
        
        return selected
    
    def _check_all_chapters_recursive(self, item: QTreeWidgetItem, checked: bool) -> int:
        """递归勾选/取消勾选所有章节"""
        count = 0
        for i in range(item.childCount()):
            child = item.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            
            if path and isinstance(path, str) and os.path.isdir(path):  # 是章节项
                child.setCheckState(0, Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
                count += 1
            else:  # 是来源项或其他，递归处理
                count += self._check_all_chapters_recursive(child, checked)
        
        return count
    
    def _collect_checked_chapters_recursive(self, item: QTreeWidgetItem, selected: List[str]):
        """递归收集所有勾选的章节"""
        for i in range(item.childCount()):
            child = item.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            
            if path and isinstance(path, str) and os.path.isdir(path):  # 是章节项
                if child.checkState(0) == Qt.CheckState.Checked:
                    selected.append(path)
            else:  # 是来源项或其他，递归处理
                self._collect_checked_chapters_recursive(child, selected)
    
    def _update_selection_count(self):
        """更新选中数量"""
        selected_chapters = self.get_selected_chapters()
        count = len(selected_chapters)
        
        # 更新统计
        total_works = len(self.folder_data)
        total_chapters = sum(
            sum(len(chapters) for chapters in data['chapters'].values())
            for data in self.folder_data.values()
        )
        self._update_stats(total_works, total_chapters, count)
        
        # 更新确定按钮、拼接按钮和拆分按钮状态
        self.ok_button.setEnabled(count > 0)
        self.stitch_button.setEnabled(count > 0)
        self.split_button.setEnabled(count > 0)
        
        # 更新已选章节列表
        self._update_selected_chapters_list(selected_chapters)
    
    def _update_stats(self, title_count: int, chapter_count: int, selected_count: int):
        """更新统计信息"""
        self.stats_label.setText(
            f"作品: {title_count} | 章节: {chapter_count} | 已选: {selected_count}"
        )
    
    def _log(self, message: str, level: str = "INFO"):
        """输出日志（带时间戳和级别）
        
        同时输出到：
        1. 对话框内的日志文本框
        2. 标准日志系统（文件和控制台）
        3. 通过信号发送到主程序（如果已连接）
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S,%f")[:-3]
        formatted = f"{timestamp} - manga_translator - {level} - {message}"
        
        # 输出到对话框日志
        try:
            self.log_text.append(formatted)
        except Exception:
            pass  # 忽略UI错误
        
        # 输出到标准日志系统
        try:
            root_logger = logging.getLogger()
            if level == "ERROR":
                root_logger.error(f"[AdvancedFolder] {message}")
            elif level == "WARNING":
                root_logger.warning(f"[AdvancedFolder] {message}")
            else:
                root_logger.info(f"[AdvancedFolder] {message}")
        except Exception:
            pass  # 忽略日志错误
        
        # 发送信号到主程序
        try:
            self.log_to_main.emit(message, level)
        except Exception:
            pass  # 信号未连接时忽略
    
    def _format_time_smart(self, timestamp: float) -> str:
        """智能格式化时间
        
        规则：
        - 今天：HH:MM (如 14:30)
        - 今年：MM-DD HH:MM (如 11-23 14:30)
        - 往年：YYYY-MM-DD (如 2024-11-23)
        """
        try:
            file_time = datetime.fromtimestamp(timestamp)
            now = datetime.now()
            
            # 判断是否是今天
            if file_time.date() == now.date():
                return file_time.strftime("%H:%M")
            
            # 判断是否是今年
            elif file_time.year == now.year:
                return file_time.strftime("%m-%d %H:%M")
            
            # 往年
            else:
                return file_time.strftime("%Y-%m-%d")
        except:
            return "未知"
    
    def toggle_log(self):
        """切换日志显示/隐藏"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        if self.log_text.isVisible():
            self.log_text.hide()
            self.log_toggle_btn.setText("展开 ▲")
            settings.setValue("log_expanded", False)
        else:
            self.log_text.show()
            self.log_toggle_btn.setText("收起 ▼")
            settings.setValue("log_expanded", True)
    
    def execute_smart_select(self):
        """执行智能选择（根据当前模式）"""
        if self.smart_select_mode == "latest":
            self.select_latest_chapters()
        else:
            self.select_recent_hour_latest()
    
    def set_smart_select_mode(self, mode: str, execute: bool = True):
        """设置智能选择模式并保存
        
        Args:
            mode: 模式名称 ('latest' 或 'recent_hour')
            execute: 是否立即执行选择，默认True
        """
        self.smart_select_mode = mode
        
        # 更新菜单项的勾选状态
        self.action_latest.setChecked(mode == "latest")
        self.action_recent_hour.setChecked(mode == "recent_hour")
        
        # 保存选择
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("smart_select_mode", mode)
        
        mode_names = {"latest": "选择所有最新章节", "recent_hour": "选择1小时内下载的章节"}
        self._log(f"✓ 智能选择模式: {mode_names.get(mode, mode)}")
        
        # 根据execute参数决定是否立即执行
        if execute:
            self.execute_smart_select()
    
    def select_latest_chapters(self):
        """智能选择：选择所有已展开作品的最新话"""
        count = self._select_latest_recursive(self.title_tree.invisibleRootItem())
        self._update_selection_count()
        self._log(f"✓ 已智能选择 {count} 个最新章节")
    
    def select_recent_hour_latest(self):
        """智能选择：选择1小时内下载的所有章节"""
        one_hour_ago = datetime.now() - timedelta(hours=1)
        count = 0
        
        # 遍历所有作品
        root = self.title_tree.invisibleRootItem()
        for i in range(root.childCount()):
            work_item = root.child(i)
            # 展开作品以加载章节
            if work_item.childCount() == 0:
                self._load_chapters_for_item(work_item)
                work_item.setExpanded(True)
            
            # 选择该作品中所有1小时内下载的章节
            count += self._select_recent_chapters(work_item, one_hour_ago)
        
        self._update_selection_count()
        self._log(f"✓ 已选择1小时内下载的 {count} 个章节")
    
    def _select_recent_chapters(self, work_item: QTreeWidgetItem, time_threshold: datetime) -> int:
        """选择作品中所有在时间阈值之后下载的章节"""
        count = 0
        
        def select_in_children(parent):
            nonlocal count
            for i in range(parent.childCount()):
                child = parent.child(i)
                chapter_path = child.data(0, Qt.ItemDataRole.UserRole)
                
                if chapter_path and os.path.isdir(chapter_path):  # 是章节项
                    try:
                        mtime = os.path.getmtime(chapter_path)
                        # 检查是否在时间阈值之后
                        if mtime > time_threshold.timestamp():
                            child.setCheckState(0, Qt.CheckState.Checked)
                            count += 1
                    except:
                        pass
                else:  # 可能是来源项，继续递归
                    select_in_children(child)
        
        select_in_children(work_item)
        return count
    
    def _find_latest_chapter_item(self, work_item: QTreeWidgetItem, time_threshold: datetime) -> Optional[QTreeWidgetItem]:
        """查找作品的最新章节项（需要在时间阈值之后）"""
        latest_item = None
        latest_time = 0
        
        # 递归查找所有章节项
        def find_chapters(parent):
            nonlocal latest_item, latest_time
            for i in range(parent.childCount()):
                child = parent.child(i)
                chapter_path = child.data(0, Qt.ItemDataRole.UserRole)
                
                if chapter_path and os.path.isdir(chapter_path):  # 是章节项
                    try:
                        mtime = os.path.getmtime(chapter_path)
                        # 检查是否在时间阈值之后
                        if mtime > time_threshold.timestamp() and mtime > latest_time:
                            latest_time = mtime
                            latest_item = child
                    except:
                        pass
                else:  # 可能是来源项，继续递归
                    find_chapters(child)
        
        find_chapters(work_item)
        return latest_item
    
    def _select_latest_recursive(self, parent: QTreeWidgetItem) -> int:
        """递归选择已展开节点的最新章节"""
        count = 0
        
        for i in range(parent.childCount()):
            item = parent.child(i)
            
            # 如果是已展开的作品项且有子项
            if item.isExpanded() and item.childCount() > 0:
                # 查找并选中最新的章节
                latest_chapter = self._find_latest_chapter_in_item(item)
                if latest_chapter:
                    latest_chapter.setCheckState(0, Qt.CheckState.Checked)
                    count += 1
                
                # 递归处理子项
                count += self._select_latest_recursive(item)
        
        return count
    
    def _find_latest_chapter_in_item(self, item: QTreeWidgetItem) -> Optional[QTreeWidgetItem]:
        """在指定项中查找最新的章节项"""
        latest_item = None
        latest_time = 0
        
        def find_in_children(parent):
            nonlocal latest_item, latest_time
            for i in range(parent.childCount()):
                child = parent.child(i)
                chapter_path = child.data(0, Qt.ItemDataRole.UserRole)
                
                if chapter_path and os.path.isdir(chapter_path):  # 是章节项
                    try:
                        mtime = os.path.getmtime(chapter_path)
                        if mtime > latest_time:
                            latest_time = mtime
                            latest_item = child
                    except:
                        pass
                else:  # 可能是来源项，继续递归
                    find_in_children(child)
        
        find_in_children(item)
        return latest_item
    
    def _load_column_widths(self):
        """加载保存的列宽"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        header = self.title_tree.header()
        
        for i in range(4):
            width = settings.value(f"column_width_{i}", type=int)
            if width:
                header.resizeSection(i, width)
    
    def _save_column_widths(self):
        """保存列宽"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        header = self.title_tree.header()
        
        for i in range(4):
            settings.setValue(f"column_width_{i}", header.sectionSize(i))
    
    def accept(self):
        """确定按钮点击"""
        self._stop_scan_if_running()
        self._stop_stitch_if_running()
        self._stop_split_if_running()
        self._save_column_widths()
        self._save_root_path()
        self._save_recent_works()  # 保存最近操作的作品
        self._clear_undo_data()    # 关闭时清理撤回备份
        super().accept()
    
    def reject(self):
        """取消按钮点击"""
        self._stop_scan_if_running()
        self._stop_stitch_if_running()
        self._stop_split_if_running()
        self._save_column_widths()
        self._save_root_path()
        self._clear_undo_data()    # 关闭时清理撤回备份
        super().reject()
    
    def closeEvent(self, event):
        """窗口关闭事件"""
        # 确保关闭窗口时清理撤回备份
        self._clear_undo_data()
        super().closeEvent(event)
    
    def _save_root_path(self):
        """保存根目录路径（兼容旧代码）"""
        self._save_root_paths()
    
    def _on_folders_dropped(self, folders: List[str]):
        """处理拖放导入的文件夹"""
        print(f"[拖放] _on_folders_dropped 收到 {len(folders)} 个文件夹")
        added_count = 0
        for folder in folders:
            print(f"[拖放] 处理: {folder}")
            if os.path.isdir(folder):
                # 检查是否已经在选中列表中
                existing_paths = [c.get('path', '') if isinstance(c, dict) else c for c in self.selected_chapters]
                if folder not in existing_paths:
                    chapter_info = {
                        'name': os.path.basename(folder),
                        'path': folder,
                        'source': 'drag_drop'
                    }
                    self.selected_chapters.append(chapter_info)
                    self.selected_chapters_list.addItem(f"📁 {chapter_info['name']}")
                    added_count += 1
                    print(f"[拖放] 已添加: {chapter_info['name']}")
        
        if added_count > 0:
            self._log(f"✓ 拖放导入 {added_count} 个文件夹")
            self._update_selection_count()
        else:
            print("[拖放] 没有新增文件夹")
    
    def _on_selected_chapter_double_clicked(self, item):
        """双击已选章节项，打开所在文件夹"""
        try:
            # 优先从 UserRole 获取路径（_update_selected_chapters_list 方法设置的）
            chapter_path = item.data(Qt.ItemDataRole.UserRole)
            
            # 如果 UserRole 没有路径，尝试通过索引获取（拖放导入的情况）
            if not chapter_path:
                row = self.selected_chapters_list.row(item)
                if 0 <= row < len(self.selected_chapters):
                    chapter_info = self.selected_chapters[row]
                    if isinstance(chapter_info, dict):
                        chapter_path = chapter_info.get('path', '')
                    else:
                        chapter_path = str(chapter_info)
            
            if chapter_path and os.path.isdir(chapter_path):
                # 在 Windows 资源管理器中打开文件夹
                import subprocess
                subprocess.Popen(['explorer', chapter_path])
                self._log(f"📂 已打开: {os.path.basename(chapter_path)}")
            else:
                self._log(f"⚠️ 文件夹不存在: {chapter_path}")
        except Exception as e:
            print(f"[双击打开] 错误: {e}")
    
    def _load_recent_works(self):
        """加载最近操作的作品（两行显示一个操作）"""
        try:
            import json
            settings = QSettings("MangaTranslator", "AdvancedFolder")
            recent_data = settings.value("recent_works", "")
            
            if not recent_data:
                return
            
            recent_operations = json.loads(recent_data)
            self.recent_works_list.clear()
            
            # 处理旧格式数据（字符串列表）兼容
            if recent_operations and isinstance(recent_operations[0], str):
                # 转换为新格式
                new_operations = []
                for work_name in recent_operations[:4]:
                    new_operations.append({
                        'works': [work_name],
                        'chapter_count': 0,
                        'time': ''
                    })
                recent_operations = new_operations
            
            # 显示最近4个操作（每个操作占两行）
            for operation in recent_operations[:4]:
                # 确保operation是字典类型
                if not isinstance(operation, dict):
                    continue
                
                item = QTreeWidgetItem(self.recent_works_list)
                
                # 统计信息（N个作品 M章）
                works = operation.get('works', [])
                chapter_count = operation.get('chapter_count', 0)
                work_count = len(works)
                
                if work_count == 1:
                    summary_text = f"{chapter_count}章"
                else:
                    summary_text = f"{work_count}个作品, {chapter_count}章"
                
                # 时间 + 章节范围
                time_str = operation.get('time', '')
                chapter_range = operation.get('chapter_range', '')
                
                if time_str and chapter_range:
                    detail_text = f"{time_str} · {chapter_range}"
                elif time_str:
                    detail_text = time_str
                elif chapter_range:
                    detail_text = chapter_range
                else:
                    detail_text = "历史记录"
                
                # 单行显示：统计信息 | 详细信息
                display_text = f"{summary_text} | {detail_text}"
                item.setText(0, display_text)
                
                # 保存完整数据
                item.setData(0, Qt.ItemDataRole.UserRole, operation)
                
                # 设置行高（单行）
                item.setSizeHint(0, QSize(0, 24))
                
        except Exception as e:
            # 错误时不输出日志，避免log_text未初始化错误
            print(f"[AdvancedFolder] 加载最近作品失败: {e}")
    
    def _save_recent_works(self):
        """保存最近操作的作品（支持多个作品一起保存，记录具体章节）"""
        try:
            import json
            from datetime import datetime
            settings = QSettings("MangaTranslator", "AdvancedFolder")
            
            # ✅ 使用 get_selected_chapters() 获取实际选择的章节（包括树中勾选的和拖放导入的）
            all_selected_paths = self.get_selected_chapters()
            if not all_selected_paths:
                return
            
            # Normalize and de-duplicate chapter paths (important for drag&drop restore)
            all_selected_paths_norm = []
            seen_paths = set()
            for p in all_selected_paths:
                if not p:
                    continue
                p_norm = os.path.abspath(str(p))
                if p_norm in seen_paths:
                    continue
                seen_paths.add(p_norm)
                all_selected_paths_norm.append(p_norm)

            # Group chapters by work name
            selected_data = []  # [{'work': name, 'chapters': [...], 'chapter_paths': [...]}, ...]
            work_chapters_map = {}  # {work_name: {'chapters': [...], 'chapter_paths': [...]}}

            # Build path -> work mapping from title_tree (if available)
            root = self.title_tree.invisibleRootItem()
            path_to_work = {}  # {chapter_path: work_name}
            for i in range(root.childCount()):
                work_item = root.child(i)
                work_name = work_item.text(0)
                self._collect_chapter_paths_to_work(work_item, work_name, path_to_work)

            # Group chapters
            for chapter_path in all_selected_paths_norm:
                chapter_name = os.path.basename(chapter_path)

                # Prefer mapping from tree; fallback to parent folder name for drag&drop cases
                work_name = path_to_work.get(chapter_path)
                if not work_name:
                    parent_dir = os.path.dirname(chapter_path)
                    work_name = os.path.basename(parent_dir) or "未分组"

                if work_name not in work_chapters_map:
                    work_chapters_map[work_name] = {"chapters": [], "chapter_paths": []}
                work_chapters_map[work_name]["chapters"].append(chapter_name)
                work_chapters_map[work_name]["chapter_paths"].append(chapter_path)

            # Convert to selected_data and count chapters
            total_chapters = 0
            for work_name, data in work_chapters_map.items():
                selected_data.append({
                    'work': work_name,
                    'chapters': data.get('chapters', []),
                    'chapter_paths': data.get('chapter_paths', [])
                })
                total_chapters += len(data.get('chapters', []))

            if not selected_data:
                return
            
            # 计算章节范围（取所有章节名中的最小和最大）
            all_chapter_names = []
            for item in selected_data:
                all_chapter_names.extend(item['chapters'])
            
            chapter_range = self._calculate_chapter_range(all_chapter_names)
            
            # 创建操作记录
            work_list = [item['work'] for item in selected_data]
            operation = {
                'works': work_list,  # 保持兼容
                'works_detail': selected_data,  # 新增：详细章节信息
                'chapter_paths': all_selected_paths_norm,  # Used for robust restore (including drag&drop)
                'chapter_count': total_chapters,
                'chapter_range': chapter_range,
                'time': datetime.now().strftime("%m-%d %H:%M")
            }
            
            # 加载现有的最近操作
            recent_data = settings.value("recent_works", "")
            if recent_data:
                recent_operations = json.loads(recent_data)
            else:
                recent_operations = []
            
            # 去重：删除相同作品组合的旧记录
            work_set = set(work_list)
            recent_operations = [
                op for op in recent_operations
                if set(op.get('works', [])) != work_set
            ]
            
            # 将新操作添加到列表开头
            recent_operations.insert(0, operation)
            
            # 只保留最近10个操作
            recent_operations = recent_operations[:10]
            
            # 保存
            settings.setValue("recent_works", json.dumps(recent_operations, ensure_ascii=False))
            settings.sync()  # ✅ 强制立即写入磁盘，防止程序退出时数据丢失
            self._log(f"✓ 已保存最近操作: {len(selected_data)}个作品, {total_chapters}章")
            
        except Exception as e:
            self._log(f"⚠️ 保存最近作品失败: {e}")
    
    def _calculate_chapter_range(self, chapter_names: List[str]) -> str:
        """计算章节范围，如 Ch.1-Ch.10"""
        if not chapter_names:
            return ""
        
        import re
        # 提取章节号
        chapter_numbers = []
        for name in chapter_names:
            # 尝试从章节名中提取数字
            match = re.search(r'(\d+)', name)
            if match:
                chapter_numbers.append(int(match.group(1)))
        
        if not chapter_numbers:
            # 无法提取数字，返回章节数量
            return f"{len(chapter_names)}章"
        
        chapter_numbers.sort()
        min_ch = chapter_numbers[0]
        max_ch = chapter_numbers[-1]
        
        if min_ch == max_ch:
            return f"Ch.{min_ch}"
        else:
            return f"Ch.{min_ch}-{max_ch}"
    
    def _has_checked_children(self, item: QTreeWidgetItem) -> bool:
        """检查项目是否有被选中的子项"""
        for i in range(item.childCount()):
            child = item.child(i)
            if child.checkState(0) == Qt.CheckState.Checked:
                return True
            if self._has_checked_children(child):
                return True
        return False
    
    def _count_checked_children(self, item: QTreeWidgetItem) -> int:
        """统计项目下被选中的子项数量"""
        count = 0
        for i in range(item.childCount()):
            child = item.child(i)
            if child.checkState(0) == Qt.CheckState.Checked:
                count += 1
            count += self._count_checked_children(child)
        return count
    
    def _get_checked_chapter_names(self, item: QTreeWidgetItem) -> List[str]:
        """获取项目下所有被选中的章节名称"""
        chapters = []
        for i in range(item.childCount()):
            child = item.child(i)
            # 检查是否是章节项（有路径数据）
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if path and isinstance(path, str) and os.path.isdir(path):
                if child.checkState(0) == Qt.CheckState.Checked:
                    chapters.append(child.text(0))  # 保存章节名
            else:
                # 递归处理子项（三层结构时的来源节点）
                chapters.extend(self._get_checked_chapter_names(child))
        return chapters
    
    def _collect_chapter_paths_to_work(self, item: QTreeWidgetItem, work_name: str, path_to_work: dict):
        """收集章节路径与作品名的映射（递归）"""
        for i in range(item.childCount()):
            child = item.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            if path and isinstance(path, str) and os.path.isdir(path):
                # 是章节项，记录路径到作品名的映射
                path_to_work[path] = work_name
            else:
                # 递归处理子项（三层结构时的来源节点）
                self._collect_chapter_paths_to_work(child, work_name, path_to_work)
    
    def on_recent_work_double_clicked(self, item: QTreeWidgetItem, column: int):
        """双击最近操作时跳转并恢复章节选中（支持多个作品）"""
        operation = item.data(0, Qt.ItemDataRole.UserRole)
        if not operation or 'works' not in operation:
            self._log("⚠️ 无效的操作数据")
            return
        
        # 获取详细信息（包含章节列表）或旧格式数据
        works_detail = operation.get('works_detail', [])
        if not works_detail:
            # 兼容旧数据：只有作品名，没有章节信息
            works = operation['works']
            works_detail = [{'work': w, 'chapters': []} for w in works]
        
        work_names = [w.get('work', '') for w in works_detail if isinstance(w, dict)]
        
        # Prefer restoring by chapter paths (robust for drag&drop selections)
        chapter_paths = operation.get('chapter_paths', [])
        if not chapter_paths and works_detail:
            for w in works_detail:
                if isinstance(w, dict):
                    chapter_paths.extend(w.get('chapter_paths', []) or [])
        
        if chapter_paths:
            self._log(f"↻ 正在按路径恢复: {len(chapter_paths)} 章")
        else:
            self._log(f"🔍 正在查找: {', '.join(work_names)}")
        
        # 清除搜索过滤，确保所有作品可见
        self.search_combo.setCurrentText("")
        self.search_combo.clearEditText()
        
        # 清除当前选中
        self.title_tree.clearSelection()
        
        # 暂时断开信号，避免批量操作时多次触发
        try:
            self.title_tree.itemChanged.disconnect(self.on_item_changed)
        except TypeError:
            # 信号未连接时忽略
            pass
        
        try:
            # 在作品列表中查找并展开所有相关作品
            root = self.title_tree.invisibleRootItem()
            found_count = 0
            first_item = None
            not_found = []
            total_checked = 0
            
            # If we have chapter paths, restore selection from paths first (works even without scan)
            if chapter_paths:
                # Clear existing checked state silently to avoid mixing selections
                for i in range(root.childCount()):
                    self._check_all_chapters_recursive(root.child(i), False)

                # Clear current drag&drop selections
                self.selected_chapters = []
                try:
                    self.selected_chapters_list.clear()
                except Exception:
                    pass

                restored_paths = []
                missing_paths = []
                seen = set()
                for p in chapter_paths:
                    if not p:
                        continue
                    p_norm = os.path.abspath(str(p))
                    if p_norm in seen:
                        continue
                    seen.add(p_norm)
                    if os.path.isdir(p_norm):
                        restored_paths.append(p_norm)
                    else:
                        missing_paths.append(p_norm)

                if not restored_paths:
                    self._log("⚠️ 最近操作的章节路径均不存在，尝试按作品/章节名恢复")
                else:
                    self.selected_chapters = [{'path': p, 'source': 'recent_restore'} for p in restored_paths]

                    # Best-effort: sync checked state in title_tree if the works are present
                    paths_by_work = {}
                    for p in restored_paths:
                        work_dir_raw = os.path.basename(os.path.dirname(p)) or "未分组"
                        work_dir = self._get_mapped_name(work_dir_raw)
                        if work_dir not in paths_by_work:
                            paths_by_work[work_dir] = []
                        paths_by_work[work_dir].append(p)

                    not_in_tree = []
                    for work_dir, paths in paths_by_work.items():
                        target_clean = work_dir.replace(' ', '').lower()
                        matched_item = None
                        for i in range(root.childCount()):
                            candidate = root.child(i)
                            item_name = candidate.text(0)
                            item_clean = item_name.replace(' ', '').lower()
                            if (item_name == work_dir or
                                work_dir in item_name or
                                item_name in work_dir or
                                item_clean == target_clean or
                                target_clean in item_clean):
                                matched_item = candidate
                                break

                        if matched_item is None:
                            not_in_tree.append(work_dir)
                            continue

                        matched_item.setSelected(True)
                        matched_item.setExpanded(True)
                        if first_item is None:
                            first_item = matched_item

                        # Load chapters lazily
                        if matched_item.childCount() == 0:
                            self._load_chapters_for_item(matched_item)

                        path_set = {os.path.abspath(x) for x in paths}
                        checked_count = self._restore_chapter_selection_by_paths(matched_item, path_set)
                        total_checked += checked_count
                        found_count += 1

                    if first_item:
                        self.title_tree.scrollToItem(first_item)
                        self.title_tree.setCurrentItem(first_item)

                    self._log(f"✓ 已恢复选中 {len(restored_paths)} 章")
                    if total_checked > 0:
                        self._log(f"  ✓ 已同步勾选树节点 {total_checked} 章")
                    if missing_paths:
                        self._log(f"  ⚠️ 路径不存在已跳过: {len(missing_paths)} 章")
                    if not_in_tree:
                        self._log(f"  💡 有 {len(not_in_tree)} 个作品不在当前扫描列表，仅恢复到已选章节")

                    return

            for work_data in works_detail:
                work_name = work_data['work']
                chapter_names = work_data.get('chapters', [])
                found = False
                
                for i in range(root.childCount()):
                    work_item = root.child(i)
                    item_name = work_item.text(0)
                    
                    # 多种匹配方式：精确匹配、包含匹配、忽略空格匹配
                    item_name_clean = item_name.replace(' ', '').lower()
                    work_name_clean = work_name.replace(' ', '').lower()
                    
                    if (item_name == work_name or 
                        work_name in item_name or 
                        item_name in work_name or
                        item_name_clean == work_name_clean or
                        work_name_clean in item_name_clean):
                        # 选中并展开
                        work_item.setSelected(True)
                        work_item.setExpanded(True)
                        
                        # 加载章节（如果还没加载）
                        if work_item.childCount() == 0:
                            self._load_chapters_for_item(work_item)
                        
                        # 恢复章节选中状态
                        if chapter_names:
                            checked_count = self._restore_chapter_selection(work_item, chapter_names)
                            total_checked += checked_count
                            self._log(f"  ✓ 找到: {item_name}，恢复选中 {checked_count} 章")
                        else:
                            self._log(f"  ✓ 找到: {item_name}")
                        
                        found_count += 1
                        found = True
                        if first_item is None:
                            first_item = work_item
                        break
                
                if not found:
                    not_found.append(work_name)
            
            # 滚动到第一个作品
            if first_item:
                self.title_tree.scrollToItem(first_item)
                # 确保第一个作品在视窗顶部
                self.title_tree.setCurrentItem(first_item)
            
            # 输出结果
            if found_count > 0:
                if total_checked > 0:
                    self._log(f"✓ 已跳转到 {found_count} 个作品，恢复选中 {total_checked} 章")
                elif found_count == 1:
                    self._log(f"✓ 已跳转到: {work_names[0]}")
                else:
                    self._log(f"✓ 已跳转到 {found_count} 个作品")
            
            if not_found:
                self._log(f"⚠️ 未找到: {', '.join(not_found)}")
        
        finally:
            # 重新连接信号
            try:
                self.title_tree.itemChanged.connect(self.on_item_changed)
            except Exception:
                # 忽略重复连接错误
                pass
            # 手动触发一次统计更新
            self._update_selection_count()
    
    def _restore_chapter_selection(self, work_item: QTreeWidgetItem, chapter_names: List[str]) -> int:
        """恢复作品下的章节选中状态"""
        checked_count = 0
        # 遍历所有子项（可能是章节或来源）
        for i in range(work_item.childCount()):
            child = work_item.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)
            
            # 如果是章节项
            if path and isinstance(path, str) and os.path.isdir(path):
                chapter_name = child.text(0)
                # 模糊匹配章节名
                if self._is_chapter_match(chapter_name, chapter_names):
                    child.setCheckState(0, Qt.CheckState.Checked)
                    checked_count += 1
            else:
                # 递归处理子项（三层结构时的来源节点）
                checked_count += self._restore_chapter_selection(child, chapter_names)
        
        return checked_count
    
    def _restore_chapter_selection_by_paths(self, work_item: QTreeWidgetItem, chapter_paths: Set[str]) -> int:
        """Restore chapter checked state by absolute folder paths."""
        checked_count = 0
        for i in range(work_item.childCount()):
            child = work_item.child(i)
            path = child.data(0, Qt.ItemDataRole.UserRole)

            if path and isinstance(path, str) and os.path.isdir(path):
                if os.path.abspath(path) in chapter_paths:
                    child.setCheckState(0, Qt.CheckState.Checked)
                    checked_count += 1
            else:
                checked_count += self._restore_chapter_selection_by_paths(child, chapter_paths)

        return checked_count
    
    def _is_chapter_match(self, chapter_name: str, chapter_names: List[str]) -> bool:
        """检查章节名是否匹配（多种匹配方式）"""
        chapter_clean = chapter_name.replace(' ', '').lower()
        for name in chapter_names:
            name_clean = name.replace(' ', '').lower()
            if (chapter_name == name or 
                name in chapter_name or 
                chapter_name in name or
                chapter_clean == name_clean or
                name_clean in chapter_clean):
                return True
        return False
    
    def _save_scan_cache(self):
        """保存扫描结果到缓存"""
        try:
            import json
            settings = QSettings("MangaTranslator", "AdvancedFolder")
            
            # 转换set为list以便JSON序列化
            cache_data = {
                'root_path': str(Path(self.root_path_edit.text().strip()).resolve()),  # 规范化路径
                'folder_data': {}
            }
            
            for title, data in self.folder_data.items():
                cache_data['folder_data'][title] = {
                    'original_names': list(data['original_names']),
                    'sources': list(data['sources']),
                    'chapters': data['chapters']  # chapters已经是可序列化的
                }
            
            settings.setValue("scan_cache", json.dumps(cache_data, ensure_ascii=False))
            self._log("✓ 已保存扫描结果")
            
        except Exception as e:
            self._log(f"⚠️ 保存扫描结果失败: {e}")
    
    def _load_scan_cache(self):
        """加载上次扫描结果"""
        try:
            import json
            settings = QSettings("MangaTranslator", "AdvancedFolder")
            cached_data = settings.value("scan_cache", "")
            
            if not cached_data:
                self._log("💡 首次使用，请点击【扫描作品】按钮开始")
                return
            
            cache = json.loads(cached_data)
            cached_root = cache.get('root_path', '')
            
            # 规范化路径进行比较
            current_root = str(Path(self.root_path_edit.text().strip()).resolve()) if self.root_path_edit.text().strip() else ""
            cached_root_normalized = str(Path(cached_root).resolve()) if cached_root else ""
            
            # 如果缓存的根目录与当前根目录不同，不加载
            if cached_root_normalized != current_root:
                self._log(f"💡 根目录已更改，请重新扫描")
                self._log(f"  缓存: {cached_root_normalized}")
                self._log(f"  当前: {current_root}")
                return
            
            # 转换list回set
            self.folder_data = {}
            for title, data in cache.get('folder_data', {}).items():
                self.folder_data[title] = {
                    'original_names': set(data.get('original_names', [])),
                    'sources': set(data.get('sources', [])),
                    'chapters': data.get('chapters', {})
                }
            
            if self.folder_data:
                self.refresh_title_list()
                self._populate_search_combo_with_mapping()
                self._log(f"✓ 已加载上次扫描结果（{len(self.folder_data)} 个作品）")
            else:
                self._log("💡 请点击【扫描作品】按钮开始")
                
        except Exception as e:
            self._log(f"⚠️ 加载扫描结果失败: {e}")
            self._log("💡 请点击【扫描作品】按钮开始")


    def _update_selected_chapters_list(self, selected_chapters: List[str]):
        """更新已选章节列表显示"""
        self.selected_chapters_list.clear()
        
        for chapter_path in selected_chapters:
            chapter_name = os.path.basename(chapter_path)
            
            # 统计图片数量
            image_count = 0
            image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
            try:
                for f in os.listdir(chapter_path):
                    if os.path.splitext(f)[1].lower() in image_extensions:
                        image_count += 1
            except:
                pass
            
            # 显示格式：ch.X 图片数
            display_text = f"{chapter_name} ({image_count}张)"
            
            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, chapter_path)
            self.selected_chapters_list.addItem(item)
    
    def _save_stitch_settings(self):
        """保存长图拼接设置"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("stitch_enabled", self.stitch_enabled_cb.isChecked())
        settings.setValue("stitch_max_height", self.stitch_max_height_spin.value())
        settings.setValue("stitch_margin", self.stitch_margin_spin.value())
        
        # 刷新已选章节列表显示
        selected_chapters = self.get_selected_chapters()
        self._update_selected_chapters_list(selected_chapters)
    
    def _save_auto_clean_setting(self):
        """保存自动清理哈希后缀设置"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("auto_clean_hash", self.auto_clean_hash_cb.isChecked())
    
    def _show_stitch_settings_popup(self):
        """显示长图拼接参数设置弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("长图拼接设置")
        dialog.setFixedSize(250, 120)
        
        layout = QFormLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)
        
        # 最大拼接高度
        height_spin = QSpinBox()
        height_spin.setRange(1000, 50000)
        height_spin.setSingleStep(1000)
        height_spin.setValue(self.stitch_max_height_spin.value())
        height_spin.setSuffix(" px")
        layout.addRow("最大拼接高度:", height_spin)
        
        # 气泡边距
        margin_spin = QSpinBox()
        margin_spin.setRange(0, 500)
        margin_spin.setSingleStep(10)
        margin_spin.setValue(self.stitch_margin_spin.value())
        margin_spin.setSuffix(" px")
        layout.addRow("气泡边距:", margin_spin)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addRow(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存设置
            self.stitch_max_height_spin.setValue(height_spin.value())
            self.stitch_margin_spin.setValue(margin_spin.value())
            self._save_stitch_settings()
            self._log(f"✓ 长图拼接参数已更新: 高度={height_spin.value()}px, 边距={margin_spin.value()}px")
    
    def _on_ok_clicked(self):
        """确定按钮点击：如启用拼接则先拼接再导入翻译器队列"""
        selected_chapters = self.get_selected_chapters()
        if not selected_chapters:
            return
        
        # 如果启用了自动清理哈希后缀，先执行清理
        if self.auto_clean_hash_cb.isChecked():
            selected_chapters = self._auto_clean_hash_on_import(selected_chapters)
        
        # 更新内部已选列表（路径可能已改变）
        self.selected_chapters = [{'path': p} for p in selected_chapters]
        
        stitch_enabled = self.stitch_enabled_cb.isChecked()
        
        if stitch_enabled:
            # 使用异步拼接，拼接完成后自动接受对话框
            self._start_stitch_async(selected_chapters, accept_after=True)
        else:
            # 直接接受对话框
            self.accept()
    
    def _on_stitch_clicked(self):
        """拼接按钮点击：仅对所选章节执行长图拼接（不关闭对话框）"""
        selected_chapters = self.get_selected_chapters()
        if not selected_chapters:
            QMessageBox.warning(self, "提示", "请先选择要拼接的章节")
            return
        
        if self._is_stitching:
            self._log("⚠️ 拼接正在进行中，请稍候...")
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, "确认拼接",
            f"确定要对 {len(selected_chapters)} 个章节执行长图拼接吗？\n\n"
            f"⚠️ 此操作会删除原图，替换为拼接后的图片！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 使用异步拼接，完成后不关闭对话框
        self._start_stitch_async(selected_chapters, accept_after=False)
    
    def _start_stitch_async(self, chapters: List[str], accept_after: bool = False):
        """启动异步拼接任务
        
        Args:
            chapters: 要拼接的章节路径列表
            accept_after: 拼接完成后是否自动接受对话框
        """
        if self._is_stitching:
            self._log("⚠️ 拼接正在进行中，请稍候...")
            return
        
        self._is_stitching = True
        self._stitch_accept_after = accept_after
        
        # 获取拼接参数
        max_height = self.stitch_max_height_spin.value()
        margin = self.stitch_margin_spin.value()
        
        # 显示进度条，禁用按钮
        self.stitch_button.setEnabled(False)
        self.stitch_button.setText("拼接中...")
        self.ok_button.setEnabled(False)
        self.stitch_progress.setRange(0, len(chapters))
        self.stitch_progress.setValue(0)
        self.stitch_progress.setVisible(True)
        self.stitch_progress_label.setText(f"拼接进度: 0/{len(chapters)}")
        self.stitch_progress_label.setVisible(True)
        
        # 创建后台线程和工作器
        self._stitch_thread = QThread()
        self._stitch_worker = StitchWorker(chapters, max_height, margin)
        self._stitch_worker.moveToThread(self._stitch_thread)
        
        # 连接信号
        self._stitch_thread.started.connect(self._stitch_worker.run)
        self._stitch_worker.finished.connect(self._on_stitch_finished)
        self._stitch_worker.progress.connect(self._log)
        self._stitch_worker.chapter_progress.connect(self._on_stitch_chapter_progress)
        self._stitch_worker.error.connect(self._on_stitch_error)
        self._stitch_worker.finished.connect(self._stitch_thread.quit)
        self._stitch_worker.finished.connect(self._stitch_worker.deleteLater)
        self._stitch_thread.finished.connect(self._stitch_thread.deleteLater)
        
        # 启动拼接
        self._stitch_thread.start()
    
    def _on_stitch_chapter_progress(self, current: int, total: int):
        """拼接章节进度更新"""
        self.stitch_progress.setValue(current)
        self.stitch_progress_label.setText(f"拼接进度: {current}/{total}")
    
    def _on_stitch_finished(self, result_files: list):
        """拼接完成回调"""
        self._is_stitching = False
        
        # 恢复UI状态
        self.stitch_button.setEnabled(True)
        self.stitch_button.setText("拼接")
        self.ok_button.setEnabled(True)
        self.stitch_progress.setVisible(False)
        self.stitch_progress_label.setVisible(False)
        
        # 刷新已选章节列表显示
        selected_chapters = self.get_selected_chapters()
        self._update_selected_chapters_list(selected_chapters)
        
        # 如果需要在拼接完成后接受对话框
        if self._stitch_accept_after:
            self._log(f"[长图拼接] ✓ 全部完成，导入翻译器队列")
            self.accept()
    
    def _on_stitch_error(self, error_msg: str):
        """拼接错误回调"""
        self._is_stitching = False
        
        # 恢复UI状态
        self.stitch_button.setEnabled(True)
        self.stitch_button.setText("拼接")
        self.ok_button.setEnabled(True)
        self.stitch_progress.setVisible(False)
        self.stitch_progress_label.setVisible(False)
        
        self._log(f"❌ {error_msg}", "ERROR")
        QMessageBox.critical(self, "拼接错误", f"拼接图片时出错:\n{error_msg}")
    
    def _stop_stitch_if_running(self):
        """如果拼接正在进行，停止它"""
        if self._is_stitching and self._stitch_worker:
            self._stitch_worker.stop()
            if self._stitch_thread and self._stitch_thread.isRunning():
                self._stitch_thread.quit()
                self._stitch_thread.wait(1000)  # 最多等待1秒
            self._is_stitching = False
    
    def _stitch_chapter_images(self, chapter_path: str, max_height: int, margin: int) -> List[str]:
        """拼接章节内的图片
        
        改进说明：
        1. 输出文件名格式：seg{batch:02d}_img{start:03d}-{end:03d}_{count}p.png
        2. 当文件名以seg开头时跳过拼接（表示已经是拼接后的文件）
        3. 当文件不满足拼接条件时（只有1张图或已是seg文件）不拼接
        4. 只有在拼接成功后才删除原图，防止文件被清零
        """
        from PIL import Image
        import re
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        image_files = []
        seg_files_exist = False
        
        # 收集所有图片文件
        for f in sorted(os.listdir(chapter_path)):
            ext = os.path.splitext(f)[1].lower()
            if ext not in image_extensions:
                continue
            
            # 检查是否是seg开头的文件（已经是拼接后的文件）
            if f.lower().startswith('seg'):
                seg_files_exist = True
                self._log(f"[长图拼接] ⚠️ 检测到已拼接文件: {f}，跳过此章节")
                break
            
            image_files.append(os.path.join(chapter_path, f))
        
        # 如果存在seg文件，说明已经拼接过，跳过
        if seg_files_exist:
            self._log(f"[长图拼接] ⚠️ 章节已拼接过，跳过")
            return []
        
        # 如果没有图片，返回空
        if not image_files:
            self._log(f"[长图拼接] ⚠️ 未找到图片文件，跳过")
            return []
        
        # 如果只有1张图片，不需要拼接
        if len(image_files) == 1:
            self._log(f"[长图拼接] ⚠️ 只有1张图片，不满足拼接条件，跳过")
            return []
        
        self._log(f"[长图拼接] 开始处理 {len(image_files)} 张图片")
        
        # 提取原始文件的序号信息
        def extract_image_number(filepath: str) -> int:
            """从文件名中提取图片序号"""
            filename = os.path.basename(filepath)
            name_without_ext = os.path.splitext(filename)[0]
            # 尝试提取数字
            match = re.search(r'(\d+)', name_without_ext)
            if match:
                return int(match.group(1))
            return 0
        
        # 读取所有图片并获取尺寸
        images = []
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                img_num = extract_image_number(img_path)
                images.append({
                    'path': img_path,
                    'image': img,
                    'number': img_num,
                    'filename': os.path.basename(img_path)
                })
            except Exception as e:
                self._log(f"[长图拼接] ⚠️ 无法读取图片: {os.path.basename(img_path)}", "WARNING")
        
        if not images:
            self._log(f"[长图拼接] ⚠️ 无法读取任何图片，跳过")
            return []
        
        # 如果读取后只有1张图片，不需要拼接
        if len(images) == 1:
            images[0]['image'].close()
            self._log(f"[长图拼接] ⚠️ 只有1张可读取的图片，不满足拼接条件，跳过")
            return []
        
        # 拼接图片
        result_files = []
        current_batch = []
        current_height = 0
        batch_index = 1
        
        for img_data in images:
            img = img_data['image']
            img_height = img.height
            
            if current_height + img_height > max_height and current_batch:
                # 保存当前批次
                result_path = self._save_stitched_image_v2(chapter_path, current_batch, batch_index, margin)
                if result_path:
                    result_files.append(result_path)
                    batch_count = len(current_batch)
                    start_num = current_batch[0]['number']
                    end_num = current_batch[-1]['number']
                    self._log(f"[长图拼接] 分段 {batch_index}: img{start_num:03d}-{end_num:03d} ({batch_count}张) → 高度 {current_height}px")
                batch_index += 1
                current_batch = [img_data]
                current_height = img_height
            else:
                current_batch.append(img_data)
                current_height += img_height
        
        # 保存最后一批
        if current_batch:
            result_path = self._save_stitched_image_v2(chapter_path, current_batch, batch_index, margin)
            if result_path:
                result_files.append(result_path)
                batch_count = len(current_batch)
                start_num = current_batch[0]['number']
                end_num = current_batch[-1]['number']
                self._log(f"[长图拼接] 分段 {batch_index}: img{start_num:03d}-{end_num:03d} ({batch_count}张) → 高度 {current_height}px")
        
        # 只有在拼接成功后才删除原图（防止文件被清零）
        if result_files:
            deleted_count = 0
            for img_data in images:
                img_data['image'].close()
                try:
                    os.remove(img_data['path'])
                    deleted_count += 1
                except Exception as e:
                    self._log(f"[长图拼接] ⚠️ 无法删除原图: {img_data['filename']}", "WARNING")
            
            self._log(f"[长图拼接] 已删除 {deleted_count} 张原图")
        else:
            # 拼接失败，只关闭图片，不删除原图
            self._log(f"[长图拼接] ⚠️ 拼接未成功，保留原图")
            for img_data in images:
                img_data['image'].close()
        
        return result_files
    
    def _save_stitched_image_v2(self, chapter_path: str, batch: List[dict], batch_index: int, margin: int) -> Optional[str]:
        """保存拼接后的图片（新版本）
        
        文件名格式：seg{batch:02d}_img{start:03d}-{end:03d}_{count}p.png
        例如：seg01_img001-005_5p.png
        """
        from PIL import Image
        
        if not batch:
            return None
        
        try:
            # 计算总高度和最大宽度
            total_height = sum(item['image'].height for item in batch) + margin * (len(batch) - 1)
            max_width = max(item['image'].width for item in batch)
            
            # 创建新图片
            stitched = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            
            # 拼接
            y_offset = 0
            for item in batch:
                img = item['image']
                # 居中放置
                x_offset = (max_width - img.width) // 2
                stitched.paste(img.convert('RGB'), (x_offset, y_offset))
                y_offset += img.height + margin
            
            # 生成文件名：seg{batch:02d}_img{start:03d}-{end:03d}_{count}p.png
            start_num = batch[0]['number']
            end_num = batch[-1]['number']
            count = len(batch)
            
            filename = f"seg{batch_index:02d}_img{start_num:03d}-{end_num:03d}_{count}p.png"
            result_path = os.path.join(chapter_path, filename)
            
            # 保存为PNG格式
            stitched.save(result_path, 'PNG', optimize=True)
            stitched.close()
            
            return result_path
            
        except Exception as e:
            self._log(f"[长图拼接] ⚠️ 保存拼接图片失败: {e}", "ERROR")
            return None
    
    def _save_stitched_image(self, chapter_path: str, batch: List, batch_index: int, margin: int) -> Optional[str]:
        """保存拼接后的图片（旧版本，保留兼容性）"""
        from PIL import Image
        
        if not batch:
            return None
        
        # 计算总高度和最大宽度
        total_height = sum(img.height for _, img in batch) + margin * (len(batch) - 1)
        max_width = max(img.width for _, img in batch)
        
        # 创建新图片
        stitched = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        
        # 拼接
        y_offset = 0
        for _, img in batch:
            # 居中放置
            x_offset = (max_width - img.width) // 2
            stitched.paste(img.convert('RGB'), (x_offset, y_offset))
            y_offset += img.height + margin
        
        # 保存
        result_path = os.path.join(chapter_path, f"stitched_{batch_index:03d}.jpg")
        stitched.save(result_path, 'JPEG', quality=95)
        stitched.close()
        
        return result_path
    
    # ==================== 长图拆分相关方法 ====================
    
    def _save_split_settings(self):
        """保存长图拆分设置"""
        settings = QSettings("MangaTranslator", "AdvancedFolder")
        settings.setValue("split_enabled", self.split_enabled_cb.isChecked())
        settings.setValue("split_skip_threshold", self.split_skip_threshold_spin.value())
        settings.setValue("split_target_height", self.split_target_height_spin.value())
        settings.setValue("split_buffer_range", self.split_buffer_range_spin.value())
        settings.setValue("split_min_segment", self.split_min_segment_spin.value())
        settings.setValue("split_naming", self.split_naming_combo.currentText())
        settings.setValue("split_index_digits", self.split_index_digits_spin.value())
        settings.setValue("split_index_start", self.split_index_start_spin.value())
        settings.setValue("split_reset_index_per_chapter", self.split_reset_index_per_chapter_cb.isChecked())
        
        # 刷新已选章节列表显示
        selected_chapters = self.get_selected_chapters()
        self._update_selected_chapters_list(selected_chapters)
    
    def _show_split_settings_popup(self):
        """显示长图拆分参数设置弹窗"""
        dialog = QDialog(self)
        dialog.setWindowTitle("长图拆分设置")
        dialog.setFixedSize(320, 360)
        
        layout = QFormLayout(dialog)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        
        # 短图阈值（低于此高度不拆分）
        skip_threshold_spin = QSpinBox()
        skip_threshold_spin.setRange(1000, 10000)
        skip_threshold_spin.setSingleStep(100)
        skip_threshold_spin.setValue(self.split_skip_threshold_spin.value())
        skip_threshold_spin.setSuffix(" px")
        skip_threshold_spin.setToolTip("图片高度低于此值时不拆分")
        layout.addRow("短图阈值:", skip_threshold_spin)
        
        # 目标片段高度
        target_height_spin = QSpinBox()
        target_height_spin.setRange(1000, 10000)
        target_height_spin.setSingleStep(100)
        target_height_spin.setValue(self.split_target_height_spin.value())
        target_height_spin.setSuffix(" px")
        target_height_spin.setToolTip("拆分后每个片段的目标高度")
        layout.addRow("目标片段高度:", target_height_spin)
        
        # 缓冲范围
        buffer_range_spin = QSpinBox()
        buffer_range_spin.setRange(50, 1000)
        buffer_range_spin.setSingleStep(50)
        buffer_range_spin.setValue(self.split_buffer_range_spin.value())
        buffer_range_spin.setSuffix(" px")
        buffer_range_spin.setToolTip("在目标高度 ± 此范围内搜索最佳切点")
        layout.addRow("缓冲范围:", buffer_range_spin)
        
        # 最小片段高度
        min_segment_spin = QSpinBox()
        min_segment_spin.setRange(500, 5000)
        min_segment_spin.setSingleStep(100)
        min_segment_spin.setValue(self.split_min_segment_spin.value())
        min_segment_spin.setSuffix(" px")
        min_segment_spin.setToolTip("防止切出过小的片段")
        layout.addRow("最小片段高度:", min_segment_spin)
        
        # 命名模式
        naming_combo = QComboBox()
        naming_combo.addItems(["序号_源文件名", "序号", "源文件名_序号", "序号_宽x高"])
        naming_combo.setCurrentText(self.split_naming_combo.currentText())
        naming_combo.setToolTip("拆分后文件的命名规则\n序号_源文件名: 001_image.jpg\n序号: 001.jpg\n源文件名_序号: image_001.jpg\n序号_宽x高: 001_800x2600.jpg")
        layout.addRow("命名模式:", naming_combo)
        
        # 序号位数
        digits_spin = QSpinBox()
        digits_spin.setRange(1, 6)
        digits_spin.setSingleStep(1)
        digits_spin.setValue(self.split_index_digits_spin.value())
        digits_spin.setToolTip("序号位数（1=不补零，3=001）")
        layout.addRow("序号位数:", digits_spin)
        
        # 起始序号
        start_spin = QSpinBox()
        start_spin.setRange(0, 9999)
        start_spin.setSingleStep(1)
        start_spin.setValue(self.split_index_start_spin.value())
        start_spin.setToolTip("序号起始值（每章从此值开始编号）")
        layout.addRow("起始序号:", start_spin)
        
        # Reset index per chapter
        reset_index_cb = QCheckBox("序号按章节重置")
        reset_index_cb.setChecked(self.split_reset_index_per_chapter_cb.isChecked())
        reset_index_cb.setToolTip("批量拆分多个章节时，每个章节的序号从起始序号重新开始")
        layout.addRow("", reset_index_cb)
        
        # 按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        
        ok_btn = QPushButton("确定")
        ok_btn.clicked.connect(dialog.accept)
        btn_layout.addWidget(ok_btn)
        
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(dialog.reject)
        btn_layout.addWidget(cancel_btn)
        
        layout.addRow(btn_layout)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            # 保存设置
            self.split_skip_threshold_spin.setValue(skip_threshold_spin.value())
            self.split_target_height_spin.setValue(target_height_spin.value())
            self.split_buffer_range_spin.setValue(buffer_range_spin.value())
            self.split_min_segment_spin.setValue(min_segment_spin.value())
            self.split_naming_combo.setCurrentText(naming_combo.currentText())
            self.split_index_digits_spin.setValue(digits_spin.value())
            self.split_index_start_spin.setValue(start_spin.value())
            self.split_reset_index_per_chapter_cb.setChecked(reset_index_cb.isChecked())
            self._save_split_settings()
            self._log(f"✓ 长图拆分参数已更新: 短图阈值={skip_threshold_spin.value()}px, 命名={naming_combo.currentText()}, 位数={digits_spin.value()}, 起始={start_spin.value()}, 按章节重置={'是' if reset_index_cb.isChecked() else '否'}")
    
    def _on_split_clicked(self):
        """拆分按钮点击：仅对所选章节执行长图拆分（不关闭对话框）"""
        selected_chapters = self.get_selected_chapters()
        if not selected_chapters:
            QMessageBox.warning(self, "提示", "请先选择要拆分的章节")
            return
        
        if self._is_splitting:
            self._log("⚠️ 拆分正在进行中，请稍候...")
            return
        
        # 确认对话框
        reply = QMessageBox.question(
            self, "确认拆分",
            f"确定要对 {len(selected_chapters)} 个章节执行长图拆分吗？\n\n"
            f"⚙️ 配置: 目标高度={self.split_target_height_spin.value()}px\n"
            f"⚠️ 此操作会删除超高原图，替换为拆分后的图片！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 使用异步拆分，完成后不关闭对话框
        self._start_split_async(selected_chapters, accept_after=False)
    
    def _start_split_async(self, chapters: List[str], accept_after: bool = False):
        """启动异步拆分任务
        
        Args:
            chapters: 要拆分的章节路径列表
            accept_after: 拆分完成后是否自动接受对话框
        """
        if self._is_splitting:
            self._log("⚠️ 拆分正在进行中，请稍候...")
            return
        
        self._is_splitting = True
        self._split_accept_after = accept_after
        
        # 清空上一次的撤回数据
        self._clear_undo_data()
        
        # 创建备份目录
        self._pending_undo_data = {}
        self._pending_chapters = chapters[:]
        self._backup_chapters_for_undo(chapters, "split")
        
        # 获取拆分参数
        skip_threshold = self.split_skip_threshold_spin.value()
        target_height = self.split_target_height_spin.value()
        buffer_range = self.split_buffer_range_spin.value()
        min_segment = self.split_min_segment_spin.value()
        
        # 获取命名模式
        from desktop_qt_ui.utils.long_image_splitter import NAMING_PRESETS
        naming_key = self.split_naming_combo.currentText()
        naming_pattern = NAMING_PRESETS.get(naming_key, "{index}_{source}")
        
        # 序号位数与起始序号
        index_digits = self.split_index_digits_spin.value()
        index_start = self.split_index_start_spin.value()
        
        # Reset index per chapter
        reset_index_per_chapter = self.split_reset_index_per_chapter_cb.isChecked()
        
        # 显示进度条，禁用按钮
        self.split_button.setEnabled(False)
        self.split_button.setText("拆分中...")
        self.stitch_button.setEnabled(False)
        self.ok_button.setEnabled(False)
        self.stitch_progress.setRange(0, len(chapters))
        self.stitch_progress.setValue(0)
        self.stitch_progress.setVisible(True)
        self.stitch_progress_label.setText(f"拆分进度: 0/{len(chapters)}")
        self.stitch_progress_label.setVisible(True)
        
        # 创建后台线程和工作器
        self._split_thread = QThread()
        self._split_worker = SplitWorker(chapters, skip_threshold, target_height, buffer_range, min_segment, naming_pattern, index_digits, index_start, reset_index_per_chapter)
        self._split_worker.moveToThread(self._split_thread)
        
        # 连接信号
        self._split_thread.started.connect(self._split_worker.run)
        self._split_worker.finished.connect(self._on_split_finished)
        self._split_worker.progress.connect(self._log)
        self._split_worker.chapter_progress.connect(self._on_split_chapter_progress)
        self._split_worker.error.connect(self._on_split_error)
        self._split_worker.finished.connect(self._split_thread.quit)
        self._split_worker.finished.connect(self._split_worker.deleteLater)
        self._split_thread.finished.connect(self._split_thread.deleteLater)
        
        # 启动拆分
        self._split_thread.start()
    
    def _on_split_chapter_progress(self, current: int, total: int):
        """拆分章节进度更新"""
        self.stitch_progress.setValue(current)
        self.stitch_progress_label.setText(f"拆分进度: {current}/{total}")
    
    def _on_split_finished(self, result_files: list):
        """拆分完成回调"""
        self._is_splitting = False
        
        # 记录生成的文件用于撤回
        if result_files and hasattr(self, '_pending_undo_data'):
            self._finalize_undo_data(result_files, "split")
        
        # 恢复UI状态
        self.split_button.setEnabled(True)
        self.split_button.setText("拆分")
        self.stitch_button.setEnabled(True)
        self.ok_button.setEnabled(True)
        self.stitch_progress.setVisible(False)
        self.stitch_progress_label.setVisible(False)
        
        # 刷新已选章节列表显示
        selected_chapters = self.get_selected_chapters()
        self._update_selected_chapters_list(selected_chapters)
        
        # 如果需要在拆分完成后接受对话框
        if self._split_accept_after:
            self._log(f"[长图拆分] ✓ 全部完成，导入翻译器队列")
            self.accept()
    
    def _on_split_error(self, error_msg: str):
        """拆分错误回调"""
        self._is_splitting = False
        
        # 恢复UI状态
        self.split_button.setEnabled(True)
        self.split_button.setText("拆分")
        self.stitch_button.setEnabled(True)
        self.ok_button.setEnabled(True)
        self.stitch_progress.setVisible(False)
        self.stitch_progress_label.setVisible(False)
        
        self._log(f"❌ {error_msg}", "ERROR")
        QMessageBox.critical(self, "拆分错误", f"拆分图片时出错:\n{error_msg}")
    
    def _stop_split_if_running(self):
        """如果拆分正在进行，停止它"""
        if self._is_splitting and self._split_worker:
            self._split_worker.stop()
            if self._split_thread and self._split_thread.isRunning():
                self._split_thread.quit()
                self._split_thread.wait(1000)  # 最多等待1秒
            self._is_splitting = False
    
    def get_import_result(self) -> Optional[List[str]]:
        """获取导入模式的结果文件列表"""
        if hasattr(self, 'import_mode') and self.import_mode:
            return getattr(self, 'stitched_result_files', None)
        return None
    
    # ==================== 撤回功能相关方法 ====================
    
    def _backup_chapters_for_undo(self, chapters: List[str], operation_type: str):
        """备份章节图片用于撤回
        
        Args:
            chapters: 章节路径列表
            operation_type: "split" 或 "stitch"
        """
        import shutil
        import tempfile
        
        self._pending_undo_data = {}
        image_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}
        
        for chapter_path in chapters:
            try:
                # 创建临时备份目录
                backup_dir = tempfile.mkdtemp(prefix=f"undo_{operation_type}_")
                backup_files = []
                
                # 备份所有图片文件
                for f in os.listdir(chapter_path):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in image_extensions:
                        src = os.path.join(chapter_path, f)
                        dst = os.path.join(backup_dir, f)
                        shutil.copy2(src, dst)
                        backup_files.append(f)
                
                if backup_files:
                    self._pending_undo_data[chapter_path] = {
                        "backup_dir": backup_dir,
                        "backup_files": backup_files,
                        "generated_files": []  # 待填充
                    }
                    self._log(f"[撤回] 已备份 {len(backup_files)} 张图片: {os.path.basename(chapter_path)}")
                else:
                    # 没有图片，删除空备份目录
                    shutil.rmtree(backup_dir, ignore_errors=True)
                    
            except Exception as e:
                self._log(f"[撤回] ⚠️ 备份失败 {os.path.basename(chapter_path)}: {e}")
    
    def _finalize_undo_data(self, generated_files: List[str], operation_type: str):
        """完成撤回数据记录
        
        Args:
            generated_files: 生成的文件路径列表
            operation_type: "split" 或 "stitch"
        """
        if not hasattr(self, '_pending_undo_data') or not self._pending_undo_data:
            return
        
        # 按章节分组生成的文件
        for file_path in generated_files:
            chapter_path = os.path.dirname(file_path)
            if chapter_path in self._pending_undo_data:
                self._pending_undo_data[chapter_path]["generated_files"].append(file_path)
        
        # 保存撤回数据
        self._undo_data = self._pending_undo_data
        self._undo_type = operation_type
        self._pending_undo_data = {}
        
        # 启用撤回按钮
        self.undo_button.setEnabled(True)
        self.undo_button.setToolTip(f"撤销最近的{"拆分" if operation_type == "split" else "拼接"}操作")
        self._log(f"[撤回] ✓ 已记录撤回数据，可点击“撤回”按钮恢复")
    
    def _clear_undo_data(self):
        """清除撤回数据并删除备份文件"""
        import shutil
        
        if self._undo_data:
            for chapter_data in self._undo_data.values():
                backup_dir = chapter_data.get("backup_dir")
                if backup_dir and os.path.isdir(backup_dir):
                    try:
                        shutil.rmtree(backup_dir)
                    except Exception:
                        pass
        
        self._undo_data = None
        self._undo_type = ""
        self.undo_button.setEnabled(False)
    
    def _on_undo_clicked(self):
        """撤回按钮点击"""
        if not self._undo_data:
            QMessageBox.information(self, "提示", "没有可撤回的操作")
            return
        
        operation_name = "拆分" if self._undo_type == "split" else "拼接"
        chapter_count = len(self._undo_data)
        
        reply = QMessageBox.question(
            self, "确认撤回",
            f"确定要撤销最近的{operation_name}操作吗？\n\n"
            f"涉及 {chapter_count} 个章节\n"
            f"• 删除{operation_name}后生成的文件\n"
            f"• 恢复原始图片",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        self._execute_undo()
    
    def _execute_undo(self):
        """执行撤回操作"""
        import shutil
        
        if not self._undo_data:
            return
        
        operation_name = "拆分" if self._undo_type == "split" else "拼接"
        success_count = 0
        
        for chapter_path, chapter_data in self._undo_data.items():
            try:
                backup_dir = chapter_data.get("backup_dir")
                backup_files = chapter_data.get("backup_files", [])
                generated_files = chapter_data.get("generated_files", [])
                
                # 1. 删除生成的文件
                for file_path in generated_files:
                    if os.path.exists(file_path):
                        try:
                            os.remove(file_path)
                        except Exception:
                            pass
                
                # 2. 恢复备份文件
                if backup_dir and os.path.isdir(backup_dir):
                    for f in backup_files:
                        src = os.path.join(backup_dir, f)
                        dst = os.path.join(chapter_path, f)
                        if os.path.exists(src):
                            shutil.copy2(src, dst)
                    
                    # 删除备份目录
                    shutil.rmtree(backup_dir, ignore_errors=True)
                
                success_count += 1
                self._log(f"[撤回] ✓ 已恢复: {os.path.basename(chapter_path)}")
                
            except Exception as e:
                self._log(f"[撤回] ⚠️ 恢复失败 {os.path.basename(chapter_path)}: {e}")
        
        # 清除撤回数据
        self._undo_data = None
        self._undo_type = ""
        self.undo_button.setEnabled(False)
        
        # 刷新已选章节列表
        selected_chapters = self.get_selected_chapters()
        self._update_selected_chapters_list(selected_chapters)
        
        self._log(f"[撤回] ✓ {operation_name}操作已撤销，恢复了 {success_count} 个章节")
        QMessageBox.information(self, "撤回完成", f"已成功撤销{operation_name}操作，恢复了 {success_count} 个章节")


def show_advanced_folder_dialog(parent=None, start_dir: str = "") -> Optional[List[str]]:
    """显示高级文件夹对话框并返回选中的章节路径"""
    # 读取上次路径
    settings = QSettings("MangaTranslator", "AdvancedFolder")
    if not start_dir:
        saved_dir = settings.value("last_dir", "")
        if saved_dir and os.path.isdir(saved_dir):
            start_dir = saved_dir
        else:
            start_dir = str(Path.home() / "Downloads")
    
    dialog = AdvancedFolderDialog(parent, start_dir)
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        selected = dialog.get_selected_chapters()
        if selected:
            return selected
    
    return None
