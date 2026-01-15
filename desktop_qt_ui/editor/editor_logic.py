import json
import os
from typing import List, Optional

from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QFileDialog

from editor.file_list_model import FileListModel, FileType, FileItem
from services import get_config_service, get_logger
from widgets.folder_dialog import select_folders


class EditorLogic(QObject):
    """
    Handles the business logic for the editor view, including file list management.
    """
    file_list_changed = pyqtSignal(list)
    file_list_with_tree_changed = pyqtSignal(list, dict)  # (files, folder_map)

    def __init__(self, controller, parent=None):
        super().__init__(parent)
        self.controller = controller
        self.config_service = get_config_service()
        self.logger = get_logger(__name__)
        
        # 使用新的文件列表模型
        self.file_model = FileListModel()
        
        # 保留树形结构支持
        self.folder_tree: dict = {}  # 保存文件夹树结构

    # --- File Management Methods ---

    @pyqtSlot()
    def open_and_add_files(self):
        """Opens a file dialog to add files to the editor's list."""
        last_dir = self.config_service.get_config().app.last_open_dir
        file_paths, _ = QFileDialog.getOpenFileNames(
            None, 
            "添加文件到编辑器", 
            last_dir, 
            "All Supported Files (*.png *.jpg *.jpeg *.bmp *.webp *.avif *.heic *.heif *.pdf *.epub *.cbz *.cbr *.zip);;"
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.avif *.heic *.heif);;"
            "PDF Files (*.pdf);;"
            "EPUB Files (*.epub);;"
            "Comic Book Archives (*.cbz *.cbr *.zip)"
        )
        if file_paths:
            self.add_files(file_paths)
            os.path.dirname(file_paths[0])
            # TODO: Find a way to save last_open_dir back to config service

    @pyqtSlot()
    def open_and_add_folder(self):
        """Opens a dialog to select folders (supports multiple selection) and adds all containing images to the list."""
        last_dir = self.config_service.get_config().app.last_open_dir

        # 使用自定义的现代化文件夹选择器
        folders = select_folders(
            parent=None,
            start_dir=last_dir,
            multi_select=True,
            config_service=self.config_service
        )

        if folders:
            # 扫描文件夹，添加所有图片文件路径
            for folder_path in folders:
                self.add_folder(folder_path)

    def add_files(self, files: List[str]):
        """添加文件到列表"""
        if not files:
            return
        
        # 使用新模型添加文件
        added_items = self.file_model.add_files(files)
        
        if added_items:
            # 检查是否是第一次添加文件
            is_first_add = len(self.file_model.files) == len(added_items)
            
            # 发射信号更新UI
            file_paths = [item.path for item in self.file_model.files]
            self.file_list_changed.emit(file_paths)
            
            # 如果是第一次添加文件，自动加载第一个
            if is_first_add and len(added_items) > 0:
                try:
                    self.load_image_into_editor(added_items[0].path)
                except Exception:
                    pass  # 静默失败，避免崩溃

    def add_folder(self, folder_path: str):
        """添加文件夹到列表"""
        if not folder_path or not os.path.isdir(folder_path):
            return
        
        # 检查是否是第一次添加文件
        is_first_add = len(self.file_model.files) == 0
        
        # 扫描文件夹中的所有图片
        image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
        files_to_add = []
        
        try:
            for root, dirs, files in os.walk(folder_path):
                # 跳过 manga_translator_work 目录
                if 'manga_translator_work' in root:
                    continue
                    
                for f in sorted(files):
                    if os.path.splitext(f)[1].lower() in image_extensions:
                        file_path = os.path.join(root, f)
                        files_to_add.append(file_path)
        except OSError as e:
            self.logger.error(f"扫描文件夹失败: {e}")
            return
        
        if files_to_add:
            # 添加文件
            added_items = self.file_model.add_files(files_to_add)
            
            # 发射信号更新UI
            file_paths = [item.path for item in self.file_model.files]
            self.file_list_changed.emit(file_paths)
            
            # 如果是第一次添加，自动加载第一个图片
            if is_first_add and added_items:
                try:
                    self.load_image_into_editor(added_items[0].path)
                except Exception:
                    pass  # 静默失败，避免崩溃

    @pyqtSlot(list)
    def add_files_from_paths(self, paths: List[str]):
        """
        从拖放的路径列表中添加文件和文件夹
        
        Args:
            paths: 拖放的文件或文件夹路径列表
        """
        files_to_add = []
        for path in paths:
            if os.path.isfile(path):
                # 验证是否是图片文件
                image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
                if os.path.splitext(path)[1].lower() in image_extensions:
                    files_to_add.append(path)
            elif os.path.isdir(path):
                # 添加文件夹中的所有图片
                self.add_folder(path)
        
        # 添加单独的文件
        if files_to_add:
            self.add_files(files_to_add)

    @pyqtSlot(str)
    def remove_file(self, file_path: str, emit_signal: bool = False):
        """
        移除文件或文件夹
        
        Args:
            file_path: 要移除的文件或文件夹路径
            emit_signal: 是否发射 file_list_changed 信号（默认 False，由视图自己处理）
        """
        norm_path = os.path.normpath(file_path)
        
        # 获取要删除的文件项，检查是否有关联的源文件或翻译文件
        file_item = self.file_model.get_file_item(file_path)
        paths_to_check = [norm_path]
        
        if file_item:
            # 如果是翻译后的文件，添加源文件路径
            if file_item.source_path:
                paths_to_check.append(os.path.normpath(file_item.source_path))
            # 如果是源文件，添加翻译后的文件路径
            if file_item.translated_path:
                paths_to_check.append(os.path.normpath(file_item.translated_path))
        
        # 检查是否是文件夹（在 folder_tree 中）
        if norm_path in self.folder_tree:
            # 移除文件夹下的所有文件
            files_to_remove = []
            for file_item in self.file_model.files:
                item_norm_path = os.path.normpath(file_item.path)
                try:
                    # 检查文件是否在该文件夹内
                    if item_norm_path.startswith(norm_path + os.sep) or item_norm_path == norm_path:
                        files_to_remove.append(file_item.path)
                except Exception:
                    pass
            
            # 批量移除文件
            for file_to_remove in files_to_remove:
                self.file_model.remove_file(file_to_remove)
                # 释放缓存
                if hasattr(self.controller, 'resource_manager'):
                    self.controller.resource_manager.release_image_from_cache(file_to_remove)
            
            # 从 folder_tree 中删除
            del self.folder_tree[norm_path]
            
            # 检查当前加载的图片是否在被删除的文件夹内
            current_image_path = self.controller.model.get_source_image_path()
            if current_image_path:
                norm_current = os.path.normpath(current_image_path)
                try:
                    if norm_current.startswith(norm_path + os.sep) or norm_current == norm_path:
                        self.controller.model.set_image(None)
                        self.controller._clear_editor_state(release_image_cache=True)
                except Exception:
                    pass
        else:
            # 移除单个文件
            removed = self.file_model.remove_file(file_path)
            
            if not removed:
                return
            
            # 检查当前加载的图片是否是被移除的文件（或其关联文件）
            current_image_path = self.controller.model.get_source_image_path()
            if current_image_path:
                norm_current = os.path.normpath(current_image_path)
                
                # 检查当前图片是否匹配要删除的文件或其关联文件
                if norm_current in paths_to_check:
                    self.controller.model.set_image(None)
                    self.controller._clear_editor_state(release_image_cache=True)
            
            # 从资源管理器的缓存中释放被移除的图片及其关联文件
            if hasattr(self.controller, 'resource_manager'):
                for path in paths_to_check:
                    self.controller.resource_manager.release_image_from_cache(path)
        
        # 检查是否还有文件，如果没有了就清空画布
        if len(self.file_model.files) == 0:
            self.controller.model.set_image(None)
            self.controller._clear_editor_state(release_image_cache=True)
            
            # 清空所有图片缓存
            if hasattr(self.controller, 'resource_manager'):
                self.controller.resource_manager.clear_image_cache()
        
        # 如果需要发射信号，更新UI
        if emit_signal:
            file_paths = [item.path for item in self.file_model.files]
            if self.folder_tree:
                self.file_list_with_tree_changed.emit(file_paths, self.folder_tree)
            else:
                self.file_list_changed.emit(file_paths)

    @pyqtSlot()
    def clear_list(self):
        """清空文件列表"""
        self.file_model.clear()
        self.folder_tree.clear()
        
        # 清空列表时发射空列表
        self.file_list_changed.emit([])
        
        # 先清空画布图片，这样后台任务会检测到图片为None而提前返回
        self.controller.model.set_image(None)
        # 然后清空编辑器状态（包括取消后台任务）
        self.controller._clear_editor_state(release_image_cache=True)
        
        # 清空所有图片缓存
        if hasattr(self.controller, 'resource_manager'):
            self.controller.resource_manager.clear_image_cache()

    # --- Image Loading Methods ---

    def load_file_lists(self, source_files: List[str], translated_files: List[str], folder_tree: dict = None, show_translated: bool = False):
        """
        从主窗口接收文件列表（用于翻译完成后进入编辑器）
        
        Args:
            source_files: 源文件列表
            translated_files: 翻译后的文件列表
            folder_tree: 文件夹树结构
            show_translated: 是否显示翻译后的文件列表
        """
        self.folder_tree = folder_tree if folder_tree else {}
        
        # 决定显示哪个文件列表
        files_to_show = translated_files if (show_translated and translated_files) else source_files
        
        # 清空文件模型
        self.file_model.clear()
        
        # 批量添加文件，避免一次性处理过多文件导致UI卡顿
        batch_size = 50  # 每批处理50个文件
        for i in range(0, len(files_to_show), batch_size):
            batch = files_to_show[i:i + batch_size]
            self.file_model.add_files(batch)
            
            # 处理事件，保持UI响应
            from PyQt6.QtWidgets import QApplication
            QApplication.processEvents()
        
        # 如果有folder_tree，使用树形结构显示
        if folder_tree:
            self.file_list_with_tree_changed.emit(files_to_show, folder_tree)
        else:
            # 否则使用平铺列表
            self.file_list_changed.emit(files_to_show)

    @pyqtSlot(str)
    def load_image_into_editor(self, file_path: str):
        """
        加载图片到编辑器（统一接口）
        
        根据文件类型自动处理：
        - 原图（有JSON）：加载原图和JSON进行编辑
        - 翻译后的图（有map）：只显示翻译后的图（查看模式）
        - 未翻译的图：提示进行翻译
        """
        # 获取文件项
        file_item = self.file_model.get_file_item(file_path)
        
        if not file_item:
            # 文件不在列表中，尝试识别
            self.file_model.add_files([file_path])
            file_item = self.file_model.get_file_item(file_path)
        
        if not file_item:
            self.logger.error(f"无法识别文件: {file_path}")
            return
        
        # 根据文件类型处理
        if file_item.file_type == FileType.SOURCE:
            # 原图：加载原图和JSON
            self.controller.load_image_and_regions(file_path)
        
        elif file_item.file_type == FileType.TRANSLATED:
            # 翻译后的图：只显示翻译后的图
            self.controller.load_image_and_regions(file_path)
        
        elif file_item.file_type == FileType.UNTRANSLATED:
            # 未翻译的图：提示进行翻译
            self.logger.warning(f"未翻译的图片: {file_path}")
            # 仍然加载图片，但不加载JSON
            self.controller.load_image_and_regions(file_path)

    def _find_file_pair(self, file_path: str) -> (str, Optional[str]):
        """Given a file path, find its source/translated pair using translation_map.json."""
        norm_path = os.path.normpath(file_path)

        # Case 1: The given file is a translated file (a key in a map)
        try:
            output_dir = os.path.dirname(norm_path)
            map_path = os.path.join(output_dir, 'translation_map.json')
            if os.path.exists(map_path):
                t_map = self.translation_map_cache.get(map_path)
                if t_map is None:
                    with open(map_path, 'r', encoding='utf-8') as f:
                        t_map = json.load(f)
                    self.translation_map_cache[map_path] = t_map
                
                if norm_path in t_map:
                    source = t_map[norm_path]
                    if os.path.exists(source):
                        return source, file_path
        except Exception:
            pass
        
        # Case 2: The given file is a source file (a value in a map)
        try:
            for trans_file in self.translated_files:
                if not trans_file: continue
                norm_trans = os.path.normpath(trans_file)
                output_dir = os.path.dirname(norm_trans)
                map_path = os.path.join(output_dir, 'translation_map.json')
                if os.path.exists(map_path):
                    t_map = self.translation_map_cache.get(map_path)
                    if t_map is None:
                        with open(map_path, 'r', encoding='utf-8') as f:
                            t_map = json.load(f)
                        self.translation_map_cache[map_path] = t_map

                    if t_map.get(norm_trans) == norm_path:
                        return file_path, trans_file
        except Exception:
            pass

        # Case 3: No pair found, it's a source file with no known translation.
        return file_path, None

    @pyqtSlot()
    def on_global_render_setting_changed(self):
        """Slot to handle changes in global render settings."""
        self.controller.handle_global_render_setting_change()