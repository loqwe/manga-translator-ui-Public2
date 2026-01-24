import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from PyQt6.QtCore import QSize, Qt, pyqtSignal, QObject, pyqtSlot, QSettings
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QVBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSplitter,
    QStyle,
    QTreeWidget,
    QTreeWidgetItem,
    QWidget,
)


# 全局线程池，用于异步加载缩略图
_thumbnail_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="thumbnail_loader")


def shutdown_thumbnail_executor():
    """关闭缩略图加载线程池"""
    global _thumbnail_executor
    if _thumbnail_executor:
        try:
            _thumbnail_executor.shutdown(wait=False)
        except Exception:
            pass


class ThumbnailSignals(QObject):
    """用于从工作线程发送信号到主线程"""
    thumbnail_loaded = pyqtSignal(str, QPixmap)  # file_path, pixmap


def natural_sort_key(path: str):
    """
    生成自然排序的键，支持数字排序
    例如: file1.jpg, file2.jpg, file10.jpg 会按 1, 2, 10 排序
    """
    filename = os.path.basename(path)
    parts = []
    for part in re.split(r'(\d+)', filename):
        if part.isdigit():
            parts.append(int(part))
        else:
            parts.append(part.lower())
    return parts


def _load_thumbnail_worker(file_path: str) -> tuple[str, Optional[QPixmap]]:
    """
    在工作线程中加载缩略图
    返回 (file_path, pixmap) 或 (file_path, None) 如果失败
    """
    try:
        img = Image.open(file_path)
        img.thumbnail((40, 40))
        
        # Convert PIL image to QPixmap
        if img.mode == 'RGB':
            q_img = QImage(img.tobytes(), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        elif img.mode == 'RGBA':
            q_img = QImage(img.tobytes(), img.width, img.height, img.width * 4, QImage.Format.Format_RGBA8888)
        else:  # Fallback for other modes like L, P, etc.
            img = img.convert('RGBA')
            q_img = QImage(img.tobytes(), img.width, img.height, img.width * 4, QImage.Format.Format_RGBA8888)

        pixmap = QPixmap.fromImage(q_img)
        return (file_path, pixmap)
    except Exception as e:
        print(f"Error loading thumbnail for {file_path}: {e}")
        return (file_path, None)


class FileItemWidget(QWidget):
    """自定义列表项，用于显示缩略图、文件名和移除按钮"""
    remove_requested = pyqtSignal(str)
    
    # 类级别的缩略图缓存
    _thumbnail_cache: Dict[str, QPixmap] = {}
    # 类级别的信号对象（所有实例共享）
    _signals = ThumbnailSignals()
    # 存储所有活动的实例，用于分发信号
    _active_instances: Dict[str, List['FileItemWidget']] = {}

    def __init__(self, file_path, is_folder=False, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.is_folder = is_folder
        self._thumbnail_loading = False

        # 注册实例
        if not is_folder and not os.path.isdir(file_path):
            if file_path not in FileItemWidget._active_instances:
                FileItemWidget._active_instances[file_path] = []
            FileItemWidget._active_instances[file_path].append(self)

        self.layout = QHBoxLayout(self)
        self.layout.setContentsMargins(5, 5, 5, 5)
        self.layout.setSpacing(10)

        # Thumbnail
        self.thumbnail_label = QLabel()
        self.thumbnail_label.setFixedSize(40, 40)
        self.thumbnail_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.thumbnail_label)

        if is_folder or os.path.isdir(self.file_path):
            style = QApplication.style()
            icon = style.standardIcon(QStyle.StandardPixmap.SP_DirIcon)
            self.thumbnail_label.setPixmap(icon.pixmap(QSize(40,40)))
        elif self._is_archive_file(self.file_path):
            # 压缩包/文档文件显示特殊图标
            style = QApplication.style()
            icon = style.standardIcon(QStyle.StandardPixmap.SP_FileIcon)
            self.thumbnail_label.setPixmap(icon.pixmap(QSize(40,40)))
        else:
            # 连接全局信号（只连接一次）
            if not hasattr(FileItemWidget, '_signals_connected'):
                FileItemWidget._signals.thumbnail_loaded.connect(FileItemWidget._dispatch_thumbnail)
                FileItemWidget._signals_connected = True
            self._load_thumbnail()

        # File Name
        display_name = os.path.basename(file_path)
        self.base_display_name = display_name  # 保存基础名称
        
        self.name_label = QLabel(display_name)
        self.name_label.setWordWrap(True)
        self.layout.addWidget(self.name_label, 1)  # Stretch factor

        # Remove Button
        self.remove_button = QPushButton("✕")
        self.remove_button.setFixedSize(20, 20)
        self.remove_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # 防止获取焦点
        self.remove_button.clicked.connect(self._emit_remove_request)
        self.layout.addWidget(self.remove_button)
    
    def __del__(self):
        """析构时从活动实例列表中移除"""
        if self.file_path in FileItemWidget._active_instances:
            try:
                FileItemWidget._active_instances[self.file_path].remove(self)
                if not FileItemWidget._active_instances[self.file_path]:
                    del FileItemWidget._active_instances[self.file_path]
            except (ValueError, KeyError):
                pass
    
    @classmethod
    def _dispatch_thumbnail(cls, file_path: str, pixmap: Optional[QPixmap]):
        """分发缩略图到所有相关实例"""
        if file_path in cls._active_instances:
            for instance in cls._active_instances[file_path]:
                instance._on_thumbnail_loaded(file_path, pixmap)

    def update_file_count(self, count: int):
        """更新文件夹显示的文件数量"""
        if self.is_folder:
            display_name = f"{self.base_display_name} ({count}个文件)"
            self.name_label.setText(display_name)

    def _load_thumbnail(self):
        """异步加载缩略图，使用缓存机制"""
        # 检查缓存
        if self.file_path in FileItemWidget._thumbnail_cache:
            self.thumbnail_label.setPixmap(FileItemWidget._thumbnail_cache[self.file_path])
            return
        
        # 显示加载中提示
        self.thumbnail_label.setText("...")
        self._thumbnail_loading = True
        
        # 提交到线程池异步加载
        future = _thumbnail_executor.submit(_load_thumbnail_worker, self.file_path)
        future.add_done_callback(self._on_thumbnail_future_done)
    
    def _on_thumbnail_future_done(self, future):
        """线程池任务完成回调"""
        try:
            file_path, pixmap = future.result()
            # 通过信号发送到主线程
            FileItemWidget._signals.thumbnail_loaded.emit(file_path, pixmap)
        except Exception as e:
            print(f"Error in thumbnail future callback: {e}")
    
    def _on_thumbnail_loaded(self, file_path: str, pixmap: Optional[QPixmap]):
        """在主线程中接收缩略图加载完成的信号"""
        self._thumbnail_loading = False
        
        # 检查 widget 是否还存在
        try:
            if pixmap:
                self.thumbnail_label.setPixmap(pixmap)
                # 缓存缩略图（只缓存一次）
                if file_path not in FileItemWidget._thumbnail_cache:
                    FileItemWidget._thumbnail_cache[file_path] = pixmap
            else:
                self.thumbnail_label.setText("ERR")
        except RuntimeError:
            # Widget 已被删除，忽略
            pass

    def _emit_remove_request(self):
        """发射删除请求信号"""
        self.remove_requested.emit(self.file_path)
    
    @staticmethod
    def _is_archive_file(file_path: str) -> bool:
        """检查文件是否是压缩包/文档格式"""
        archive_extensions = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}
        ext = os.path.splitext(file_path)[1].lower()
        return ext in archive_extensions

    def get_path(self):
        return self.file_path
    
    @classmethod
    def clear_thumbnail_cache(cls):
        """清空缩略图缓存"""
        cls._thumbnail_cache.clear()
    
    @classmethod
    def remove_from_cache(cls, file_path: str):
        """从缓存中移除指定文件的缩略图"""
        if file_path in cls._thumbnail_cache:
            del cls._thumbnail_cache[file_path]


class FileListView(QTreeWidget):
    """显示文件列表的自定义控件（支持文件夹分组）"""
    file_remove_requested = pyqtSignal(str)
    file_selected = pyqtSignal(str)
    files_dropped = pyqtSignal(list)  # 新增：拖放文件信号
    _folders_scanned = pyqtSignal(list)  # 内部信号：文件夹扫描完成

    def __init__(self, model, parent=None):
        super().__init__(parent)
        self.model = model
        
        # 导入i18n
        from services import get_i18n_manager
        self.i18n = get_i18n_manager()
        
        # 设置树形控件属性
        self.setHeaderHidden(True)  # 隐藏标题栏
        self.setIndentation(20)  # 设置缩进
        self.setAnimated(True)  # 启用展开/折叠动画
        
        # 启用拖放
        self.setAcceptDrops(True)
        self.setDragEnabled(False)  # 禁用拖出，只允许拖入
        
        # 存储文件夹到树节点的映射
        self.folder_nodes: Dict[str, QTreeWidgetItem] = {}
        
        # 连接选择信号
        self.itemSelectionChanged.connect(self._on_selection_changed)
        
        # 连接内部信号（确保在主线程中处理）
        self._folders_scanned.connect(self._on_folders_scanned)
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key
    
    def refresh_ui_texts(self):
        """刷新UI文本（用于语言切换）"""
        # 强制重绘以更新拖拽提示文本
        self.viewport().update()
        self.update()

    def paintEvent(self, event):
        """重写绘制事件，在列表为空时显示提示"""
        super().paintEvent(event)
        
        # 只在列表为空时显示提示
        if self.topLevelItemCount() == 0:
            from PyQt6.QtGui import QPainter, QColor, QFont
            from PyQt6.QtCore import Qt, QRect
            
            painter = QPainter(self.viewport())
            painter.setPen(QColor(150, 150, 150))  # 灰色
            
            # 设置字体
            font = QFont()
            font.setPointSize(10)
            painter.setFont(font)
            
            # 绘制提示文本
            rect = self.viewport().rect()
            text = self._t("Drag and drop files or folders here\nor click the buttons above to add")
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
            
            painter.end()

    def _on_selection_changed(self):
        """处理选择变化"""
        selected_items = self.selectedItems()
        if not selected_items:
            return
        
        tree_item = selected_items[0]
        file_path = tree_item.data(0, Qt.ItemDataRole.UserRole)
        
        # 只有当选中的是文件（不是文件夹节点）时才发出信号
        if file_path and not os.path.isdir(file_path):
            self.file_selected.emit(file_path)

    def add_files(self, file_paths: List[str]):
        """添加多个文件/文件夹到列表（异步处理大文件夹）"""
        folders_to_add = []
        files_to_add = []
        
        for path in file_paths:
            norm_path = os.path.normpath(path)
            if os.path.isdir(norm_path):
                folders_to_add.append(norm_path)
            else:
                files_to_add.append(norm_path)
        
        # 立即添加单个文件（不会阻塞）
        for file_path in files_to_add:
            self._add_single_file(file_path)
        
        # 异步添加文件夹（避免阻塞UI）
        if folders_to_add:
            # 显示加载提示
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            
            # 在后台线程中扫描文件夹结构
            def scan_folder_structure():
                result = []
                for folder_path in folders_to_add:
                    if folder_path in self.folder_nodes:
                        continue  # 文件夹已存在
                    
                    # 在后台线程中扫描文件夹结构
                    folder_data = self._scan_folder_structure(folder_path)
                    result.append((folder_path, folder_data))
                return result
            
            future = _thumbnail_executor.submit(scan_folder_structure)
            future.add_done_callback(lambda f: self._folders_scanned.emit(f.result()))
    
    def _scan_folder_structure(self, folder_path: str):
        """在后台线程中扫描文件夹结构（不创建UI元素）"""
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
            archive_extensions = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}
            all_extensions = image_extensions | archive_extensions
            structure = {'subdirs': [], 'files': [], 'subdir_data': {}}
            
            items = os.listdir(folder_path)
            for item in items:
                if item == 'manga_translator_work':
                    continue
                    
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    structure['subdirs'].append(item_path)
                elif os.path.splitext(item)[1].lower() in all_extensions:
                    structure['files'].append(item_path)
            
            # 排序
            structure['subdirs'].sort(key=natural_sort_key)
            structure['files'].sort(key=natural_sort_key)
            
            # 递归扫描子文件夹
            for subdir in structure['subdirs']:
                structure['subdir_data'][subdir] = self._scan_folder_structure(subdir)
            
            return structure
        except Exception as e:
            print(f"Error scanning folder {folder_path}: {e}")
            return {'subdirs': [], 'files': [], 'subdir_data': {}}
    
    @pyqtSlot(list)
    def _on_folders_scanned(self, folder_data_list):
        """文件夹扫描完成后的回调（在主线程中创建UI元素）"""
        try:
            for folder_path, folder_data in folder_data_list:
                self._add_folder_tree_from_data(folder_path, folder_data)
        finally:
            # 恢复光标
            QApplication.restoreOverrideCursor()
    
    def _add_folder_tree_from_data(self, folder_path: str, folder_data: dict):
        """从扫描的数据创建文件夹树（在主线程中执行）"""
        if folder_path in self.folder_nodes:
            return
        
        # 创建顶层文件夹节点
        folder_item = QTreeWidgetItem(self)
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(folder_item)
        self.setItemWidget(folder_item, 0, folder_widget)
        self.folder_nodes[folder_path] = folder_item
        
        # 递归添加子文件夹和文件
        self._populate_folder_tree_from_data(folder_item, folder_path, folder_data)
        
        # 更新文件数量
        file_count = self._count_files_in_tree(folder_item)
        folder_widget.update_file_count(file_count)
    
    def _populate_folder_tree_from_data(self, parent_item: QTreeWidgetItem, folder_path: str, folder_data: dict):
        """从扫描的数据填充文件夹树（在主线程中执行）"""
        # 添加子文件夹
        for subdir in folder_data.get('subdirs', []):
            subdir_item = QTreeWidgetItem(parent_item)
            subdir_item.setData(0, Qt.ItemDataRole.UserRole, subdir)
            
            subdir_widget = FileItemWidget(subdir, is_folder=True)
            subdir_widget.remove_requested.connect(self.file_remove_requested.emit)
            
            parent_item.addChild(subdir_item)
            self.setItemWidget(subdir_item, 0, subdir_widget)
            self.folder_nodes[subdir] = subdir_item
            
            # 递归处理子文件夹
            subdir_data = folder_data.get('subdir_data', {}).get(subdir)
            if subdir_data:
                self._populate_folder_tree_from_data(subdir_item, subdir, subdir_data)
            
            # 更新子文件夹的文件数量
            file_count = self._count_files_in_tree(subdir_item)
            subdir_widget.update_file_count(file_count)
        
        # 添加文件
        for file_path in folder_data.get('files', []):
            file_item = QTreeWidgetItem(parent_item)
            file_item.setData(0, Qt.ItemDataRole.UserRole, file_path)
            
            file_widget = FileItemWidget(file_path, is_folder=False)
            file_widget.remove_requested.connect(self.file_remove_requested.emit)
            
            parent_item.addChild(file_item)
            self.setItemWidget(file_item, 0, file_widget)
        
        # 触发重绘
        self.viewport().update()
    
    def add_files_from_tree(self, folder_tree: dict):
        """
        从完整的树结构添加文件
        folder_tree: {folder_path: {'files': [...], 'subfolders': [...]}}
        """
        if not folder_tree:
            return
        
        # 找到所有根文件夹（没有父文件夹在tree中的文件夹）
        all_folders = set(folder_tree.keys())
        root_folders = []
        
        for folder in all_folders:
            is_root = True
            for other_folder in all_folders:
                if folder != other_folder and folder.startswith(other_folder + os.sep):
                    is_root = False
                    break
            if is_root:
                root_folders.append(folder)
        
        # 按自然排序
        root_folders.sort(key=natural_sort_key)
        
        # 为每个根文件夹创建树
        for root_folder in root_folders:
            self._create_folder_node_from_tree(root_folder, folder_tree, None)
    
    def _create_folder_node_from_tree(self, folder_path: str, folder_tree: dict, parent_item: QTreeWidgetItem = None):
        """从树结构递归创建文件夹节点"""
        if folder_path not in folder_tree:
            return
        
        # 创建文件夹节点
        if parent_item is None:
            folder_item = QTreeWidgetItem(self)
            self.addTopLevelItem(folder_item)
        else:
            folder_item = QTreeWidgetItem(parent_item)
            parent_item.addChild(folder_item)
        
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        # 创建文件夹控件
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        self.setItemWidget(folder_item, 0, folder_widget)
        
        # 保存文件夹节点
        self.folder_nodes[folder_path] = folder_item
        
        # 添加该文件夹直接包含的文件
        folder_data = folder_tree[folder_path]
        for file_path in folder_data.get('files', []):
            self._add_file_to_folder(file_path, folder_item)
        
        # 递归添加子文件夹
        subfolders = folder_data.get('subfolders', [])
        subfolders.sort(key=natural_sort_key)
        for subfolder in subfolders:
            self._create_folder_node_from_tree(subfolder, folder_tree, folder_item)
        
        # 更新文件数量
        file_count = self._count_files_in_tree(folder_item)
        folder_widget.update_file_count(file_count)
    
    def add_files_with_tree(self, file_paths: List[str], folder_map: dict = None):
        """
        添加文件列表，并根据folder_map创建树形结构
        只添加file_paths中的文件，不扫描整个文件夹
        
        Args:
            file_paths: 要添加的文件列表
            folder_map: 文件到文件夹的映射 {file_path: folder_path}
        """
        if not folder_map:
            # 如果没有folder_map，使用普通的add_files
            self.add_files(file_paths)
            return
        
        # 按文件夹分组
        folder_groups = {}  # 文件按其直接父文件夹分组
        single_files = []
        all_folders = set()
        
        # 首先收集所有文件的直接父文件夹
        for file_path in file_paths:
            norm_file_path = os.path.normpath(file_path)
            mapped_folder = folder_map.get(file_path)
            
            if mapped_folder:
                norm_mapped_folder = os.path.normpath(mapped_folder)
                
                # 文件应该被添加到其直接父文件夹
                file_dir = os.path.dirname(norm_file_path)
                if file_dir not in folder_groups:
                    folder_groups[file_dir] = []
                folder_groups[file_dir].append(norm_file_path)
                
                # 添加直接父文件夹
                all_folders.add(file_dir)
                
                # 添加 folder_map 中映射的文件夹
                all_folders.add(norm_mapped_folder)
            else:
                single_files.append(norm_file_path)
        
        # 为每个文件夹添加从其到顶层文件夹的所有中间文件夹
        if all_folders:
            expanded_folders = set(all_folders)  # 先包含所有已知文件夹
            
            # 为每个文件夹添加所有父文件夹，直到到达顶层文件夹或根目录
            for folder in list(all_folders):
                current = folder
                while True:
                    parent = os.path.dirname(current)
                    # 如果父文件夹为空或与当前相同（到达根目录），停止
                    if not parent or parent == current:
                        break
                    expanded_folders.add(parent)
                    current = parent
                    # 如果父文件夹已经在原始 all_folders 中，说明这是顶层文件夹，停止
                    if parent in all_folders:
                        break
            
            all_folders = expanded_folders
        
        # 构建文件夹层级关系
        folder_hierarchy = self._build_folder_hierarchy(all_folders)
        
        # 按层级创建文件夹树
        self._create_folder_tree_with_files(folder_hierarchy, folder_groups)
        
        # 添加单独的文件
        for file_path in single_files:
            self._add_single_file(file_path)
    
    def _build_folder_hierarchy(self, folders: set) -> dict:
        """
        构建文件夹的层级关系
        返回: {parent_folder: [child_folders]}
        """
        hierarchy = {}
        root_folders = []
        
        for folder in folders:
            # 查找父文件夹
            parent = None
            for other_folder in folders:
                if folder != other_folder and folder.startswith(other_folder + os.sep):
                    # folder 是 other_folder 的子文件夹
                    if parent is None or len(other_folder) > len(parent):
                        parent = other_folder
            
            if parent:
                if parent not in hierarchy:
                    hierarchy[parent] = []
                hierarchy[parent].append(folder)
            else:
                root_folders.append(folder)
        
        return {'__root__': root_folders, **hierarchy}
    
    def _create_folder_tree_with_files(self, hierarchy: dict, folder_groups: dict, parent_item=None, parent_folder=None):
        """
        递归创建文件夹树形结构
        使用自然排序（数字排序：1, 2, 10 而不是 1, 10, 2）
        """
        folders_to_process = hierarchy.get(parent_folder or '__root__', [])
        
        # 使用自然排序
        for folder_path in sorted(folders_to_process, key=natural_sort_key):
            # 创建文件夹节点
            if parent_item is None:
                # 顶层文件夹
                folder_item = QTreeWidgetItem(self)
                self.addTopLevelItem(folder_item)
            else:
                # 子文件夹
                folder_item = QTreeWidgetItem(parent_item)
                parent_item.addChild(folder_item)
            
            folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
            
            # 创建文件夹控件
            folder_widget = FileItemWidget(folder_path, is_folder=True)
            folder_widget.remove_requested.connect(self.file_remove_requested.emit)
            self.setItemWidget(folder_item, 0, folder_widget)
            
            # 保存文件夹节点
            self.folder_nodes[folder_path] = folder_item
            
            # 添加该文件夹中的文件
            if folder_path in folder_groups:
                for file_path in folder_groups[folder_path]:
                    self._add_file_to_folder(file_path, folder_item)
            
            # 递归处理子文件夹
            if folder_path in hierarchy:
                self._create_folder_tree_with_files(hierarchy, folder_groups, folder_item, folder_path)
            
            # 更新文件数量（包括子文件夹中的文件）
            file_count = len(folder_groups.get(folder_path, []))
            # 递归统计子文件夹中的文件数量
            for child_folder in hierarchy.get(folder_path, []):
                file_count += self._count_files_in_hierarchy(child_folder, hierarchy, folder_groups)
            folder_widget.update_file_count(file_count)
    
    def _count_files_in_hierarchy(self, folder_path: str, hierarchy: dict, folder_groups: dict) -> int:
        """递归统计文件夹及其子文件夹中的文件数量"""
        count = len(folder_groups.get(folder_path, []))
        for child_folder in hierarchy.get(folder_path, []):
            count += self._count_files_in_hierarchy(child_folder, hierarchy, folder_groups)
        return count
    
    def _add_folder_with_files(self, folder_path: str, file_list: List[str]):
        """
        添加文件夹及其指定的文件（不扫描整个文件夹）
        
        Args:
            folder_path: 文件夹路径
            file_list: 要添加的文件列表
        """
        if folder_path in self.folder_nodes:
            # 文件夹已存在，只添加新文件
            folder_item = self.folder_nodes[folder_path]
            for file_path in file_list:
                self._add_file_to_folder(file_path, folder_item)
            # 更新文件数量
            folder_widget = self.itemWidget(folder_item, 0)
            if isinstance(folder_widget, FileItemWidget):
                folder_widget.update_file_count(len(file_list))
            return
        
        # 创建顶层文件夹节点
        folder_item = QTreeWidgetItem(self)
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        # 创建文件夹项的自定义控件
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(folder_item)
        self.setItemWidget(folder_item, 0, folder_widget)
        
        # 保存文件夹节点
        self.folder_nodes[folder_path] = folder_item
        
        # 添加文件到文件夹
        for file_path in file_list:
            self._add_file_to_folder(file_path, folder_item)
        
        # 更新文件数量显示
        folder_widget.update_file_count(len(file_list))
    
    def _add_folder_tree(self, folder_path: str):
        """添加文件夹及其完整的树形结构"""
        if folder_path in self.folder_nodes:
            return  # 文件夹已存在
        
        # 创建顶层文件夹节点
        folder_item = QTreeWidgetItem(self)
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        # 创建文件夹项的自定义控件
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(folder_item)
        self.setItemWidget(folder_item, 0, folder_widget)
        
        # 保存文件夹节点
        self.folder_nodes[folder_path] = folder_item
        
        # 递归添加子文件夹和文件
        self._populate_folder_tree(folder_item, folder_path)
        
        # 更新文件数量显示
        file_count = self._count_files_recursive(folder_path)
        folder_widget.update_file_count(file_count)
    
    def _count_files_recursive(self, folder_path: str) -> int:
        """递归统计文件夹中的图片文件数量"""
        if not os.path.isdir(folder_path):
            return 0
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
            archive_extensions = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}
            all_extensions = image_extensions | archive_extensions
            count = 0
            for root, dirs, files in os.walk(folder_path):
                # 忽略 manga_translator_work 目录
                if 'manga_translator_work' in dirs:
                    dirs.remove('manga_translator_work')
                    
                for filename in files:
                    if os.path.splitext(filename)[1].lower() in all_extensions:
                        count += 1
            return count
        except:
            return 0
    
    def _populate_folder_tree(self, parent_item: QTreeWidgetItem, folder_path: str):
        """递归填充文件夹树形结构"""
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
            archive_extensions = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}
            all_extensions = image_extensions | archive_extensions
            
            # 获取当前文件夹的直接子项
            items = os.listdir(folder_path)
            
            # 分离文件夹和文件
            subdirs = []
            files = []
            
            for item in items:
                # 忽略 manga_translator_work 目录
                if item == 'manga_translator_work':
                    continue
                    
                item_path = os.path.join(folder_path, item)
                if os.path.isdir(item_path):
                    subdirs.append(item_path)
                elif os.path.splitext(item)[1].lower() in all_extensions:
                    files.append(item_path)
            
            # 先添加子文件夹
            for subdir in sorted(subdirs, key=natural_sort_key):
                subdir_item = QTreeWidgetItem(parent_item)
                subdir_item.setData(0, Qt.ItemDataRole.UserRole, subdir)
                
                subdir_widget = FileItemWidget(subdir, is_folder=True)
                subdir_widget.remove_requested.connect(self.file_remove_requested.emit)
                
                parent_item.addChild(subdir_item)
                self.setItemWidget(subdir_item, 0, subdir_widget)
                
                # 保存子文件夹节点
                self.folder_nodes[subdir] = subdir_item
                
                # 递归处理子文件夹
                self._populate_folder_tree(subdir_item, subdir)
                
                # 更新子文件夹的文件数量显示
                file_count = self._count_files_recursive(subdir)
                subdir_widget.update_file_count(file_count)
            
            # 再添加文件
            for file_path in sorted(files, key=natural_sort_key):
                file_item = QTreeWidgetItem(parent_item)
                file_item.setData(0, Qt.ItemDataRole.UserRole, file_path)
                
                file_widget = FileItemWidget(file_path, is_folder=False)
                file_widget.remove_requested.connect(self.file_remove_requested.emit)
                
                parent_item.addChild(file_item)
                self.setItemWidget(file_item, 0, file_widget)
                
        except Exception as e:
            print(f"Error populating folder tree for {folder_path}: {e}")
        
        # 触发重绘以隐藏占位提示
        self.viewport().update()

    def _add_folder(self, folder_path: str):
        """添加文件夹及其包含的所有图片文件"""
        if folder_path in self.folder_nodes:
            return  # 文件夹已存在
        
        # 创建文件夹节点
        folder_item = QTreeWidgetItem(self)
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        # 创建文件夹项的自定义控件
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(folder_item)
        self.setItemWidget(folder_item, 0, folder_widget)
        
        # 保存文件夹节点
        self.folder_nodes[folder_path] = folder_item
        
        # 添加文件夹中的文件
        try:
            image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.avif'}
            archive_extensions = {'.pdf', '.epub', '.cbz', '.cbr', '.zip'}
            all_extensions = image_extensions | archive_extensions
            files = [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if os.path.splitext(f)[1].lower() in all_extensions
            ]
            
            for file_path in sorted(files, key=natural_sort_key):
                self._add_file_to_folder(file_path, folder_item)
            
            # 更新文件夹显示的文件数
            self._update_folder_count(folder_item)
        except Exception as e:
            print(f"Error loading files from folder {folder_path}: {e}")
    
    def _add_folder_group(self, folder_path: str, files: List[str]):
        """添加文件夹分组（使用提供的文件列表）"""
        if folder_path in self.folder_nodes:
            # 文件夹已存在，添加新文件
            folder_item = self.folder_nodes[folder_path]
            existing_files = set()
            for i in range(folder_item.childCount()):
                child = folder_item.child(i)
                existing_files.add(child.data(0, Qt.ItemDataRole.UserRole))
            
            for file_path in files:
                if file_path not in existing_files:
                    self._add_file_to_folder(file_path, folder_item)
            
            # 更新文件夹显示的文件数
            self._update_folder_count(folder_item)
            return
        
        # 创建文件夹节点
        folder_item = QTreeWidgetItem(self)
        folder_item.setData(0, Qt.ItemDataRole.UserRole, folder_path)
        
        # 创建文件夹项的自定义控件
        folder_widget = FileItemWidget(folder_path, is_folder=True)
        folder_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(folder_item)
        self.setItemWidget(folder_item, 0, folder_widget)
        
        # 保存文件夹节点
        self.folder_nodes[folder_path] = folder_item
        
        # 添加文件列表
        for file_path in sorted(files, key=natural_sort_key):
            self._add_file_to_folder(file_path, folder_item)
        
        # 更新文件夹显示的文件数
        self._update_folder_count(folder_item)

    def _add_file_to_folder(self, file_path: str, parent_item: QTreeWidgetItem):
        """将文件添加到文件夹节点下"""
        norm_path = os.path.normpath(file_path)
        
        file_item = QTreeWidgetItem(parent_item)
        file_item.setData(0, Qt.ItemDataRole.UserRole, norm_path)
        
        file_widget = FileItemWidget(norm_path, is_folder=False)
        file_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        parent_item.addChild(file_item)
        self.setItemWidget(file_item, 0, file_widget)

    def _add_single_file(self, file_path: str):
        """添加单个文件（不属于任何文件夹）"""
        norm_path = os.path.normpath(file_path)
        
        # 检查文件是否已存在于顶层项
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            item_path = item.data(0, Qt.ItemDataRole.UserRole)
            if item_path == norm_path:
                return  # 文件已存在于顶层
            
            # 检查文件是否已存在于文件夹节点中（递归搜索）
            if self._file_exists_in_tree(item, norm_path):
                return  # 文件已存在于某个文件夹中
        
        file_item = QTreeWidgetItem(self)
        file_item.setData(0, Qt.ItemDataRole.UserRole, norm_path)
        
        file_widget = FileItemWidget(file_path, is_folder=False)
        file_widget.remove_requested.connect(self.file_remove_requested.emit)
        
        self.addTopLevelItem(file_item)
        self.setItemWidget(file_item, 0, file_widget)
    
    def _file_exists_in_tree(self, item: QTreeWidgetItem, file_path: str) -> bool:
        """递归检查文件是否已存在于树中"""
        norm_path = os.path.normpath(file_path)
        for i in range(item.childCount()):
            child = item.child(i)
            child_path = child.data(0, Qt.ItemDataRole.UserRole)
            # 规范化后比较
            if child_path and os.path.normpath(child_path) == norm_path:
                return True
            # 递归检查子项
            if self._file_exists_in_tree(child, norm_path):
                return True
        return False

    def remove_file(self, file_path: str):
        """移除指定文件或文件夹"""
        norm_path = os.path.normpath(file_path)
        
        # 临时断开选择信号，避免删除时触发选择事件
        try:
            self.itemSelectionChanged.disconnect(self._on_selection_changed)
        except:
            pass
        
        try:
            # 递归查找并移除项
            def find_and_remove_item(parent_item: Optional[QTreeWidgetItem] = None) -> tuple[bool, Optional[QTreeWidgetItem]]:
                if parent_item is None:
                    # 搜索顶层项
                    for i in range(self.topLevelItemCount()):
                        item = self.topLevelItem(i)
                        item_path = item.data(0, Qt.ItemDataRole.UserRole)
                        
                        if item_path == norm_path:
                            # 找到了，删除这个顶层项
                            self.takeTopLevelItem(i)
                            # 如果是文件夹，从folder_nodes中移除
                            if norm_path in self.folder_nodes:
                                del self.folder_nodes[norm_path]
                            # 递归删除所有子文件夹的引用
                            self._remove_folder_nodes_recursive(item)
                            return True, None
                        
                        # 递归搜索子项
                        result, parent = find_and_remove_item(item)
                        if result:
                            return True, parent
                    
                    return False, None
                else:
                    # 搜索子项
                    for i in range(parent_item.childCount()):
                        child = parent_item.child(i)
                        child_path = child.data(0, Qt.ItemDataRole.UserRole)
                        
                        if child_path == norm_path:
                            # 找到了，删除这个子项
                            parent_item.removeChild(child)
                            # 如果是文件夹，从folder_nodes中移除
                            if norm_path in self.folder_nodes:
                                del self.folder_nodes[norm_path]
                            # 递归删除所有子文件夹的引用
                            self._remove_folder_nodes_recursive(child)
                            # 递归向上更新所有父文件夹的文件数量
                            self._update_all_parent_counts(parent_item)
                            return True, parent_item
                        
                        # 递归搜索更深层的子项
                        result, parent = find_and_remove_item(child)
                        if result:
                            return True, parent
                    
                    return False, None
            
            find_and_remove_item()
            
            # 删除后清除选择状态，避免自动触发加载
            self.clearSelection()
            
        finally:
            # 重新连接选择信号
            try:
                self.itemSelectionChanged.connect(self._on_selection_changed)
            except:
                pass
    
    def _remove_folder_nodes_recursive(self, item: QTreeWidgetItem):
        """递归移除文件夹节点的所有子文件夹引用"""
        for i in range(item.childCount()):
            child = item.child(i)
            child_path = child.data(0, Qt.ItemDataRole.UserRole)
            if child_path in self.folder_nodes:
                del self.folder_nodes[child_path]
            # 递归处理子项
            self._remove_folder_nodes_recursive(child)

    def _update_folder_count(self, folder_item: QTreeWidgetItem) -> bool:
        """
        更新文件夹显示的文件数量（递归统计）
        返回: True 如果文件夹被删除，False 否则
        """
        if folder_item:
            widget = self.itemWidget(folder_item, 0)
            if isinstance(widget, FileItemWidget) and widget.is_folder:
                # 递归统计所有文件数量
                count = self._count_files_in_tree(folder_item)
                folder_path = folder_item.data(0, Qt.ItemDataRole.UserRole)
                
                # 如果文件夹为空（计数为0），删除该文件夹节点
                if count == 0:
                    # 从 folder_nodes 中移除
                    if folder_path in self.folder_nodes:
                        del self.folder_nodes[folder_path]
                    
                    # 从树中移除
                    parent = folder_item.parent()
                    if parent:
                        parent.removeChild(folder_item)
                    else:
                        # 顶层项
                        index = self.indexOfTopLevelItem(folder_item)
                        if index >= 0:
                            self.takeTopLevelItem(index)
                    return True
                else:
                    widget.update_file_count(count)
        return False
    
    def _update_all_parent_counts(self, item: QTreeWidgetItem):
        """递归向上更新所有父文件夹的文件数量，如果文件夹为空则删除"""
        current = item
        while current:
            parent = current.parent()  # 先保存父节点，因为 current 可能被删除
            was_deleted = self._update_folder_count(current)
            if was_deleted:
                # 如果当前文件夹被删除了，继续检查父文件夹
                current = parent
            else:
                # 如果没被删除，继续向上更新
                current = parent
    
    def _count_files_in_tree(self, tree_item: QTreeWidgetItem) -> int:
        """递归统计树节点中的文件数量"""
        count = 0
        for i in range(tree_item.childCount()):
            child = tree_item.child(i)
            child_path = child.data(0, Qt.ItemDataRole.UserRole)
            if child_path and os.path.isfile(child_path):
                count += 1
            elif child_path and os.path.isdir(child_path):
                # 递归统计子文件夹
                count += self._count_files_in_tree(child)
        return count

    def clear(self, clear_cache: bool = False):
        """
        清空所有项
        
        Args:
            clear_cache: 是否同时清空缩略图缓存（默认 False，保留缓存以便重用）
        """
        super().clear()
        self.folder_nodes.clear()
        
        if clear_cache:
            FileItemWidget.clear_thumbnail_cache()
        
        # 触发重绘以显示占位提示
        self.viewport().update()

    # 拖放事件处理
    def dragEnterEvent(self, event):
        """拖入事件：检查是否包含文件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        """拖动移动事件"""
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        """放下事件：处理拖入的文件和文件夹"""
        if event.mimeData().hasUrls():
            paths = []
            for url in event.mimeData().urls():
                path = url.toLocalFile()
                if path:
                    paths.append(path)
            
            if paths:
                # 发射信号，让业务逻辑层处理
                self.files_dropped.emit(paths)
            
            event.acceptProposedAction()
        else:
            event.ignore()


class _FailedRecordItemWidget(QWidget):
    """失败记录行组件：按钮默认隐藏，悬停时显示。"""

    def __init__(
        self,
        label_text: str,
        on_double_click,
        on_retry,
        on_delete,
        tooltip_retry: str,
        tooltip_delete: str,
        parent=None,
    ):
        super().__init__(parent)

        self._on_double_click = on_double_click
        self._on_retry = on_retry
        self._on_delete = on_delete

        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 2, 5, 2)
        layout.setSpacing(5)

        # 记录文本
        self.label = QLabel(label_text)
        self.label.setStyleSheet("QLabel { color: #666; }")
        # 让父Widget接收鼠标事件（用于悬停显示按钮）
        self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.label, 1)

        # 重试按钮
        self.retry_btn = QPushButton("↻")
        self.retry_btn.setFixedSize(20, 20)
        self.retry_btn.setToolTip(tooltip_retry)
        self.retry_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: #999;
                font-size: 14px;
            }
            QPushButton:hover {
                color: #4CAF50;
            }
        """)
        self.retry_btn.clicked.connect(self._on_retry)
        layout.addWidget(self.retry_btn)

        # 删除按钮
        self.delete_btn = QPushButton("✕")
        self.delete_btn.setFixedSize(20, 20)
        self.delete_btn.setToolTip(tooltip_delete)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                border: none;
                color: #999;
                font-size: 12px;
            }
            QPushButton:hover {
                color: #f00;
            }
        """)
        self.delete_btn.clicked.connect(self._on_delete)
        layout.addWidget(self.delete_btn)

        # 默认隐藏按钮
        self._set_buttons_visible(False)
        self.setMouseTracking(True)

    def _set_buttons_visible(self, visible: bool):
        self.retry_btn.setVisible(visible)
        self.delete_btn.setVisible(visible)

    def enterEvent(self, event):
        self._set_buttons_visible(True)
        return super().enterEvent(event)

    def leaveEvent(self, event):
        self._set_buttons_visible(False)
        return super().leaveEvent(event)

    def mouseDoubleClickEvent(self, event):
        try:
            if callable(self._on_double_click):
                self._on_double_click()
        finally:
            return super().mouseDoubleClickEvent(event)


class FailedRecordsList(QWidget):
    """失败记录列表组件 - 显示翻译失败的图片记录，支持双击重新导入"""
    import_requested = pyqtSignal(dict)  # 发射导入请求信号，携带失败记录数据
    retry_requested = pyqtSignal(dict)   # 发射重试请求信号，直接进入延迟重试流程
    
    MAX_RECORDS = 10  # 最大记录数
    SETTINGS_KEY = "failed_records"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # 导入i18n
        from services import get_i18n_manager
        self.i18n = get_i18n_manager()
        
        self.records: List[dict] = []  # 存储失败记录
        
        self._init_ui()
        self._load_records()
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key
    
    def _init_ui(self):
        """初始化UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 0)
        layout.setSpacing(2)
        
        # 标题栏
        header = QWidget()
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(5, 0, 5, 0)
        header_layout.setSpacing(5)
        
        title_label = QLabel("📋 " + self._t("Failed Records"))
        header_layout.addWidget(title_label)
        header_layout.addStretch()
        
        # 清空按钮
        self.clear_btn = QPushButton(self._t("Clear"))
        self.clear_btn.setFixedSize(50, 20)
        self.clear_btn.clicked.connect(self._clear_all_records)
        header_layout.addWidget(self.clear_btn)
        
        layout.addWidget(header)
        
        # 记录列表
        self.record_list = QListWidget()
        self.record_list.setFixedHeight(100)  # 固定高度
        self.record_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.record_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.record_list.setToolTip(self._t("Double-click to re-import failed images"))
        layout.addWidget(self.record_list)
    
    def _load_records(self):
        """从QSettings加载失败记录"""
        settings = QSettings("MangaTranslator", "UI")
        records_json = settings.value(self.SETTINGS_KEY, "[]")
        try:
            self.records = json.loads(records_json)
            # 验证并过滤无效记录
            self.records = [r for r in self.records if self._validate_record(r)]
            self._refresh_list()
        except json.JSONDecodeError:
            self.records = []
    
    def _save_records(self):
        """保存失败记录到QSettings"""
        settings = QSettings("MangaTranslator", "UI")
        settings.setValue(self.SETTINGS_KEY, json.dumps(self.records, ensure_ascii=False))
    
    def _validate_record(self, record: dict) -> bool:
        """验证记录格式是否正确"""
        required_keys = ["time", "folders", "total_count", "label"]
        return all(key in record for key in required_keys)
    
    def add_record(self, failed_files: Dict[str, List[str]]):
        """
        添加失败记录
        
        Args:
            failed_files: {folder_path: [failed_file_names, ...]}
        """
        if not failed_files:
            return
        
        # 过滤空文件夹
        failed_files = {k: v for k, v in failed_files.items() if v and len(v) > 0}
        if not failed_files:
            return
        
        # 计算总数
        total_count = sum(len(files) for files in failed_files.values())
        
        # 生成标签：提取公共前缀和数字后缀
        folder_names = [os.path.basename(f) for f in failed_files.keys()]
        label = self._generate_folder_label(folder_names, total_count)
        
        # 创建记录
        record = {
            "time": datetime.now().strftime("%m/%d %H:%M"),
            "folders": failed_files,
            "total_count": total_count,
            "label": label
        }
        
        # 添加到列表开头
        self.records.insert(0, record)
        
        # 限制最大记录数
        if len(self.records) > self.MAX_RECORDS:
            self.records = self.records[:self.MAX_RECORDS]
        
        self._save_records()
        self._refresh_list()
    
    def _generate_folder_label(self, folder_names: List[str], total_count: int) -> str:
        """
        生成文件夹标签
        
        示例：
        - ["Chapter 9"] -> "(2张) Chapter 9"
        - ["Chapter 8", "Chapter 9", "Chapter 10"] -> "(5张) Chapter 8, 9, 10"
        - ["Folder A", "Folder B"] -> "(3张) Folder A, Folder B"
        """
        import re
        
        if len(folder_names) == 1:
            return f"({total_count}张) {folder_names[0]}"
        
        # 尝试提取公共前缀和数字后缀
        # 匹配模式：前缀 + 数字（如 "Chapter 9", "Vol 10"）
        pattern = r'^(.+?)\s*(\d+)$'
        
        parsed = []
        for name in folder_names:
            match = re.match(pattern, name)
            if match:
                parsed.append((match.group(1).strip(), int(match.group(2))))
            else:
                parsed.append((name, None))
        
        # 检查是否所有文件夹都有相同的前缀
        prefixes = [p[0] for p in parsed if p[1] is not None]
        numbers = [p[1] for p in parsed if p[1] is not None]
        
        if len(prefixes) == len(folder_names) and len(set(prefixes)) == 1:
            # 所有文件夹都有相同前缀，可以简化显示
            prefix = prefixes[0]
            numbers.sort()
            number_str = ", ".join(str(n) for n in numbers)
            return f"({total_count}张) {prefix} {number_str}"
        else:
            # 前缀不同，显示完整名称
            if len(folder_names) <= 3:
                return f"({total_count}张) {', '.join(folder_names)}"
            else:
                return f"({total_count}张) {folder_names[0]} 等{len(folder_names)}个文件夹"
    
    def _refresh_list(self):
        """刷新显示列表"""
        self.record_list.clear()
        
        for i, record in enumerate(self.records):
            # 添加到列表
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, i)

            label_text = f"{record['time']}  {record['label']}"
            item_widget = _FailedRecordItemWidget(
                label_text=label_text,
                on_double_click=lambda _item=item: self._on_item_double_clicked(_item),
                on_retry=lambda checked=False, idx=i: self._retry_record(idx),
                on_delete=lambda checked=False, idx=i: self._delete_record(idx),
                tooltip_retry=self._t("Retry"),
                tooltip_delete=self._t("Delete"),
                parent=self.record_list,
            )

            item.setSizeHint(item_widget.sizeHint())
            self.record_list.addItem(item)
            self.record_list.setItemWidget(item, item_widget)
    
    def _delete_record(self, index: int):
        """删除指定索引的记录"""
        if 0 <= index < len(self.records):
            self.records.pop(index)
            self._save_records()
            self._refresh_list()
    
    def _retry_record(self, index: int):
        """重试指定索引的记录，直接进入延迟重试流程"""
        if index is not None and 0 <= index < len(self.records):
            record = self.records[index]
            
            # 检查文件是否存在
            valid_folders = {}
            for folder_path, files in record["folders"].items():
                valid_files = []
                for filename in files:
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        valid_files.append(filename)
                if valid_files:
                    valid_folders[folder_path] = valid_files
            
            if not valid_folders:
                # 所有文件都不存在
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    self._t("Warning"),
                    self._t("All files in this record no longer exist.")
                )
                return
            
            # 发射重试请求信号
            self.retry_requested.emit({
                "folders": valid_folders,
                "total_count": sum(len(f) for f in valid_folders.values()),
                "record_index": index  # 传递索引以便重试成功后删除记录
            })
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """双击记录项，重新导入失败的图片"""
        index = item.data(Qt.ItemDataRole.UserRole)
        if index is not None and 0 <= index < len(self.records):
            record = self.records[index]
            
            # 检查文件是否存在
            valid_folders = {}
            for folder_path, files in record["folders"].items():
                valid_files = []
                for filename in files:
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path):
                        valid_files.append(filename)
                if valid_files:
                    valid_folders[folder_path] = valid_files
            
            if not valid_folders:
                # 所有文件都不存在
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.warning(
                    self,
                    self._t("Warning"),
                    self._t("All files in this record no longer exist.")
                )
                return
            
            # 发射导入请求信号
            self.import_requested.emit({
                "folders": valid_folders,
                "total_count": sum(len(f) for f in valid_folders.values())
            })
            
            # 导入后不删除记录（用户手动删除）
    
    def _clear_all_records(self):
        """清空所有记录"""
        self.records.clear()
        self._save_records()
        self._refresh_list()
    
    def refresh_ui_texts(self):
        """刷新UI文本（用于语言切换）"""
        # 刷新提示文本
        self.record_list.setToolTip(self._t("Double-click to re-import failed images"))
        self.clear_btn.setText(self._t("Clear"))


class FileListContainer(QWidget):
    """文件列表容器 - 包含 FileListView 和 FailedRecordsList"""
    
    # 转发 FileListView 的信号
    file_remove_requested = pyqtSignal(str)
    file_selected = pyqtSignal(str)
    files_dropped = pyqtSignal(list)
    
    # 失败记录导入请求信号
    failed_import_requested = pyqtSignal(dict)
    
    # 失败记录重试请求信号（直接进入延迟重试流程）
    failed_retry_requested = pyqtSignal(dict)
    
    def __init__(self, model, parent=None):
        super().__init__(parent)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # 文件列表视图
        self.file_list_view = FileListView(model, self)
        
        # 失败记录列表
        self.failed_records_list = FailedRecordsList(self)
        
        # 使用 QSplitter 让用户可以调整两部分的高度
        self.splitter = QSplitter(Qt.Orientation.Vertical)
        self.splitter.addWidget(self.file_list_view)
        self.splitter.addWidget(self.failed_records_list)
        self.splitter.setStretchFactor(0, 1)  # 文件列表可伸缩
        self.splitter.setStretchFactor(1, 0)  # 失败记录固定高度
        self.splitter.setCollapsible(0, False)  # 文件列表不可折叠
        self.splitter.setCollapsible(1, True)  # 失败记录可折叠
        
        layout.addWidget(self.splitter)
        
        # 连接信号
        self.file_list_view.file_remove_requested.connect(self.file_remove_requested.emit)
        self.file_list_view.file_selected.connect(self.file_selected.emit)
        self.file_list_view.files_dropped.connect(self.files_dropped.emit)
        self.failed_records_list.import_requested.connect(self._on_failed_import_requested)
        self.failed_records_list.retry_requested.connect(self._on_failed_retry_requested)
    
    def _on_failed_import_requested(self, record: dict):
        """处理失败记录导入请求"""
        folders_data = record.get("folders", {})
        if not folders_data:
            return
        
        # 收集所有文件路径
        all_file_paths = []
        
        # 仅添加最后一层文件夹节点（不展开上级目录）
        # 目标效果：
        # - Chapter 9
        #   - 012__001.jpg
        #   - 012__002.jpg
        for folder_path, filenames in folders_data.items():
            if not folder_path or not filenames:
                continue
            
            file_paths = [os.path.join(folder_path, fn) for fn in filenames]
            all_file_paths.extend(file_paths)
            self.file_list_view._add_folder_group(folder_path, file_paths)
        
        # 通过 files_dropped 信号通知 AppLogic 添加文件到 source_files
        # 这样点击"开始翻译"时才能找到这些文件
        if all_file_paths:
            self.files_dropped.emit(all_file_paths)
        
        # 同时发射信号通知外部（如果需要额外处理）
        self.failed_import_requested.emit(record)
    
    def _on_failed_retry_requested(self, record: dict):
        """处理失败记录重试请求，复用双击导入逻辑后转发重试信号"""
        folders_data = record.get("folders", {})
        if not folders_data:
            return
        
        # 复用双击导入的逻辑：添加文件到 UI 列表
        all_file_paths = []
        for folder_path, filenames in folders_data.items():
            if not folder_path or not filenames:
                continue
            
            file_paths = [os.path.join(folder_path, fn) for fn in filenames]
            all_file_paths.extend(file_paths)
            self.file_list_view._add_folder_group(folder_path, file_paths)
        
        # 通过 files_dropped 信号通知 AppLogic 添加文件到 source_files
        if all_file_paths:
            self.files_dropped.emit(all_file_paths)
        
        # 转发重试信号给 app_logic，触发自动启动翻译
        self.failed_retry_requested.emit(record)
    
    def delete_failed_record(self, index: int):
        """删除指定索引的失败记录"""
        self.failed_records_list._delete_record(index)
    
    def add_failed_record(self, failed_files: Dict[str, List[str]]):
        """添加失败记录"""
        self.failed_records_list.add_record(failed_files)
    
    # 代理 FileListView 的方法
    def add_files(self, file_paths: List[str]):
        """添加文件到列表"""
        self.file_list_view.add_files(file_paths)
    
    def add_files_with_tree(self, file_paths: List[str], folder_map: dict = None):
        """添加文件并保持树形结构"""
        self.file_list_view.add_files_with_tree(file_paths, folder_map)
    
    def add_files_from_tree(self, folder_tree: dict):
        """从树结构添加文件"""
        self.file_list_view.add_files_from_tree(folder_tree)
    
    def remove_file(self, file_path: str):
        """移除文件"""
        self.file_list_view.remove_file(file_path)
    
    def clear(self, clear_cache: bool = False):
        """清空文件列表"""
        self.file_list_view.clear(clear_cache)
    
    def topLevelItemCount(self) -> int:
        """获取顶层项数量"""
        return self.file_list_view.topLevelItemCount()
    
    def refresh_ui_texts(self):
        """刷新UI文本（用于语言切换）"""
        self.file_list_view.refresh_ui_texts()
        self.failed_records_list.refresh_ui_texts()
