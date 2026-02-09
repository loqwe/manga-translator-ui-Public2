#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
自定义漫画管理面板（工具箱）
整合一键处理、章节分析功能
名称映射管理已移动到独立标签页 (NameMappingPanel)
"""

import os
import json
from pathlib import Path
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings, QSortFilterProxyModel
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QPushButton, QLineEdit, QTextEdit, QLabel, QTreeView,
    QHeaderView, QAbstractItemView, QFileDialog,
    QMessageBox, QSplitter, QScrollArea, QCheckBox
)
from PyQt6.QtGui import QDragEnterEvent, QDropEvent, QStandardItemModel, QStandardItem

# 导入工具模块
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

try:
    from utils.one_click_processor import OneClickProcessor
    from utils.comic_analyzer import ComicAnalyzer
    from utils.name_replacer import NameReplacer
except ImportError:
    # 降级处理
    OneClickProcessor = None
    ComicAnalyzer = None
    NameReplacer = None


class MappingFilterProxy(QSortFilterProxyModel):
    """多列、递归过滤代理：匹配父或任一子项均显示父项"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._kw = ""
        self._status_filter = ""  # Empty = all, "连载" or "完结"

    def setFilterKeyword(self, text: str):
        self._kw = (text or "").strip().lower()
        self.invalidateFilter()

    def setStatusFilter(self, status: str):
        """Set status filter: '' = all, '连载' or '完结'"""
        self._status_filter = status
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        model = self.sourceModel()
        idx0 = model.index(source_row, 0, source_parent)
        def _t(idx):
            v = model.data(idx, Qt.ItemDataRole.DisplayRole)
            return (v or "").lower()
        
        # Status filter: only apply to top-level (parent) rows
        if self._status_filter and not source_parent.isValid():
            idx2 = model.index(source_row, 2, source_parent)
            status_text = model.data(idx2, Qt.ItemDataRole.DisplayRole) or "连载"
            if status_text != self._status_filter:
                return False
        
        # Keyword filter
        if not self._kw:
            return True
        idx1 = model.index(source_row, 1, source_parent)
        idx2 = model.index(source_row, 2, source_parent)
        # 当前行匹配（名称、生肉名称、状态列）
        if self._kw in _t(idx0) or self._kw in _t(idx1) or self._kw in _t(idx2):
            return True
        # 子项匹配则保留父项
        child_count = model.rowCount(idx0)
        for i in range(child_count):
            if self.filterAcceptsRow(i, idx0):
                return True
        return False


class FolderLineEdit(QLineEdit):
    """支持拖拽文件夹路径的输入框"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        md = event.mimeData()
        if md.hasUrls():
            urls = md.urls()
            # 接受第一个本地路径
            if urls and urls[0].isLocalFile():
                event.acceptProposedAction()
                return
        event.ignore()

    def dropEvent(self, event: QDropEvent):
        md = event.mimeData()
        if md.hasUrls():
            urls = md.urls()
            if urls and urls[0].isLocalFile():
                path = urls[0].toLocalFile()
                self.setText(path)
                event.acceptProposedAction()
                return
        event.ignore()

class ProcessThread(QThread):
    """处理线程"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(dict)
    
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.processor.progress_callback = self.report_progress
    
    def report_progress(self, message: str):
        self.progress.emit(message)
    
    def run(self):
        results = self.processor.process_all()
        self.finished.emit(results)


class CustomComicPanel(QWidget):
    """自定义漫画管理面板（工具箱）
    
    包含一键处理和章节分析功能。
    名称映射管理已移动到独立的标签页 (NameMappingPanel)。
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.process_thread = None
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 分隔上部功能和下部结果
        self.splitter = QSplitter(Qt.Orientation.Vertical, self)
        main_layout.addWidget(self.splitter)

        # 顶部功能容器（放入可滚动区域）
        top_widget = QWidget()
        top_layout = QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(self._create_one_click_section())
        top_layout.addWidget(self._create_analysis_section(), 1)  # 章节分析向上扩展，拉伸填充剩余空间
        # 可滚动容器
        top_scroll = QScrollArea(self.splitter)
        top_scroll.setWidgetResizable(True)
        top_scroll.setWidget(top_widget)

        # 底部结果容器
        bottom_widget = QWidget(self.splitter)
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(10)
        result_group = QGroupBox("处理结果")
        result_layout = QVBoxLayout(result_group)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setPlaceholderText("处理结果将显示在这里...")
        result_layout.addWidget(self.result_text)
        bottom_layout.addWidget(result_group)

        self.splitter.addWidget(top_scroll)
        self.splitter.addWidget(bottom_widget)
        self.splitter.setSizes([600, 200])
        self.splitter.splitterMoved.connect(lambda pos, index: self._save_splitter_sizes())
        # 分割条移动时也保存展开状态
        self.splitter.splitterMoved.connect(lambda: self._save_analysis_tree_state() if hasattr(self, 'analysis_view') else None)
    
    def showEvent(self, event):
        """首次显示时恢复设置"""
        super().showEvent(event)
        if not hasattr(self, '_settings_restored'):
            self._settings_restored = True
            self._restore_basic_settings()
            # 自动开始分析（如果有配置的文件夹）
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(200, self._auto_analyze_on_open)
    
    # =========================================================
    # 名称映射相关方法已移除（迁移到 NameMappingPanel）
    # =========================================================
    
    def _create_one_click_section(self) -> QWidget:
        """创建一键处理区域"""
        group = QGroupBox("一键处理（替换→压缩→转移）")
        layout = QFormLayout(group)
        
        self.input_folder_edit = FolderLineEdit()
        self.input_folder_edit.setPlaceholderText("输入文件夹（翻译输出）")
        input_btn = QPushButton("浏览...")
        input_btn.clicked.connect(lambda: self._browse_folder(self.input_folder_edit))
        self.input_folder_edit.textChanged.connect(self._save_paths)
        
        input_layout = QHBoxLayout()
        input_layout.addWidget(self.input_folder_edit)
        input_layout.addWidget(input_btn)
        layout.addRow("输入文件夹:", input_layout)
        
        self.output_folder_edit = FolderLineEdit()
        self.output_folder_edit.setPlaceholderText("输出文件夹（存储）")
        output_btn = QPushButton("浏览...")
        output_btn.clicked.connect(lambda: self._browse_folder(self.output_folder_edit))
        self.output_folder_edit.textChanged.connect(self._save_paths)
        
        output_layout = QHBoxLayout()
        output_layout.addWidget(self.output_folder_edit)
        output_layout.addWidget(output_btn)
        layout.addRow("输出文件夹:", output_layout)
        
        # 语言后缀选项
        self.language_suffix_checkbox = QCheckBox("非韩文漫画章节名加 [R] 后缀")
        self.language_suffix_checkbox.setToolTip("韩文漫画不加后缀，其他语言加 [R] 标记")
        self.language_suffix_checkbox.stateChanged.connect(self._save_paths)
        layout.addRow("", self.language_suffix_checkbox)
        
        btn_layout = QHBoxLayout()
        self.process_btn = QPushButton("开始一键处理")
        self.process_btn.clicked.connect(self.start_processing)
        self.process_btn.setStyleSheet("background-color: #0078d4; color: white; padding: 5px;")
        btn_layout.addWidget(self.process_btn)
        layout.addRow("", btn_layout)
        
        return group
    
    def _create_analysis_section(self) -> QWidget:
        """创建章节分析区域（表格视图）"""
        group = QGroupBox("章节分析")
        layout = QVBoxLayout(group)
        
        # 文件夹选择
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("文件夹:"))
        self.analysis_folder_edit = FolderLineEdit()
        self.analysis_folder_edit.setPlaceholderText("要分析的文件夹")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(lambda: self._browse_folder(self.analysis_folder_edit))
        self.analysis_folder_edit.textChanged.connect(self._save_paths)
        folder_layout.addWidget(self.analysis_folder_edit, 1)
        folder_layout.addWidget(browse_btn)
        layout.addLayout(folder_layout)
        
        # 按钮行
        btn_layout = QHBoxLayout()
        analyze_btn = QPushButton("开始分析")
        analyze_btn.clicked.connect(self.analyze_chapters)
        self.sort_asc_btn = QPushButton("章节⇑")
        self.sort_asc_btn.setToolTip("章节按升序排列")
        self.sort_asc_btn.clicked.connect(lambda: self._sort_chapters(ascending=True))
        self.sort_desc_btn = QPushButton("章节⇓")
        self.sort_desc_btn.setToolTip("章节按降序排列")
        self.sort_desc_btn.clicked.connect(lambda: self._sort_chapters(ascending=False))
        expand_all_btn = QPushButton("展开全部")
        expand_all_btn.clicked.connect(lambda: self.analysis_view.expandAll() if hasattr(self, 'analysis_view') else None)
        collapse_all_btn = QPushButton("折叠全部")
        collapse_all_btn.clicked.connect(lambda: self.analysis_view.collapseAll() if hasattr(self, 'analysis_view') else None)
        export_btn = QPushButton("导出报告")
        export_btn.clicked.connect(self._export_analysis_report)
        # Restore saved status filter mode
        saved_mode = self._settings().value("analysis/status_filter", "全部")
        if saved_mode not in ("全部", "连载", "完结"):
            saved_mode = "全部"
        self._status_filter_mode = saved_mode
        self.status_filter_btn = QPushButton(saved_mode)
        self.status_filter_btn.setToolTip("筛选状态：全部 / 连载 / 完结")
        self.status_filter_btn.clicked.connect(self._cycle_status_filter)
        btn_layout.addWidget(analyze_btn)
        btn_layout.addWidget(self.status_filter_btn)
        btn_layout.addWidget(self.sort_asc_btn)
        btn_layout.addWidget(self.sort_desc_btn)
        btn_layout.addWidget(expand_all_btn)
        btn_layout.addWidget(collapse_all_btn)
        btn_layout.addWidget(export_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)
        
        # 搜索与计数
        search_layout = QHBoxLayout()
        search_layout.addWidget(QLabel("搜索:"))
        self.analysis_search_edit = QLineEdit()
        self.analysis_search_edit.setPlaceholderText("过滤漫画/章节/CBZ...")
        self.analysis_search_edit.textChanged.connect(self.apply_analysis_filter)
        search_layout.addWidget(self.analysis_search_edit, 1)
        self.analysis_count_label = QLabel("显示 0 / 总 0")
        search_layout.addWidget(self.analysis_count_label)
        layout.addLayout(search_layout)
        
        # 表格视图
        self.analysis_view = QTreeView()
        self.analysis_view.setRootIsDecorated(True)
        self.analysis_view.setAlternatingRowColors(True)
        self.analysis_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.analysis_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.analysis_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.analysis_view.setUniformRowHeights(True)  # 优化性能
        self.analysis_view.doubleClicked.connect(self._on_analysis_double_click)
        self.analysis_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.analysis_view.customContextMenuRequested.connect(self._show_analysis_context_menu)
        self.analysis_view.expanded.connect(lambda: self._save_analysis_tree_state())
        self.analysis_view.collapsed.connect(lambda: self._save_analysis_tree_state())
        # Ctrl+C copy current cell
        from PyQt6.QtGui import QShortcut, QKeySequence
        copy_shortcut = QShortcut(QKeySequence.StandardKey.Copy, self.analysis_view)
        copy_shortcut.activated.connect(self._copy_current_cell)
        # 调淡选中/悬停行颜色
        self.analysis_view.setStyleSheet("""
            QTreeView::item:hover {
                background-color: #cce4f7;
            }
            QTreeView::item:selected {
                background-color: #5a9fd4;
                color: white;
            }
            QTreeView::item:selected:hover {
                background-color: #4a8fc4;
                color: white;
            }
            QTreeView::item:selected:!active {
                background-color: #7ab8e0;
            }
        """)
        
        layout.addWidget(self.analysis_view)
        
        # 初始化空模型
        empty_model = QStandardItemModel()
        empty_model.setHorizontalHeaderLabels(["名称", "生肉名称", "状态", "最新章节", "CBZ章节"])
        self.analysis_view.setModel(empty_model)
        
        # 设置表头的基本属性
        header = self.analysis_view.header()
        header.setDefaultSectionSize(100)
        header.setMinimumSectionSize(50)
        header.setStretchLastSection(True)  # 最后一列自动填充剩余空间
        header.setSectionsMovable(False)  # 禁止列移动
        
        # 连接信号并优化性能
        header.sectionResized.connect(self._save_analysis_column_widths)
        
        self._set_default_analysis_column_widths()
        self._analysis_columns_initialized = True  # 标记已初始化
        
        return group
    
    def _browse_folder(self, line_edit: QLineEdit):
        """浏览文件夹（记忆上次目录）"""
        settings = QSettings("MangaTranslator", "CustomPanel")
        start_dir = settings.value("last_dir", os.path.expanduser("~"))
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", start_dir)
        if folder:
            line_edit.setText(folder)
            settings.setValue("last_dir", folder)
    
    def start_processing(self):
        """开始一键处理"""
        if not OneClickProcessor:
            QMessageBox.warning(self, "错误", "无法加载处理模块，请检查依赖文件")
            return
        
        input_folder = self.input_folder_edit.text()
        output_folder = self.output_folder_edit.text()
        
        if not input_folder or not os.path.exists(input_folder):
            QMessageBox.warning(self, "错误", "输入文件夹不存在")
            return
        
        self.result_text.clear()
        self.result_text.append("开始一键处理...")
        self.process_btn.setEnabled(False)
        
        add_suffix = self.language_suffix_checkbox.isChecked()
        processor = OneClickProcessor(input_folder, output_folder, add_language_suffix=add_suffix)
        self.process_thread = ProcessThread(processor)
        self.process_thread.progress.connect(self.on_progress)
        self.process_thread.finished.connect(self.on_finished)
        self.process_thread.start()
    
    def on_progress(self, message: str):
        """处理进度"""
        self.result_text.append(message)
    
    def on_finished(self, results: dict):
        """处理完成"""
        self.process_btn.setEnabled(True)
        total = (len(results.get('organized', [])) + 
                len(results.get('compressed', [])) + 
                len(results.get('transferred', [])))
        self.result_text.append(f"\n✓ 处理完成！共完成 {total} 项操作")
        QMessageBox.information(self, "完成", f"一键处理完成\n共处理 {total} 项")

    # ============ Settings persistence ============
    def _settings(self) -> QSettings:
        return QSettings("MangaTranslator", "CustomPanel")

    def _restore_basic_settings(self):
        s = self._settings()
        # Paths - 阻止信号避免触发保存
        self.input_folder_edit.blockSignals(True)
        self.output_folder_edit.blockSignals(True)
        self.analysis_folder_edit.blockSignals(True)
        self.language_suffix_checkbox.blockSignals(True)
        
        self.input_folder_edit.setText(s.value("paths/input", ""))
        self.output_folder_edit.setText(s.value("paths/output", ""))
        self.analysis_folder_edit.setText(s.value("paths/analysis", ""))
        self.language_suffix_checkbox.setChecked(s.value("options/language_suffix", False, type=bool))
        
        self.input_folder_edit.blockSignals(False)
        self.output_folder_edit.blockSignals(False)
        self.analysis_folder_edit.blockSignals(False)
        self.language_suffix_checkbox.blockSignals(False)
        
        # Splitter sizes - 延迟恢复确保 UI已完全初始化
        from PyQt6.QtCore import QTimer
        def restore_splitter():
            sizes = s.value("ui/splitter_sizes")
            if isinstance(sizes, list) and len(sizes) == 2:
                try:
                    sizes = [int(x) for x in sizes]
                    self.splitter.setSizes(sizes)
                except Exception:
                    pass
        QTimer.singleShot(100, restore_splitter)

    def _save_splitter_sizes(self):
        """保存自定义面板分割条位置"""
        try:
            s = self._settings()
            sizes = self.splitter.sizes()
            s.setValue("ui/splitter_sizes", sizes)
        except Exception:
            pass

    def _save_paths(self):
        s = self._settings()
        s.setValue("paths/input", self.input_folder_edit.text())
        s.setValue("paths/output", self.output_folder_edit.text())
        s.setValue("paths/analysis", self.analysis_folder_edit.text())
        s.setValue("options/language_suffix", self.language_suffix_checkbox.isChecked())
    
    def analyze_chapters(self):
        """分析章节并填充表格"""
        folder = self.analysis_folder_edit.text()
        if not folder or not os.path.exists(folder):
            QMessageBox.warning(self, "错误", "文件夹不存在")
            return
        
        try:
            folder_path = Path(folder)
            results = {}
            
            for comic_folder in folder_path.iterdir():
                if not comic_folder.is_dir():
                    continue
                
                chapters = []
                cbz_files = []
                
                for item in comic_folder.iterdir():
                    if item.is_dir():
                        chapters.append(item.name)
                    elif item.suffix.lower() == '.cbz':
                        cbz_files.append(item.stem)
                
                if chapters or cbz_files:
                    results[comic_folder.name] = {
                        'chapters': sorted(chapters),
                        'cbz_files': sorted(cbz_files),
                        'path': str(comic_folder)
                    }
            
            # 构建表格模型
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["名称", "生肉名称", "状态", "最新章节", "CBZ章节"])
            
            # Reverse lookup: Chinese name -> raw name (Korean preferred)
            replacer = NameReplacer() if NameReplacer else None
            comic_status = self._load_comic_status()
            
            total_comics = len(results)
            for comic in sorted(results.keys()):
                data = results[comic]
                all_chapters = data['chapters'] + data['cbz_files']
                latest = self._get_latest_chapter(all_chapters) if all_chapters else ""
                cbz_count = len(data['cbz_files'])
                
                # Reverse lookup raw name
                raw_name = replacer.get_raw_name(comic) if replacer else ""
                status = comic_status.get(comic, "连载")
                
                # 父项（漫画）
                name_item = QStandardItem(comic)
                name_item.setData(data['path'], Qt.ItemDataRole.UserRole)  # 存储路径
                raw_name_item = QStandardItem(raw_name)
                status_item = QStandardItem(status)
                latest_item = QStandardItem(latest)
                cbz_count_item = QStandardItem(str(cbz_count))
                
                row_items = [name_item, raw_name_item, status_item, latest_item, cbz_count_item]
                for it in row_items:
                    it.setEditable(False)
                self._apply_status_style(row_items, status)
                
                model.appendRow(row_items)
                
                # 子项（章节文件夹）
                for ch in data['chapters']:
                    ch_path = os.path.join(data['path'], ch)
                    r0 = QStandardItem(ch)
                    r0.setData(ch_path, Qt.ItemDataRole.UserRole)
                    r1 = QStandardItem("")
                    r2 = QStandardItem("")
                    r3 = QStandardItem("")
                    r4 = QStandardItem("")
                    for it in (r0, r1, r2, r3, r4):
                        it.setEditable(False)
                    name_item.appendRow([r0, r1, r2, r3, r4])
                
                # 子项（CBZ章节）
                for cbz in data['cbz_files']:
                    cbz_path = os.path.join(data['path'], cbz + '.cbz')
                    r0 = QStandardItem(cbz)
                    r0.setData(cbz_path, Qt.ItemDataRole.UserRole)
                    r1 = QStandardItem("")
                    r2 = QStandardItem("")
                    r3 = QStandardItem("")
                    r4 = QStandardItem("")
                    for it in (r0, r1, r2, r3, r4):
                        it.setEditable(False)
                    name_item.appendRow([r0, r1, r2, r3, r4])
            
            # 设置代理和模型
            if not hasattr(self, 'analysis_proxy') or self.analysis_proxy is None:
                self.analysis_proxy = MappingFilterProxy(self)
            self.analysis_proxy.setSourceModel(model)
            self.analysis_view.setModel(self.analysis_proxy)
            
            # 设置表头和连接信号
            header = self.analysis_view.header()
            header.setDefaultSectionSize(100)
            header.setMinimumSectionSize(50)
            header.setStretchLastSection(True)  # 最后一列自动填充剩余空间
            header.setSectionsMovable(False)  # 禁止列移动
            
            # 确保信号只连接一次
            try:
                header.sectionResized.disconnect()
            except Exception:
                pass
            header.sectionResized.connect(self._save_analysis_column_widths)
            
            # 只在第一次设置列宽，之后保留用户调整
            if not hasattr(self, '_analysis_columns_initialized'):
                self._set_default_analysis_column_widths()
                self._analysis_columns_initialized = True
            
            # 恢复保存的列宽和展开状态
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self._restore_analysis_column_widths)
            QTimer.singleShot(100, self._restore_analysis_tree_state)
            
            # 恢复状态筛选
            self._apply_current_status_filter()
            
            # 更新计数
            self._analysis_total_comics = total_comics
            self.apply_analysis_filter()
            
            # 保存分析结果到JSON文件
            self._save_analysis_results(results)
            
            # 输出到结果区
            total_chapters = sum(len(d['chapters']) + len(d['cbz_files']) for d in results.values())
            self.result_text.append(f"✓ 分析完成：{total_comics} 部漫画，共 {total_chapters} 个章节")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"分析失败: {str(e)}")
            self.result_text.append(f"❌ 分析失败: {str(e)}")
    
    def apply_analysis_filter(self, text: Optional[str] = None):
        """过滤章节分析结果"""
        if text is None:
            text = self.analysis_search_edit.text() if hasattr(self, 'analysis_search_edit') else ""
        if hasattr(self, 'analysis_proxy') and self.analysis_proxy is not None:
            self.analysis_proxy.setFilterKeyword(text)
        
        # 统计可见漫画数
        visible = 0
        proxy = getattr(self, 'analysis_proxy', None)
        if proxy is not None:
            visible = proxy.rowCount()
        total = getattr(self, '_analysis_total_comics', visible)
        if hasattr(self, 'analysis_count_label'):
            self.analysis_count_label.setText(f"显示 {visible} / 总 {total}")
    
    def _cycle_status_filter(self):
        """循环切换状态筛选：全部 → 连载 → 完结 → 全部"""
        cycle = ["全部", "连载", "完结"]
        idx = cycle.index(self._status_filter_mode) if self._status_filter_mode in cycle else 0
        self._status_filter_mode = cycle[(idx + 1) % len(cycle)]
        self.status_filter_btn.setText(self._status_filter_mode)
        
        # Apply to proxy
        self._apply_current_status_filter()
        
        # Persist
        self._settings().setValue("analysis/status_filter", self._status_filter_mode)
        
        # Update count
        self.apply_analysis_filter()
    
    def _apply_current_status_filter(self):
        """Apply the current status filter mode to proxy"""
        proxy = getattr(self, 'analysis_proxy', None)
        if proxy is not None:
            status = "" if self._status_filter_mode == "全部" else self._status_filter_mode
            proxy.setStatusFilter(status)
    
    def _set_default_analysis_column_widths(self):
        """设置默认列宽"""
        if not hasattr(self, 'analysis_view'):
            return
        header = self.analysis_view.header()
        # 默认列宽: 名称(200), 生肉名称(180), 状态(60), 最新章节(130), CBZ章节(80)
        default_widths = [200, 180, 60, 130, 80]
        header.blockSignals(True)
        
        # 全部设置为Interactive模式，不使用Stretch
        for i in range(header.count()):
            header.setSectionResizeMode(i, QHeaderView.ResizeMode.Interactive)
        
        # 设置各列宽度
        for i, width in enumerate(default_widths):
            if i < header.count():
                header.resizeSection(i, width)
        
        header.blockSignals(False)
    
    def _save_analysis_column_widths(self):
        """保存章节分析列宽"""
        if not hasattr(self, 'analysis_view'):
            return
        # 防抖动：避免频繁保存
        if not hasattr(self, '_save_column_timer'):
            from PyQt6.QtCore import QTimer
            self._save_column_timer = QTimer()
            self._save_column_timer.setSingleShot(True)
            self._save_column_timer.timeout.connect(self._do_save_column_widths)
        self._save_column_timer.start(500)  # 500ms后保存
    
    def _do_save_column_widths(self):
        """实际执行保存操作"""
        try:
            header = self.analysis_view.header()
            widths = [header.sectionSize(i) for i in range(header.count())]
            s = self._settings()
            s.setValue("analysis/column_widths", widths)
            s.sync()
        except Exception as e:
            print(f"保存列宽失败: {e}")
    
    def _restore_analysis_column_widths(self):
        """恢复章节分析列宽"""
        if not hasattr(self, 'analysis_view'):
            return
        try:
            s = self._settings()
            widths = s.value("analysis/column_widths")
            header = self.analysis_view.header()
            col_count = header.count()
            # 旧的 5 列布局保存值，重置为新的 6 列默认宽度
            if widths and isinstance(widths, list) and len(widths) < col_count:
                self._set_default_analysis_column_widths()
                return
            if widths and isinstance(widths, list) and len(widths) >= col_count:
                # 阻止信号避免触发保存
                header.blockSignals(True)
                for i, w in enumerate(widths):
                    if i < header.count():  # 恢复所有列
                        try:
                            w_int = int(w)
                            if w_int > 20:  # 最小列宽20px
                                header.resizeSection(i, w_int)
                        except (ValueError, TypeError):
                            pass
                header.blockSignals(False)
                print(f"✓ 已恢复列宽: {widths}")
            else:
                print("⚠ 无保存的列宽，使用默认值")
        except Exception as e:
            print(f"恢复列宽失败: {e}")
    
    def _restore_analysis_tree_state(self):
        """恢复章节分析树的展开状态"""
        s = self._settings()
        expanded = s.value("analysis/expanded", [])
        if not isinstance(expanded, list):
            return
        expanded_set = set(expanded)
        
        proxy = getattr(self, 'analysis_proxy', None)
        if proxy is not None:
            top_rows = proxy.rowCount()
            for r in range(top_rows):
                p_idx = proxy.index(r, 0)
                src_idx = proxy.mapToSource(p_idx)
                model = src_idx.model()
                comic_name = model.data(src_idx, Qt.ItemDataRole.DisplayRole)
                if comic_name in expanded_set:
                    self.analysis_view.expand(p_idx)
    
    def _save_analysis_tree_state(self):
        """保存章节分析树的展开状态"""
        expanded = []
        proxy = getattr(self, 'analysis_proxy', None)
        if proxy is not None:
            top_rows = proxy.rowCount()
            for r in range(top_rows):
                p_idx = proxy.index(r, 0)
                if self.analysis_view.isExpanded(p_idx):
                    src_idx = proxy.mapToSource(p_idx)
                    model = src_idx.model()
                    comic_name = model.data(src_idx, Qt.ItemDataRole.DisplayRole)
                    if comic_name:
                        expanded.append(comic_name)
        s = self._settings()
        s.setValue("analysis/expanded", expanded)
    
    def _on_analysis_double_click(self, index):
        """双击打开文件夹"""
        if not index.isValid():
            return
        src_idx = self.analysis_proxy.mapToSource(index)
        model = src_idx.model()
        path = model.data(model.index(src_idx.row(), 0, src_idx.parent()), Qt.ItemDataRole.UserRole)
        if path and os.path.exists(path):
            import subprocess
            import platform
            try:
                if platform.system() == 'Windows':
                    os.startfile(path)
                elif platform.system() == 'Darwin':  # macOS
                    subprocess.run(['open', path])
                else:  # Linux
                    subprocess.run(['xdg-open', path])
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开: {str(e)}")
    
    def _show_analysis_context_menu(self, position):
        """显示章节分析右键菜单"""
        index = self.analysis_view.indexAt(position)
        if not index.isValid():
            return
        
        from PyQt6.QtWidgets import QMenu
        from PyQt6.QtGui import QAction
        
        menu = QMenu(self)
        
        # 复制名称
        copy_action = QAction("复制名称", self)
        copy_action.triggered.connect(lambda: self._copy_analysis_name(index))
        menu.addAction(copy_action)
        
        # 复制生肉名称（仅父行且有生肉名时显示）
        src_idx_for_raw = self.analysis_proxy.mapToSource(index)
        if not src_idx_for_raw.parent().isValid():
            raw_idx = src_idx_for_raw.model().index(src_idx_for_raw.row(), 1)
            raw_text = src_idx_for_raw.model().data(raw_idx, Qt.ItemDataRole.DisplayRole) or ""
            if raw_text:
                copy_raw_action = QAction("复制生肉名称", self)
                copy_raw_action.triggered.connect(lambda checked, t=raw_text: self._copy_text(t))
                menu.addAction(copy_raw_action)
        
        # 复制生肉名称子菜单（展示当前漫画的全部生肉名称供选择）
        if not src_idx_for_raw.parent().isValid():
            comic_name = src_idx_for_raw.model().item(src_idx_for_raw.row(), 0).text()
            if NameReplacer:
                replacer = NameReplacer()
                all_raw_names = replacer.get_all_raw_names(comic_name)
                if all_raw_names:
                    raw_names_submenu = QMenu("复制生肉名称...", self)
                    for rn in all_raw_names:
                        action = QAction(rn, self)
                        action.triggered.connect(lambda checked, t=rn: self._copy_text(t))
                        raw_names_submenu.addAction(action)
                    menu.addMenu(raw_names_submenu)
        
        # 打开文件夹
        open_action = QAction("打开文件夹", self)
        open_action.triggered.connect(lambda: self._on_analysis_double_click(index))
        menu.addAction(open_action)
        
        # 完结/连载 切换（仅父行）
        src_idx = self.analysis_proxy.mapToSource(index)
        is_parent = not src_idx.parent().isValid()
        if is_parent:
            menu.addSeparator()
            model = src_idx.model()
            status_idx = model.index(src_idx.row(), 2)
            current_status = model.data(status_idx, Qt.ItemDataRole.DisplayRole) or "连载"
            label = "标记为完结" if current_status == "连载" else "标记为连载"
            status_action = QAction(label, self)
            # Capture index by value
            status_action.triggered.connect(lambda checked, idx=index: self._toggle_comic_status(idx))
            menu.addAction(status_action)
        
        menu.addSeparator()
        
        # 展开/折叠全部
        expand_action = QAction("展开全部", self)
        expand_action.triggered.connect(self.analysis_view.expandAll)
        menu.addAction(expand_action)
        
        collapse_action = QAction("折叠全部", self)
        collapse_action.triggered.connect(self.analysis_view.collapseAll)
        menu.addAction(collapse_action)
        
        menu.exec(self.analysis_view.viewport().mapToGlobal(position))
    
    def _copy_analysis_name(self, index):
        """复制分析结果名称到剪贴板"""
        if not index.isValid():
            return
        text = index.data(Qt.ItemDataRole.DisplayRole)
        if text:
            self._copy_text(text)
    
    def _copy_text(self, text: str):
        """复制文本到剪贴板"""
        from PyQt6.QtWidgets import QApplication
        QApplication.clipboard().setText(text)
    
    def _copy_current_cell(self):
        """复制当前选中单元格的内容（Ctrl+C）"""
        index = self.analysis_view.currentIndex()
        if index.isValid():
            text = index.data(Qt.ItemDataRole.DisplayRole)
            if text:
                self._copy_text(text)
                # Show brief tooltip feedback
                from PyQt6.QtWidgets import QToolTip
                pos = self.analysis_view.visualRect(index).center()
                global_pos = self.analysis_view.viewport().mapToGlobal(pos)
                tip = f"已复制: {text[:30]}{'...' if len(text) > 30 else ''}"
                QToolTip.showText(global_pos, tip, self.analysis_view, self.analysis_view.visualRect(index), 1500)
    
    def _save_analysis_results(self, results: dict):
        """保存分析结果到JSON文件"""
        try:
            from datetime import datetime
            analysis_file = Path("analysis_results.json")
            
            # 构建保存数据
            save_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "folder": self.analysis_folder_edit.text(),
                "comics": {}
            }
            
            for comic, data in results.items():
                save_data["comics"][comic] = {
                    "chapters": data['chapters'],
                    "cbz_files": data['cbz_files'],
                    "path": data['path'],
                    "latest": self._get_latest_chapter(data['chapters'] + data['cbz_files']) if (data['chapters'] or data['cbz_files']) else ""
                }
            
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"保存分析结果失败: {e}")
    
    def _auto_analyze_on_open(self):
        """打开时自动开始分析（如果有配置的文件夹）"""
        try:
            folder = self.analysis_folder_edit.text()
            if folder and os.path.exists(folder):
                # 有有效的分析文件夹，自动开始分析
                self.result_text.append("\u25b6 自动开始分析...")
                self.analyze_chapters()
            else:
                # 没有配置文件夹，尝试加载上次结果作为备选
                self._load_previous_analysis()
        except Exception as e:
            print(f"自动分析失败: {e}")
            # 失败时尝试加载上次结果
            self._load_previous_analysis()
    
    def _load_previous_analysis(self):
        """加载上次分析结果"""
        try:
            analysis_file = Path("analysis_results.json")
            if not analysis_file.exists():
                return
            
            with open(analysis_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not data.get('comics'):
                return
            
            # 构建表格模型
            model = QStandardItemModel()
            model.setHorizontalHeaderLabels(["名称", "生肉名称", "状态", "最新章节", "CBZ章节"])
            
            # Reverse lookup: Chinese name -> raw name (Korean preferred)
            replacer = NameReplacer() if NameReplacer else None
            comic_status = self._load_comic_status()
            
            total_comics = len(data['comics'])
            for comic, comic_data in sorted(data['comics'].items()):
                latest = comic_data.get('latest', '')
                cbz_count = len(comic_data.get('cbz_files', []))
                comic_path = comic_data.get('path', '')
                
                # Reverse lookup raw name
                raw_name = replacer.get_raw_name(comic) if replacer else ""
                status = comic_status.get(comic, "连载")
                
                # 父项（漫画）
                name_item = QStandardItem(comic)
                name_item.setData(comic_path, Qt.ItemDataRole.UserRole)
                raw_name_item = QStandardItem(raw_name)
                status_item = QStandardItem(status)
                latest_item = QStandardItem(latest)
                cbz_count_item = QStandardItem(str(cbz_count))
                
                row_items = [name_item, raw_name_item, status_item, latest_item, cbz_count_item]
                for it in row_items:
                    it.setEditable(False)
                self._apply_status_style(row_items, status)
                
                model.appendRow(row_items)
                
                # 子项（章节文件夹）
                for ch in comic_data.get('chapters', []):
                    ch_path = os.path.join(comic_path, ch) if comic_path else ch
                    r0 = QStandardItem(ch)
                    r0.setData(ch_path, Qt.ItemDataRole.UserRole)
                    r1 = QStandardItem("")
                    r2 = QStandardItem("")
                    r3 = QStandardItem("")
                    r4 = QStandardItem("")
                    for it in (r0, r1, r2, r3, r4):
                        it.setEditable(False)
                    name_item.appendRow([r0, r1, r2, r3, r4])
                
                # 子项（CBZ章节）
                for cbz in comic_data.get('cbz_files', []):
                    cbz_path = os.path.join(comic_path, cbz + '.cbz') if comic_path else cbz
                    r0 = QStandardItem(cbz)
                    r0.setData(cbz_path, Qt.ItemDataRole.UserRole)
                    r1 = QStandardItem("")
                    r2 = QStandardItem("")
                    r3 = QStandardItem("")
                    r4 = QStandardItem("")
                    for it in (r0, r1, r2, r3, r4):
                        it.setEditable(False)
                    name_item.appendRow([r0, r1, r2, r3, r4])
            
            # 设置代理和模型
            if not hasattr(self, 'analysis_proxy') or self.analysis_proxy is None:
                self.analysis_proxy = MappingFilterProxy(self)
            self.analysis_proxy.setSourceModel(model)
            self.analysis_view.setModel(self.analysis_proxy)
            
            # 设置表头和连接信号
            header = self.analysis_view.header()
            header.setDefaultSectionSize(100)
            header.setMinimumSectionSize(50)
            header.setStretchLastSection(True)  # 最后一列自动填充剩余空间
            header.setSectionsMovable(False)  # 禁止列移动
            
            # 确保信号只连接一次
            try:
                header.sectionResized.disconnect()
            except Exception:
                pass
            header.sectionResized.connect(self._save_analysis_column_widths)
            
            # 只在第一次设置列宽，之后保留用户调整
            if not hasattr(self, '_analysis_columns_initialized'):
                self._set_default_analysis_column_widths()
                self._analysis_columns_initialized = True
            
            # 恢复保存的列宽和展开状态（延迟执行）
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(50, self._restore_analysis_column_widths)
            QTimer.singleShot(100, self._restore_analysis_tree_state)
            
            # 恢复状态筛选
            self._apply_current_status_filter()
            
            # 更新计数
            self._analysis_total_comics = total_comics
            self.apply_analysis_filter()
            
            # 提示加载成功
            timestamp = data.get('timestamp', '')
            total_chapters = sum(len(d.get('chapters', [])) + len(d.get('cbz_files', [])) for d in data['comics'].values())
            self.result_text.append(f"↻ 已加载上次分析结果 ({timestamp})")
            self.result_text.append(f"   {total_comics} 部漫画，共 {total_chapters} 个章节")
            
        except Exception as e:
            print(f"加载上次分析结果失败: {e}")
    
    def _sort_chapters(self, ascending: bool = True):
        """按章节编号排序"""
        if not hasattr(self, 'analysis_proxy') or self.analysis_proxy is None:
            return
        
        try:
            import re
            source_model = self.analysis_proxy.sourceModel()
            if source_model is None:
                return
            
            # 遍历每个漫画的子项并排序
            for row in range(source_model.rowCount()):
                parent_item = source_model.item(row, 0)
                if parent_item is None:
                    continue
                
                # 获取所有子项的数据
                children_data = []
                for i in range(parent_item.rowCount()):
                    row_data = {
                        'name': parent_item.child(i, 0).text(),
                        'path': parent_item.child(i, 0).data(Qt.ItemDataRole.UserRole),
                        'type': parent_item.child(i, 2).text() if parent_item.child(i, 2) else ""
                    }
                    children_data.append(row_data)
                
                # 按章节号排序
                def extract_number(data):
                    name = data['name']
                    match = re.search(r'(\d+\.?\d*)', name)
                    return float(match.group(1)) if match else 0
                
                children_data.sort(key=extract_number, reverse=not ascending)
                
                # 移除所有子项
                parent_item.removeRows(0, parent_item.rowCount())
                
                # 重新添加排序后的子项
                for data in children_data:
                    r0 = QStandardItem(data['name'])
                    r0.setData(data['path'], Qt.ItemDataRole.UserRole)
                    r1 = QStandardItem("")  # raw name
                    r2 = QStandardItem(data['type'])
                    r3 = QStandardItem("")
                    r4 = QStandardItem("")
                    for it in (r0, r1, r2, r3, r4):
                        it.setEditable(False)
                    parent_item.appendRow([r0, r1, r2, r3, r4])
            
            # 刷新视图
            self.analysis_view.viewport().update()
            self.result_text.append(f"✓ 章节已按{'升' if ascending else '降'}序重新排列")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"排序失败: {str(e)}")
    
    def _export_analysis_report(self):
        """导出分析报告为CSV文件"""
        if not hasattr(self, 'analysis_proxy') or self.analysis_proxy is None:
            QMessageBox.warning(self, "提示", "请先进行章节分析")
            return
        
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            default_name = f"chapter_analysis_{timestamp}.csv"
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "导出报告", default_name,
                "CSV文件 (*.csv);;所有文件 (*.*)"
            )
            
            if not file_path:
                return
            
            source_model = self.analysis_proxy.sourceModel()
            with open(file_path, 'w', encoding='utf-8-sig', newline='') as f:
                import csv
                writer = csv.writer(f)
                writer.writerow(['漫画名', '生肉名称', '状态', '章节名', '最新章节', 'CBZ章节数'])
                
                for row in range(source_model.rowCount()):
                    comic_name = source_model.item(row, 0).text()
                    raw_name = source_model.item(row, 1).text() if source_model.item(row, 1) else ""
                    status = source_model.item(row, 2).text() if source_model.item(row, 2) else ""
                    latest = source_model.item(row, 3).text()
                    cbz_count = source_model.item(row, 4).text()
                    
                    parent_item = source_model.item(row, 0)
                    for i in range(parent_item.rowCount()):
                        chapter_name = parent_item.child(i, 0).text()
                        writer.writerow([comic_name, raw_name, status, chapter_name, latest, cbz_count])
            
            self.result_text.append(f"✓ 报告已导出: {file_path}")
            QMessageBox.information(self, "成功", f"报告已导出至\n{file_path}")
            
        except Exception as e:
            QMessageBox.warning(self, "错误", f"导出失败: {str(e)}")
    
    def _get_latest_chapter(self, chapters: list) -> str:
        """获取最新章节"""
        import re
        def extract_number(name):
            match = re.search(r'(\d+\.?\d*)', name)
            return float(match.group(1)) if match else 0
        sorted_chapters = sorted(chapters, key=extract_number)
        return sorted_chapters[-1] if sorted_chapters else "未知"
    
    # ============ 完结/连载 状态管理 ============
    _STATUS_FILE = Path(__file__).parent.parent.parent / "comic_status.json"
    
    def _load_comic_status(self) -> dict:
        """Load comic completion status from JSON file"""
        try:
            if self._STATUS_FILE.exists():
                with open(self._STATUS_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            print(f"加载漫画状态失败: {e}")
        return {}
    
    def _save_comic_status(self, status: dict):
        """Save comic completion status to JSON file"""
        try:
            with open(self._STATUS_FILE, 'w', encoding='utf-8') as f:
                json.dump(status, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存漫画状态失败: {e}")
    
    def _toggle_comic_status(self, proxy_index):
        """切换漫画的完结/连载状态"""
        if not proxy_index.isValid():
            return
        src_idx = self.analysis_proxy.mapToSource(proxy_index)
        if src_idx.parent().isValid():
            return  # Only toggle parent (manga) rows
        
        model = src_idx.model()
        row = src_idx.row()
        
        # Get comic name and current status
        comic_name = model.item(row, 0).text()
        status_item = model.item(row, 2)
        current = status_item.text() if status_item else "连载"
        new_status = "完结" if current == "连载" else "连载"
        
        # Update model items
        status_item.setText(new_status)
        row_items = [model.item(row, col) for col in range(model.columnCount())]
        self._apply_status_style(row_items, new_status)
        
        # Persist
        all_status = self._load_comic_status()
        if new_status == "连载":
            all_status.pop(comic_name, None)  # Default is 连载, remove entry
        else:
            all_status[comic_name] = new_status
        self._save_comic_status(all_status)
    
    def _apply_status_style(self, row_items, status: str):
        """根据状态设置行的视觉样式（完结=灰色）"""
        from PyQt6.QtGui import QColor, QBrush
        if status == "完结":
            color = QBrush(QColor("#999999"))
        else:
            color = QBrush(QColor("#000000"))
        for item in row_items:
            if item:
                item.setForeground(color)
