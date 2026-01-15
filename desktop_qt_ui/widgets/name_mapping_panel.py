#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称映射管理面板（独立标签页）
从 CustomComicPanel 中抽离，作为独立的标签页显示
"""

import os
from pathlib import Path
from typing import Optional, List
from PyQt6.QtCore import Qt, QSettings, QSortFilterProxyModel
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLineEdit, QTextEdit, QLabel, QTreeView,
    QHeaderView, QAbstractItemView, QMessageBox
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem

try:
    from utils.name_replacer import NameReplacer
except ImportError:
    NameReplacer = None

from widgets.name_mapping_dialog import NameMappingDialog


class MappingFilterProxy(QSortFilterProxyModel):
    """多列、递归过滤代理：匹配父或任一子项均显示父项"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self._kw = ""

    def setFilterKeyword(self, text: str):
        self._kw = (text or "").strip().lower()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if not self._kw:
            return True
        model = self.sourceModel()
        idx0 = model.index(source_row, 0, source_parent)
        idx1 = model.index(source_row, 1, source_parent)
        def _t(idx):
            v = model.data(idx, Qt.ItemDataRole.DisplayRole)
            return (v or "").lower()
        # 当前行匹配
        if self._kw in _t(idx0) or self._kw in _t(idx1):
            return True
        # 子项匹配则保留父项
        child_count = model.rowCount(idx0)
        for i in range(child_count):
            if self.filterAcceptsRow(i, idx0):
                return True
        return False


class NameMappingPanel(QWidget):
    """名称映射管理面板"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.replacer = NameReplacer() if NameReplacer else None
        self.mapping_groups = {}
        self.mapping_proxy = None
        self._total_mapping_rows = 0
        self._header_connected = False
        self._init_ui()
    
    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # 名称映射管理区域
        group = QGroupBox("名称映射管理")
        layout = QVBoxLayout(group)

        # 顶部筛选与计数
        search_layout = QHBoxLayout()
        search_layout.setContentsMargins(0, 0, 0, 0)
        search_layout.setSpacing(8)
        search_layout.addWidget(QLabel("搜索:"))
        self.mapping_search_edit = QLineEdit()
        self.mapping_search_edit.setPlaceholderText("输入关键字过滤生肉/熟肉名称…")
        self.mapping_search_edit.textChanged.connect(self.apply_mapping_filter)
        self.mapping_search_edit.textChanged.connect(lambda: self._save_mapping_search())
        search_layout.addWidget(self.mapping_search_edit, 1)
        self.mapping_count_label = QLabel("显示 0 / 总 0")
        search_layout.addWidget(self.mapping_count_label)
        layout.addLayout(search_layout)

        # 分组树视图
        self.mapping_view = QTreeView()
        self.mapping_view.setRootIsDecorated(True)
        self.mapping_view.setAlternatingRowColors(True)
        self.mapping_view.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.mapping_view.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.mapping_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.mapping_view.doubleClicked.connect(self.edit_mapping)
        self.mapping_view.expanded.connect(self._on_tree_expanded)
        self.mapping_view.collapsed.connect(self._on_tree_collapsed)
        # 调淡选中/悬停行颜色
        self.mapping_view.setStyleSheet("""
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
        layout.addWidget(self.mapping_view)

        # 操作按钮
        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        add_btn = QPushButton("添加映射")
        add_btn.clicked.connect(self.add_mapping)
        edit_btn = QPushButton("编辑")
        edit_btn.clicked.connect(lambda: self.edit_mapping())
        delete_btn = QPushButton("删除映射")
        delete_btn.clicked.connect(self.delete_mapping)
        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self.load_mappings)

        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(edit_btn)
        btn_layout.addWidget(delete_btn)
        btn_layout.addWidget(refresh_btn)
        layout.addLayout(btn_layout)

        main_layout.addWidget(group, 1)  # 拉伸填充空间

        self.load_mappings()
    
    def showEvent(self, event):
        """首次显示时恢复设置"""
        super().showEvent(event)
        if not hasattr(self, '_settings_restored'):
            self._settings_restored = True
            self._restore_settings()

    # ============ Settings persistence ============
    def _settings(self) -> QSettings:
        return QSettings("MangaTranslator", "NameMappingPanel")

    def _restore_settings(self):
        s = self._settings()
        # 搜索框
        self.mapping_search_edit.blockSignals(True)
        self.mapping_search_edit.setText(s.value("mapping/search", ""))
        self.mapping_search_edit.blockSignals(False)
        # 应用筛选
        self.apply_mapping_filter()

    def _save_mapping_search(self):
        s = self._settings()
        s.setValue("mapping/search", self.mapping_search_edit.text())

    def _expanded_set(self) -> set:
        s = self._settings()
        items = s.value("mapping/expanded", [])
        try:
            return set(items) if isinstance(items, list) else set()
        except Exception:
            return set()

    def _save_expanded_set(self, expanded: set):
        s = self._settings()
        s.setValue("mapping/expanded", list(expanded))

    def _on_tree_expanded(self, proxy_index):
        try:
            src_idx = self.mapping_proxy.mapToSource(proxy_index)
            model = src_idx.model()
            col0 = model.index(src_idx.row(), 0, src_idx.parent())
            trans_name = model.data(col0, Qt.ItemDataRole.UserRole)
            if trans_name:
                st = self._expanded_set()
                st.add(trans_name)
                self._save_expanded_set(st)
        except Exception:
            pass

    def _on_tree_collapsed(self, proxy_index):
        try:
            src_idx = self.mapping_proxy.mapToSource(proxy_index)
            model = src_idx.model()
            col0 = model.index(src_idx.row(), 0, src_idx.parent())
            trans_name = model.data(col0, Qt.ItemDataRole.UserRole)
            if trans_name:
                st = self._expanded_set()
                if trans_name in st:
                    st.remove(trans_name)
                    self._save_expanded_set(st)
        except Exception:
            pass

    def _save_column_widths(self):
        header = self.mapping_view.header()
        widths = [header.sectionSize(0), header.sectionSize(1)]
        s = self._settings()
        s.setValue("mapping/column_widths", widths)

    def _restore_column_widths(self):
        s = self._settings()
        widths = s.value("mapping/column_widths")
        if isinstance(widths, list) and len(widths) == 2:
            try:
                header = self.mapping_view.header()
                header.resizeSection(0, int(widths[0]))
                header.resizeSection(1, int(widths[1]))
            except Exception:
                pass

    def _restore_tree_state(self):
        # 展开状态
        expanded = self._expanded_set()
        proxy = self.mapping_proxy
        if proxy is not None:
            top_rows = proxy.rowCount()
            for r in range(top_rows):
                p_idx = proxy.index(r, 0)
                src_idx = proxy.mapToSource(p_idx)
                model = src_idx.model()
                col0 = model.index(src_idx.row(), 0, src_idx.parent())
                trans_name = model.data(col0, Qt.ItemDataRole.UserRole)
                if trans_name in expanded:
                    self.mapping_view.expand(p_idx)
                else:
                    self.mapping_view.collapse(p_idx)
        # 列宽
        self._restore_column_widths()
        # 选中项
        s = self._settings()
        last_trans = s.value("mapping/last_selected_trans", "")
        if last_trans and proxy is not None:
            top_rows = proxy.rowCount()
            for r in range(top_rows):
                p_idx = proxy.index(r, 0)
                src_idx = proxy.mapToSource(p_idx)
                model = src_idx.model()
                col0 = model.index(src_idx.row(), 0, src_idx.parent())
                trans_name = model.data(col0, Qt.ItemDataRole.UserRole)
                if trans_name == last_trans:
                    self.mapping_view.setCurrentIndex(p_idx)
                    self.mapping_view.scrollTo(p_idx)
                    break

    def _on_selection_changed(self, current, previous):
        try:
            proxy = self.mapping_proxy
            if proxy is None or not current.isValid():
                return
            src_idx = proxy.mapToSource(current)
            model = src_idx.model()
            parent_idx = src_idx.parent()
            if parent_idx.isValid():
                trans_name = model.data(model.index(parent_idx.row(), 0, parent_idx.parent()), Qt.ItemDataRole.UserRole)
            else:
                trans_name = model.data(model.index(src_idx.row(), 0, src_idx.parent()), Qt.ItemDataRole.UserRole)
            if trans_name:
                s = self._settings()
                s.setValue("mapping/last_selected_trans", trans_name)
        except Exception:
            pass

    # ============ 映射操作 ============
    def load_mappings(self):
        """加载映射到树（分组：熟肉名 -> 生肉名列表）"""
        if not self.replacer:
            return

        mappings = self.replacer.get_all_mappings()

        # 构建分组 {熟肉名: [生肉名列表]}
        self.mapping_groups = {}
        for raw_name, trans_name in mappings.items():
            if '|' in raw_name:
                variants = [v.strip() for v in raw_name.split('|') if v.strip()]
                self.mapping_groups.setdefault(trans_name, []).extend(variants)
            else:
                self.mapping_groups.setdefault(trans_name, []).append(raw_name)

        # 模型
        model = QStandardItemModel()
        model.setHorizontalHeaderLabels(["生肉名称", "熟肉名称"])

        total_rows = 0
        for trans_name in sorted(self.mapping_groups.keys()):
            raws = sorted(self.mapping_groups[trans_name])
            total_rows += len(raws)
            parent_name_item = QStandardItem(f"{trans_name} ({len(raws)})")
            parent_name_item.setEditable(False)
            parent_name_item.setData(trans_name, Qt.ItemDataRole.UserRole)
            parent_trans_item = QStandardItem("")
            parent_trans_item.setEditable(False)
            model.appendRow([parent_name_item, parent_trans_item])
            for raw in raws:
                raw_item = QStandardItem(raw)
                raw_item.setEditable(False)
                trans_item = QStandardItem(trans_name)
                trans_item.setEditable(False)
                parent_name_item.appendRow([raw_item, trans_item])

        # 代理
        if self.mapping_proxy is None:
            self.mapping_proxy = MappingFilterProxy(self)
        self.mapping_proxy.setSourceModel(model)
        self.mapping_view.setModel(self.mapping_proxy)
        header = self.mapping_view.header()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        if not self._header_connected:
            header.sectionResized.connect(lambda logicalIndex, oldSize, newSize: self._save_column_widths())
            self._header_connected = True
        self._restore_tree_state()

        self._total_mapping_rows = total_rows
        self.apply_mapping_filter()

        sel_model = self.mapping_view.selectionModel()
        if sel_model is not None:
            try:
                sel_model.currentChanged.disconnect(self._on_selection_changed)
            except Exception:
                pass
            sel_model.currentChanged.connect(self._on_selection_changed)

    def apply_mapping_filter(self, text: Optional[str] = None):
        """根据搜索关键字过滤树并更新计数标签"""
        if text is None:
            text = self.mapping_search_edit.text() if hasattr(self, 'mapping_search_edit') else ""
        if self.mapping_proxy is not None:
            self.mapping_proxy.setFilterKeyword(text)
        # 统计可见子项数量
        visible = 0
        proxy = self.mapping_proxy
        if proxy is not None:
            top_rows = proxy.rowCount()
            for r in range(top_rows):
                parent_idx = proxy.index(r, 0)
                visible += proxy.rowCount(parent_idx)
        total = self._total_mapping_rows
        self.mapping_count_label.setText(f"显示 {visible} / 总 {total}")
    
    def add_mapping(self):
        """新建映射"""
        if not self.replacer:
            return
        
        dlg = NameMappingDialog(self)
        if dlg.exec():
            trans_name, raw_list = dlg.get_mapping()
            if not trans_name or not raw_list:
                return
            for raw in raw_list:
                self.replacer.add_mapping(raw, trans_name)
            self.load_mappings()
    
    def delete_mapping(self):
        """删除选中的单条映射（按行删除一个生肉名）"""
        if not self.replacer:
            return
        
        idx = self.mapping_view.currentIndex()
        if not idx.isValid():
            QMessageBox.warning(self, "未选中", "请先选择要删除的项（父项删除整组，子项删除单条）")
            return
        src_idx = self.mapping_proxy.mapToSource(idx)
        model = src_idx.model()
        parent_idx = src_idx.parent()
        if parent_idx.isValid():
            # 删除单条映射（子项）
            raw_name = model.data(src_idx, Qt.ItemDataRole.DisplayRole)
            trans_name = model.data(model.index(parent_idx.row(), 0, parent_idx.parent()), Qt.ItemDataRole.UserRole)
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定删除该映射？\n{raw_name} → {trans_name}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self.replacer.remove_mapping(raw_name)
            self.load_mappings()
        else:
            # 删除整组（父项）
            parent_col0 = model.index(src_idx.row(), 0, src_idx.parent())
            trans_name = model.data(parent_col0, Qt.ItemDataRole.UserRole)
            raw_list = self.mapping_groups.get(trans_name, [])
            reply = QMessageBox.question(
                self, "确认删除",
                f"确定要删除「{trans_name}」的所有映射（{len(raw_list)}个）吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            for raw in raw_list:
                self.replacer.remove_mapping(raw)
            self.load_mappings()

    def edit_mapping(self, index=None):
        """编辑映射"""
        if not self.replacer:
            return
        if isinstance(index, bool):
            index = None
        if index is None:
            idx = self.mapping_view.currentIndex()
        else:
            idx = index
        if not idx.isValid():
            QMessageBox.warning(self, "未选中", "请先选择要编辑的项（双击子项或选择父项）")
            return
        src_idx = self.mapping_proxy.mapToSource(idx)
        model = src_idx.model()
        parent_idx = src_idx.parent()
        if parent_idx.isValid():
            trans_name = model.data(model.index(parent_idx.row(), 0, parent_idx.parent()), Qt.ItemDataRole.UserRole)
        else:
            parent_col0 = model.index(src_idx.row(), 0, src_idx.parent())
            trans_name = model.data(parent_col0, Qt.ItemDataRole.UserRole)
        raw_list = self.mapping_groups.get(trans_name, [])

        dlg = NameMappingDialog(self, trans_name=trans_name, raw_list=raw_list)
        if dlg.exec():
            new_trans, new_raws = dlg.get_mapping()
            if not new_trans or not new_raws:
                return
            # 删除旧映射
            for old_raw in raw_list:
                self.replacer.remove_mapping(old_raw)
            # 添加新映射
            for raw in new_raws:
                self.replacer.add_mapping(raw, new_trans)
            self.load_mappings()
