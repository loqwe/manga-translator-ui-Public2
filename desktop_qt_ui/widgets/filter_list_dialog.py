#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
过滤列表编辑对话框
- 编辑 watermark_filter.json 文件
- 支持部分匹配和完整匹配两种模式
"""
import json
import os
from typing import List, Optional
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QWidget, QDialogButtonBox, QMessageBox, QApplication,
    QCheckBox, QListWidget, QListWidgetItem, QInputDialog,
    QTreeView, QHeaderView, QAbstractItemView, QMenu
)


class FilterListDialog(QDialog):
    """过滤列表编辑对话框"""

    def __init__(self, parent=None, config_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("过滤列表管理")
        self.setModal(True)
        self.resize(820, 580)
        
        self.config_path = config_path

        # 分组数据模型
        self._groups: List[dict] = []
        self._current_group_index: int = -1

        # 右侧规则数据模型（当前分组）
        self._rules_model: Optional[QStandardItemModel] = None

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 标题
        title = QLabel("编辑水印/广告过滤规则")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 说明文本
        info_label = QLabel(
            "匹配的文本区域会被跳过（不翻译、不擦除、不渲染）\n"
            "• 支持字符串匹配、正则表达式（勾选“正则”）\n"
            "• 默认不区分大小写，勾选“Aa”后区分大小写"
        )
        info_label.setStyleSheet("color: #666; font-size: 12px; padding: 8px; background: #f5f5f5; border-radius: 4px;")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # 中间区域：左侧分组 + 右侧规则编辑
        center = QWidget()
        center_layout = QHBoxLayout(center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(12)
        main_layout.addWidget(center, 1)

        # 左侧：分组列表
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        left_layout.addWidget(QLabel("分组："))
        self.group_list = QListWidget()
        self.group_list.currentRowChanged.connect(self._on_group_changed)
        left_layout.addWidget(self.group_list, 1)

        group_btn_row = QHBoxLayout()
        self.group_add_btn = QPushButton("新增")
        self.group_add_btn.clicked.connect(self._add_group)
        self.group_rename_btn = QPushButton("重命名")
        self.group_rename_btn.clicked.connect(self._rename_group)
        self.group_del_btn = QPushButton("删除")
        self.group_del_btn.clicked.connect(self._delete_group)
        group_btn_row.addWidget(self.group_add_btn)
        group_btn_row.addWidget(self.group_rename_btn)
        group_btn_row.addWidget(self.group_del_btn)
        left_layout.addLayout(group_btn_row)

        left_panel.setFixedWidth(180)
        center_layout.addWidget(left_panel)

        # 右侧：规则编辑区（编辑当前分组）
        right_panel = QWidget()
        center_layout.addWidget(right_panel, 1)
        self._setup_rules_panel(right_panel)

        # 加载现有数据（含旧格式迁移）
        self._load_groups_from_config()
        self._refresh_group_list(select_index=0)

        # 确认/取消按钮
        btn_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self
        )
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)
        
        # 恢复对话框几何
        self._restore_geometry()

    def _setup_rules_panel(self, parent: QWidget):
        """设置右侧规则编辑面板"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel("过滤规则："))
        btn_row.addStretch()
        
        add_btn = QPushButton("新增")
        add_btn.clicked.connect(lambda: self._add_rule_field(""))
        btn_row.addWidget(add_btn)

        del_btn = QPushButton("删除选中")
        del_btn.clicked.connect(self._delete_selected_rules)
        btn_row.addWidget(del_btn)
        
        paste_btn = QPushButton("从剪贴板粘贴")
        paste_btn.clicked.connect(self._add_from_clipboard)
        btn_row.addWidget(paste_btn)
        
        clear_btn = QPushButton("清空全部")
        clear_btn.clicked.connect(self._clear_all_rules)
        btn_row.addWidget(clear_btn)
        
        layout.addLayout(btn_row)

        # 规则表格（QTreeView 更工整）
        self.rules_view = QTreeView()
        self.rules_view.setRootIsDecorated(False)
        self.rules_view.setIndentation(0)
        self.rules_view.setAlternatingRowColors(True)
        self.rules_view.setUniformRowHeights(True)
        self.rules_view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.rules_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.rules_view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed |
            QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        self.rules_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.rules_view.customContextMenuRequested.connect(self._show_rules_context_menu)
        # 调淡选中/悬停行颜色
        self.rules_view.setStyleSheet("""
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
        layout.addWidget(self.rules_view, 1)

        self._rules_model = QStandardItemModel(0, 3, self)
        self._rules_model.setHorizontalHeaderLabels(["规则", "正则", "Aa"])
        self.rules_view.setModel(self._rules_model)

        header = self.rules_view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 60)
        header.resizeSection(2, 60)

    def _add_rule_field(self, text: str = ""):
        """新增一条规则（QTreeView 行）"""
        if self._rules_model is None:
            return

        # 兼容：text 可能是 str 或 dict
        pattern = ""
        use_regex = False
        case_sensitive = False
        if isinstance(text, dict):
            pattern = str(text.get('pattern', '') or '')
            use_regex = bool(text.get('regex', False))
            case_sensitive = bool(text.get('case_sensitive', False))
        else:
            pattern = str(text or "")

        item_pattern = QStandardItem(pattern)
        item_pattern.setEditable(True)

        item_regex = QStandardItem("")
        item_regex.setEditable(False)
        item_regex.setCheckable(True)
        item_regex.setCheckState(Qt.CheckState.Checked if use_regex else Qt.CheckState.Unchecked)

        item_case = QStandardItem("")
        item_case.setEditable(False)
        item_case.setCheckable(True)
        item_case.setCheckState(Qt.CheckState.Checked if case_sensitive else Qt.CheckState.Unchecked)

        self._rules_model.appendRow([item_pattern, item_regex, item_case])

    def _delete_selected_rules(self):
        """删除选中的规则行"""
        if self._rules_model is None or not hasattr(self, 'rules_view'):
            return
        sel = self.rules_view.selectionModel()
        if sel is None:
            return
        rows = sel.selectedRows()
        if not rows:
            return
        for idx in sorted(rows, key=lambda x: x.row(), reverse=True):
            self._rules_model.removeRow(idx.row())
        if self._rules_model.rowCount() == 0:
            self._add_rule_field("")

    def _show_rules_context_menu(self, pos):
        """右键菜单"""
        menu = QMenu(self)
        act_del = menu.addAction("删除选中")
        act = menu.exec(self.rules_view.viewport().mapToGlobal(pos))
        if act == act_del:
            self._delete_selected_rules()

    def _add_from_clipboard(self):
        """从剪贴板按行添加多个规则"""
        if self._rules_model is None:
            return
        text = QApplication.clipboard().text() or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            QMessageBox.information(self, "剪贴板为空", "剪贴板没有可用的文本。")
            return

        # 优先填充现有空行
        idx = 0
        for r in range(self._rules_model.rowCount()):
            if idx >= len(lines):
                break
            it = self._rules_model.item(r, 0)
            if it is not None and not (it.text() or "").strip():
                it.setText(lines[idx])
                idx += 1

        # 其余行追加
        for i in range(idx, len(lines)):
            self._add_rule_field(lines[i])

    def _clear_all_rules(self):
        """清空所有规则"""
        if self._rules_model is None:
            return
        self._rules_model.removeRows(0, self._rules_model.rowCount())
        self._add_rule_field("")

    def _is_category_comment(self, text: str) -> bool:
        """检查是否是类别注释（旧格式：以 ===== 开头和结尾）"""
        if isinstance(text, str):
            stripped = text.strip()
            return stripped.startswith("=====") and stripped.endswith("=====")
        return False

    def _parse_category_name(self, text: str) -> str:
        """从 ===== xxx ===== 提取分组名"""
        s = (text or "").strip()
        if not self._is_category_comment(s):
            return ""
        s = s.strip("=").strip()
        return s or "未命名"

    def _ensure_min_group(self):
        if not self._groups:
            self._groups = [{
                'name': '默认',
                'patterns': [],
            }]

    def _load_groups_from_config(self):
        """从配置文件加载分组（兼容旧格式并做一次性迁移）"""
        self._groups = []

        if not self.config_path or not os.path.exists(self.config_path):
            self._ensure_min_group()
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # 新格式：groups
            if isinstance(config, dict) and isinstance(config.get('groups'), list):
                for g in config.get('groups'):
                    if not isinstance(g, dict):
                        continue
                    name = str(g.get('name', '')).strip() or '未命名'
                    # 合并 partial + exact 为统一 patterns
                    merged_patterns = []
                    for p in (g.get('partial_match_patterns', []) or []):
                        merged_patterns.append(p)
                    for p in (g.get('exact_match_patterns', []) or []):
                        merged_patterns.append(p)
                    self._groups.append({
                        'name': name,
                        'patterns': merged_patterns,
                    })
                self._ensure_min_group()
                return

            # 旧格式：平铺数组 + 类别注释行
            partial_patterns = []
            exact_patterns = []
            if isinstance(config, dict):
                partial_patterns = config.get('partial_match_patterns', []) or []
                exact_patterns = config.get('exact_match_patterns', []) or []

            current_group = {
                'name': '默认',
                'patterns': [],
            }
            have_any_group = False

            for item in partial_patterns:
                if isinstance(item, str) and self._is_category_comment(item):
                    # 切换分组
                    if current_group['patterns'] or have_any_group:
                        self._groups.append(current_group)
                        have_any_group = True
                    current_group = {
                        'name': self._parse_category_name(item),
                        'patterns': [],
                    }
                    continue
                current_group['patterns'].append(item)

            # 收尾
            if current_group['patterns'] or not have_any_group:
                self._groups.append(current_group)

            # 旧格式的 exact 统一放入“默认”分组
            if exact_patterns:
                self._groups[0]['patterns'].extend(exact_patterns)

            self._ensure_min_group()

        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载配置文件：{e}")
            self._ensure_min_group()

    def _refresh_group_list(self, select_index: int = 0):
        self.group_list.blockSignals(True)
        self.group_list.clear()
        for g in self._groups:
            item = QListWidgetItem(str(g.get('name', '未命名')))
            self.group_list.addItem(item)
        self.group_list.blockSignals(False)

        if self.group_list.count() > 0:
            select_index = max(0, min(select_index, self.group_list.count() - 1))
            self.group_list.setCurrentRow(select_index)
        else:
            self._current_group_index = -1


    def _save_current_group_to_model(self):
        if self._current_group_index < 0 or self._current_group_index >= len(self._groups):
            return
        g = self._groups[self._current_group_index]
        g['patterns'] = self._get_patterns()

    def _load_group_to_ui(self, group_index: int):
        if self._rules_model is None:
            return

        # 清空当前表格
        self._rules_model.removeRows(0, self._rules_model.rowCount())

        if group_index < 0 or group_index >= len(self._groups):
            self._add_rule_field("")
            return

        g = self._groups[group_index]
        patterns = g.get('patterns', []) or []

        if patterns:
            for pattern in patterns:
                self._add_rule_field(pattern)
        else:
            self._add_rule_field("")

    def _on_group_changed(self, new_index: int):
        # 先保存当前分组编辑内容
        if self._current_group_index != -1:
            self._save_current_group_to_model()

        self._current_group_index = new_index
        self._load_group_to_ui(new_index)

    def _add_group(self):
        name, ok = QInputDialog.getText(self, "新增分组", "分组名称：")
        if not ok:
            return
        name = (name or "").strip() or "未命名"
        self._groups.append({
            'name': name,
            'patterns': [],
        })
        self._refresh_group_list(select_index=len(self._groups) - 1)

    def _rename_group(self):
        idx = self.group_list.currentRow()
        if idx < 0 or idx >= len(self._groups):
            return
        current_name = str(self._groups[idx].get('name', ''))
        name, ok = QInputDialog.getText(self, "重命名分组", "分组名称：", text=current_name)
        if not ok:
            return
        name = (name or "").strip() or "未命名"
        self._groups[idx]['name'] = name
        self._refresh_group_list(select_index=idx)

    def _delete_group(self):
        idx = self.group_list.currentRow()
        if idx < 0 or idx >= len(self._groups):
            return
        if len(self._groups) <= 1:
            QMessageBox.information(self, "无法删除", "至少需要保留一个分组。")
            return
        name = str(self._groups[idx].get('name', '未命名'))
        ret = QMessageBox.question(self, "确认删除", f"确定要删除分组：{name} ？")
        if ret != QMessageBox.StandardButton.Yes:
            return

        # 如果删除的是当前分组，先保存一次当前内容避免丢失（用户可取消）
        if idx == self._current_group_index:
            self._save_current_group_to_model()

        del self._groups[idx]
        next_idx = min(idx, len(self._groups) - 1)
        self._refresh_group_list(select_index=next_idx)

    def _on_accept(self):
        """保存配置并关闭对话框"""
        # 先把当前分组内容回写到模型
        self._save_current_group_to_model()

        # 构建新格式配置数据
        config = {
            "version": 2,
            "groups": []
        }
        for g in self._groups:
            patterns = g.get('patterns', []) or []
            # 为兼容后端，保存时仍然写回 partial + exact 分开
            config["groups"].append({
                "name": str(g.get('name', '未命名')),
                "partial_match_patterns": patterns,
                "exact_match_patterns": [],
            })

        # 保存到文件
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            
            QMessageBox.information(self, "保存成功", "过滤规则已保存。")
            self._save_geometry()
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"无法保存配置文件：{e}")

    def _get_patterns(self) -> List[str]:
        """获取当前分组的所有规则"""
        if self._rules_model is None:
            return []

        patterns = []
        for r in range(self._rules_model.rowCount()):
            it_pattern = self._rules_model.item(r, 0)
            if it_pattern is None:
                continue
            t = (it_pattern.text() or "").strip()
            if not t:
                continue

            it_regex = self._rules_model.item(r, 1)
            it_case = self._rules_model.item(r, 2)
            use_regex = bool(it_regex.checkState() == Qt.CheckState.Checked) if it_regex is not None else False
            case_sensitive = bool(it_case.checkState() == Qt.CheckState.Checked) if it_case is not None else False

            if use_regex or case_sensitive:
                patterns.append({
                    'pattern': t,
                    'regex': use_regex,
                    'case_sensitive': case_sensitive,
                })
            else:
                patterns.append(t)

        # 去重，保持相对顺序
        seen = set()
        unique = []
        for p in patterns:
            key = json.dumps(p, ensure_ascii=False, sort_keys=True) if isinstance(p, dict) else p
            if key not in seen:
                seen.add(key)
                unique.append(p)
        return unique

    # ----- Settings for dialog geometry -----
    def _settings(self) -> QSettings:
        return QSettings("MangaTranslator", "FilterList")

    def _restore_geometry(self):
        try:
            s = self._settings()
            data = s.value("dialog/filter_list_geometry")
            if data:
                self.restoreGeometry(data)
        except Exception:
            pass

    def _save_geometry(self):
        try:
            s = self._settings()
            s.setValue("dialog/filter_list_geometry", self.saveGeometry())
        except Exception:
            pass

    def reject(self) -> None:
        self._save_geometry()
        return super().reject()

    def closeEvent(self, event):
        self._save_geometry()
        return super().closeEvent(event)
