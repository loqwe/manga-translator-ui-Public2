#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
名称映射编辑对话框（PyQt6 原生组件）
- 每个映射都有专门的编辑对话框
- 每个生肉名都有独立输入框（可增删）
- 不需要分隔符，简单直观
"""
from typing import List, Optional, Tuple
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QWidget, QScrollArea, QDialogButtonBox, QMessageBox, QApplication
)


class NameMappingDialog(QDialog):
    """名称映射编辑对话框"""

    def __init__(self, parent=None, trans_name: str = "", raw_list: Optional[List[str]] = None):
        super().__init__(parent)
        self.setWindowTitle("名称映射")
        self.setModal(True)
        self.resize(520, 520)

        self._raw_rows: List[QWidget] = []

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 标题
        title = QLabel("编辑名称映射" if trans_name else "新建名称映射")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)

        # 熟肉名输入
        main_layout.addWidget(QLabel("熟肉名称（中文名）:"))
        self.trans_edit = QLineEdit(self)
        self.trans_edit.setPlaceholderText("例如：青春猪头少年")
        self.trans_edit.setText(trans_name)
        main_layout.addWidget(self.trans_edit)

        # 生肉名列表（可滚动）
        raw_label_row = QHBoxLayout()
        raw_label_row.addWidget(QLabel("生肉名称（可添加多个）："))
        raw_label_row.addStretch()
        self.add_raw_btn = QPushButton("新增生肉名")
        self.add_raw_btn.clicked.connect(lambda: self.add_raw_field(""))
        raw_label_row.addWidget(self.add_raw_btn)
        self.paste_btn = QPushButton("从剪贴板粘贴")
        self.paste_btn.clicked.connect(self.add_raws_from_clipboard)
        raw_label_row.addWidget(self.paste_btn)
        self.clear_btn = QPushButton("清空全部")
        self.clear_btn.clicked.connect(self.clear_all_raws)
        raw_label_row.addWidget(self.clear_btn)
        main_layout.addLayout(raw_label_row)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        main_layout.addWidget(scroll, 1)

        self.raw_container = QWidget()
        self.raw_layout = QVBoxLayout(self.raw_container)
        self.raw_layout.setContentsMargins(0, 0, 0, 0)
        self.raw_layout.setSpacing(8)
        scroll.setWidget(self.raw_container)

        # 初始化生肉名行
        initial_list = raw_list or [""]
        if not initial_list:
            initial_list = [""]
        for name in initial_list:
            self.add_raw_field(name)

        # 确认/取消按钮
        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, parent=self)
        btn_box.accepted.connect(self._on_accept)
        btn_box.rejected.connect(self.reject)
        main_layout.addWidget(btn_box)
        # 恢复对话框几何
        self._restore_geometry()

    def add_raw_field(self, text: str = ""):
        """新增一个生肉名输入行"""
        row_widget = QWidget(self.raw_container)
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)

        line_edit = QLineEdit(row_widget)
        line_edit.setPlaceholderText("例如：Seishun Buta Yarou / かぐや様 / Solo Leveling")
        line_edit.setText(text)
        row_layout.addWidget(line_edit, 1)

        remove_btn = QPushButton("删除", row_widget)
        remove_btn.clicked.connect(lambda: self._remove_row(row_widget))
        row_layout.addWidget(remove_btn)

        self.raw_layout.addWidget(row_widget)
        self._raw_rows.append(row_widget)

    def _remove_row(self, row_widget: QWidget):
        """删除一行生肉名输入"""
        if row_widget in self._raw_rows:
            self._raw_rows.remove(row_widget)
        row_widget.setParent(None)
        row_widget.deleteLater()

        # 保证至少有一行
        if not self._raw_rows:
            self.add_raw_field("")

    def add_raws_from_clipboard(self):
        """从剪贴板按行添加多个生肉名"""
        text = QApplication.clipboard().text() or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            QMessageBox.information(self, "剪贴板为空", "剪贴板没有可用的文本。")
            return
        # 优先填充现有的空行
        empties: List[QLineEdit] = []
        for w in self._raw_rows:
            le = w.findChild(QLineEdit)
            if le and not le.text().strip():
                empties.append(le)
        idx = 0
        for le in empties:
            if idx >= len(lines):
                break
            le.setText(lines[idx])
            idx += 1
        # 其余行创建新输入框
        for i in range(idx, len(lines)):
            self.add_raw_field(lines[i])

    def clear_all_raws(self):
        """清空所有生肉名输入，保留一空行"""
        for w in list(self._raw_rows):
            self._remove_row(w)
        if not self._raw_rows:
            self.add_raw_field("")

    def _on_accept(self):
        """校验并关闭对话框"""
        trans, raws = self.get_mapping()
        if not trans:
            QMessageBox.warning(self, "输入错误", "请填写熟肉名称")
            return
        if not raws:
            QMessageBox.warning(self, "输入错误", "请至少添加一个生肉名称")
            return
        self._save_geometry()
        self.accept()

    # ----- Settings for dialog geometry -----
    def _settings(self) -> QSettings:
        return QSettings("MangaTranslator", "CustomPanel")

    def _restore_geometry(self):
        try:
            s = self._settings()
            data = s.value("dialog/name_mapping_geometry")
            if data:
                self.restoreGeometry(data)
        except Exception:
            pass

    def _save_geometry(self):
        try:
            s = self._settings()
            s.setValue("dialog/name_mapping_geometry", self.saveGeometry())
        except Exception:
            pass

    def reject(self) -> None:
        self._save_geometry()
        return super().reject()

    def closeEvent(self, event):
        self._save_geometry()
        return super().closeEvent(event)

    def get_mapping(self) -> Tuple[str, List[str]]:
        """返回 (熟肉名, 生肉名列表)"""
        trans_name = self.trans_edit.text().strip()
        raw_names: List[str] = []
        for w in self._raw_rows:
            # 第一个子控件是 QLineEdit
            le = w.findChild(QLineEdit)
            if not le:
                continue
            t = le.text().strip()
            if t:
                raw_names.append(t)
        # 去重，保持相对顺序
        seen = set()
        unique_raws = []
        for r in raw_names:
            if r not in seen:
                seen.add(r)
                unique_raws.append(r)
        return trans_name, unique_raws
