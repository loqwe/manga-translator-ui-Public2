#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""术语日志对话框（表格式，参考使用日志）。"""

from __future__ import annotations

import json
import os
import sys
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, List, Optional

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    QSettings,
    QByteArray,
)
from PyQt6.QtGui import QColor, QPen, QFontMetrics, QPainter
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QHeaderView,
)


@dataclass(frozen=True)
class GlossaryLogRecord:
    time_str: str
    time_ts: float
    tag: str
    level: str
    work: str
    raw_work: str
    message: str
    logger_name: str
    raw_line: str


def _default_t(key: str, **kwargs) -> str:
    return key


def _to_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, int):
        return val != 0
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ('1', 'true', 'yes', 'y', 'on'):
            return True
        if s in ('0', 'false', 'no', 'n', 'off', ''):
            return False
        # Fallback: any other non-empty string treated as True
        return True
    return bool(val)


def _get_result_log_dir() -> str:
    """获取 UI 写入的 result 日志目录。"""
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), '_internal', 'result')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'result'))


def _get_glossary_db_path() -> str:
    return os.path.join(_get_result_log_dir(), 'glossary_log.db')


def _load_glossary_records(limit: int = 500) -> tuple[List[GlossaryLogRecord], Optional[str]]:
    """从数据库读取术语日志记录。"""
    db_path = _get_glossary_db_path()
    if not os.path.isfile(db_path):
        return [], None

    records: List[GlossaryLogRecord] = []

    try:
        conn = sqlite3.connect(db_path, timeout=1.5)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts, tag, original, translation, category, work, raw_work, level, message, extra_json
            FROM glossary_log
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        for row in rows:
            ts, tag, original, translation, category, work, raw_work, level, message, extra_json = row
            time_str = ts or ''

            dt = None
            try:
                dt = datetime.fromisoformat(time_str)
            except Exception:
                dt = None

            if dt is None:
                try:
                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S,%f')
                except Exception:
                    dt = None

            if dt is None:
                try:
                    dt = datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                except Exception:
                    dt = None

            if dt is not None:
                time_ts = dt.timestamp()
                time_str_disp = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                time_ts = 0.0
                time_str_disp = time_str

            # 构建 raw_line （用于详情查看）
            raw_line = ''
            try:
                if extra_json:
                    raw_line = extra_json
            except Exception:
                raw_line = message or ''

            records.append(
                GlossaryLogRecord(
                    time_str=time_str_disp,
                    time_ts=time_ts,
                    tag=tag or '术语提取',
                    level=level or 'INFO',
                    work=work or '',
                    raw_work=raw_work or '',
                    message=message or '',
                    logger_name='glossary_log_db',
                    raw_line=raw_line,
                )
            )
    except Exception:
        return [], db_path
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return records, db_path


class GlossaryLogTableModel(QAbstractTableModel):
    COL_TIME = 0
    COL_TAG = 1
    COL_WORK = 2
    COL_RAW = 3
    COL_LEVEL = 4
    COL_MESSAGE = 5

    def __init__(self, records: List[GlossaryLogRecord], t: Callable[[str], str] = _default_t, parent=None):
        super().__init__(parent)
        self._records = records
        self._t = t

    def set_records(self, records: List[GlossaryLogRecord]):
        self.beginResetModel()
        self._records = records
        self.endResetModel()

    def record_at(self, row: int) -> Optional[GlossaryLogRecord]:
        if row < 0 or row >= len(self._records):
            return None
        return self._records[row]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else 6

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole or orientation != Qt.Orientation.Horizontal:
            return None
        headers = [
            self._t('col_time'),
            '标签',
            '作品',
            '生肉名',
            '级别',
            '消息',
        ]
        return headers[section] if 0 <= section < len(headers) else None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rec = self._records[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.UserRole:
            return rec

        if role == Qt.ItemDataRole.EditRole:
            if col == self.COL_TIME:
                return rec.time_ts
            if col == self.COL_TAG:
                return rec.tag
            if col == self.COL_WORK:
                return rec.work
            if col == self.COL_RAW:
                return rec.raw_work
            if col == self.COL_LEVEL:
                return rec.level
            if col == self.COL_MESSAGE:
                return rec.message

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (self.COL_TIME, self.COL_LEVEL):
                return int(Qt.AlignmentFlag.AlignCenter)
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        if role == Qt.ItemDataRole.DisplayRole:
            if col == self.COL_TIME:
                return rec.time_str
            if col == self.COL_TAG:
                return rec.tag
            if col == self.COL_WORK:
                return rec.work
            if col == self.COL_RAW:
                return rec.raw_work
            if col == self.COL_LEVEL:
                return rec.level
            if col == self.COL_MESSAGE:
                return rec.message
        return None


class GlossaryLogProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search = ''
        self._tag = 'ALL'
        self._level = 'ALL'
        self._only_errors = False
        self._only_extract = False

    def set_search(self, text: str):
        self._search = (text or '').lower().strip()
        self.invalidateFilter()

    def set_tag_filter(self, tag: str):
        self._tag = tag or 'ALL'
        self.invalidateFilter()

    def set_level_filter(self, lvl: str):
        self._level = lvl or 'ALL'
        self.invalidateFilter()

    def set_only_errors(self, enabled: bool):
        self._only_errors = enabled
        self.invalidateFilter()

    def set_only_extract(self, enabled: bool):
        self._only_extract = enabled
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        src: GlossaryLogTableModel = self.sourceModel()  # type: ignore
        rec = src.record_at(source_row)
        if rec is None:
            return False

        if self._tag != 'ALL' and rec.tag != self._tag:
            return False
        if self._level != 'ALL' and rec.level.lower() != self._level.lower():
            return False
        if self._only_errors and rec.level.lower() != 'error':
            return False
        if self._only_extract and rec.tag not in ('术语提取', 'Glossary'):
            return False

        if not self._search:
            return True
        hay = f"{rec.time_str} {rec.tag} {rec.work} {rec.raw_work} {rec.level} {rec.message}"
        return self._search in hay.lower()


class TagDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option, index: QModelIndex):
        rec = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(rec, GlossaryLogRecord):
            return super().paint(painter, option, index)

        col = index.column()
        if col not in (GlossaryLogTableModel.COL_TAG, GlossaryLogTableModel.COL_LEVEL):
            return super().paint(painter, option, index)

        painter.save()
        opt = QStyleOptionViewItem(option)
        opt.text = ''
        super().paint(painter, opt, index)

        fm = QFontMetrics(painter.font())
        text = rec.tag if col == GlossaryLogTableModel.COL_TAG else rec.level
        bg = QColor('#eef2f7') if col == GlossaryLogTableModel.COL_TAG else QColor('#f1f5f9')
        fg = QColor('#334155')
        padding_x = 8
        padding_y = 4
        w = fm.horizontalAdvance(text) + padding_x * 2
        h = fm.height() + padding_y * 2
        x = option.rect.x() + 8
        y = option.rect.y() + (option.rect.height() - h) // 2

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(x, y, w, h, 8, 8)

        painter.setPen(QPen(fg))
        painter.drawText(x + padding_x, y + padding_y + fm.ascent(), text)
        painter.restore()


class GlossaryLogDialog(QDialog):
    def __init__(self, parent=None, translate: Callable[[str], str] = _default_t):
        super().__init__(parent)
        self._t = translate or _default_t
        self._settings = QSettings("manga-translator", "glossary_log_dialog")

        self.setWindowTitle(self._t('btn_glossary_log'))
        self.setModal(True)
        self.resize(960, 560)

        self._records: List[GlossaryLogRecord] = []
        self._log_path: Optional[str] = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        title = QLabel(self._t('btn_glossary_log'))
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        layout.addWidget(title)

        toolbar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText('搜索（作品/生肉/消息）...')
        toolbar.addWidget(self.search_input, 1)

        self.tag_filter = QComboBox()
        self.tag_filter.addItems(['ALL', '术语', '术语缓存', '术语提取', 'Glossary'])
        toolbar.addWidget(self.tag_filter)

        self.level_filter = QComboBox()
        self.level_filter.addItems(['ALL', 'INFO', 'DEBUG', 'WARNING', 'ERROR'])
        toolbar.addWidget(self.level_filter)

        self.error_only_cb = QCheckBox('仅错误')
        self.extract_only_cb = QCheckBox('仅术语提取')
        toolbar.addWidget(self.error_only_cb)
        toolbar.addWidget(self.extract_only_cb)

        # Density / row height
        self.density_combo = QComboBox()
        self.density_combo.addItems(['紧凑', '标准', '宽松'])
        self.density_combo.setToolTip('调整表格行高')
        toolbar.addWidget(self.density_combo)

        self.refresh_btn = QPushButton(self._t('Refresh'))
        self.auto_refresh_cb = QCheckBox(self._t('usage_auto_refresh'))
        self.auto_refresh_cb.setChecked(True)
        self.copy_btn = QPushButton(self._t('Copy'))
        self.export_btn = QPushButton('导出JSON')
        self.save_btn = QPushButton('保存JSON')
        self.close_btn = QPushButton(self._t('Close'))
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.auto_refresh_cb)
        toolbar.addWidget(self.copy_btn)
        toolbar.addWidget(self.export_btn)
        toolbar.addWidget(self.save_btn)
        toolbar.addWidget(self.close_btn)
        layout.addLayout(toolbar)

        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: #666; font-size: 12px;')
        layout.addWidget(self.status_label)

        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(30)

        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        # Default widths (user can drag to adjust; we also persist header state)
        header.resizeSection(GlossaryLogTableModel.COL_TIME, 92)
        header.resizeSection(GlossaryLogTableModel.COL_TAG, 92)
        header.resizeSection(GlossaryLogTableModel.COL_WORK, 120)
        header.resizeSection(GlossaryLogTableModel.COL_RAW, 120)
        header.resizeSection(GlossaryLogTableModel.COL_LEVEL, 78)

        self.table.setShowGrid(False)
        mono_css = "font-family: 'Cascadia Mono', 'Consolas', 'Roboto Mono', monospace;"
        self.table.setStyleSheet(self.table.styleSheet() + f" QTableView {{ {mono_css} }}")

        layout.addWidget(self.table, 1)

        self.model = GlossaryLogTableModel([], t=self._t, parent=self)
        self.proxy = GlossaryLogProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.EditRole)
        self.table.setModel(self.proxy)

        delegate = TagDelegate(self.table)
        self.table.setItemDelegateForColumn(GlossaryLogTableModel.COL_TAG, delegate)
        self.table.setItemDelegateForColumn(GlossaryLogTableModel.COL_LEVEL, delegate)

        # timers & signals
        self._refresh_guard = False
        self._last_snapshot_key = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(2000)
        self._auto_timer.timeout.connect(self._on_auto_refresh)

        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        self.auto_refresh_cb.toggled.connect(self._set_auto_refresh)
        self.copy_btn.clicked.connect(self.copy_selected)
        self.export_btn.clicked.connect(self.export_json)
        self.save_btn.clicked.connect(self.save_json)
        self.close_btn.clicked.connect(self.accept)
        self.search_input.textChanged.connect(self.proxy.set_search)
        self.tag_filter.currentTextChanged.connect(self.proxy.set_tag_filter)
        self.level_filter.currentTextChanged.connect(self.proxy.set_level_filter)
        self.error_only_cb.toggled.connect(self.proxy.set_only_errors)
        self.extract_only_cb.toggled.connect(self.proxy.set_only_extract)
        self.density_combo.currentTextChanged.connect(self._apply_density)
        self.table.doubleClicked.connect(self._show_row_detail)

        self._load_settings()
        self.refresh(force=True)
        self._set_auto_refresh(self.auto_refresh_cb.isChecked())

    def _snapshot_key(self, records: List[GlossaryLogRecord]) -> str:
        if not records:
            return '0'
        head = records[0]
        return f"{len(records)}|{head.time_str}|{head.tag}|{head.level}|{head.message[:30]}"

    def _apply_density(self, label: str):
        # Row height presets
        mapping = {
            '紧凑': 24,
            '标准': 30,
            '宽松': 38,
        }
        h = mapping.get(label, 30)
        try:
            self.table.verticalHeader().setDefaultSectionSize(h)
        except Exception:
            pass

    def _get_view_records(self) -> List[GlossaryLogRecord]:
        # Export current filtered view
        out: List[GlossaryLogRecord] = []
        try:
            for row in range(self.proxy.rowCount()):
                idx = self.proxy.index(row, 0)
                rec = idx.data(Qt.ItemDataRole.UserRole)
                if isinstance(rec, GlossaryLogRecord):
                    out.append(rec)
        except Exception:
            return []
        return out

    def _set_auto_refresh(self, enabled: bool):
        if enabled:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _on_auto_refresh(self):
        self.refresh(force=False)

    def refresh(self, force: bool = False):
        if self._refresh_guard:
            return
        self._refresh_guard = True
        try:
            records, path = _load_glossary_records(limit=800)
            snap = self._snapshot_key(records)
            if not force and snap == self._last_snapshot_key:
                return
            self._last_snapshot_key = snap
            self._records = records
            self._log_path = path
            self.model.set_records(records)
            file_part = os.path.basename(path) if path else ''
            self.status_label.setText(self._t('usage_status', count=len(records), file=file_part))
        finally:
            self._refresh_guard = False

    def copy_selected(self):
        sel = self.table.selectionModel().selectedRows()
        if not sel:
            return
        lines = ['\t'.join(['时间', '标签', '作品', '生肉名', '级别', '消息'])]
        for proxy_idx in sel:
            src_idx = self.proxy.mapToSource(proxy_idx)
            rec = self.model.record_at(src_idx.row())
            if rec is None:
                continue
            lines.append('\t'.join([
                rec.time_str,
                rec.tag,
                rec.work,
                rec.raw_work,
                rec.level,
                rec.message,
            ]))
        QApplication.clipboard().setText('\n'.join(lines))

    def export_json(self):
        data = []
        for rec in self._get_view_records():
            data.append({
                'time': rec.time_str,
                'tag': rec.tag,
                'work': rec.work,
                'raw_work': rec.raw_work,
                'level': rec.level,
                'message': rec.message,
                'logger': rec.logger_name,
            })
        try:
            json_text = json.dumps(data, ensure_ascii=False, indent=2)
            QApplication.clipboard().setText(json_text)
            QMessageBox.information(self, self._t('btn_glossary_log'), '已复制 JSON 到剪贴板')
        except Exception as e:
            QMessageBox.warning(self, self._t('btn_glossary_log'), str(e))

    def save_json(self):
        data = []
        for rec in self._get_view_records():
            data.append({
                'time': rec.time_str,
                'tag': rec.tag,
                'work': rec.work,
                'raw_work': rec.raw_work,
                'level': rec.level,
                'message': rec.message,
                'logger': rec.logger_name,
                'raw_line': rec.raw_line,
            })

        default_name = 'glossary_log.json'
        if self._log_path:
            base = os.path.splitext(os.path.basename(self._log_path))[0]
            default_name = f"{base}_glossary.json"

        path, _ = QFileDialog.getSaveFileName(
            self,
            self._t('btn_glossary_log'),
            default_name,
            'JSON Files (*.json);;All Files (*)',
        )
        if not path:
            return

        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json.dumps(data, ensure_ascii=False, indent=2))
            QMessageBox.information(self, self._t('btn_glossary_log'), f"已保存: {path}")
        except Exception as e:
            QMessageBox.warning(self, self._t('btn_glossary_log'), str(e))

    def _show_row_detail(self, proxy_index: QModelIndex):
        src_index = self.proxy.mapToSource(proxy_index)
        rec = self.model.record_at(src_index.row())
        if rec is None:
            return
        msg = [
            f"时间: {rec.time_str}",
            f"标签: {rec.tag}",
            f"级别: {rec.level}",
            f"作品: {rec.work}",
            f"生肉名: {rec.raw_work}",
            f"logger: {rec.logger_name}",
            '',
            '原始日志:',
            rec.raw_line,
        ]
        QMessageBox.information(self, self._t('btn_glossary_log'), '\n'.join(msg))

    def _load_settings(self):
        try:
            geo = self._settings.value("geometry")
            if isinstance(geo, (QByteArray, bytes)):
                self.restoreGeometry(geo)
            header_state = self._settings.value("header_state")
            if isinstance(header_state, (QByteArray, bytes)):
                self.table.horizontalHeader().restoreState(header_state)
            # Ensure user can always drag-resize columns even if a previous state stored a non-interactive mode
            try:
                header = self.table.horizontalHeader()
                header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
                header.setStretchLastSection(True)
            except Exception:
                pass
            auto = self._settings.value("auto_refresh", None)
            if auto is not None:
                self.auto_refresh_cb.setChecked(_to_bool(auto))
            tag = self._settings.value("tag_filter", None)
            if tag:
                idx = self.tag_filter.findText(tag)
                if idx >= 0:
                    self.tag_filter.setCurrentIndex(idx)
            lvl = self._settings.value("level_filter", None)
            if lvl:
                idx = self.level_filter.findText(lvl)
                if idx >= 0:
                    self.level_filter.setCurrentIndex(idx)
            err_only = self._settings.value("only_errors", None)
            if err_only is not None:
                self.error_only_cb.setChecked(_to_bool(err_only))
            ext_only = self._settings.value("only_extract", None)
            if ext_only is not None:
                self.extract_only_cb.setChecked(_to_bool(ext_only))

            density = self._settings.value("density", None)
            if density:
                idx = self.density_combo.findText(str(density))
                if idx >= 0:
                    self.density_combo.setCurrentIndex(idx)

            # Ensure row height is applied even if settings were missing
            self._apply_density(self.density_combo.currentText())
        except Exception:
            pass

    def _save_settings(self):
        try:
            self._settings.setValue("geometry", self.saveGeometry())
            self._settings.setValue("header_state", self.table.horizontalHeader().saveState())
            self._settings.setValue("auto_refresh", int(self.auto_refresh_cb.isChecked()))
            self._settings.setValue("tag_filter", self.tag_filter.currentText())
            self._settings.setValue("level_filter", self.level_filter.currentText())
            self._settings.setValue("only_errors", int(self.error_only_cb.isChecked()))
            self._settings.setValue("only_extract", int(self.extract_only_cb.isChecked()))
            self._settings.setValue("density", self.density_combo.currentText())
            try:
                self._settings.sync()
            except Exception:
                pass
        except Exception:
            pass

    def closeEvent(self, event):
        # Persist settings even when the user closes the dialog via the window close button
        try:
            self._save_settings()
        except Exception:
            pass
        return super().closeEvent(event)

    def done(self, r: int):
        try:
            self._set_auto_refresh(False)
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass
        return super().done(r)
