#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用日志（AI 调用指标）对话框

目标：把日志中的 [AI_METRICS] 解析为结构化表格，并用更“看板化”的样式显示。

- 数据来源：desktop_qt_ui/main.py 写入的 result/log_*.txt
- 每行记录来自一条 [AI_METRICS] 日志

注意：本模块只读日志文件，不修改任何业务逻辑。
"""

from __future__ import annotations

import os
import re
import sys
import sqlite3
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional

from PyQt6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QSortFilterProxyModel,
    Qt,
    QTimer,
    QSettings,
    QByteArray,
)
from PyQt6.QtGui import QColor, QFontMetrics, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QStyledItemDelegate,
    QStyle,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QHeaderView,
)


@dataclass(frozen=True)
class UsageLogRecord:
    # 展示字段
    time_str: str
    time_ts: float
    preset: str
    model: str
    duration_s: float
    first_byte_s: Optional[float]  # 首字节时间（秒）
    stream: bool
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    status: str

    # 附加字段（用于详情/排查）
    request_id: str
    raw_line: str
    logger_name: str


def _default_t(key: str, **kwargs) -> str:
    return key


def _get_result_log_dir() -> str:
    """获取 UI 写入的 result 日志目录。

    desktop_qt_ui/main.py 中：
    - frozen: <exe_dir>/_internal/result
    - dev:    <repo_root>/result
    """
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), '_internal', 'result')
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'result'))


def _pick_latest_log_file(result_dir: str) -> Optional[str]:
    if not os.path.isdir(result_dir):
        return None

    files = []
    for name in os.listdir(result_dir):
        # 优先匹配 UI 生成的日志命名
        if name.startswith('log_') and name.lower().endswith('.txt'):
            full = os.path.join(result_dir, name)
            try:
                files.append((os.path.getmtime(full), full))
            except OSError:
                continue

    if not files:
        return None

    files.sort(key=lambda x: x[0], reverse=True)

    # 优先找到包含 AI_METRICS 的最新日志
    for _, path in files[:10]:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    if '[AI_METRICS]' in line:
                        return path
        except Exception:
            continue

    # 找不到就退回到最新文件
    return files[0][1]


def _parse_time_prefix(line: str) -> tuple[str, float]:
    """解析日志前缀时间。

    例：2026-01-10 23:14:13,418 - INFO - [xxx] - ...
    """
    # 默认值：原样 + 0
    raw = line.split(' - ', 1)[0].strip()

    # 标准化显示（去掉毫秒）
    display = raw
    try:
        # 兼容 "YYYY-mm-dd HH:MM:SS,mmm"
        dt = datetime.strptime(raw, '%Y-%m-%d %H:%M:%S,%f')
        display = dt.strftime('%Y-%m-%d %H:%M:%S')
        return display, dt.timestamp()
    except Exception:
        return display, 0.0


def _int_or_none(val: str) -> Optional[int]:
    try:
        if not val or val == 'n/a':
            return None
        return int(val)
    except Exception:
        return None


def get_latest_ui_log_path() -> Optional[str]:
    """返回最新（优先含 AI_METRICS）的 UI 日志文件路径。"""
    return _pick_latest_log_file(_get_result_log_dir())


def _get_cache_db_path() -> str:
    return os.path.join(_get_result_log_dir(), 'ai_metrics.db')


def _load_usage_records_from_db(limit: int = 500) -> tuple[List[UsageLogRecord], Optional[str]]:
    db_path = _get_cache_db_path()
    if not os.path.isfile(db_path):
        return [], None

    records: List[UsageLogRecord] = []
    try:
        conn = sqlite3.connect(db_path, timeout=1.5)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT ts, preset, model, status, send_ts, recv_ts, duration_ms, first_byte_ms,
                   prompt_tokens, completion_tokens, total_tokens, extra_json, raw_line
            FROM ai_metrics
            ORDER BY ts DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        for row in rows:
            ts, preset, model, status, send_ts, recv_ts, dur_ms, fb_ms, p_tok, c_tok, t_tok, extra_json, raw_line = row
            time_str = ts or recv_ts or ''

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

            rid = ''
            try:
                if extra_json:
                    extra_obj = json.loads(extra_json)
                    rid = extra_obj.get('request_id') or ''
            except Exception:
                pass

            # Determine if stream mode based on first_byte_ms presence (must be positive)
            is_stream = fb_ms is not None and fb_ms > 0
            first_byte_s = (float(fb_ms) / 1000.0) if (fb_ms is not None and fb_ms > 0) else None

            records.append(
                UsageLogRecord(
                    time_str=time_str_disp,
                    time_ts=time_ts,
                    preset=preset or '',
                    model=model or '',
                    duration_s=(float(dur_ms) if dur_ms is not None else 0.0) / 1000.0,
                    first_byte_s=first_byte_s,
                    stream=is_stream,
                    input_tokens=_int_or_none(str(p_tok) if p_tok is not None else ''),
                    output_tokens=_int_or_none(str(c_tok) if c_tok is not None else ''),
                    status=status or '',
                    request_id=rid,
                    raw_line=raw_line or '',
                    logger_name='ai_metrics_cache',
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


def load_usage_records(limit: int = 500) -> tuple[List[UsageLogRecord], Optional[str], str]:
    """从数据库读取使用日志记录。

    返回: (records, db_path, 'cache')
    """
    db_records, db_path = _load_usage_records_from_db(limit=limit)
    return db_records, db_path, 'cache'


class UsageLogTableModel(QAbstractTableModel):
    COL_TIME = 0
    COL_PRESET = 1
    COL_MODEL = 2
    COL_DURATION = 3
    COL_INPUT = 4
    COL_OUTPUT = 5
    COL_STATUS = 6

    def __init__(self, records: List[UsageLogRecord], t: Callable[[str], str] = _default_t, parent=None):
        super().__init__(parent)
        self._records = records
        self._t = t

    def set_records(self, records: List[UsageLogRecord]):
        self.beginResetModel()
        self._records = records
        self.endResetModel()

    def record_at(self, row: int) -> Optional[UsageLogRecord]:
        if row < 0 or row >= len(self._records):
            return None
        return self._records[row]

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return len(self._records)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        if parent.isValid():
            return 0
        return 7

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation != Qt.Orientation.Horizontal:
            return None

        headers = [
            self._t('col_time'),
            self._t('col_preset'),
            self._t('col_model'),
            self._t('col_duration_first'),
            self._t('col_input'),
            self._t('col_output'),
            self._t('col_status'),
        ]
        if 0 <= section < len(headers):
            return headers[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        rec = self._records[index.row()]
        col = index.column()

        if role == Qt.ItemDataRole.UserRole:
            return rec

        if role == Qt.ItemDataRole.ToolTipRole:
            tip = [
                f"preset: {rec.preset}",
                f"model: {rec.model}",
                f"status: {rec.status}",
                f"request_id: {rec.request_id}",
                f"duration: {rec.duration_s:.3f}s",
            ]
            if rec.input_tokens is not None or rec.output_tokens is not None:
                tip.append(f"tokens: in={rec.input_tokens}, out={rec.output_tokens}")
            if rec.logger_name:
                tip.append(f"logger: {rec.logger_name}")
            return "\n".join(tip)

        if role == Qt.ItemDataRole.TextAlignmentRole:
            if col in (self.COL_INPUT, self.COL_OUTPUT):
                return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignCenter)
            if col in (self.COL_DURATION, self.COL_STATUS):
                return int(Qt.AlignmentFlag.AlignCenter)
            return int(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)

        # Use EditRole as sort role (configured in proxy model)
        if role == Qt.ItemDataRole.EditRole:
            if col == self.COL_TIME:
                return rec.time_ts
            if col == self.COL_PRESET:
                return rec.preset or ''
            if col == self.COL_MODEL:
                return rec.model or ''
            if col == self.COL_DURATION:
                return rec.duration_s
            if col == self.COL_INPUT:
                return rec.input_tokens or 0
            if col == self.COL_OUTPUT:
                return rec.output_tokens or 0
            if col == self.COL_STATUS:
                return rec.status or ''
            return ''

        if role != Qt.ItemDataRole.DisplayRole:
            return None

        # For badge-rendered columns, return empty string to avoid double painting
        if col in (self.COL_PRESET, self.COL_MODEL, self.COL_DURATION, self.COL_STATUS):
            return ''

        if col == self.COL_TIME:
            return rec.time_str
        if col == self.COL_INPUT:
            return '' if rec.input_tokens is None else str(rec.input_tokens)
        if col == self.COL_OUTPUT:
            return '' if rec.output_tokens is None else str(rec.output_tokens)
        return ''


class UsageLogProxyModel(QSortFilterProxyModel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._search_text = ''
        self._model_filter = ''

    def set_search_text(self, text: str):
        self._search_text = (text or '').strip().lower()
        self.invalidateFilter()

    def set_model_filter(self, model_name: str):
        self._model_filter = (model_name or '').strip()
        self.invalidateFilter()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex) -> bool:
        src = self.sourceModel()
        if not isinstance(src, UsageLogTableModel):
            return True

        rec = src.record_at(source_row)
        if rec is None:
            return False

        if self._model_filter and self._model_filter != 'ALL':
            if rec.model != self._model_filter:
                return False

        if not self._search_text:
            return True

        hay = f"{rec.time_str} {rec.preset} {rec.model} {rec.status} {rec.request_id}"
        return self._search_text in hay.lower()


class BadgeDelegate:
    """简易徽章绘制器（用于让表格更像看板）。"""

    def __init__(self):
        # 颜色配置（偏保守，避免过饱和）
        self._colors = {
            'token_bg': QColor('#f3f4f6'),
            'token_fg': QColor('#374151'),
            'level1_bg': QColor('#eef2f7'),
            'level1_fg': QColor('#4b5563'),
            'level2_bg': QColor('#e9f5ff'),
            'level2_fg': QColor('#1f7acb'),
            'level3_bg': QColor('#e7f8f1'),
            'level3_fg': QColor('#0f9c5a'),
            'kind_bg': QColor('#fff1d6'),
            'kind_fg': QColor('#c46b1a'),
            'model_bg': QColor('#f1f3f5'),
            'model_fg': QColor('#3f3d56'),
            'dur_ok_bg': QColor('#e7f8f1'),
            'dur_ok_fg': QColor('#0b7a56'),
            'dur_warn_bg': QColor('#fff1d6'),
            'dur_warn_fg': QColor('#c46b1a'),
            'dur_bad_bg': QColor('#ffe1e1'),
            'dur_bad_fg': QColor('#c12a2a'),
            'tag_bg': QColor('#f2f0ff'),
            'tag_fg': QColor('#5a46c6'),
        }

    def paint_cell(self, painter: QPainter, rect, badges: List[Dict[str, Any]], selected: bool, align: str = 'left'):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        fm = QFontMetrics(painter.font())
        center_y = rect.y() + rect.height() // 2

        # Pre-calc badge sizes for alignment
        padding_x = 8
        padding_y = 3
        gap = 6

        sized = []
        total_w = 0
        for b in badges:
            text = b.get('text', '')
            w = fm.horizontalAdvance(text) + padding_x * 2
            h = fm.height() + padding_y * 2
            sized.append((text, w, h, b.get('bg'), b.get('fg')))
            total_w += w
        if sized:
            total_w += gap * (len(sized) - 1)

        # Align
        if align == 'center':
            x = rect.x() + max(8, (rect.width() - total_w) // 2)
        elif align == 'right':
            x = rect.x() + max(8, rect.width() - total_w - 8)
        else:
            x = rect.x() + 8

        for (text, w, h, bg, fg) in sized:
            y = center_y - h // 2

            # Background
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(bg)
            painter.drawRoundedRect(x, y, w, h, 10, 10)

            # Text
            painter.setPen(QPen(fg))
            painter.drawText(x + padding_x, y + padding_y + fm.ascent(), text)

            x += w + gap

        painter.restore()


class UsageLogStyledDelegate(QStyledItemDelegate):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._badge = BadgeDelegate()

    def paint(self, painter: QPainter, option, index: QModelIndex):
        rec = index.data(Qt.ItemDataRole.UserRole)
        if not isinstance(rec, UsageLogRecord):
            return super().paint(painter, option, index)

        col = index.column()
        selected = bool(option.state & QStyle.StateFlag.State_Selected)

        # 先让基类绘制背景（清空文本，避免和徽章重复）
        opt = QStyleOptionViewItem(option)
        opt.text = ''
        super().paint(painter, opt, index)

        badges: List[Dict[str, Any]] = []
        align = 'left'

        if col == UsageLogTableModel.COL_PRESET:
            badges = [{
                'text': rec.preset or '默认',
                'bg': self._badge._colors['token_bg'],
                'fg': self._badge._colors['token_fg'],
            }]
        elif col == UsageLogTableModel.COL_MODEL:
            badges = [{'text': rec.model, 'bg': self._badge._colors['model_bg'], 'fg': self._badge._colors['model_fg']}]
        elif col == UsageLogTableModel.COL_DURATION:
            align = 'center'
            # 用时颜色分级
            if rec.duration_s >= 120:
                bg = self._badge._colors['dur_bad_bg']
                fg = self._badge._colors['dur_bad_fg']
            elif rec.duration_s >= 30:
                bg = self._badge._colors['dur_warn_bg']
                fg = self._badge._colors['dur_warn_fg']
            else:
                bg = self._badge._colors['dur_ok_bg']
                fg = self._badge._colors['dur_ok_fg']

            badges = []
            # 首字节时间（仅流式且为正数时显示）- 独立颜色分级
            if rec.first_byte_s is not None and rec.first_byte_s > 0:
                if rec.first_byte_s >= 10:
                    fb_bg = self._badge._colors['dur_bad_bg']
                    fb_fg = self._badge._colors['dur_bad_fg']
                elif rec.first_byte_s >= 5:
                    fb_bg = self._badge._colors['dur_warn_bg']
                    fb_fg = self._badge._colors['dur_warn_fg']
                else:
                    fb_bg = self._badge._colors['dur_ok_bg']
                    fb_fg = self._badge._colors['dur_ok_fg']
                badges.append({'text': f"{rec.first_byte_s:.1f} s", 'bg': fb_bg, 'fg': fb_fg})
            # 总时间
            badges.append({'text': f"{rec.duration_s:.1f} s", 'bg': self._badge._colors['token_bg'], 'fg': self._badge._colors['token_fg']})
            # 流式/非流标签
            if rec.stream:
                badges.append({'text': '流', 'bg': QColor('#e0f2fe'), 'fg': QColor('#0369a1')})
            else:
                badges.append({'text': '非流', 'bg': QColor('#f3e8ff'), 'fg': QColor('#7c3aed')})
        elif col == UsageLogTableModel.COL_STATUS:
            align = 'center'
            st_raw = (rec.status or '').strip()
            st_key = st_raw.lower()
            label_map = {
                'ok': '成功',
                'timeout': '超时',
                'rate_limit': '限速',
                'content_filter': '拦截',
                'error': '错误',
            }
            color_map = {
                'ok': (QColor('#e7f8f1'), QColor('#0b7a56')),
                'timeout': (QColor('#fff1d6'), QColor('#c46b1a')),
                'rate_limit': (QColor('#fff1d6'), QColor('#c46b1a')),
                'content_filter': (QColor('#ffe1e1'), QColor('#c12a2a')),
                'error': (QColor('#ffe1e1'), QColor('#c12a2a')),
            }
            bg, fg = color_map.get(st_key, (QColor('#eef2f7'), QColor('#4b5563')))
            badges = [{
                'text': label_map.get(st_key, st_raw or '未知'),
                'bg': bg,
                'fg': fg,
            }]

        if badges:
            self._badge.paint_cell(painter, option.rect, badges, selected, align=align)


class UsageLogDialog(QDialog):
    def __init__(self, parent=None, translate: Callable[[str], str] = _default_t):
        super().__init__(parent)
        self._t = translate or _default_t

        self.setWindowTitle(self._t('btn_usage_log'))
        self.setModal(True)
        self.resize(980, 560)

        self._records: List[UsageLogRecord] = []
        self._log_path: Optional[str] = None
        self._settings = QSettings("manga-translator", "usage_log_dialog")

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(10)

        # 标题
        title = QLabel(self._t('btn_usage_log'))
        title.setStyleSheet('font-size: 18px; font-weight: bold;')
        main_layout.addWidget(title)

        info = QLabel(self._t('usage_log_tip'))
        info.setWordWrap(True)
        info.setStyleSheet('color: #666; font-size: 12px; padding: 8px; background: #f5f5f5; border-radius: 4px;')
        main_layout.addWidget(info)

        # 工具栏
        toolbar = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText(self._t('usage_search_placeholder'))
        toolbar.addWidget(self.search_input, 1)

        self.refresh_btn = QPushButton(self._t('Refresh'))
        self.auto_refresh_cb = QCheckBox(self._t('usage_auto_refresh'))
        self.auto_refresh_cb.setChecked(True)
        self.copy_btn = QPushButton(self._t('Copy'))
        self.close_btn = QPushButton(self._t('Close'))
        toolbar.addWidget(self.refresh_btn)
        toolbar.addWidget(self.auto_refresh_cb)
        toolbar.addWidget(self.copy_btn)
        toolbar.addWidget(self.close_btn)
        main_layout.addLayout(toolbar)

        # 状态栏
        self.status_label = QLabel('')
        self.status_label.setStyleSheet('color: #666; font-size: 12px;')
        main_layout.addWidget(self.status_label)

        # 表格
        self.table = QTableView()
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(32)
        header = self.table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        # 时间、预设、模型适度自适应；输入/输出保持右对齐并自适应
        header.setSectionResizeMode(UsageLogTableModel.COL_TIME, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_PRESET, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_MODEL, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_DURATION, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_INPUT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_OUTPUT, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(UsageLogTableModel.COL_STATUS, QHeaderView.ResizeMode.ResizeToContents)
        self.table.setShowGrid(False)

        # 为数字列设置等宽字体，提升对齐
        num_font_css = "font-family: 'Cascadia Mono', 'Consolas', 'Roboto Mono', monospace;"
        self.table.setStyleSheet(self.table.styleSheet() + f" QTableView {{ {num_font_css} }}")

        # 更接近“看板”的轻量样式
        self.table.setStyleSheet(
            """
            QTableView {
                background: white;
                border: 1px solid #e9ecef;
                border-radius: 6px;
                selection-background-color: transparent;
            }
            QTableView::item {
                padding: 6px;
            }
            QTableView::item:selected {
                background: #f5f5f7;
            }
            QHeaderView::section {
                background: #f8f9fa;
                padding: 8px;
                border: none;
                border-bottom: 1px solid #e9ecef;
                font-weight: 600;
            }
            """
        )

        main_layout.addWidget(self.table, 1)

        # 模型/代理/委托
        self.model = UsageLogTableModel([], t=self._t, parent=self)
        self.proxy = UsageLogProxyModel(self)
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.ItemDataRole.EditRole)
        self.table.setModel(self.proxy)

        self.delegate = UsageLogStyledDelegate(self.table)
        for c in (
            UsageLogTableModel.COL_PRESET,
            UsageLogTableModel.COL_MODEL,
            UsageLogTableModel.COL_DURATION,
            UsageLogTableModel.COL_STATUS,
        ):
            self.table.setItemDelegateForColumn(c, self.delegate)

        # Auto refresh
        self._refresh_guard = False
        self._last_snapshot_key = None
        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(2000)
        self._auto_timer.timeout.connect(self._on_auto_refresh_tick)

        # 恢复窗口/表头状态
        self._load_settings()

        # 信号
        self.refresh_btn.clicked.connect(lambda: self.refresh(force=True))
        self.auto_refresh_cb.toggled.connect(self._set_auto_refresh)
        self.copy_btn.clicked.connect(self.copy_selected)
        self.close_btn.clicked.connect(self.accept)
        self.search_input.textChanged.connect(self.proxy.set_search_text)
        self.table.doubleClicked.connect(self._show_row_detail)

        # 首次加载
        self.refresh(force=True)
        self._set_auto_refresh(self.auto_refresh_cb.isChecked())

    def _make_snapshot_key(self, records: List[UsageLogRecord]) -> str:
        if not records:
            return '0'
        head = records[0]
        return f"{len(records)}|{head.time_str}|{head.model}|{head.status}|{head.raw_line}"

    def _load_settings(self):
        try:
            geo = self._settings.value("geometry")
            if isinstance(geo, (QByteArray, bytes)):
                self.restoreGeometry(geo)
            header_state = self._settings.value("header_state")
            if isinstance(header_state, (QByteArray, bytes)):
                self.table.horizontalHeader().restoreState(header_state)
            auto = self._settings.value("auto_refresh", None)
            if auto is not None:
                self.auto_refresh_cb.setChecked(bool(auto))
        except Exception:
            pass

    def _save_settings(self):
        try:
            self._settings.setValue("geometry", self.saveGeometry())
            self._settings.setValue("header_state", self.table.horizontalHeader().saveState())
            self._settings.setValue("auto_refresh", self.auto_refresh_cb.isChecked())
        except Exception:
            pass

    def _set_auto_refresh(self, enabled: bool):
        if enabled:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _on_auto_refresh_tick(self):
        self.refresh(force=False)

    def refresh(self, force: bool = False):
        if self._refresh_guard:
            return

        self._refresh_guard = True
        try:
            records, path, source = load_usage_records(limit=500)
            snapshot_key = self._make_snapshot_key(records)
            if not force and snapshot_key == self._last_snapshot_key:
                return

            self._last_snapshot_key = snapshot_key
            self._records = records
            self._log_path = path
            self.model.set_records(records)

            file_part = os.path.basename(path) if path else ''
            self.status_label.setText(self._t('usage_status', count=len(records), file=file_part))
        finally:
            self._refresh_guard = False

    def copy_selected(self):
        selection = self.table.selectionModel().selectedRows()
        if not selection:
            return

        # TSV：按可见列顺序
        lines: List[str] = []
        header = [
            self._t('col_time'),
            self._t('col_preset'),
            self._t('col_model'),
            self._t('col_duration_first'),
            self._t('col_input'),
            self._t('col_output'),
            self._t('col_status'),
        ]
        lines.append('\t'.join(header))

        for proxy_index in selection:
            src_index = self.proxy.mapToSource(proxy_index)
            rec = self.model.record_at(src_index.row())
            if rec is None:
                continue
            fb_part = f"{rec.first_byte_s:.1f}s/" if rec.first_byte_s is not None else ''
            stream_part = ' 流' if rec.stream else ''
            row = [
                rec.time_str,
                rec.preset,
                rec.model,
                f"{fb_part}{rec.duration_s:.1f}s{stream_part}",
                '' if rec.input_tokens is None else str(rec.input_tokens),
                '' if rec.output_tokens is None else str(rec.output_tokens),
                rec.status,
            ]
            lines.append('\t'.join(row))

        QApplication.clipboard().setText('\n'.join(lines))

    def done(self, r: int):
        # Ensure the timer stops once the dialog is closed/hidden.
        try:
            self._set_auto_refresh(False)
        except Exception:
            pass
        try:
            self._save_settings()
        except Exception:
            pass
        return super().done(r)

    def _show_row_detail(self, proxy_index: QModelIndex):
        src_index = self.proxy.mapToSource(proxy_index)
        rec = self.model.record_at(src_index.row())
        if rec is None:
            return

        # 轻量弹窗：展示 request_id + 原始行
        msg = [
            f"时间: {rec.time_str}",
            f"预设: {rec.preset}",
            f"模型: {rec.model}",
            f"状态: {rec.status}",
            f"请求ID: {rec.request_id}",
            f"耗时: {rec.duration_s:.3f}s",
            f"输入tokens: {rec.input_tokens}",
            f"输出tokens: {rec.output_tokens}",
            "",
            "原始日志:",
            rec.raw_line,
        ]
        QMessageBox.information(self, self._t('btn_usage_log'), '\n'.join(msg))
