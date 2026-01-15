#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
术语参考过滤表编辑对话框
- 编辑 dict/prompt_example.json 文件中的 glossary 部分
- 按类别管理术语（人名、地名、组织、物品、技能、生物）
- 支持按作品名分类管理术语
"""
import json
import os
import copy
from pathlib import Path
from typing import List, Dict, Optional
from PyQt6.QtCore import Qt, QSettings
from PyQt6.QtGui import QStandardItemModel, QStandardItem
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QPushButton,
    QWidget, QDialogButtonBox, QMessageBox, QApplication,
    QTabWidget, QListWidget, QListWidgetItem, QFrame, QMenu, QInputDialog,
    QCheckBox, QTabBar, QTreeView, QHeaderView, QAbstractItemView
)

try:
    # 注意：此文件在运行时通常以 "widgets.xxx" 方式导入，不能用相对导入 ..utils
    from utils.name_replacer import NameReplacer
except Exception:
    NameReplacer = None

try:
    # 可选依赖：更快的语言识别（pycld2）
    import pycld2 as cld2
except Exception:
    cld2 = None


# 默认类别映射
DEFAULT_CATEGORIES = {
    "Person": "人名",
    "Location": "地名",
    "Org": "组织",
    "Item": "物品",
    "Skill": "技能",
    "Creature": "生物"
}

# 支持的语言列表
# 注意：术语场景下不区分 en/es/id（拉丁字母类）——统一归类为 "en"
SUPPORTED_LANGUAGES = {
    "__ALL__": "全部语言",
    "ko": "韩语 (ko)",
    "ja": "日语 (ja)",
    "en": "拉丁字母 (en/es/id)",
    "auto": "其他 (auto)"
}

# 全局标记：是否启用作品分类模式
WORK_BASED_MODE = True


def normalize_term_lang(lang: str) -> str:
    """规范化术语语言代码。

    - 合并 en/es/id -> en
    - 归一化 zh-xx -> zh
    - 其他无法识别的返回 auto
    """
    if not lang:
        return "auto"

    base = str(lang).strip().lower()
    if not base:
        return "auto"

    base = base.split("-", 1)[0]

    if base in ("en", "es", "id"):
        return "en"
    if base in ("ko", "ja", "auto"):
        return base
    if base == "zh":
        return "auto"  # 术语原文通常不会是中文，归为其他

    return "auto"


def detect_language(text: str) -> str:
    """根据文本内容自动检测语言（用于术语）。

    Returns:
        语言代码: 'ko', 'ja', 'zh', 'en', 'auto'

    说明:
        - 术语不区分 en/es/id：统一返回 'en'
        - 优先用文字脚本快速判定（韩文/假名）
        - 若安装了 pycld2，会优先用它辅助判定
    """
    if not text:
        return "auto"

    has_hangul = False
    has_kana = False
    has_han = False
    has_latin = False

    for ch in text:
        # 韩文谚文范围: AC00-D7AF (Hangul Syllables), 1100-11FF (Hangul Jamo)
        if "\uAC00" <= ch <= "\uD7AF" or "\u1100" <= ch <= "\u11FF":
            has_hangul = True
            break

        # 日文平假名: 3040-309F, 片假名: 30A0-30FF
        if "\u3040" <= ch <= "\u309F" or "\u30A0" <= ch <= "\u30FF":
            has_kana = True
            # 不 break：继续扫描是否混有韩文（极少），但不影响最终结果

        # 汉字范围: 4E00-9FFF (CJK Unified Ideographs)
        if "\u4E00" <= ch <= "\u9FFF":
            has_han = True

        # 拉丁字母
        if ch.isascii() and ch.isalpha():
            has_latin = True

    if has_hangul:
        return "ko"
    if has_kana:
        return "ja"

    # 尝试用 pycld2（若可用）进一步判定
    if cld2 is not None:
        try:
            # pycld2.detect 返回 (is_reliable, text_bytes, details)
            # details 是 [(lang_name, lang_code, percent, score), ...]
            _, _, details = cld2.detect(text)
            if details and len(details) > 0:
                lang_code = details[0][1]  # 第一个检测结果的语言代码
                lang = normalize_term_lang(lang_code)
                # 如果包含汉字但 cld2 判成拉丁，通常是短文本误判，回退
                if has_han and lang == "en":
                    lang = "auto"
                if lang != "auto":
                    return lang
        except Exception:
            pass

    # 兜底：按脚本粗分
    if has_han:
        return "auto"  # 术语原文通常不会是中文，归为其他
    if has_latin:
        return "en"

    return "auto"


class GlossaryFilterDialog(QDialog):
    """术语参考过滤表编辑对话框"""

    def __init__(self, parent=None, config_path: str = ""):
        super().__init__(parent)
        self.setWindowTitle("术语参考管理")
        self.setModal(True)
        self.resize(900, 600)  # 加宽以容纳左侧作品列表
        
        self.config_path = config_path
        # 根据提示词文件名推断语言（仅用于默认选择；实际可选语言列表来自 dict/prompts/*.json）
        self._prompt_language = self._detect_prompt_language(config_path)
        self._available_prompt_languages = self._get_available_prompt_languages()
        
        # 右侧每个类别使用 QTreeView + QStandardItemModel（更工整，性能更好）
        self._category_views: Dict[str, QTreeView] = {}
        self._category_models: Dict[str, QStandardItemModel] = {}
        self._system_prompt = ""
        
        # 作品数据存储：{work_name: {category: [{original, translation}, ...]}}
        self._works_data: Dict[str, Dict[str, List[Dict]]] = {}
        # 无作品区（仅存放，不参与翻译应用）：{category: [{original, translation}, ...]}
        self._unassigned_key = "__UNASSIGNED__"  # 特殊键：表示“无作品”
        self._unassigned_data: Dict[str, List[Dict]] = {cat: [] for cat in DEFAULT_CATEGORIES}

        self._current_work: Optional[str] = None  # 当前选中的作品
        self._all_terms_key = "__ALL__"  # 特殊键：表示"全部术语"
        # 默认语言：若能从文件名推断到具体语言则默认选中，否则默认“全部语言”
        self._current_language = self._prompt_language if self._prompt_language != "__ALL__" else "__ALL__"
        
        # 撤销历史记录（最多保留 50 步）
        self._undo_history: List[Dict] = []
        self._max_undo_steps = 50
        
        # 加载名称映射器（用于获取所有熟肉名）
        self._name_replacer = None
        if NameReplacer:
            # 优先使用相对路径；若找不到，则按项目根目录推断
            mapping_path = Path("name_mapping.json")
            if not mapping_path.exists():
                # glossary_filter_dialog.py -> widgets -> desktop_qt_ui -> 项目根目录
                base_path = Path(__file__).resolve().parents[2]
                cand1 = base_path / "name_mapping.json"
                cand2 = base_path / "examples" / "config" / "name_mapping.json"
                if cand1.exists():
                    mapping_path = cand1
                elif cand2.exists():
                    mapping_path = cand2
            try:
                self._name_replacer = NameReplacer(mapping_file=str(mapping_path))
            except Exception:
                self._name_replacer = None

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 标题
        title = QLabel("编辑术语参考表")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 说明文本
        info_label = QLabel(
            "术语参考会在翻译时提供给AI，帮助保持专有名词翻译的一致性。\n"
            "• 原文：OCR识别的原始文本（如韩文、日文）\n"
            "• 译文：期望的翻译结果（中文）"
        )
        info_label.setStyleSheet("color: #666; font-size: 12px; padding: 8px; background: #f5f5f5; border-radius: 4px;")
        info_label.setWordWrap(True)
        main_layout.addWidget(info_label)

        # 中间内容区域（左右分栏）
        content_layout = QHBoxLayout()
        content_layout.setSpacing(12)
        
        # === 左侧作品列表 ===
        self._setup_work_list_panel(content_layout)
        
        # === 右侧术语编辑区 ===
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        
        # 语言筛选条
        self._setup_language_filter(right_layout)
        
        # 标签页
        self.tab_widget = QTabWidget()
        right_layout.addWidget(self.tab_widget, 1)
        
        content_layout.addWidget(right_widget, 3)  # 右侧占比更大
        
        main_layout.addLayout(content_layout, 1)

        # 为每个类别创建标签页
        for cat_key, cat_name in DEFAULT_CATEGORIES.items():
            tab = QWidget()
            self.tab_widget.addTab(tab, cat_name)
            self._setup_category_tab(tab, cat_key, cat_name)

        # 底部按钮区域
        bottom_layout = QHBoxLayout()
        
        # 统计信息（先创建，在 _load_glossary 之前）
        self.stats_label = QLabel("加载中...")
        bottom_layout.addWidget(self.stats_label)
        bottom_layout.addStretch()
        
        # 撤销按钮
        self.undo_btn = QPushButton("撤销")
        self.undo_btn.setToolTip("撤销上一步操作 (Ctrl+Z)")
        self.undo_btn.setEnabled(False)
        self.undo_btn.clicked.connect(self._undo)
        self.undo_btn.setShortcut("Ctrl+Z")
        bottom_layout.addWidget(self.undo_btn)
        
        # 关闭按钮（自动保存模式下不需要确认）
        close_btn = QPushButton("关闭")
        close_btn.clicked.connect(self._on_close)
        bottom_layout.addWidget(close_btn)
        
        main_layout.addLayout(bottom_layout)
        
        # 加载现有数据（在 stats_label 创建之后）
        self._load_glossary()
        
        # 更新统计
        self._update_stats()
        
        # 恢复对话框几何
        self._restore_geometry()
    
    def _detect_prompt_language(self, path: str) -> str:
        """根据提示词文件名推断语言（如 ko.json / ja.json）。

        注意：术语语言中 en/es/id 会合并为 en。
        """
        if not path:
            return "__ALL__"

        fname = Path(path).stem.lower().strip()
        base = fname.split("_", 1)[0]

        if base in ("en", "es", "id"):
            return "en"
        if base in ("ko", "ja"):
            return base
        if base == "default":
            return "__ALL__"

        return "__ALL__"

    def _get_available_prompt_languages(self) -> List[str]:
        """从 dict/prompts/*.json 收集可用语言代码（ko/ja/en/es/id/...）。"""
        try:
            base_path = Path(__file__).resolve().parents[2]
            prompts_dir = base_path / "dict" / "prompts"
            if not prompts_dir.exists() or not prompts_dir.is_dir():
                return []

            codes = []
            for p in prompts_dir.glob("*.json"):
                raw = p.stem.lower().strip()
                if raw == "default":
                    continue

                # 术语语言不区分 en/es/id，合并到 en
                code = normalize_term_lang(raw)

                if code in SUPPORTED_LANGUAGES and code != "__ALL__":
                    codes.append(code)

            order = ["ko", "ja", "en", "auto"]
            uniq = sorted(set(codes), key=lambda c: order.index(c) if c in order else 999)
            return uniq
        except Exception:
            return []

    def _setup_work_list_panel(self, parent_layout: QHBoxLayout):
        """设置左侧作品列表面板"""
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        
        # 标题
        work_label = QLabel("作品分类")
        work_label.setStyleSheet("font-weight: bold; font-size: 13px;")
        left_layout.addWidget(work_label)
        
        # 作品列表
        self.work_list = QListWidget()
        # 左侧尽量紧凑（作品名较长时会被省略显示）
        self.work_list.setMinimumWidth(95)
        self.work_list.setMaximumWidth(130)
        self.work_list.currentItemChanged.connect(self._on_work_selected)
        left_layout.addWidget(self.work_list, 1)
        
        # 提示信息
        hint_label = QLabel("来源: 名称映射")
        hint_label.setStyleSheet("color: #888; font-size: 10px;")
        left_layout.addWidget(hint_label)

        # 一键转移：无作品区 → 指定作品
        self.bulk_transfer_btn = QPushButton("无作品→作品...")
        self.bulk_transfer_btn.setToolTip("将无作品区的所有术语一次性转移到指定作品（无作品区仅存放不应用）。")
        self.bulk_transfer_btn.clicked.connect(self._bulk_transfer_unassigned_to_work)
        self.bulk_transfer_btn.setEnabled(False)
        left_layout.addWidget(self.bulk_transfer_btn)
        
        # 分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        
        parent_layout.addWidget(left_widget)
        parent_layout.addWidget(separator)
    
    def _setup_language_filter(self, parent_layout: QVBoxLayout):
        """设置语言筛选标签页（按提示词语言分类：ko/ja/en/es/id...）"""
        filter_row = QHBoxLayout()
        filter_row.setSpacing(8)
        
        # 语言标签页
        self.language_tabs = QTabBar()
        self.language_tabs.setExpanding(False)  # 不自动扩展填充
        self.language_tabs.setDocumentMode(True)  # 更简洁的样式
        
        # 存储语言代码与标签页索引的映射
        self._lang_code_to_tab_index = {}
        self._tab_index_to_lang_code = {}
        
        # 添加语言标签页：全部 + prompts 目录中的语言
        lang_candidates = ["__ALL__"] + (self._available_prompt_languages or [])
        seen = set()
        tab_index = 0
        for code in lang_candidates:
            if code in seen:
                continue
            if code in SUPPORTED_LANGUAGES:
                # 使用简短的标签名称
                label = SUPPORTED_LANGUAGES[code]
                # 对于带括号的名称，只取括号前的部分
                if "(" in label:
                    label = label.split("(")[0].strip()
                self.language_tabs.addTab(label)
                self._lang_code_to_tab_index[code] = tab_index
                self._tab_index_to_lang_code[tab_index] = code
                seen.add(code)
                tab_index += 1
        
        self.language_tabs.currentChanged.connect(self._on_language_tab_changed)
        # 默认选中 self._current_language（若不存在则保持第一个）
        default_idx = self._lang_code_to_tab_index.get(self._current_language, 0)
        self.language_tabs.setCurrentIndex(default_idx)
        
        filter_row.addWidget(self.language_tabs)
        
        filter_row.addStretch()
        
        # 语言统计标签
        self.lang_stats_label = QLabel("")
        self.lang_stats_label.setStyleSheet("color: #888; font-size: 11px;")
        filter_row.addWidget(self.lang_stats_label)
        
        parent_layout.addLayout(filter_row)
    
    def _on_language_tab_changed(self, index: int):
        """语言标签页切换时触发"""
        lang_code = self._tab_index_to_lang_code.get(index, "__ALL__")
        if lang_code == self._current_language:
            return
        
        # 先保存当前术语到内存
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()

        self._current_language = lang_code
        
        # 重新加载当前作品的术语（按新语言筛选）
        if self._current_work:
            self._load_work_terms_to_ui(self._current_work)
        
        self._update_language_stats()
    
    def _update_language_stats(self):
        """更新语言统计信息"""
        if not hasattr(self, 'lang_stats_label'):
            return
        
        # 统计当前作品各语言的术语数量
        if not self._current_work or self._current_work == self._all_terms_key:
            # 全部作品时不显示详细统计
            self.lang_stats_label.setText("")
            return
        
        if self._current_work == self._unassigned_key:
            work_data = self._unassigned_data or {}
        else:
            work_data = self._works_data.get(self._current_work, {})
        
        # 统计各语言数量
        lang_counts = {}
        for cat_key in DEFAULT_CATEGORIES:
            terms = work_data.get(cat_key, [])
            for term in terms:
                if isinstance(term, dict):
                    lang = normalize_term_lang(term.get('lang', 'auto'))
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        if lang_counts:
            parts = []
            for lang, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
                lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                # 去掉括号中的代码，只显示中文名称
                lang_short = lang_name.split('(')[0].strip() if isinstance(lang_name, str) else str(lang_name)
                parts.append(f"{lang_short}:{count}")
            self.lang_stats_label.setText("  ".join(parts[:4]))  # 最多显示4种语言
        else:
            self.lang_stats_label.setText("")

    def _setup_category_tab(self, parent: QWidget, cat_key: str, cat_name: str):
        """设置类别标签页"""
        layout = QVBoxLayout(parent)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        # 按钮行
        btn_row = QHBoxLayout()
        btn_row.addWidget(QLabel(f"{cat_name}术语列表："))
        btn_row.addStretch()
        
        add_btn = QPushButton("新增")
        add_btn.clicked.connect(lambda: self._on_add_term_clicked(cat_key))
        btn_row.addWidget(add_btn)
        
        paste_btn = QPushButton("从剪贴板粘贴")
        paste_btn.clicked.connect(lambda: self._add_from_clipboard(cat_key))
        btn_row.addWidget(paste_btn)
        
        clear_btn = QPushButton("清空全部")
        clear_btn.clicked.connect(lambda: self._clear_all(cat_key))
        btn_row.addWidget(clear_btn)
        
        dedup_btn = QPushButton("去重")
        dedup_btn.setToolTip("删除完全相同的重复术语")
        dedup_btn.clicked.connect(lambda: self._deduplicate(cat_key))
        btn_row.addWidget(dedup_btn)

        del_btn = QPushButton("删除选中行")
        del_btn.setToolTip("删除右侧表格中选中的行（用于删除某几条术语）")
        del_btn.clicked.connect(lambda: self._delete_selected_rows(cat_key))
        btn_row.addWidget(del_btn)
        
        layout.addLayout(btn_row)
        
        # 选择操作行
        select_row = QHBoxLayout()
        
        select_all_btn = QPushButton("全选")
        select_all_btn.setFixedWidth(50)
        select_all_btn.clicked.connect(lambda: self._select_all(cat_key, True))
        select_row.addWidget(select_all_btn)
        
        deselect_all_btn = QPushButton("全不选")
        deselect_all_btn.setFixedWidth(60)
        deselect_all_btn.clicked.connect(lambda: self._select_all(cat_key, False))
        select_row.addWidget(deselect_all_btn)
        
        invert_btn = QPushButton("反选")
        invert_btn.setFixedWidth(50)
        invert_btn.clicked.connect(lambda: self._invert_selection(cat_key))
        select_row.addWidget(invert_btn)
        
        select_row.addStretch()
        
        transfer_selected_btn = QPushButton("转移选中→作品...")
        transfer_selected_btn.setToolTip("将当前类别中勾选的术语批量转移到指定作品")
        transfer_selected_btn.clicked.connect(lambda: self._transfer_selected_terms(cat_key))
        select_row.addWidget(transfer_selected_btn)
        
        layout.addLayout(select_row)

        # 术语表格（QTreeView 更工整）
        view = QTreeView()
        view.setRootIsDecorated(False)
        view.setIndentation(0)
        view.setAlternatingRowColors(True)
        view.setUniformRowHeights(True)
        view.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        view.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked |
            QAbstractItemView.EditTrigger.EditKeyPressed |
            QAbstractItemView.EditTrigger.AnyKeyPressed
        )
        view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        view.customContextMenuRequested.connect(lambda pos, ck=cat_key: self._show_terms_context_menu(ck, pos))
        # 调淡选中/悬停行颜色
        view.setStyleSheet("""
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
        layout.addWidget(view, 1)

        model = QStandardItemModel(0, 4, self)
        model.setHorizontalHeaderLabels(["选中", "作品名", "原文", "译文"])
        view.setModel(model)

        header = view.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.resizeSection(0, 55)
        header.resizeSection(1, 140)

        # 默认隐藏“作品名”列（仅无作品区时显示）
        view.setColumnHidden(1, True)

        self._category_views[cat_key] = view
        self._category_models[cat_key] = model

    def _add_term_field(self, cat_key: str, original: str = "", translation: str = "", 
                        readonly: bool = False, lang: str = "auto", raw_work_name: str = ""):
        """新增一个术语行（QTreeView 行）
        
        列结构固定为：
        - 0: 选中（勾选框，用于批量转移）
        - 1: 作品名（仅“无作品区”显示）
        - 2: 原文
        - 3: 译文
        """
        model = self._category_models.get(cat_key)
        view = self._category_views.get(cat_key)
        if model is None:
            return

        is_unassigned = (self._current_work == self._unassigned_key)
        term_lang = normalize_term_lang(lang)

        # 选中列（只读模式下不提供勾选）
        item_sel = QStandardItem("")
        item_sel.setEditable(False)
        if not readonly:
            item_sel.setCheckable(True)
            item_sel.setCheckState(Qt.CheckState.Unchecked)

        # 作品名列（仅无作品区显示）
        work_text = ""
        if is_unassigned:
            work_text = (raw_work_name or "").strip() or "无法确定"
        item_work = QStandardItem(work_text)
        item_work.setEditable(bool(is_unassigned and not readonly))

        # 原文/译文
        item_orig = QStandardItem(str(original or ""))
        item_orig.setEditable(not readonly)
        item_trans = QStandardItem(str(translation or ""))
        item_trans.setEditable(not readonly)

        # 在行数据上保存语言信息（用于语言筛选保存）
        for it in (item_sel, item_work, item_orig, item_trans):
            it.setData(term_lang, Qt.ItemDataRole.UserRole + 1)

        model.appendRow([item_sel, item_work, item_orig, item_trans])

        # 选中并滚动到新行
        if view is not None:
            try:
                idx = model.index(model.rowCount() - 1, 2)
                view.setCurrentIndex(idx)
                view.scrollTo(idx)
            except Exception:
                pass

        self._update_stats()

    def _delete_selected_rows(self, cat_key: str):
        """删除右侧表格中选中的行（不影响勾选框逻辑）"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "【全部术语】为只读视图，无法删除。")
            return

        model = self._category_models.get(cat_key)
        view = self._category_views.get(cat_key)
        if model is None or view is None:
            return

        sel = view.selectionModel()
        if sel is None:
            return
        rows = sel.selectedRows(2) or sel.selectedRows(3) or sel.selectedRows(0)
        if not rows:
            return

        # 自动保存（在删除前记录状态）
        self._push_undo_state()

        for idx in sorted(rows, key=lambda x: x.row(), reverse=True):
            model.removeRow(idx.row())

        # 保底：至少保留一行空行（非只读）
        if model.rowCount() == 0:
            new_lang = self._current_language if self._current_language != "__ALL__" else "auto"
            self._add_term_field(cat_key, "", "", lang=new_lang)

        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        self._save_to_file()
        self._update_stats()

    def _show_terms_context_menu(self, cat_key: str, pos):
        """右键菜单（QTreeView）"""
        view = self._category_views.get(cat_key)
        if view is None:
            return

        is_readonly = (self._current_work == self._all_terms_key)

        menu = QMenu(view)
        if not is_readonly:
            act_del = menu.addAction("删除选中行")
            act_transfer = menu.addAction("转移当前术语到其他作品...")
        else:
            act_del = None
            act_transfer = None

        action = menu.exec(view.viewport().mapToGlobal(pos))
        if action is None:
            return

        if action == act_del:
            self._delete_selected_rows(cat_key)
        elif action == act_transfer:
            self._transfer_current_term(cat_key)

    def _transfer_current_term(self, cat_key: str):
        """转移当前选中的术语到其他作品（QTreeView 行）"""
        if self._current_work == self._all_terms_key:
            return

        model = self._category_models.get(cat_key)
        view = self._category_views.get(cat_key)
        if model is None or view is None:
            return

        idx = view.currentIndex()
        if not idx.isValid():
            return
        row = idx.row()

        orig_item = model.item(row, 2)
        trans_item = model.item(row, 3)
        work_item = model.item(row, 1)

        orig_text = (orig_item.text() if orig_item else "").strip()
        trans_text = (trans_item.text() if trans_item else "").strip()
        if not orig_text and not trans_text:
            QMessageBox.information(self, "提示", "该术语为空，无法转移。")
            return

        # 收集可转移的目标作品（排除当前作品和"全部"）
        target_display_list = []
        target_display_to_key = {}
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if not key:
                continue
            if key == self._all_terms_key or key == self._current_work:
                continue
            display = item.text().strip()
            target_display_list.append(display)
            target_display_to_key[display] = key

        if not target_display_list:
            QMessageBox.information(self, "提示", "没有其他作品可供转移。")
            return

        target_display, ok = QInputDialog.getItem(
            self, "转移术语",
            f"将术语 「{orig_text} → {trans_text}」 转移到：",
            target_display_list, 0, False
        )
        if not ok or not target_display:
            return
        target_work = target_display_to_key.get(target_display)
        if not target_work:
            return

        # 获取语言信息：优先用行数据，否则按原文检测
        term_lang = ""
        if orig_item is not None:
            term_lang = normalize_term_lang(orig_item.data(Qt.ItemDataRole.UserRole + 1) or "")
        if not term_lang or term_lang == "auto":
            if self._current_language != "__ALL__":
                term_lang = normalize_term_lang(self._current_language)
            else:
                term_lang = detect_language(orig_text)

        term_data = {"original": orig_text, "translation": trans_text, "lang": term_lang}

        # 确保目标作品结构存在
        if target_work == self._unassigned_key:
            if not isinstance(self._unassigned_data, dict):
                self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}
            if cat_key not in self._unassigned_data or not isinstance(self._unassigned_data.get(cat_key), list):
                self._unassigned_data[cat_key] = []
            # 保留 raw_work_name（如果有）
            raw_work_name_val = (work_item.text() if work_item else "").strip()
            if raw_work_name_val and raw_work_name_val != "无法确定":
                term_data["raw_work_name"] = raw_work_name_val
            self._unassigned_data[cat_key].append(term_data)
        else:
            if target_work not in self._works_data:
                self._works_data[target_work] = {cat: [] for cat in DEFAULT_CATEGORIES}
            self._works_data[target_work][cat_key].append(term_data)

        # 从当前作品删除（删表格行）
        model.removeRow(row)

        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()

        self._refresh_work_list()
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self._current_work:
                self.work_list.setCurrentItem(item)
                break

        self._auto_save()
        QMessageBox.information(self, "转移成功", f"术语已转移到 「{target_display}」。")
    
    def _on_add_term_clicked(self, cat_key: str):
        """新增术语按钮点击事件"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "请先选择一个具体作品后再添加术语。")
            return
        # 新增术语时使用当前筛选语言（如果是全部则用auto）
        new_lang = self._current_language if self._current_language != "__ALL__" else "auto"
        self._add_term_field(cat_key, "", "", lang=new_lang)

    def _add_from_clipboard(self, cat_key: str):
        """从剪贴板按行添加术语（格式：原文<tab>译文 或 原文=译文）"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "请先选择一个具体作品后再添加术语。")
            return
        
        text = QApplication.clipboard().text() or ""
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            QMessageBox.information(self, "剪贴板为空", "剪贴板没有可用的文本。")
            return
        
        added = 0
        for line in lines:
            # 尝试解析不同格式
            original, translation = "", ""
            if '\t' in line:
                parts = line.split('\t', 1)
                original = parts[0].strip()
                translation = parts[1].strip() if len(parts) > 1 else ""
            elif '=' in line:
                parts = line.split('=', 1)
                original = parts[0].strip()
                translation = parts[1].strip() if len(parts) > 1 else ""
            elif ':' in line and not line.startswith('http'):
                parts = line.split(':', 1)
                original = parts[0].strip()
                translation = parts[1].strip() if len(parts) > 1 else ""
            else:
                original = line
            
            if original:
                # 语言分类按提示词语言：优先用当前筛选语言（若为“全部语言”则回退到自动检测）
                if self._current_language and self._current_language != "__ALL__":
                    lang_code = self._current_language
                else:
                    lang_code = detect_language(original)
                self._add_term_field(cat_key, original, translation, lang=lang_code)
                added += 1
        
        if added > 0:
            # 自动保存
            self._auto_save()
            QMessageBox.information(self, "导入成功", f"已导入 {added} 条术语。")

    def _clear_all(self, cat_key: str):
        """清空该类别所有术语"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "请先选择一个具体作品后再进行操作。")
            return

        model = self._category_models.get(cat_key)
        if model is None:
            return

        reply = QMessageBox.question(
            self, "确认清空",
            f"确定要清空所有{DEFAULT_CATEGORIES.get(cat_key, cat_key)}术语吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 自动保存（在清空前记录状态）
        self._push_undo_state()

        model.removeRows(0, model.rowCount())

        # 添加一个空行
        new_lang = self._current_language if self._current_language != "__ALL__" else "auto"
        self._add_term_field(cat_key, "", "", lang=new_lang)

        # 保存到文件
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        self._save_to_file()
        self._update_stats()
    
    def _deduplicate(self, cat_key: str):
        """去除该类别中完全相同的重复术语"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "请先选择一个具体作品后再进行操作。")
            return

        model = self._category_models.get(cat_key)
        if model is None:
            return

        seen = set()
        dup_rows = []
        for r in range(model.rowCount()):
            orig = (model.item(r, 2).text() if model.item(r, 2) else "").strip()
            trans = (model.item(r, 3).text() if model.item(r, 3) else "").strip()
            if not orig and not trans:
                continue
            key = (orig, trans)
            if key in seen:
                dup_rows.append(r)
            else:
                seen.add(key)

        if not dup_rows:
            QMessageBox.information(self, "去重结果", "没有发现重复的术语。")
            return

        self._push_undo_state()

        for r in sorted(dup_rows, reverse=True):
            model.removeRow(r)

        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        self._save_to_file()
        self._update_stats()

        QMessageBox.information(self, "去重完成", f"已删除 {len(dup_rows)} 条重复术语。")
    
    def _select_all(self, cat_key: str, checked: bool):
        """全选/全不选指定类别的术语（勾选框列）"""
        model = self._category_models.get(cat_key)
        if model is None:
            return
        for r in range(model.rowCount()):
            it = model.item(r, 0)
            if it is not None and it.isCheckable():
                it.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)
    
    def _invert_selection(self, cat_key: str):
        """反选指定类别的术语（勾选框列）"""
        model = self._category_models.get(cat_key)
        if model is None:
            return
        for r in range(model.rowCount()):
            it = model.item(r, 0)
            if it is not None and it.isCheckable():
                it.setCheckState(
                    Qt.CheckState.Unchecked if it.checkState() == Qt.CheckState.Checked else Qt.CheckState.Checked
                )
    
    def _transfer_selected_terms(self, cat_key: str):
        """批量转移选中的术语到指定作品（基于勾选框列）"""
        # 检查是否在"全部术语"视图
        if self._current_work == self._all_terms_key:
            QMessageBox.information(self, "提示", "请先选择一个具体作品后再进行操作。")
            return

        model = self._category_models.get(cat_key)
        if model is None:
            return

        # 收集勾选的行号与数据
        picked = []  # (row, orig, trans, lang)
        for r in range(model.rowCount()):
            it_sel = model.item(r, 0)
            if it_sel is None or not it_sel.isCheckable() or it_sel.checkState() != Qt.CheckState.Checked:
                continue
            orig_item = model.item(r, 2)
            trans_item = model.item(r, 3)
            orig = (orig_item.text() if orig_item else "").strip()
            trans = (trans_item.text() if trans_item else "").strip()
            if not (orig or trans):
                continue

            row_lang = ""
            if orig_item is not None:
                row_lang = normalize_term_lang(orig_item.data(Qt.ItemDataRole.UserRole + 1) or "")
            if not row_lang or row_lang == "auto":
                row_lang = detect_language(orig)

            picked.append((r, orig, trans, row_lang))

        if not picked:
            QMessageBox.information(self, "提示", "请先勾选要转移的术语。")
            return
        
        # 收集可转移的目标作品（排除当前作品和"全部"）
        target_display_list = []
        target_display_to_key = {}
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if not key:
                continue
            if key == self._all_terms_key or key == self._current_work:
                continue
            display = item.text().strip()
            target_display_list.append(display)
            target_display_to_key[display] = key
        
        if not target_display_list:
            QMessageBox.information(self, "提示", "没有其他作品可供转移。")
            return
        
        # 让用户选择目标作品
        target_display, ok = QInputDialog.getItem(
            self, "批量转移术语",
            f"将 {len(picked)} 条选中术语转移到：",
            target_display_list, 0, False
        )
        if not ok or not target_display:
            return
        target_work = target_display_to_key.get(target_display)
        if not target_work:
            return
        
        # 自动保存（在转移前记录状态）
        self._push_undo_state()
        
        # 确保目标作品有数据结构
        if target_work == self._unassigned_key:
            if not isinstance(self._unassigned_data, dict):
                self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}
            if cat_key not in self._unassigned_data or not isinstance(self._unassigned_data.get(cat_key), list):
                self._unassigned_data[cat_key] = []
        else:
            if target_work not in self._works_data:
                self._works_data[target_work] = {cat: [] for cat in DEFAULT_CATEGORIES}
            if cat_key not in self._works_data[target_work]:
                self._works_data[target_work][cat_key] = []
        
        # 转移术语（保留语言信息）
        moved = 0
        for row, orig_text, trans_text, term_lang in picked:
            term_data = {"original": orig_text, "translation": trans_text, "lang": term_lang}
            if target_work == self._unassigned_key:
                self._unassigned_data[cat_key].append(term_data)
            else:
                self._works_data[target_work][cat_key].append(term_data)
            moved += 1

        # 从当前界面删除（倒序删除避免行号变化）
        for row, _, _, _ in sorted(picked, key=lambda x: x[0], reverse=True):
            model.removeRow(row)

        # 保底：若空了，留一行
        if model.rowCount() == 0:
            new_lang = self._current_language if self._current_language != "__ALL__" else "auto"
            self._add_term_field(cat_key, "", "", lang=new_lang)
        
        # 把当前界面状态写回内存
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        
        # 刷新作品列表（更新计数）
        self._refresh_work_list()
        
        # 重新选中当前作品
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self._current_work:
                self.work_list.setCurrentItem(item)
                break
        
        # 保存到文件
        self._save_to_file()
        
        QMessageBox.information(self, "转移成功", f"已将 {moved} 条术语转移到「{target_display}」。")

    # ========== 作品列表操作方法 ==========
    
    def _on_work_selected(self, current: QListWidgetItem, previous: QListWidgetItem):
        """作品选中变化时触发"""
        # 先保存当前作品的术语到内存（"全部" 为只读视图，不需要保存）
        if previous and self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        
        if not current:
            self._current_work = None
            return
        
        work_name = current.data(Qt.ItemDataRole.UserRole)
        self._current_work = work_name
        
        # 加载该作品的术语到界面
        self._load_work_terms_to_ui(work_name)
    
    def _save_current_terms_to_memory(self):
        """将当前界面上的术语保存到内存（从 QTreeView 模型读取）。

        注意：当有语言筛选时，只更新当前筛选语言的术语，保留其他语言。
        """
        if not self._current_work or self._current_work == self._all_terms_key:
            return

        # 获取原有数据
        if self._current_work == self._unassigned_key:
            existing_data = self._unassigned_data or {}
        else:
            existing_data = self._works_data.get(self._current_work, {})

        filter_lang = self._current_language

        work_data: Dict[str, List[Dict]] = {cat: [] for cat in DEFAULT_CATEGORIES}

        for cat_key in DEFAULT_CATEGORIES:
            # 如果有语言筛选，先保留其他语言的术语
            if filter_lang != "__ALL__":
                for term in existing_data.get(cat_key, []):
                    if isinstance(term, dict):
                        term_lang = normalize_term_lang(term.get('lang', 'auto'))
                        if term_lang != filter_lang:
                            work_data[cat_key].append(term)

            model = self._category_models.get(cat_key)
            if model is None:
                continue

            for r in range(model.rowCount()):
                orig_item = model.item(r, 2)
                trans_item = model.item(r, 3)
                work_item = model.item(r, 1)

                orig = (orig_item.text() if orig_item else "").strip()
                trans = (trans_item.text() if trans_item else "").strip()
                if not (orig or trans):
                    continue

                # 语言：优先使用行数据；缺失则按当前筛选语言或检测
                row_lang = ""
                if orig_item is not None:
                    row_lang = normalize_term_lang(orig_item.data(Qt.ItemDataRole.UserRole + 1) or "")
                if not row_lang or row_lang == "auto":
                    if filter_lang != "__ALL__":
                        row_lang = normalize_term_lang(filter_lang)
                    else:
                        row_lang = detect_language(orig)

                term_entry: Dict[str, str] = {
                    "original": orig,
                    "translation": trans,
                    "lang": row_lang
                }

                # unassigned 视图下保存 raw_work_name
                if self._current_work == self._unassigned_key:
                    raw_work_name_val = (work_item.text() if work_item else "").strip()
                    if raw_work_name_val and raw_work_name_val != "无法确定":
                        term_entry["raw_work_name"] = raw_work_name_val

                work_data[cat_key].append(term_entry)

        if self._current_work == self._unassigned_key:
            self._unassigned_data = work_data
        else:
            self._works_data[self._current_work] = work_data
    
    def _load_work_terms_to_ui(self, work_name: str):
        """加载指定作品的术语到界面（支持语言筛选）。

        UI 为 QTreeView + QStandardItemModel。
        """
        # 判断是否是只读模式（"全部术语"视图）
        is_readonly = (work_name == self._all_terms_key)
        is_unassigned = (work_name == self._unassigned_key)

        # 当前语言筛选
        filter_lang = self._current_language

        # 选择数据源
        if work_name == self._all_terms_key:
            merged_data: Dict[str, List[Dict]] = {cat: [] for cat in DEFAULT_CATEGORIES}
            for wn, wd in self._works_data.items():
                if wn in (self._all_terms_key, self._unassigned_key):
                    continue
                for ck in DEFAULT_CATEGORIES:
                    merged_data[ck].extend(wd.get(ck, []))
            work_data = merged_data
        elif work_name == self._unassigned_key:
            work_data = self._unassigned_data or {}
        else:
            work_data = self._works_data.get(work_name, {})

        # 清空并重建每个类别表格
        for cat_key in DEFAULT_CATEGORIES:
            model = self._category_models.get(cat_key)
            view = self._category_views.get(cat_key)
            if model is None or view is None:
                continue

            model.removeRows(0, model.rowCount())

            # 控制列显示：作品名列仅在 unassigned 视图显示
            view.setColumnHidden(1, not is_unassigned)

            terms = work_data.get(cat_key, [])
            added_count = 0

            for term in (terms or []):
                if not isinstance(term, dict):
                    continue

                orig = term.get('original', '')
                trans = term.get('translation', '')
                has_lang = ('lang' in term)
                term_lang = normalize_term_lang(term.get('lang', 'auto'))

                raw_work_name = term.get('raw_work_name', '') if is_unassigned else ''

                # 语言筛选：旧数据缺少 lang 时，在当前筛选语言下显示，并自动归类到该语言
                if filter_lang != "__ALL__":
                    if has_lang:
                        if term_lang != filter_lang:
                            continue
                    else:
                        term_lang = filter_lang

                if (orig or trans):
                    self._add_term_field(cat_key, orig, trans, readonly=is_readonly, lang=term_lang, raw_work_name=raw_work_name)
                    added_count += 1

            # 如果没有任何术语，添加一个空行（只读模式下不添加）
            if added_count == 0 and not is_readonly:
                new_lang = filter_lang if filter_lang != "__ALL__" else "auto"
                self._add_term_field(cat_key, "", "", lang=new_lang)

            # 只读模式下：确保不提供勾选框
            if is_readonly:
                for r in range(model.rowCount()):
                    it0 = model.item(r, 0)
                    if it0 is not None and it0.isCheckable():
                        it0.setCheckable(False)

        self._update_stats()
        self._update_bulk_transfer_button()
        self._update_language_stats()
    
    def _refresh_work_list(self):
        """刷新作品列表 - 从名称映射加载所有熟肉名"""
        self.work_list.blockSignals(True)
        self.work_list.clear()

        # 添加"全部术语"选项（只读汇总视图）
        all_count = 0
        try:
            for work_name, work_data in self._works_data.items():
                if work_name in (self._all_terms_key, self._unassigned_key):
                    continue
                all_count += sum(len(work_data.get(cat, [])) for cat in DEFAULT_CATEGORIES)
        except Exception:
            all_count = 0
        all_text = "📚 全部" + (f" ({all_count})" if all_count > 0 else "")
        all_item = QListWidgetItem(all_text)
        all_item.setData(Qt.ItemDataRole.UserRole, self._all_terms_key)
        self.work_list.addItem(all_item)

        # 添加"未确定作品名"选项（仅存放不应用）
        un_count = 0
        try:
            un_count = sum(len(self._unassigned_data.get(cat, [])) for cat in DEFAULT_CATEGORIES)
        except Exception:
            un_count = 0
        un_text = "🗃 未确定作品名" + (f" ({un_count})" if un_count > 0 else "")
        un_item = QListWidgetItem(un_text)
        un_item.setData(Qt.ItemDataRole.UserRole, self._unassigned_key)
        self.work_list.addItem(un_item)
        
        # 收集所有作品名：名称映射中的熟肉名 + 已有术语的作品名
        all_work_names = set()
        
        # 从名称映射获取所有熟肉名
        if self._name_replacer:
            mappings = self._name_replacer.get_all_mappings()
            for translated_name in mappings.values():
                if translated_name:  # 过滤空值
                    all_work_names.add(translated_name)
        
        # 添加已有术语的作品名
        for work_name in self._works_data.keys():
            if work_name not in (self._all_terms_key, self._unassigned_key):
                all_work_names.add(work_name)
        
        # 添加各作品
        for work_name in sorted(all_work_names):
            # 统计该作品的术语数量
            if work_name in self._works_data:
                count = sum(len(self._works_data[work_name].get(cat, [])) for cat in DEFAULT_CATEGORIES)
                if count > 0:
                    item = QListWidgetItem(f"{work_name} ({count})")
                else:
                    item = QListWidgetItem(work_name)
            else:
                item = QListWidgetItem(work_name)
            item.setData(Qt.ItemDataRole.UserRole, work_name)
            self.work_list.addItem(item)
        
        self.work_list.blockSignals(False)
    
    # ========== 加载/保存方法 ==========
    
    def _load_glossary(self):
        """从配置文件加载术语"""
        if not self.config_path or not os.path.exists(self.config_path):
            # 默认创建一个空作品
            self._works_data = {}
            self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}
            self._refresh_work_list()
            # 选中"全部术语"
            if self.work_list.count() > 0:
                self.work_list.setCurrentRow(0)
            return

        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # 保存 system_prompt
            self._system_prompt = config.get('system_prompt', '')
            
            # 加载 glossary - 仅支持新的作品分类格式（works/unassigned）
            glossary = config.get('glossary', {})

            # 默认初始化无作品区
            self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}

            # 标记当前文件格式是否支持（旧格式不再自动转换，避免误覆盖文件）
            self._glossary_format_supported = True

            # 新格式: {"works": {...}, "unassigned": {...}, 以及根级分类(向后兼容)}
            if "works" in glossary and isinstance(glossary.get("works"), dict):
                self._works_data = glossary.get("works", {})
                # 确保每个作品都有所有类别
                for work_name in list(self._works_data.keys()):
                    for cat in DEFAULT_CATEGORIES:
                        if cat not in self._works_data[work_name]:
                            self._works_data[work_name][cat] = []

                # 加载无作品区（可选）
                unassigned = glossary.get("unassigned", {})
                if isinstance(unassigned, dict):
                    for cat in DEFAULT_CATEGORIES:
                        v = unassigned.get(cat, [])
                        self._unassigned_data[cat] = v if isinstance(v, list) else []

                # 自动迁移"默认"作品到"未确定作品名"区
                if "默认" in self._works_data:
                    default_data = self._works_data.pop("默认")
                    for cat in DEFAULT_CATEGORIES:
                        terms = default_data.get(cat, [])
                        if isinstance(terms, list) and terms:
                            self._unassigned_data[cat].extend(terms)

                # ✅ 自动迁移：把 unassigned 中 raw_work_name 可映射到熟肉名的条目转移到对应作品
                moved = self._auto_migrate_unassigned_by_name_mapping()
                if moved > 0:
                    # 直接落盘，避免用户还要手动“无作品→作品”
                    self._save_to_file()
            else:
                # 不再支持旧格式（没有 works 结构）
                self._glossary_format_supported = False
                QMessageBox.warning(
                    self,
                    "术语格式不支持",
                    "当前术语文件不是按作品分组的格式（缺少 glossary.works）。\n"
                    "已停止加载，并且不会自动转换/保存，避免覆盖原文件。\n\n"
                    "请将术语文件升级为按作品分组格式后再打开。"
                )
                self._works_data = {}
                self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}
                self._refresh_work_list()
                if self.work_list.count() > 0:
                    self.work_list.setCurrentRow(0)
                return
            
            self._refresh_work_list()

            # 默认选中第一个“作品”（不选中“全部/无作品”）
            selected = False
            for i in range(self.work_list.count()):
                item = self.work_list.item(i)
                key = item.data(Qt.ItemDataRole.UserRole)
                if key and key not in (self._all_terms_key, self._unassigned_key):
                    self.work_list.setCurrentRow(i)
                    selected = True
                    break
            if not selected:
            # 若没有任何作品，选中“未确定作品名”
                if self.work_list.count() > 0:
                    self.work_list.setCurrentRow(0)

        except Exception as e:
            QMessageBox.warning(self, "加载失败", f"无法加载配置文件：{e}")
            self._works_data = {}
            self._unassigned_data = {cat: [] for cat in DEFAULT_CATEGORIES}
            self._refresh_work_list()
            if self.work_list.count() > 0:
                self.work_list.setCurrentRow(0)

    def _auto_migrate_unassigned_by_name_mapping(self) -> int:
        """自动迁移 unassigned -> works。

        规则：
        - 仅迁移带 raw_work_name 的条目
        - raw_work_name 能通过 name_mapping.json 映射到熟肉名时，迁移到该熟肉作品
        - raw_work_name 本身已是熟肉名（存在于映射 values 或已存在于 works）时，也迁移

        Returns:
            实际迁移的条目数量
        """
        if not isinstance(self._unassigned_data, dict):
            return 0
        if not isinstance(self._works_data, dict):
            self._works_data = {}

        if not self._name_replacer:
            return 0

        try:
            mapped_values = set((self._name_replacer.mapping or {}).values())
        except Exception:
            mapped_values = set()

        def _resolve_target_work(raw_work_name: str) -> Optional[str]:
            raw = (raw_work_name or "").strip()
            if not raw:
                return None

            # raw 已经是一个已存在作品 key
            if raw in self._works_data:
                return raw

            # 尝试映射
            mapped = self._name_replacer.get_translated_name(raw)
            if mapped and mapped != raw:
                return mapped

            # raw 本身就是熟肉名（出现在映射 value 中）
            if raw in mapped_values:
                return raw

            return None

        moved = 0

        for cat_key in DEFAULT_CATEGORIES:
            src_terms = self._unassigned_data.get(cat_key, [])
            if not isinstance(src_terms, list) or not src_terms:
                continue

            new_src = []

            for t in src_terms:
                if not isinstance(t, dict):
                    new_src.append(t)
                    continue

                o = (t.get('original') or '').strip()
                tr = (t.get('translation') or '').strip()
                if not (o or tr):
                    continue

                raw_work_name = (t.get('raw_work_name') or '').strip()
                target_work = _resolve_target_work(raw_work_name)
                if not target_work:
                    new_src.append(t)
                    continue

                # 确保目标作品结构存在
                if target_work not in self._works_data or not isinstance(self._works_data.get(target_work), dict):
                    self._works_data[target_work] = {cat: [] for cat in DEFAULT_CATEGORIES}
                for ck in DEFAULT_CATEGORIES:
                    if ck not in self._works_data[target_work] or not isinstance(self._works_data[target_work].get(ck), list):
                        self._works_data[target_work][ck] = []

                dst_terms = self._works_data[target_work].get(cat_key, [])
                if not isinstance(dst_terms, list):
                    dst_terms = []
                    self._works_data[target_work][cat_key] = dst_terms

                # 去重：按 (original, translation) 去重
                seen = set()
                for dt in dst_terms:
                    if isinstance(dt, dict):
                        oo = (dt.get('original') or '').strip()
                        tt = (dt.get('translation') or '').strip()
                        if oo or tt:
                            seen.add((oo, tt))

                key = (o, tr)
                if key not in seen:
                    term_lang = normalize_term_lang(t.get('lang', 'auto'))
                    dst_terms.append({"original": o, "translation": tr, "lang": term_lang})
                moved += 1

            self._unassigned_data[cat_key] = new_src

        return moved

    def _push_undo_state(self):
        """保存当前状态到撤销历史"""
        # 先把当前界面写回内存
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        
        state = {
            "works_data": copy.deepcopy(self._works_data),
            "unassigned_data": copy.deepcopy(self._unassigned_data)
        }
        self._undo_history.append(state)
        
        # 限制历史记录数量
        if len(self._undo_history) > self._max_undo_steps:
            self._undo_history.pop(0)
        
        self._update_undo_button()
    
    def _undo(self):
        """撤销上一步操作"""
        if not self._undo_history:
            return
        
        state = self._undo_history.pop()
        self._works_data = state["works_data"]
        self._unassigned_data = state["unassigned_data"]
        
        # 刷新界面
        self._refresh_work_list()
        
        # 重新选中当前作品
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == self._current_work:
                self.work_list.setCurrentItem(item)
                break
        else:
            # 如果当前作品不存在了，选中第一个
            if self.work_list.count() > 0:
                self.work_list.setCurrentRow(0)
        
        # 保存到文件
        self._save_to_file()
        self._update_undo_button()
        self._update_stats()
    
    def _update_undo_button(self):
        """更新撤销按钮状态"""
        if hasattr(self, 'undo_btn'):
            self.undo_btn.setEnabled(len(self._undo_history) > 0)
            if self._undo_history:
                self.undo_btn.setText(f"撤销 ({len(self._undo_history)})")
            else:
                self.undo_btn.setText("撤销")
    
    def _save_to_file(self):
        """保存当前数据到文件（自动保存）"""
        if not self.config_path:
            return False
        
        # 旧格式不再自动转换：为避免覆盖原文件，直接禁止保存
        if hasattr(self, '_glossary_format_supported') and not self._glossary_format_supported:
            return False
        
        try:
            # 构建 glossary 数据
            glossary = {
                "works": self._works_data,
                "unassigned": self._unassigned_data
            }
            
            # 同时保留【作品区】术语合并后的平坦结构（向后兼容）
            merged_terms = {cat: [] for cat in DEFAULT_CATEGORIES}
            for work_name, work_data in self._works_data.items():
                if work_name in (self._all_terms_key, self._unassigned_key):
                    continue
                for cat_key in DEFAULT_CATEGORIES:
                    merged_terms[cat_key].extend(work_data.get(cat_key, []))
            
            for cat_key in DEFAULT_CATEGORIES:
                glossary[cat_key] = merged_terms[cat_key]
            
            config = {
                "system_prompt": self._system_prompt,
                "glossary": glossary
            }
            
            os.makedirs(os.path.dirname(os.path.abspath(self.config_path)), exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            print(f"Auto-save failed: {e}")
            return False
    
    def _auto_save(self):
        """自动保存：先记录撤销状态，再保存到文件"""
        self._push_undo_state()
        
        # 先把当前界面写回内存
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        
        self._save_to_file()
    
    def _on_close(self):
        """关闭对话框（自动保存模式，直接关闭）"""
        # 确保最后的修改已保存
        if self._current_work and self._current_work != self._all_terms_key:
            self._save_current_terms_to_memory()
        self._save_to_file()
        self._save_geometry()
        self.accept()

    def _on_accept(self):
        """保存配置并关闭对话框（兼容旧逻辑）"""
        self._on_close()

    def _update_stats(self):
        """更新统计信息"""
        total = 0
        details = []

        for cat_key, cat_name in DEFAULT_CATEGORIES.items():
            model = self._category_models.get(cat_key)
            if model is None:
                continue

            count = 0
            for r in range(model.rowCount()):
                orig = (model.item(r, 2).text() if model.item(r, 2) else "").strip()
                trans = (model.item(r, 3).text() if model.item(r, 3) else "").strip()
                if orig or trans:
                    count += 1

            if count > 0:
                details.append(f"{cat_name}:{count}")
                total += count

        # 显示当前作品名称
        work_info = ""
        if self._current_work:
            if self._current_work == self._all_terms_key:
                work_info = f"【全部作品: {len(self._works_data)}个】 "
            elif self._current_work == self._unassigned_key:
                work_info = "【无作品】 "
            else:
                work_info = f"【{self._current_work}】 "

        if details:
            self.stats_label.setText(f"{work_info}共 {total} 条术语 ({', '.join(details)})")
        else:
            self.stats_label.setText(f"{work_info}暂无术语")

        self._update_bulk_transfer_button()

    # ----- Settings for dialog geometry -----
    def _settings(self) -> QSettings:
        return QSettings("MangaTranslator", "GlossaryFilter")

    def _restore_geometry(self):
        try:
            s = self._settings()
            data = s.value("dialog/glossary_filter_geometry")
            if data:
                self.restoreGeometry(data)
        except Exception:
            pass

    def _save_geometry(self):
        try:
            s = self._settings()
            s.setValue("dialog/glossary_filter_geometry", self.saveGeometry())
        except Exception:
            pass

    def _update_bulk_transfer_button(self):
        """更新“无作品→作品”按钮可用状态。"""
        btn = getattr(self, 'bulk_transfer_btn', None)
        if not btn:
            return

        # 仅在当前选中“无作品”且确实有术语时启用
        is_unassigned = (self._current_work == self._unassigned_key)
        un_total = 0
        try:
            un_total = sum(len(self._unassigned_data.get(cat, [])) for cat in DEFAULT_CATEGORIES)
        except Exception:
            un_total = 0

        # 需要存在至少一个可转移目标作品
        has_target = False
        try:
            for i in range(self.work_list.count()):
                item = self.work_list.item(i)
                key = item.data(Qt.ItemDataRole.UserRole)
                if key and key not in (self._all_terms_key, self._unassigned_key):
                    has_target = True
                    break
        except Exception:
            has_target = False

        btn.setEnabled(bool(is_unassigned and un_total > 0 and has_target))

    def _bulk_transfer_unassigned_to_work(self):
        """把“无作品区”的所有术语一键转移到指定作品。"""
        if self._current_work != self._unassigned_key:
            QMessageBox.information(self, "提示", "请先在左侧选择“无作品”，再进行一键转移。")
            return

        # 先把界面内容写回内存（避免用户未保存的编辑丢失）
        self._save_current_terms_to_memory()

        # 统计无作品区术语数量
        un_total = 0
        try:
            un_total = sum(len(self._unassigned_data.get(cat, [])) for cat in DEFAULT_CATEGORIES)
        except Exception:
            un_total = 0

        if un_total <= 0:
            QMessageBox.information(self, "提示", "无作品区没有可转移的术语。")
            return

        # 构造目标作品列表（排除：全部/无作品）
        target_display_list = []
        target_display_to_key = {}
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            key = item.data(Qt.ItemDataRole.UserRole)
            if not key:
                continue
            if key in (self._all_terms_key, self._unassigned_key):
                continue
            display = item.text().strip()
            target_display_list.append(display)
            target_display_to_key[display] = key

        if not target_display_list:
            QMessageBox.information(self, "提示", "没有可用的目标作品。")
            return

        target_display, ok = QInputDialog.getItem(
            self,
            "一键转移无作品术语",
            "请选择要接收的作品（将把无作品区所有术语转移过去，并清空无作品区）：",
            target_display_list,
            0,
            False
        )
        if not ok or not target_display:
            return

        target_work = target_display_to_key.get(target_display)
        if not target_work:
            return

        # 二次确认
        reply = QMessageBox.question(
            self,
            "确认转移",
            f"确认将无作品区的 {un_total} 条术语全部转移到「{target_display}」吗？\n\n"
            "提示：无作品区术语仅存放不应用；转移后将从无作品区移除。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 自动保存（在转移前记录状态）
        self._push_undo_state()
        
        # 确保目标作品存在
        if target_work not in self._works_data:
            self._works_data[target_work] = {cat: [] for cat in DEFAULT_CATEGORIES}
        for cat in DEFAULT_CATEGORIES:
            if cat not in self._works_data[target_work] or not isinstance(self._works_data[target_work].get(cat), list):
                self._works_data[target_work][cat] = []

        moved = 0
        skipped = 0

        for cat_key in DEFAULT_CATEGORIES:
            src_terms = self._unassigned_data.get(cat_key, [])
            if not isinstance(src_terms, list) or not src_terms:
                self._unassigned_data[cat_key] = []
                continue

            dst_terms = self._works_data[target_work].get(cat_key, [])
            if not isinstance(dst_terms, list):
                dst_terms = []
                self._works_data[target_work][cat_key] = dst_terms

            # 去重：按 (original, translation) 去重，避免完全重复项
            seen = set()
            for t in dst_terms:
                if isinstance(t, dict):
                    o = (t.get('original') or '').strip()
                    tr = (t.get('translation') or '').strip()
                    if o or tr:
                        seen.add((o, tr))

            for t in src_terms:
                if not isinstance(t, dict):
                    continue
                o = (t.get('original') or '').strip()
                tr = (t.get('translation') or '').strip()
                if not (o or tr):
                    continue
                k = (o, tr)
                if k in seen:
                    skipped += 1
                    continue
                # 保留语言信息
                term_lang = normalize_term_lang(t.get('lang', 'auto'))
                dst_terms.append({"original": o, "translation": tr, "lang": term_lang})
                seen.add(k)
                moved += 1

            # 清空无作品区
            self._unassigned_data[cat_key] = []

        # 保存到文件
        self._save_to_file()
        
        # 刷新列表计数，并跳转到目标作品以便检查
        self._refresh_work_list()
        for i in range(self.work_list.count()):
            item = self.work_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == target_work:
                self.work_list.setCurrentItem(item)
                break

        QMessageBox.information(
            self,
            "转移完成",
            f"已转移 {moved} 条术语到「{target_display}」。\n"
            f"跳过重复：{skipped} 条。\n"
            "无作品区已清空。"
        )

    def reject(self) -> None:
        self._save_geometry()
        return super().reject()

    def closeEvent(self, event):
        self._save_geometry()
        return super().closeEvent(event)
