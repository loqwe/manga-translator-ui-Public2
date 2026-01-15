
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QSlider,
    QToolButton,
    QWidget,
)

from services import get_i18n_manager


class EditorToolbar(QWidget):
    """
    编辑器顶部工具栏，包含返回、导出、撤销/重做、缩放、视图模式等全局操作。
    """
    # --- Define signals for all actions ---
    back_requested = pyqtSignal()
    export_requested = pyqtSignal()
    save_json_requested = pyqtSignal()
    edit_file_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    redo_requested = pyqtSignal()
    edit_geometry_requested = pyqtSignal(bool)
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    fit_window_requested = pyqtSignal()
    display_mode_changed = pyqtSignal(str)
    original_image_alpha_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.i18n = get_i18n_manager()
        self._init_ui()
        self._connect_signals()
    
    def _t(self, key: str, **kwargs) -> str:
        """翻译辅助方法"""
        if self.i18n:
            return self.i18n.translate(key, **kwargs)
        return key

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(10)

        # --- File Actions ---
        self.back_button = QToolButton()
        self.back_button.setText(self._t("Back"))
        self.back_button.setToolTip(self._t("Back to Main"))
        layout.addWidget(self.back_button)

        self.export_button = QToolButton()
        self.export_button.setText(self._t("Export Image"))
        self.export_button.setToolTip(self._t("Export current rendered image") + " (Ctrl+Q)")
        layout.addWidget(self.export_button)
        
        self.save_json_button = QToolButton()
        self.save_json_button.setText(self._t("Save JSON"))
        self.save_json_button.setToolTip(self._t("Save translation data to JSON file") + " (Ctrl+W)")
        layout.addWidget(self.save_json_button)
        
        self.edit_file_button = QToolButton()
        self.edit_file_button.setText(self._t("Edit Original"))
        self.edit_file_button.setToolTip(self._t("Switch to source file of current translation for editing") + " (Ctrl+E)")
        layout.addWidget(self.edit_file_button)

        layout.addWidget(self._create_separator())

        # --- Edit Actions ---
        self.undo_button = QToolButton()
        self.undo_button.setText(self._t("Undo"))
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip(self._t("Undo last operation") + " (Ctrl+Z)")
        layout.addWidget(self.undo_button)

        self.redo_button = QToolButton()
        self.redo_button.setText(self._t("Redo"))
        self.redo_button.setEnabled(False)
        self.redo_button.setToolTip(self._t("Redo last undone operation") + " (Ctrl+Y)")
        layout.addWidget(self.redo_button)

        self.edit_geometry_button = QToolButton()
        self.edit_geometry_button.setText(self._t("Edit Shape"))
        self.edit_geometry_button.setToolTip(self._t("Add new associated shape for selected text box"))
        self.edit_geometry_button.setCheckable(True)
        layout.addWidget(self.edit_geometry_button)

        layout.addWidget(self._create_separator())

        # --- View Actions ---
        self.zoom_out_button = QToolButton()
        self.zoom_out_button.setText(self._t("Zoom Out (-)"))
        layout.addWidget(self.zoom_out_button)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(40)
        self.zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.zoom_label)

        self.zoom_in_button = QToolButton()
        self.zoom_in_button.setText(self._t("Zoom In (+)"))
        layout.addWidget(self.zoom_in_button)

        self.fit_window_button = QToolButton()
        self.fit_window_button.setText(self._t("Fit to Window"))
        layout.addWidget(self.fit_window_button)
        
        layout.addWidget(self._create_separator())

        # --- Display Mode ---
        # 创建一个容器来包装显示模式控件，确保它们作为一个整体
        display_mode_container = QWidget()
        display_mode_layout = QHBoxLayout(display_mode_container)
        display_mode_layout.setContentsMargins(0, 0, 0, 0)
        display_mode_layout.setSpacing(5)
        
        self.display_mode_label = QLabel(self._t("Display Mode:"))
        display_mode_layout.addWidget(self.display_mode_label)
        
        self.display_mode_combo = QComboBox()
        self.display_mode_combo.addItems([
            self._t("Show Text and Boxes"),
            self._t("Show Text Only"),
            self._t("Show Boxes Only"),
            self._t("Show Nothing")
        ])
        # 设置固定宽度，不使用自适应（自适应会增加额外空间）
        self.display_mode_combo.setFixedWidth(110)
        # 通过样式表缩短箭头和文字之间的距离
        self.display_mode_combo.setStyleSheet("""
            QComboBox {
                padding-right: 2px;  /* 最小化右侧内边距 */
                padding-left: 3px;
            }
            QComboBox::drop-down {
                width: 16px;  /* 缩小箭头区域 */
                border: none;
            }
        """)
        display_mode_layout.addWidget(self.display_mode_combo)
        
        # 添加分隔符到容器内
        display_mode_layout.addWidget(self._create_separator())
        
        # 设置容器的尺寸策略，防止被压缩
        from PyQt6.QtWidgets import QSizePolicy
        display_mode_container.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        
        # 将整个容器添加到主布局
        layout.addWidget(display_mode_container, 0)

        self.opacity_label = QLabel(self._t("Original Image Opacity:"))
        layout.addWidget(self.opacity_label)
        self.original_image_alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.original_image_alpha_slider.setRange(0, 100)
        self.original_image_alpha_slider.setValue(0) # Default to 0 (fully transparent, show inpainted)
        # 设置滑块自适应，较小的最小宽度
        self.original_image_alpha_slider.setMinimumWidth(80)
        layout.addWidget(self.original_image_alpha_slider)

        layout.addStretch() # Pushes everything to the left

    def _create_separator(self):
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.VLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        separator.setLineWidth(1)
        separator.setMidLineWidth(0)
        separator.setFixedWidth(2)  # 分隔符可以固定宽度，因为它只是一条线
        # 设置分隔符的最小高度，确保它垂直显示
        separator.setMinimumHeight(20)
        return separator

    def _connect_signals(self):
        self.back_button.clicked.connect(self.back_requested)
        self.export_button.clicked.connect(self.export_requested)
        self.save_json_button.clicked.connect(self.save_json_requested)
        self.edit_file_button.clicked.connect(self.edit_file_requested)
        self.undo_button.clicked.connect(self.undo_requested)
        self.redo_button.clicked.connect(self.redo_requested)
        self.edit_geometry_button.toggled.connect(self.edit_geometry_requested)
        self.zoom_in_button.clicked.connect(self.zoom_in_requested)
        self.zoom_out_button.clicked.connect(self.zoom_out_requested)
        self.fit_window_button.clicked.connect(self.fit_window_requested)
        self.display_mode_combo.currentTextChanged.connect(self.display_mode_changed)
        self.original_image_alpha_slider.valueChanged.connect(self.original_image_alpha_changed)

    # --- Public Slots ---
    def update_undo_redo_state(self, can_undo: bool, can_redo: bool):
        self.undo_button.setEnabled(can_undo)
        self.redo_button.setEnabled(can_redo)

    def set_original_image_alpha_slider(self, alpha: float):
        """同步滑块值（alpha: 0.0-1.0）"""
        # 转换：alpha 0.0 = slider 0（完全透明），alpha 1.0 = slider 100（完全不透明）
        slider_value = int(alpha * 100)
        self.original_image_alpha_slider.blockSignals(True)
        self.original_image_alpha_slider.setValue(slider_value)
        self.original_image_alpha_slider.blockSignals(False)
        # 强制更新UI
        self.undo_button.update()
        self.redo_button.update()

    def update_zoom_level(self, zoom_level: float):
        self.zoom_label.setText(f"{zoom_level:.0%}")
    
    def set_export_enabled(self, enabled: bool):
        """设置导出按钮的启用状态"""
        self.export_button.setEnabled(enabled)
    
    def refresh_ui_texts(self):
        """刷新所有UI文本（用于语言切换）"""
        # 刷新按钮文本
        self.back_button.setText(self._t("Back"))
        self.back_button.setToolTip(self._t("Back to Main"))
        self.export_button.setText(self._t("Export Image"))
        self.export_button.setToolTip(self._t("Export current rendered image") + " (Ctrl+Q)")
        self.save_json_button.setText(self._t("Save JSON"))
        self.save_json_button.setToolTip(self._t("Save translation data to JSON file") + " (Ctrl+W)")
        self.edit_file_button.setText(self._t("Edit Original"))
        self.edit_file_button.setToolTip(self._t("Switch to source file of current translation for editing") + " (Ctrl+E)")
        self.undo_button.setText(self._t("Undo"))
        self.undo_button.setToolTip(self._t("Undo last operation") + " (Ctrl+Z)")
        self.redo_button.setText(self._t("Redo"))
        self.redo_button.setToolTip(self._t("Redo last undone operation") + " (Ctrl+Y)")
        self.edit_geometry_button.setText(self._t("Edit Shape"))
        self.edit_geometry_button.setToolTip(self._t("Add new associated shape for selected text box"))
        self.zoom_out_button.setText(self._t("Zoom Out (-)"))
        self.zoom_in_button.setText(self._t("Zoom In (+)"))
        self.fit_window_button.setText(self._t("Fit to Window"))
        
        # 刷新下拉菜单
        current_index = self.display_mode_combo.currentIndex()
        self.display_mode_combo.blockSignals(True)
        self.display_mode_combo.clear()
        self.display_mode_combo.addItems([
            self._t("Show Text and Boxes"),
            self._t("Show Text Only"),
            self._t("Show Boxes Only"),
            self._t("Show Nothing")
        ])
        self.display_mode_combo.setCurrentIndex(current_index)
        self.display_mode_combo.blockSignals(False)
        
        # 刷新标签
        if hasattr(self, 'display_mode_label'):
            self.display_mode_label.setText(self._t("Display Mode:"))
        if hasattr(self, 'opacity_label'):
            self.opacity_label.setText(self._t("Original Image Opacity:"))
