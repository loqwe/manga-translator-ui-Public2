from typing import Any, Dict, List, Optional

from PyQt6.QtCore import QObject, pyqtSignal


class EditorModel(QObject):
    """
    编辑器数据模型 (Model)。
    负责封装和管理所有核心数据，如图像、区域、蒙版等。
    当数据变化时，通过信号通知视图更新。
    """
    # --- 定义信号 ---
    image_changed = pyqtSignal(object)
    regions_changed = pyqtSignal(list)
    raw_mask_changed = pyqtSignal(object)
    refined_mask_changed = pyqtSignal(object)
    display_mask_type_changed = pyqtSignal(str)
    selection_changed = pyqtSignal(list)
    inpainted_image_changed = pyqtSignal(object)
    region_display_mode_changed = pyqtSignal(str) # New signal
    show_removed_mask_changed = pyqtSignal(bool) # New signal
    source_image_path_changed = pyqtSignal(str)
    original_image_alpha_changed = pyqtSignal(float)
    region_text_updated = pyqtSignal(int) # New signal for targeted text updates
    region_style_updated = pyqtSignal(int) # NEW SIGNAL for targeted style updates
    active_tool_changed = pyqtSignal(str)
    brush_size_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_image_path: Optional[str] = None
        self._image: Optional[Any] = None
        self._regions: List[Dict[str, Any]] = []
        self._raw_mask: Optional[Any] = None
        self._refined_mask: Optional[Any] = None
        self._inpainted_image: Optional[Any] = None
        self._inpainted_image_path: Optional[str] = None  # 新增：存储inpainted图片路径
        self._display_mask_type: str = 'none'
        self._selected_indices: List[int] = []
        self._region_display_mode: str = 'full' # New property
        self._show_removed_mask: bool = False # New property
        self._original_image_alpha: float = 0.0  # 默认完全透明，显示inpainted图片
        self._active_tool: str = 'select'
        self._brush_size: int = 30
        self.controller = None  # 添加controller引用，用于命令模式

    # --- Getter / Setter 方法 ---

    def set_source_image_path(self, path: str):
        if self._source_image_path != path:
            self._source_image_path = path
            self.source_image_path_changed.emit(path)

    def get_source_image_path(self) -> Optional[str]:
        return self._source_image_path

    def set_image(self, image: Any):
        if self._image is not image:
            self._image = image
            self.image_changed.emit(image)

    def get_image(self) -> Optional[Any]:
        return self._image

    def set_regions(self, regions: List[Dict[str, Any]]):
        self._regions = regions
        self.regions_changed.emit(regions)

    def get_regions(self) -> List[Dict[str, Any]]:
        return self._regions

    def set_raw_mask(self, mask: Any):
        self._raw_mask = mask
        self.raw_mask_changed.emit(mask)

    def get_raw_mask(self) -> Optional[Any]:
        return self._raw_mask

    def set_refined_mask(self, mask: Any):
        self._refined_mask = mask
        self.refined_mask_changed.emit(mask)
        # Force immediate display update if this is the current display type
        if self._display_mask_type == 'refined':
            self.display_mask_type_changed.emit('refined')

    def get_refined_mask(self) -> Optional[Any]:
        return self._refined_mask

    def set_display_mask_type(self, mask_type: str):
        """Sets which mask ('raw', 'refined', or 'none') should be displayed."""
        if mask_type not in ['raw', 'refined', 'none']:
            return
        
        if self._display_mask_type != mask_type:
            self._display_mask_type = mask_type
            self.display_mask_type_changed.emit(mask_type)

    def get_display_mask_type(self) -> str:
        return self._display_mask_type

    def set_inpainted_image_path(self, path: Optional[str]):
        """设置inpainted图片路径"""
        self._inpainted_image_path = path

    def get_inpainted_image_path(self) -> Optional[str]:
        """获取inpainted图片路径"""
        return self._inpainted_image_path

    def set_removed_mask_visible(self, visible: bool):
        if self._show_removed_mask != visible:
            self._show_removed_mask = visible
            self.show_removed_mask_changed.emit(visible)

    def get_show_removed_mask(self) -> bool:
        return self._show_removed_mask

    def set_selection(self, indices: List[int]):
        if self._selected_indices != indices:
            self._selected_indices = sorted(indices)
            self.selection_changed.emit(self._selected_indices)

    def get_selection(self) -> List[int]:
        return self._selected_indices

    def get_region_by_index(self, index: int) -> Optional[Dict[str, Any]]:
        """通过索引安全地获取区域数据"""
        if 0 <= index < len(self._regions):
            return self._regions[index]
        return None

    def set_inpainted_image(self, image: Any):
        self._inpainted_image = image
        self.inpainted_image_changed.emit(image)

    def get_inpainted_image(self) -> Optional[Any]:
        return self._inpainted_image

    def set_region_display_mode(self, mode: str):
        """设置区域显示模式 ('full', 'text_only', 'box_only', 'none')"""
        if self._region_display_mode != mode:
            self._region_display_mode = mode
            self.region_display_mode_changed.emit(mode)

    def get_region_display_mode(self) -> str:
        return self._region_display_mode

    def set_original_image_alpha(self, alpha: float):
        if self._original_image_alpha != alpha:
            self._original_image_alpha = alpha
            self.original_image_alpha_changed.emit(alpha)

    def get_original_image_alpha(self) -> float:
        return self._original_image_alpha

    def set_active_tool(self, tool: str):
        if self._active_tool != tool:
            self._active_tool = tool
            self.active_tool_changed.emit(tool)

    def get_active_tool(self) -> str:
        return self._active_tool

    def set_brush_size(self, size: int):
        if self._brush_size != size:
            self._brush_size = size
            self.brush_size_changed.emit(size)

    def get_brush_size(self) -> int:
        return self._brush_size

    def update_region_text(self, index: int, key: str, value: str):
        """Updates only the text of a specific region and emits a targeted signal."""
        if 0 <= index < len(self._regions):
            if self._regions[index].get(key) != value:
                self._regions[index][key] = value
                self.region_text_updated.emit(index) # Emit targeted signal

    def update_region_style(self, index: int, key: str, value: Any):
        """Updates only the style of a specific region and emits a targeted signal."""
        if 0 <= index < len(self._regions):
            if self._regions[index].get(key) != value:
                self._regions[index][key] = value
                self.region_style_updated.emit(index) # Emit targeted signal

    def update_region_data(self, index: int, key: str, value: Any):
        """安全地更新单个区域的特定字段，但不发出信号。"""
        if 0 <= index < len(self._regions):
            if self._regions[index].get(key) != value:
                self._regions[index][key] = value

    def update_region_silent(self, index: int, new_data: dict):
        """静默更新整个区域数据,不发出信号"""
        if 0 <= index < len(self._regions):
            self._regions[index].update(new_data)
