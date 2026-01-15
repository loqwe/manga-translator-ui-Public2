
import os
import cv2
import numpy as np
from manga_translator.rendering import resize_regions_to_font_size
from manga_translator.utils import TextBlock
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QCursor,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
    QTransform,
)
from PyQt6.QtCore import QRectF
from PyQt6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView, QMenu

# --- 新增Imports for Refactoring ---
from editor import text_renderer_backend
from editor.editor_model import EditorModel
from editor.graphics_items import RegionTextItem, TransparentPixmapItem
from services import get_render_parameter_service

# --- 结束新增 ---

class GraphicsView(QGraphicsView):
    """
    自定义画布视图，继承自 QGraphicsView。
    负责显示和交互式操作图形项（如图片、文本框）。
    """
    region_geometry_changed = pyqtSignal(int, dict)
    geometry_added = pyqtSignal(int, list)
    _layout_result_ready = pyqtSignal(list)  # 布局计算结果信号

    def __init__(self, model: EditorModel, parent=None):
        super().__init__(parent)
        self.model = model

        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self._image_item: QGraphicsPixmapItem = None
        self._raw_mask_item: QGraphicsPixmapItem = None
        self._refined_mask_item: QGraphicsPixmapItem = None
        self._removed_mask_item: QGraphicsPixmapItem = None # For removed area
        self._inpainted_image_item: QGraphicsPixmapItem = None
        self._q_image_ref = None # Keep a reference to the QImage to prevent crashes
        self._inpainted_q_image_ref = None
        self._preview_item: QGraphicsPixmapItem = None # Dedicated item for drawing previews

        self._region_items = []
        self._image_np = None
        self._last_edited_region_index = None

        # --- Mask Editing State ---
        self._active_tool = 'select'
        self._brush_size = 30
        self._is_drawing = False
        self._mask_image: QImage = None # Holds the mask for painting
        self._last_pos = None
        self._current_draw_path: QPainterPath = None

        # --- Geometry Editing State ---
        self._is_drawing_geometry = False
        self._geometry_start_pos = None
        self._geometry_preview_item = None
        
        # 拖动相关状态
        self._potential_drag = False  # 是否可能开始拖动
        self._drag_start_pos = None  # 拖动起始位置
        self._drag_threshold = 5  # 拖动阈值（像素）

        # --- 框选状态 ---
        self._is_box_selecting = False  # 是否正在框选
        self._box_select_start_pos = None  # 框选起始位置
        self._box_select_rect_item = None  # 框选矩形显示

        # --- Text Box Drawing State ---
        self._is_drawing_textbox = False
        self._textbox_start_pos = None
        self._textbox_preview_item = None

        # --- Text Rendering Cache ---
        self._text_render_cache = {}

        # --- Debounce timer for rendering ---
        self.render_debounce_timer = QTimer(self)
        self.render_debounce_timer.setSingleShot(True)
        self.render_debounce_timer.setInterval(150) # 150ms debounce time
        self.render_debounce_timer.timeout.connect(self._perform_render_update)


        # --- 新增渲染器和缓存 ---
        self._text_blocks_cache: list[TextBlock] = []
        self._dst_points_cache: list[np.ndarray] = []
        # --- 结束新增 ---

        self._setup_view()
        self._connect_model_signals()
        
        # 连接布局结果信号
        self._layout_result_ready.connect(self._apply_layout_result)

    def _setup_view(self):
        """配置视图属性"""
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        
        # 性能优化配置
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)
        self.setCacheMode(QGraphicsView.CacheModeFlag.CacheBackground)
        self.setOptimizationFlag(QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing, True)

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        # 启用滚动条
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.scene.setBackgroundBrush(Qt.GlobalColor.darkGray)

    def _connect_model_signals(self):
        """连接模型信号到视图的槽"""
        self.model.image_changed.connect(self.on_image_changed)
        self.model.regions_changed.connect(self.on_regions_changed)
        self.model.raw_mask_changed.connect(lambda mask: self.on_mask_data_changed('raw', mask))
        self.model.refined_mask_changed.connect(lambda mask: self.on_mask_data_changed('refined', mask))
        self.model.display_mask_type_changed.connect(self.on_display_mask_type_changed)
        self.model.show_removed_mask_changed.connect(self.on_show_removed_mask_changed)
        self.model.inpainted_image_changed.connect(self.on_inpainted_image_changed)
        self.model.region_display_mode_changed.connect(self.on_region_display_mode_changed)
        self.model.original_image_alpha_changed.connect(self.on_original_image_alpha_changed)
        self.model.region_text_updated.connect(self.on_region_text_updated) # New connection
        self.model.region_style_updated.connect(self.on_region_style_updated) # NEW: For targeted style updates
        self.model.selection_changed.connect(self._on_selection_changed) # NEW: 同步selection到Qt item

        # --- Mask Editing Connections ---
        self.model.active_tool_changed.connect(self._on_active_tool_changed)
        self.model.brush_size_changed.connect(self._on_brush_size_changed)

    def _scale_mask_item(self, mask_item: QGraphicsPixmapItem):
        """Helper function to scale a mask item to match the base image item."""
        if not self._image_item or not mask_item:
            return

        img_rect = self._image_item.boundingRect()
        mask_rect = mask_item.boundingRect()

        if mask_rect.width() > 0 and mask_rect.height() > 0:
            scale_x = img_rect.width() / mask_rect.width()
            scale_y = img_rect.height() / mask_rect.height()
            transform = QTransform()
            transform.scale(scale_x, scale_y)
            mask_item.setTransform(transform)

    def clear_all_state(self):
        """清空所有状态,包括items、缓存、计时器"""
        try:
            # 停止防抖计时器
            if hasattr(self, 'render_debounce_timer') and self.render_debounce_timer.isActive():
                self.render_debounce_timer.stop()

            # 清空region items - 使用副本遍历避免修改时出错
            items_to_remove = list(self._region_items) if hasattr(self, '_region_items') else []
            for item in items_to_remove:
                try:
                    if item and hasattr(item, 'scene') and item.scene():
                        self.scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
            if hasattr(self, '_region_items'):
                self._region_items.clear()

            # 清空image items
            if self._image_item and self._image_item.scene():
                self.scene.removeItem(self._image_item)
                self._image_item = None

            if self._inpainted_image_item and self._inpainted_image_item.scene():
                self.scene.removeItem(self._inpainted_image_item)
                self._inpainted_image_item = None

            # 清空mask items
            if self._raw_mask_item and self._raw_mask_item.scene():
                self.scene.removeItem(self._raw_mask_item)
                self._raw_mask_item = None

            if self._refined_mask_item and self._refined_mask_item.scene():
                self.scene.removeItem(self._refined_mask_item)
                self._refined_mask_item = None

            if self._removed_mask_item and self._removed_mask_item.scene():
                self.scene.removeItem(self._removed_mask_item)
                self._removed_mask_item = None

            # 清空预览项（文本框、几何编辑、蒙版绘制等）
            if self._textbox_preview_item and self._textbox_preview_item.scene():
                self.scene.removeItem(self._textbox_preview_item)
                self._textbox_preview_item = None

            if self._geometry_preview_item and self._geometry_preview_item.scene():
                self.scene.removeItem(self._geometry_preview_item)
                self._geometry_preview_item = None

            if self._preview_item and self._preview_item.scene():
                self.scene.removeItem(self._preview_item)
                self._preview_item = None

            # 清空框选矩形
            if self._box_select_rect_item:
                try:
                    if self._box_select_rect_item.scene():
                        self.scene.removeItem(self._box_select_rect_item)
                except RuntimeError:
                    pass  # C++ 对象已被删除
            self._box_select_rect_item = None
            self._is_box_selecting = False
            self._box_select_start_pos = None

            # 清空缓存
            if hasattr(self, '_text_render_cache'):
                self._text_render_cache.clear()
            self._text_blocks_cache = []
            self._dst_points_cache = []

            # 重置所有绘制状态
            self._is_drawing = False
            self._is_drawing_geometry = False
            self._is_drawing_textbox = False
            self._last_edited_region_index = None
            
            # 关闭线程池（如果存在）
            if hasattr(self, '_render_executor'):
                try:
                    self._render_executor.shutdown(wait=False)
                    del self._render_executor
                except Exception:
                    pass
        except (RuntimeError, AttributeError) as e:
            # 清理过程中可能遇到已删除的对象
            print(f"[View] Warning: Error during clear_all_state: {e}")

    def on_image_changed(self, image):
        """槽：当模型中的图像变化时更新背景"""

        # 先清空所有状态
        self.clear_all_state()

        # 清空场景
        self.scene.clear()
        self._image_item = None
        self._raw_mask_item = None
        self._refined_mask_item = None
        self._removed_mask_item = None
        self._inpainted_image_item = None
        self._preview_item = None
        self._image_np = None
        self._box_select_rect_item = None  # scene.clear() 会删除所有 item

        if image is None:
            return

        try:
            self._image_np = np.array(image.convert("RGB"))
        except Exception as convert_error:
            print(f"[VIEW WARN] Failed to convert image to numpy array: {convert_error}")
            self._image_np = None

        # 修复：ImageQt 不支持 'LA' 模式，需要转换为支持的模式
        # 支持的模式：'1', 'L', 'P', 'RGB', 'RGBA'
        if image.mode not in ('1', 'L', 'P', 'RGB', 'RGBA'):
            if image.mode == 'LA':
                # LA (灰度+Alpha) -> RGBA
                image = image.convert('RGBA')
            elif 'A' in image.mode:
                # 其他带Alpha通道的模式 -> RGBA
                image = image.convert('RGBA')
            else:
                # 其他模式 -> RGB
                image = image.convert('RGB')
        
        self._q_image_ref = ImageQt(image)
        pixmap = QPixmap.fromImage(self._q_image_ref)

        self._image_item = self.scene.addPixmap(pixmap)
        self._image_item.setZValue(2) # Place it above the inpainted image (Z=1)
        # 设置原图透明度（从模型获取）
        alpha = self.model.get_original_image_alpha()
        self._image_item.setOpacity(alpha)
        # 不设置场景矩形，让它自动管理
        self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

        # 竞态条件修复：图片加载后，强制触发一次区域重绘和渲染数据计算
        # 以确保在文本区域先于图片加载的情况下，文本也能被正确渲染
        self.on_regions_changed(self.model.get_regions())

    def on_mask_data_changed(self, mask_type: str, mask_array: np.ndarray):
        """当任一蒙版数据在模型中更新时，更新对应的PixmapItem"""
        target_item = self._raw_mask_item if mask_type == 'raw' else self._refined_mask_item

        if mask_array is None or mask_array.size == 0:
            if target_item:
                target_item.setPixmap(QPixmap())
            return

        h, w = mask_array.shape[:2]
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        color_mask[mask_array > 128] = [255, 0, 0, 128]
        # 【关键修复】使用.copy()确保QImage拥有自己的内存，防止numpy数组被回收后崩溃
        q_image = QImage(color_mask.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        pixmap = QPixmap.fromImage(q_image)

        if target_item is None or target_item.scene() is None:
            # Create or recreate the mask item if it doesn't exist or was removed from scene
            if mask_type == 'raw':
                if self._raw_mask_item and self._raw_mask_item.scene():
                    self.scene.removeItem(self._raw_mask_item)
                self._raw_mask_item = TransparentPixmapItem()
                self._raw_mask_item.setPixmap(pixmap)
                self._raw_mask_item.setZValue(10)
                self.scene.addItem(self._raw_mask_item)
                self._scale_mask_item(self._raw_mask_item)
                self._raw_mask_item.setVisible(self.model.get_display_mask_type() == 'raw')
                target_item = self._raw_mask_item
            else:
                if self._refined_mask_item and self._refined_mask_item.scene():
                    self.scene.removeItem(self._refined_mask_item)
                self._refined_mask_item = TransparentPixmapItem()
                self._refined_mask_item.setPixmap(pixmap)
                self._refined_mask_item.setZValue(11)
                self.scene.addItem(self._refined_mask_item)
                self._scale_mask_item(self._refined_mask_item)
                self._refined_mask_item.setVisible(self.model.get_display_mask_type() == 'refined')
                target_item = self._refined_mask_item
        else:
            target_item.setPixmap(pixmap)
            self._scale_mask_item(target_item) # Rescale on update too

        # Force update after mask change
        self.scene.update()
        self.viewport().update()  # Also update the viewport
        self.update()

        # Additional refresh for visibility if this mask type should be visible
        current_display_type = self.model.get_display_mask_type()
        if mask_type == current_display_type and target_item:
            target_item.setVisible(True)

        # Update the removed mask view if it's active
        self.on_show_removed_mask_changed(self.model.get_show_removed_mask())

    @pyqtSlot(bool)
    def on_show_removed_mask_changed(self, visible: bool):
        """Calculates and shows the difference between raw and refined masks."""
        if not visible:
            if self._removed_mask_item:
                self._removed_mask_item.setVisible(False)
            return

        raw_mask = self.model.get_raw_mask()
        refined_mask = self.model.get_refined_mask()

        if raw_mask is None or refined_mask is None:
            if self._removed_mask_item:
                self._removed_mask_item.setVisible(False)
            return

        # Ensure masks are 2D and same shape
        if len(raw_mask.shape) > 2: raw_mask = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY)
        if len(refined_mask.shape) > 2: refined_mask = cv2.cvtColor(refined_mask, cv2.COLOR_BGR2GRAY)
        
        if raw_mask.shape != refined_mask.shape:
            # Handle potential shape mismatch, e.g., by resizing
            refined_mask = cv2.resize(refined_mask, (raw_mask.shape[1], raw_mask.shape[0]))

        # Calculate difference: pixels in raw but not in refined
        removed_mask = cv2.subtract(raw_mask, refined_mask)

        h, w = removed_mask.shape
        color_mask = np.zeros((h, w, 4), dtype=np.uint8)
        color_mask[removed_mask > 128] = [0, 0, 255, 128]  # Blue for removed parts
        # 【关键修复】使用.copy()确保QImage拥有自己的内存
        q_image = QImage(color_mask.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
        pixmap = QPixmap.fromImage(q_image)

        if self._removed_mask_item is None:
            self._removed_mask_item = TransparentPixmapItem()
            self._removed_mask_item.setPixmap(pixmap)
            self._removed_mask_item.setZValue(12) # On top of other masks
            self.scene.addItem(self._removed_mask_item)
            self._scale_mask_item(self._removed_mask_item)
        else:
            self._removed_mask_item.setPixmap(pixmap)

        self._removed_mask_item.setVisible(True)

    def on_display_mask_type_changed(self, mask_type: str):
        """根据模型状态，切换哪个蒙版图层可见"""
        # --- Defensive Fix: Ensure mask items exist if their data exists ---
        if mask_type == 'raw' and self._raw_mask_item is None and self.model.get_raw_mask() is not None:
            self.on_mask_data_changed('raw', self.model.get_raw_mask())

        if mask_type == 'refined' and self._refined_mask_item is None and self.model.get_refined_mask() is not None:
            self.on_mask_data_changed('refined', self.model.get_refined_mask())
        # --- End Defensive Fix ---

        if self._raw_mask_item:
            is_visible = (mask_type == 'raw')
            self._raw_mask_item.setVisible(is_visible)
        if self._refined_mask_item:
            is_visible = (mask_type == 'refined')
            self._refined_mask_item.setVisible(is_visible)

        # Force scene refresh
        self.scene.update()
        self.viewport().update()  # Also update the viewport
        self.update()

        # Additional repaint call for immediate visual update
        self.repaint()

    def on_inpainted_image_changed(self, image):
        """槽：当模型中的修复后图像变化时更新"""
        if self._image_item is None:
            return # Cannot display inpainted if there is no base image

        if image is None:
            if self._inpainted_image_item:
                self._inpainted_image_item.setVisible(False)
            self._inpainted_q_image_ref = None
            return

        # 修复：ImageQt 不支持 'LA' 模式，需要转换为支持的模式
        if image.mode not in ('1', 'L', 'P', 'RGB', 'RGBA'):
            if image.mode == 'LA':
                image = image.convert('RGBA')
            elif 'A' in image.mode:
                image = image.convert('RGBA')
            else:
                image = image.convert('RGB')
        
        self._inpainted_q_image_ref = ImageQt(image)
        pixmap = QPixmap.fromImage(self._inpainted_q_image_ref)

        if self._inpainted_image_item is None:
            self._inpainted_image_item = TransparentPixmapItem()
            self._inpainted_image_item.setPixmap(pixmap)
            self._inpainted_image_item.setZValue(1) # Below masks and text
            self._inpainted_image_item.setOpacity(1.0) # 始终完全不透明
            self.scene.addItem(self._inpainted_image_item)
        else:
            self._inpainted_image_item.setPixmap(pixmap)
            self._inpainted_image_item.setOpacity(1.0) # 始终完全不透明

        self._inpainted_image_item.setVisible(True)

    @pyqtSlot(int)
    def on_region_text_updated(self, region_index: int):
        """Slot for targeted update when only text changes."""
        self._perform_single_item_update(region_index)

    @pyqtSlot(int)
    def on_region_style_updated(self, region_index: int):
        """Slot for targeted update when only style changes."""
        self._perform_single_item_update(region_index)

    @pyqtSlot(float)
    def on_original_image_alpha_changed(self, alpha: float):
        """槽：当原图透明度变化时更新"""
        if self._image_item:
            self._image_item.setOpacity(alpha)

    def on_region_display_mode_changed(self, mode: str):
        """槽：当区域显示模式变化时更新所有文本项的可见性"""
        for item in self.scene.items():
            if isinstance(item, RegionTextItem):
                if mode == "full":
                    item.setVisible(True)
                    item.set_text_visible(True)
                    item.set_box_visible(True)
                    # 显示绿框和白框
                    item.set_green_box_visible(True)
                    item.set_white_box_visible(True)
                elif mode == "text_only":
                    item.setVisible(True)
                    item.set_text_visible(True)
                    item.set_box_visible(False)
                    # 隐藏绿框和白框
                    item.set_green_box_visible(False)
                    item.set_white_box_visible(False)
                elif mode == "box_only":
                    item.setVisible(True)
                    item.set_text_visible(False)
                    item.set_box_visible(True)
                    # 显示绿框和白框
                    item.set_green_box_visible(True)
                    item.set_white_box_visible(True)
                elif mode == "none":
                    item.setVisible(False)
                    # 隐藏绿框和白框
                    item.set_green_box_visible(False)
                    item.set_white_box_visible(False)



    def on_regions_changed(self, regions):
        """槽：当模型中的区域数据变化时，根据情况选择性更新或完全更新。"""
        if self._last_edited_region_index is not None:
            # A specific item was manipulated in the view. Perform a targeted update.
            self._perform_single_item_update(self._last_edited_region_index)
            self._last_edited_region_index = None # Reset after use
        else:
            # A general change occurred (e.g., new translation). Perform full debounced update.
            self._text_render_cache.clear()
            self.render_debounce_timer.start()

    def _perform_single_item_update(self, index):
        """Delegates the update logic to the RegionTextItem itself."""
        try:
            if not (0 <= index < len(self._region_items)):
                return

            region_data = self.model.get_region_by_index(index)
            item = self._region_items[index]

            # 安全检查：确保item仍然有效
            if item is None:
                return
            
            # 检查item是否仍在场景中（可能已被删除）
            if not hasattr(item, 'scene') or item.scene() is None:
                return

            if region_data and item:
                # The item is now responsible for updating itself from the new data.
                item.update_from_data(region_data)

                # 重新计算单个区域的渲染数据
                self._recalculate_single_region_render_data(index)

                # 重新渲染单个区域的文字
                self._update_single_region_text_visual(index)

                # 重新渲染后,再次更新白框
                updated_region_data = self.model.get_region_by_index(index)
                if updated_region_data and item.scene() is not None:
                    item.update_from_data(updated_region_data)

                if item.scene() is not None:
                    item.update() # Trigger a repaint for the item.
        except (RuntimeError, AttributeError) as e:
            # Item可能在更新过程中被删除
            print(f"[View] Warning: Item update failed for index {index}: {e}")



    def _perform_render_update(self):
        """执行实际的渲染更新，由防抖计时器调用。"""
        try:
            # Reuse existing items to improve performance and stability
            regions = self.model.get_regions()
            current_items = self._region_items
            
            # 1. Remove excess items
            while len(current_items) > len(regions):
                item = current_items.pop()
                try:
                    if item and hasattr(item, 'scene') and item.scene():
                        self.scene.removeItem(item)
                except (RuntimeError, AttributeError):
                    pass
            
            # 2. Create new items if needed
            while len(current_items) < len(regions):
                i = len(current_items)
                region_data = regions[i]
                if not region_data.get('lines'):
                    # Placeholder or skip? If we skip, indices might mismatch.
                    # Better to create it but maybe hide it if invalid?
                    # For now, assume valid or handle in update.
                    pass
                
                item = RegionTextItem(
                    region_data,
                    i,
                    geometry_callback=self._on_region_geometry_changed,
                )
                item.set_image_item(self._image_item)
                item.setZValue(100)
                self.scene.addItem(item)
                current_items.append(item)
            
            # 3. Update all items with new data
            for i, region_data in enumerate(regions):
                if i < len(current_items):
                    item = current_items[i]
                    # Ensure image item reference is up to date
                    item.set_image_item(self._image_item)
                    # Update index (crucial if items were shifted)
                    item.region_index = i
                    # Update content
                    item.update_from_data(region_data)

            # After updating items, recalculate all rendering data (异步执行)
            # _update_text_visuals 会在 recalculate_render_data 完成后自动调用
            self.recalculate_render_data()
        except Exception as e:
            print(f"[View] Warning: Render update failed: {e}")

    def _update_text_visuals(self):
        try:
            if self.model.get_region_display_mode() in ["box_only", "none"]:
                for item in self._region_items:
                    if item and hasattr(item, 'text_item') and item.text_item and item.scene():
                        item.text_item.setVisible(False)
                return
            else:
                for item in self._region_items:
                    if item and hasattr(item, 'text_item') and item.text_item and item.scene():
                        item.text_item.setVisible(True)

            render_parameter_service = get_render_parameter_service()

            for i, text_block in enumerate(self._text_blocks_cache):
                item = self._region_items[i] if i < len(self._region_items) else None
                if not item or item.scene() is None:
                    continue

                if text_block is None or i >= len(self._dst_points_cache) or self._dst_points_cache[i] is None:
                    if item.scene() is not None:
                        item.update_text_pixmap(QPixmap(), QPointF(0, 0))
                        item.set_dst_points(None)  # 清除绿框数据
                    continue

                region_data = self.model.get_region_by_index(i)
                if not region_data:
                    continue

                # --- FIX: Create a temporary un-rotated text block for rendering ---
                render_region_data = region_data.copy()
                render_region_data['angle'] = 0

                constructor_args = render_region_data.copy()
                if 'lines' in constructor_args and isinstance(constructor_args['lines'], list):
                    constructor_args['lines'] = np.array(constructor_args['lines'])
                if 'font_color' in constructor_args and isinstance(constructor_args['font_color'], str):
                    hex_color = constructor_args.pop('font_color')
                    try:
                        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                        constructor_args['fg_color'] = (r, g, b)
                    except (ValueError, TypeError):
                        constructor_args['fg_color'] = (0, 0, 0)
                elif 'fg_colors' in constructor_args:
                    constructor_args['fg_color'] = constructor_args.pop('fg_colors')
                if 'bg_colors' in constructor_args:
                    constructor_args['bg_color'] = constructor_args.pop('bg_colors')

                # 使用缓存中计算好的 font_size（经过 resize_regions_to_font_size 计算的）
                if text_block is not None and hasattr(text_block, 'font_size'):
                    constructor_args['font_size'] = text_block.font_size

                try:
                    unrotated_text_block = TextBlock(**constructor_args)
                except Exception as e:
                    print(f"[View] Failed to create unrotated TextBlock for full update on index {i}: {e}")
                    item.update_text_pixmap(QPixmap(), QPointF(0, 0))
                    item.set_dst_points(None)  # 清除绿框数据
                    continue
                # --- END FIX ---

                render_params = render_parameter_service.export_parameters_for_backend(i, region_data)
                # 同步算法计算的 font_size 到 render_params，确保缓存键使用正确的值
                render_params['font_size'] = unrotated_text_block.font_size

                # Set region-specific font_path for cache key
                region_font_path = region_data.get('font_path', '')
                if region_font_path and os.path.exists(region_font_path):
                    render_params['font_path'] = region_font_path

                cache_key = (
                    unrotated_text_block.get_translation_for_rendering(),
                    tuple(map(tuple, self._dst_points_cache[i].reshape(-1, 2))),
                    render_params.get('font_path'),
                    render_params.get('font_size'),
                    render_params.get('bold'),
                    render_params.get('italic'),
                    render_params.get('font_weight'),
                    tuple(render_params.get('font_color', (0,0,0))),
                    tuple(render_params.get('text_stroke_color', (0,0,0))),
                    render_params.get('opacity'),
                    render_params.get('alignment'),
                    render_params.get('direction'),
                    render_params.get('vertical'),
                    render_params.get('line_spacing'),
                    render_params.get('letter_spacing'),
                    render_params.get('layout_mode'),
                    render_params.get('disable_auto_wrap'),
                    render_params.get('hyphenate'),
                    render_params.get('font_size_offset'),
                    render_params.get('font_size_minimum'),
                    render_params.get('max_font_size'),
                    render_params.get('font_scale_ratio'),
                    render_params.get('center_text_in_bubble'),
                    render_params.get('text_stroke_width'),
                    render_params.get('shadow_radius'),
                    render_params.get('shadow_strength'),
                    tuple(render_params.get('shadow_color', (0,0,0))),
                    tuple(render_params.get('shadow_offset', [0.0, 0.0])),
                    render_params.get('disable_font_border'),
                    render_params.get('auto_rotate_symbols'),
                )
                cached_result = self._text_render_cache.get(cache_key)

                if cached_result is None:
                    # Set font for this region if specified
                    region_font_path = region_data.get('font_path', '')
                    if region_font_path and os.path.exists(region_font_path):
                        text_renderer_backend.update_font_config(region_font_path)
                    else:
                        # Use default font from global parameters
                        default_params_obj = render_parameter_service.get_default_parameters()
                        font_path = default_params_obj.font_path
                        if font_path:
                            text_renderer_backend.update_font_config(font_path)

                    identity_transform = QTransform()
                    render_result = text_renderer_backend.render_text_for_region(
                        unrotated_text_block,
                        self._dst_points_cache[i],
                        identity_transform,
                        render_params,
                        pure_zoom=1.0,
                        total_regions=len(self._text_blocks_cache)
                    )

                    if render_result:
                        pixmap, pos = render_result
                        self._text_render_cache[cache_key] = (pixmap, pos)
                        cached_result = (pixmap, pos)

                if cached_result:
                    pixmap, pos = cached_result
                    # 不传递 angle，因为 item 已经通过 setRotation() 设置了旋转
                    pivot = None

                    if item.scene() is not None:
                        item.update_text_pixmap(pixmap, pos, 0, pivot)
                else:
                    if item.scene() is not None:
                        item.update_text_pixmap(QPixmap(), QPointF(0, 0))

                # 无论是否有渲染结果,都设置绿框数据(即使 translation 为空)
                if item.scene() is not None and i < len(self._dst_points_cache):
                    item.set_dst_points(self._dst_points_cache[i])

        except (RuntimeError, AttributeError) as e:
            # 处理item在更新过程中被删除的情况
            print(f"[View] Warning: Text visuals update failed: {e}")

    def _recalculate_single_region_render_data(self, index):
        """重新计算单个区域的渲染数据"""
        regions = self.model.get_regions()
        if self._image_np is None or not regions or not (0 <= index < len(regions)):
            return

        # 确保缓存列表足够大
        while len(self._text_blocks_cache) <= index:
            self._text_blocks_cache.append(None)
        while len(self._dst_points_cache) <= index:
            self._dst_points_cache.append(None)

        region_dict = regions[index]

        # 1. 将字典转换为 TextBlock 对象（保留原始的 angle）
        constructor_args = region_dict.copy()
        
        if 'lines' in constructor_args and isinstance(constructor_args['lines'], list):
            constructor_args['lines'] = np.array(constructor_args['lines'])

        # 确保 texts 不为 None
        if constructor_args.get('texts') is None:
            constructor_args['texts'] = []

        # 转换颜色格式和键名
        if 'font_color' in constructor_args and isinstance(constructor_args['font_color'], str):
            hex_color = constructor_args.pop('font_color')
            try:
                r = int(hex_color[1:3], 16)
                g = int(hex_color[3:5], 16)
                b = int(hex_color[5:7], 16)
                constructor_args['fg_color'] = (r, g, b)
            except (ValueError, TypeError):
                constructor_args['fg_color'] = (0, 0, 0)
        elif 'fg_colors' in constructor_args:
            constructor_args['fg_color'] = constructor_args.pop('fg_colors')

        if 'bg_colors' in constructor_args:
            constructor_args['bg_color'] = constructor_args.pop('bg_colors')

        # 【关键修复】转换direction字段：'horizontal'→'h', 'vertical'→'v'
        # TextBlock.direction只认'h', 'v', 'hr', 'vr'，不认'horizontal'/'vertical'
        if 'direction' in constructor_args:
            dir_val = constructor_args['direction']
            if dir_val == 'horizontal':
                constructor_args['direction'] = 'h'
            elif dir_val == 'vertical':
                constructor_args['direction'] = 'v'
            # 'h', 'v', 'hr', 'vr', 'auto' 保持不变

        # 创建未旋转的 text block 用于计算 dst_points
        # 因为 Qt 会通过 setRotation() 来应用旋转,所以这里需要使用 angle=0
        constructor_args['angle'] = 0

        try:
            text_block = TextBlock(**constructor_args)
            self._text_blocks_cache[index] = text_block
        except Exception as e:
            print(f"[View] Failed to create TextBlock for region {index}: {e}")
            import traceback
            traceback.print_exc()
            self._text_blocks_cache[index] = None
            return

        # 2. 获取全局渲染参数并设置字体
        render_parameter_service = get_render_parameter_service()
        default_params_obj = render_parameter_service.get_default_parameters()
        global_params_dict = default_params_obj.to_dict()

        font_path = default_params_obj.font_path
        if font_path:
            text_renderer_backend.update_font_config(font_path)

        from manga_translator.config import Config, RenderConfig

        # Hotfix for direction enum validation
        if global_params_dict.get('direction') == 'h':
            global_params_dict['direction'] = 'horizontal'
        elif global_params_dict.get('direction') == 'v':
            global_params_dict['direction'] = 'vertical'

        config_obj = Config(render=RenderConfig(**global_params_dict))

        # 3. 调用后端进行布局计算（只计算单个区域）
        try:
            single_region_dst_points = resize_regions_to_font_size(self._image_np, [text_block], config_obj, self._image_np)
            if single_region_dst_points and len(single_region_dst_points) > 0:
                self._dst_points_cache[index] = single_region_dst_points[0]
            else:
                self._dst_points_cache[index] = None
        except Exception as e:
            print(f"[View] Failed to calculate dst_points for region {index}: {e}")
            import traceback
            traceback.print_exc()
            self._dst_points_cache[index] = None

    def _update_single_region_text_visual(self, index):
        """重新渲染单个区域的文字"""
        try:
            if not (0 <= index < len(self._region_items)):
                return

            item = self._region_items[index]
            
            # 安全检查：确保item有效且在场景中
            if item is None or not hasattr(item, 'scene') or item.scene() is None:
                return
            
            if not hasattr(item, 'text_item') or item.text_item is None:
                return

            if self.model.get_region_display_mode() in ["box_only", "none"]:
                item.text_item.setVisible(False)
                return
            else:
                item.text_item.setVisible(True)

            if index >= len(self._text_blocks_cache) or index >= len(self._dst_points_cache):
                return

            text_block = self._text_blocks_cache[index] if index < len(self._text_blocks_cache) else None
            dst_points = self._dst_points_cache[index] if index < len(self._dst_points_cache) else None

            if text_block is None or dst_points is None:
                item.update_text_pixmap(QPixmap(), QPointF(0, 0))
                item.set_dst_points(None)
                return

            region_data = self.model.get_region_by_index(index)
            if not region_data:
                return

            # 创建未旋转的 text block 用于渲染
            render_region_data = region_data.copy()
            render_region_data['angle'] = 0

            constructor_args = render_region_data.copy()
            if 'lines' in constructor_args and isinstance(constructor_args['lines'], list):
                constructor_args['lines'] = np.array(constructor_args['lines'])
            if 'font_color' in constructor_args and isinstance(constructor_args['font_color'], str):
                hex_color = constructor_args.pop('font_color')
                try:
                    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
                    constructor_args['fg_color'] = (r, g, b)
                except (ValueError, TypeError):
                    constructor_args['fg_color'] = (0, 0, 0)
            elif 'fg_colors' in constructor_args:
                constructor_args['fg_color'] = constructor_args.pop('fg_colors')
            if 'bg_colors' in constructor_args:
                constructor_args['bg_color'] = constructor_args.pop('bg_colors')

            # 【关键修复】转换direction字段：'horizontal'→'h', 'vertical'→'v'
            if 'direction' in constructor_args:
                dir_val = constructor_args['direction']
                if dir_val == 'horizontal':
                    constructor_args['direction'] = 'h'
                elif dir_val == 'vertical':
                    constructor_args['direction'] = 'v'

            # 使用缓存中计算好的 font_size（经过 resize_regions_to_font_size 计算的）
            if text_block is not None and hasattr(text_block, 'font_size'):
                constructor_args['font_size'] = text_block.font_size

            try:
                unrotated_text_block = TextBlock(**constructor_args)
            except Exception as e:
                print(f"[View] Failed to create unrotated TextBlock for region {index}: {e}")
                item.update_text_pixmap(QPixmap(), QPointF(0, 0))
                item.set_dst_points(None)
                return

            render_parameter_service = get_render_parameter_service()
            render_params = render_parameter_service.export_parameters_for_backend(index, region_data)
            # 同步算法计算的 font_size 到 render_params
            render_params['font_size'] = unrotated_text_block.font_size

            # Set font for this region if specified
            region_font_path = region_data.get('font_path', '')
            if region_font_path and os.path.exists(region_font_path):
                text_renderer_backend.update_font_config(region_font_path)
            else:
                # Use default font from global parameters
                default_params_obj = render_parameter_service.get_default_parameters()
                font_path = default_params_obj.font_path
                if font_path:
                    text_renderer_backend.update_font_config(font_path)

            # 渲染文字（不使用缓存，因为几何已经改变）
            identity_transform = QTransform()
            render_result = text_renderer_backend.render_text_for_region(
                unrotated_text_block,
                self._dst_points_cache[index],
                identity_transform,
                render_params,
                pure_zoom=1.0,
                total_regions=len(self._text_blocks_cache)
            )

            if render_result:
                pixmap, pos = render_result
                # 不传递 angle，因为 item 已经通过 setRotation() 设置了旋转
                if item.scene() is not None:  # 再次检查item是否仍有效
                    item.update_text_pixmap(pixmap, pos, 0, None)
                    item.set_dst_points(dst_points)
            else:
                if item.scene() is not None:
                    item.update_text_pixmap(QPixmap(), QPointF(0, 0))
                    item.set_dst_points(None)
        except (RuntimeError, AttributeError) as e:
            # Item可能在渲染过程中被删除
            print(f"[View] Warning: Text visual update failed for index {index}: {e}")
        except Exception as e:
            print(f"[View] Error in _update_single_region_text_visual for index {index}: {e}")

    def recalculate_render_data(self):
        """
        执行昂贵的布局计算并将结果缓存。
        这个方法应该在 regions 数据变化后被调用。
        使用线程池异步执行以避免阻塞UI。
        """
        regions = self.model.get_regions()
        if self._image_np is None or not regions:
            self._text_blocks_cache = []
            self._dst_points_cache = []
            return

        # 1. 将字典转换为 TextBlock 对象 (从旧版UI移植的正确逻辑)
        text_blocks = []
        for region_dict in regions:
            constructor_args = region_dict.copy()
            if 'lines' in constructor_args and isinstance(constructor_args['lines'], list):
                constructor_args['lines'] = np.array(constructor_args['lines'])

            # 转换颜色格式和键名
            if 'font_color' in constructor_args and isinstance(constructor_args['font_color'], str):
                hex_color = constructor_args.pop('font_color')
                try:
                    r = int(hex_color[1:3], 16)
                    g = int(hex_color[3:5], 16)
                    b = int(hex_color[5:7], 16)
                    constructor_args['fg_color'] = (r, g, b)
                except (ValueError, TypeError):
                    constructor_args['fg_color'] = (0, 0, 0)
            elif 'fg_colors' in constructor_args:
                 constructor_args['fg_color'] = constructor_args.pop('fg_colors')
            
            if 'bg_colors' in constructor_args:
                constructor_args['bg_color'] = constructor_args.pop('bg_colors')

            # 创建未旋转的 text block 用于计算 dst_points
            # 因为 Qt 会通过 setRotation() 来应用旋转,所以这里需要使用 angle=0
            constructor_args['angle'] = 0

            try:
                text_block = TextBlock(**constructor_args)
                text_blocks.append(text_block)
            except Exception as e:
                print(f"[View] Failed to create TextBlock for region: {e}")
                text_blocks.append(None)
        self._text_blocks_cache = text_blocks

        # 2. 获取全局渲染参数
        render_parameter_service = get_render_parameter_service()
        default_params_obj = render_parameter_service.get_default_parameters()
        global_params_dict = default_params_obj.to_dict()
        
        # 关键修复：在调用任何渲染函数之前，确保字体已设置
        font_path = default_params_obj.font_path
        if font_path:
            text_renderer_backend.update_font_config(font_path)
        
        from manga_translator.config import Config, RenderConfig
        
        # Hotfix for direction enum validation
        if global_params_dict.get('direction') == 'h':
            global_params_dict['direction'] = 'horizontal'
        elif global_params_dict.get('direction') == 'v':
            global_params_dict['direction'] = 'vertical'
            
        config_obj = Config(render=RenderConfig(**global_params_dict))

        # 3. 异步执行昂贵的布局计算
        valid_blocks = [b for b in self._text_blocks_cache if b is not None]
        if not valid_blocks:
            self._dst_points_cache = []
            return
        
        # 使用线程池异步计算
        import concurrent.futures
        if not hasattr(self, '_render_executor'):
            self._render_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        # 保存当前状态用于回调
        image_np = self._image_np
        
        def compute_layout():
            try:
                return resize_regions_to_font_size(image_np, valid_blocks, config_obj, image_np)
            except Exception as e:
                print(f"[View] Error during resize_regions_to_font_size: {e}")
                import traceback
                traceback.print_exc()
                return [None] * len(valid_blocks)
        
        def on_layout_complete(future):
            try:
                result = future.result()
                # 使用信号在主线程更新缓存和UI（信号是线程安全的）
                self._layout_result_ready.emit(result)
            except Exception as e:
                print(f"[View] Layout computation failed: {e}")
        
        future = self._render_executor.submit(compute_layout)
        future.add_done_callback(on_layout_complete)
    
    @pyqtSlot(list)
    def _apply_layout_result(self, dst_points_cache):
        """在主线程应用布局计算结果"""
        try:
            self._dst_points_cache = dst_points_cache
            self._update_text_visuals()
            self.scene.update()
        except Exception as e:
            print(f"[View] Error applying layout result: {e}")



    def _on_region_geometry_changed(self, region_index, new_region_data):
        self._last_edited_region_index = region_index
        self.region_geometry_changed.emit(region_index, new_region_data)



    def resizeEvent(self, event):
        """处理视图大小改变事件"""
        # 不自动适应视图，保持用户的缩放级别
        super().resizeEvent(event)

    def wheelEvent(self, event):
        """处理鼠标滚轮事件以实现缩放"""
        zoom_in_factor = 1.15
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)
        
        self._update_cursor() # Update cursor size on zoom

    def mousePressEvent(self, event):
        """处理鼠标按下事件以实现平移、选择和开始绘图"""
        # 确保点击画布时，保存文本框的编辑内容
        parent_view = self.parent()
        if hasattr(parent_view, 'force_save_property_panel_edits'):
            parent_view.force_save_property_panel_edits()
        
        # 让画布获取焦点
        self.setFocus()
        
        if self._active_tool == 'geometry_edit' and event.button() == Qt.MouseButton.LeftButton:
            if len(self.model.get_selection()) == 1:
                self._is_drawing_geometry = True
                self._geometry_start_pos = self.mapToScene(event.pos())
                
                if self._geometry_preview_item is None:
                    pen = QPen(QColor(0, 255, 255, 200)) # Cyan, semi-transparent
                    pen.setWidth(2)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    self._geometry_preview_item = self.scene.addPolygon(QPolygonF(), pen)
                    self._geometry_preview_item.setZValue(200)
                
                self._geometry_preview_item.setPolygon(QPolygonF())
                self._geometry_preview_item.setVisible(True)
                event.accept()
                return

        if self._active_tool == 'draw_textbox' and event.button() == Qt.MouseButton.LeftButton:
            self._start_drawing_textbox(event.pos())
            event.accept()
            return

        if self._active_tool in ['pen', 'eraser', 'brush'] and event.button() == Qt.MouseButton.LeftButton:
            self._start_drawing(event.pos())
            event.accept()
            return

        # --- Original Selection Logic ---
        if event.button() == Qt.MouseButton.LeftButton and self._active_tool == 'select':
            # 不再依赖 scene.items()，直接让事件传播
            # Qt 的事件分发系统会自动找到正确的 item（通过 shape().contains()）
            # 如果有 item 处理了事件，事件会被 accept
            # 如果没有 item 处理，事件会返回到这里，我们再处理空白点击
            pass  # 让事件继续传播

        # --- Panning and Item Interaction Logic ---
        if event.button() == Qt.MouseButton.MiddleButton:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            # Create a new event with LeftButton instead of modifying the original
            from PyQt6.QtGui import QMouseEvent
            dummy_event = QMouseEvent(
                event.type(),
                event.position(),
                event.globalPosition(),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                event.modifiers()
            )
            super().mousePressEvent(dummy_event)
        elif event.button() == Qt.MouseButton.LeftButton:
            # 检查是否点击在空白区域
            item_at_pos = self.itemAt(event.pos())
            
            # 检查是否点击在 RegionTextItem 上（包括其子元素）
            clicked_region_item = False
            if item_at_pos:
                # 向上查找父元素，看是否是 RegionTextItem
                check_item = item_at_pos
                while check_item:
                    if isinstance(check_item, RegionTextItem):
                        clicked_region_item = True
                        break
                    check_item = check_item.parentItem()

            # 如果点击在空白区域（没有 item 或只有图片），开始框选
            if item_at_pos is None or item_at_pos == self._image_item:
                self._is_box_selecting = True
                self._box_select_start_pos = self.mapToScene(event.pos())
                
                # 创建框选矩形（检查是否需要重新创建）
                need_create = self._box_select_rect_item is None
                if not need_create:
                    # 检查 C++ 对象是否仍然有效
                    try:
                        _ = self._box_select_rect_item.scene()
                    except RuntimeError:
                        need_create = True
                        self._box_select_rect_item = None
                
                if need_create:
                    pen = QPen(QColor(0, 120, 215, 180))  # 蓝色半透明
                    pen.setWidth(2)
                    pen.setStyle(Qt.PenStyle.DashLine)
                    brush = QBrush(QColor(0, 120, 215, 30))  # 浅蓝色填充
                    self._box_select_rect_item = self.scene.addRect(0, 0, 0, 0, pen, brush)
                    self._box_select_rect_item.setZValue(300)  # 在最上层
                
                self._box_select_rect_item.setVisible(True)
                event.accept()
                return

            # 如果点击在 RegionTextItem 上，让 item 处理事件
            if clicked_region_item:
                super().mousePressEvent(event)
                # 不要在这里清除选择，让 RegionTextItem 自己处理
            else:
                super().mousePressEvent(event)
                
                # 如果点击空白且不是 Ctrl，清除选择
                if not event.isAccepted():
                    ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if not ctrl_pressed:
                        self.model.set_selection([])
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move for drawing."""
        # 处理框选
        if self._is_box_selecting and self._box_select_start_pos:
            # 检查框选矩形是否仍然有效（可能在切换图片时被删除）
            if self._box_select_rect_item is None:
                self._is_box_selecting = False
                self._box_select_start_pos = None
                return
            try:
                current_pos = self.mapToScene(event.pos())
                # 更新框选矩形
                rect = QRectF(self._box_select_start_pos, current_pos).normalized()
                self._box_select_rect_item.setRect(rect)
                event.accept()
                return
            except RuntimeError:
                # C++ 对象已被删除，重置状态
                self._box_select_rect_item = None
                self._is_box_selecting = False
                self._box_select_start_pos = None
                return
        
        if self._is_drawing_textbox:
            self._update_textbox_drawing(event.pos())
            event.accept()
            return

        if self._is_drawing_geometry:
            if self._geometry_preview_item and self._image_item:
                selected_indices = self.model.get_selection()
                if selected_indices:
                    region_data = self.model.get_region_by_index(selected_indices[0])
                    angle = region_data.get('angle', 0) if region_data else 0
                else:
                    angle = 0
                start_pos_image = self._image_item.mapFromScene(self._geometry_start_pos)
                current_pos_scene = self.mapToScene(event.pos())
                current_pos_image = self._image_item.mapFromScene(current_pos_scene)
                from .desktop_ui_geometry import calculate_rectangle_from_diagonal
                poly_coords_image = calculate_rectangle_from_diagonal(
                    (start_pos_image.x(), start_pos_image.y()),
                    (current_pos_image.x(), current_pos_image.y()),
                    angle
                )
                poly_scene = QPolygonF()
                for p in poly_coords_image:
                    poly_scene.append(self._image_item.mapToScene(QPointF(p[0], p[1])))
                self._geometry_preview_item.setPolygon(poly_scene)
            event.accept()
            return

        if self._is_drawing:
            self._update_preview_drawing(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """处理鼠标释放事件"""
        # 处理框选完成
        if self._is_box_selecting and event.button() == Qt.MouseButton.LeftButton:
            self._is_box_selecting = False
            self._box_select_start_pos = None
            
            # 隐藏框选矩形（检查对象是否仍然有效）
            if self._box_select_rect_item:
                try:
                    select_rect = self._box_select_rect_item.rect()
                    self._box_select_rect_item.setVisible(False)
                    
                    # 查找框内的所有 RegionTextItem
                    selected_indices = []
                    for i, item in enumerate(self._region_items):
                        if isinstance(item, RegionTextItem):
                            # 检查 item 的边界是否与选择框相交
                            item_rect = item.sceneBoundingRect()
                            if select_rect.intersects(item_rect):
                                selected_indices.append(i)
                    
                    # 更新选择
                    ctrl_pressed = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
                    if ctrl_pressed:
                        # Ctrl+框选：添加到现有选择
                        current_selection = self.model.get_selection()
                        new_selection = list(set(current_selection + selected_indices))
                        self.model.set_selection(new_selection)
                    else:
                        # 普通框选：替换选择
                        self.model.set_selection(selected_indices)
                except RuntimeError:
                    # C++ 对象已被删除，重置引用
                    self._box_select_rect_item = None
            
            event.accept()
            return
        
        if self._is_drawing_textbox and event.button() == Qt.MouseButton.LeftButton:
            self._finish_textbox_drawing()
            event.accept()
            return

        if self._is_drawing_geometry and event.button() == Qt.MouseButton.LeftButton:
            self._is_drawing_geometry = False
            if self._geometry_preview_item:
                self._geometry_preview_item.setVisible(False)

                selected_indices = self.model.get_selection()
                if selected_indices and self._image_item:
                    region_data = self.model.get_region_by_index(selected_indices[0])
                    angle = region_data.get('angle', 0) if region_data else 0

                    start_pos_image = self._image_item.mapFromScene(self._geometry_start_pos)
                    current_pos_scene = self.mapToScene(event.pos())
                    current_pos_image = self._image_item.mapFromScene(current_pos_scene)

                    from .desktop_ui_geometry import calculate_rectangle_from_diagonal

                    final_poly_image = calculate_rectangle_from_diagonal(
                        (start_pos_image.x(), start_pos_image.y()),
                        (current_pos_image.x(), current_pos_image.y()),
                        angle
                    )

                    self._last_edited_region_index = selected_indices[0]
                    self.geometry_added.emit(selected_indices[0], final_poly_image)
            event.accept()
            return

        if self._is_drawing and event.button() == Qt.MouseButton.LeftButton:
            self._finish_drawing()

        if event.button() == Qt.MouseButton.MiddleButton or event.button() == Qt.MouseButton.LeftButton:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            # 清除拖动状态
            if self._drag_start_pos:
                self._potential_drag = False
                self._drag_start_pos = None
        super().mouseReleaseEvent(event)

    def _start_drawing(self, pos):
        if self._image_item is None:
            return
        self._is_drawing = True
        self._current_draw_path = QPainterPath()
        self._current_draw_path.moveTo(self.mapToScene(pos))

        # Ensure the preview item exists and is ready
        if self._preview_item is None:
            pixmap = QPixmap(self._image_item.pixmap().size())
            pixmap.fill(Qt.GlobalColor.transparent)
            self._preview_item = self.scene.addPixmap(pixmap)
            self._preview_item.setZValue(150) # Ensure preview is on top of everything
            self._scale_mask_item(self._preview_item)
        self._preview_item.setVisible(True)

    def _update_preview_drawing(self, pos):
        if not self._is_drawing:
            return
        self._current_draw_path.lineTo(self.mapToScene(pos))

        # Draw the path on the temporary preview item
        pixmap = self._preview_item.pixmap()
        pixmap.fill(Qt.GlobalColor.transparent) # Clear previous preview
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Different preview for different tools
        if self._active_tool in ['pen', 'brush']:
            # Pen preview: semi-transparent red for addition
            preview_pen = QPen(QColor(255, 0, 0, 128), self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(preview_pen)
            painter.drawPath(self._current_draw_path)
        elif self._active_tool == 'eraser':
            # Eraser preview: semi-transparent blue/gray for removal indication
            preview_pen = QPen(QColor(0, 150, 255, 100), self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setPen(preview_pen)
            painter.drawPath(self._current_draw_path)

        painter.end()

        self._preview_item.setPixmap(pixmap)
        # Force immediate scene update for real-time feedback
        self.scene.update()
        self.viewport().update()

    def _finish_drawing(self):
        if not self._is_drawing or self._current_draw_path is None or self._current_draw_path.isEmpty():
            self._is_drawing = False
            # Clear preview even if no valid path was drawn
            self._clear_preview()
            return
        self._is_drawing = False

        # Get the current mask or create a new one
        current_mask = self.model.get_refined_mask()
        if current_mask is None:
            old_mask_np = None
            h, w = self._image_item.pixmap().height(), self._image_item.pixmap().width()
            mask_image = QImage(w, h, QImage.Format.Format_Grayscale8)
            mask_image.fill(Qt.GlobalColor.black)
        else:
            # 重要：创建原始蒙版的副本，避免引用问题
            old_mask_np = current_mask.copy() if current_mask is not None else None
            h, w = old_mask_np.shape
            mask_image = QImage(old_mask_np.data, w, h, w, QImage.Format.Format_Grayscale8).copy()

        # Paint the recorded path onto the actual mask image
        painter = QPainter(mask_image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        if self._active_tool in ['pen', 'brush']:
            final_pen = QPen(Qt.GlobalColor.white, self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        elif self._active_tool == 'eraser':
            final_pen = QPen(Qt.GlobalColor.black, self._brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source) # Draw black to erase
        else:
            painter.end()
            self._current_draw_path = None # Clean up path
            self._clear_preview()
            return

        painter.setPen(final_pen)
        painter.drawPath(self._current_draw_path)
        painter.end()

        # Clear the preview immediately
        self._clear_preview()

        # Convert final mask to numpy and update model
        ptr = mask_image.constBits()
        ptr.setsize(mask_image.sizeInBytes())
        # Use bytesPerLine to handle row padding correctly
        bytes_per_line = mask_image.bytesPerLine()
        new_mask_np = np.array(ptr).reshape(mask_image.height(), bytes_per_line)
        # Crop to actual width if there's padding
        new_mask_np = new_mask_np[:, :mask_image.width()].copy()

        # --- Refactored to Command Pattern ---
        from .commands import MaskEditCommand
        command = MaskEditCommand(
            model=self.model,
            old_mask=old_mask_np,
            new_mask=new_mask_np.copy()
        )
        
        # Access controller through model's controller reference
        controller = getattr(self.model, 'controller', None)
        if controller and hasattr(controller, 'execute_command'):
            controller.execute_command(command)
        else:
            # Fallback: try to find controller through parent hierarchy
            # GraphicsView -> QSplitter -> EditorView
            editor_view = self.parent()
            while editor_view and not hasattr(editor_view, 'controller'):
                editor_view = editor_view.parent()
            
            if editor_view and hasattr(editor_view, 'controller'):
                editor_view.controller.execute_command(command)
            else:
                # Last resort: directly update the model (no undo support)
                self.model.set_refined_mask(new_mask_np.copy())
                import logging
                logging.warning("Could not find controller, updated mask directly without undo support")

        self._current_draw_path = None

    def _clear_preview(self):
        """Clear the preview item immediately."""
        if self._preview_item:
            self._preview_item.pixmap().fill(Qt.GlobalColor.transparent)
            self._preview_item.setVisible(False)
            # Force immediate update
            self.scene.update()
            self.viewport().update()




    # --- Mask Editing Slots & Methods ---

    @pyqtSlot(str)
    def _on_active_tool_changed(self, tool: str):
        """Slot to handle tool changes from the model."""
        self._active_tool = tool
        self._update_cursor()
        # Force immediate cursor update
        self.viewport().update()

    @pyqtSlot(int)
    def _on_brush_size_changed(self, size: int):
        """Slot to handle brush size changes from the model."""
        self._brush_size = size
        self._update_cursor()

    def _on_selection_changed(self, selected_indices: list):
        """同步model的selection到Qt item的selected状态"""
        try:
            # 先清除所有item的selection - 使用安全遍历
            for item in self._region_items:
                try:
                    if item and hasattr(item, 'scene') and item.scene() and item.isSelected():
                        item.setSelected(False)
                        item.update()  # 强制重绘
                except (RuntimeError, AttributeError):
                    # Item可能已被删除
                    pass

            # 设置新选中的items
            for idx in selected_indices:
                if 0 <= idx < len(self._region_items):
                    item = self._region_items[idx]
                    try:
                        if item and hasattr(item, 'scene') and item.scene():
                            item.setSelected(True)
                            item.update()  # 强制重绘
                    except (RuntimeError, AttributeError):
                        pass
            
            # 强制场景更新
            if self.scene:
                self.scene.update()
            self.viewport().update()
        except Exception as e:
            print(f"[View] Warning: Selection change failed: {e}")

    def _update_cursor(self):
        """Updates the cursor to match the selected tool and brush size."""
        if self._active_tool in ['pen', 'eraser', 'brush']:
            size = max(10, int(self._brush_size * self.transform().m11()))

            # 创建自定义光标
            cursor_size = size + 6
            pixmap = QPixmap(cursor_size, cursor_size)
            pixmap.fill(Qt.GlobalColor.transparent)

            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            center = cursor_size // 2
            radius = size // 2

            if self._active_tool in ['pen', 'brush']:
                # 画笔：红色圆圈
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                painter.setBrush(Qt.GlobalColor.transparent)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                painter.setPen(QPen(Qt.GlobalColor.red, 1))
                painter.drawEllipse(center - radius + 1, center - radius + 1, (radius - 1) * 2, (radius - 1) * 2)
            else:  # eraser
                # 橡皮擦：蓝色圆圈
                painter.setPen(QPen(Qt.GlobalColor.black, 2))
                painter.setBrush(Qt.GlobalColor.transparent)
                painter.drawEllipse(center - radius, center - radius, radius * 2, radius * 2)
                painter.setPen(QPen(Qt.GlobalColor.blue, 1))
                painter.drawEllipse(center - radius + 1, center - radius + 1, (radius - 1) * 2, (radius - 1) * 2)

            painter.end()

            cursor = QCursor(pixmap, center, center)

            # 使用应用级别覆盖确保光标显示
            from PyQt6.QtWidgets import QApplication
            QApplication.setOverrideCursor(cursor)
        elif self._active_tool == 'geometry_edit':
            from PyQt6.QtWidgets import QApplication
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))
        elif self._active_tool == 'draw_textbox':
            from PyQt6.QtWidgets import QApplication
            QApplication.setOverrideCursor(QCursor(Qt.CursorShape.CrossCursor))
        else:
            # 对于选择工具或其他工具，恢复默认箭头光标
            from PyQt6.QtWidgets import QApplication
            # 清除所有应用级别的光标覆盖
            while QApplication.overrideCursor():
                QApplication.restoreOverrideCursor()
            # 确保设置为默认箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def enterEvent(self, event):
        """Handle mouse enter event."""
        self._update_cursor()
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Handle mouse leave event."""
        # Reset cursor when leaving the view
        self.unsetCursor()
        super().leaveEvent(event)


    # --- Public Slots for Toolbar ---
    @pyqtSlot()
    def zoom_in(self):
        self.scale(1.15, 1.15)

    @pyqtSlot()
    def zoom_out(self):
        self.scale(1 / 1.15, 1 / 1.15)

    @pyqtSlot()
    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.AspectRatioMode.KeepAspectRatio)

    def contextMenuEvent(self, event):
        """处理右键菜单事件"""
        # 检查是否有选中的区域
        selected_regions = self.model.get_selection()
        selection_count = len(selected_regions)
        
        # 创建右键菜单
        menu = QMenu(self)
        
        if selection_count > 0:
            # 有选中区域时的菜单项
            menu.addAction("🔍 OCR识别选中项", self._ocr_selected_regions)
            menu.addAction("🌐 翻译选中项", self._translate_selected_regions)
            menu.addSeparator()
            
            if selection_count == 1:
                # 单选时的额外选项
                menu.addAction("📋 复制区域", self._copy_selected_region)
                menu.addAction("🎨 粘贴样式", self._paste_region_style)
                menu.addSeparator()
            
            # 删除选项
            menu.addAction(f"🗑️ 删除选中的 {selection_count} 个区域", self._delete_selected_regions)
        else:
            # 无选中区域时的菜单项
            menu.addAction("➕ 添加文本框", self._add_text_box)
            menu.addAction("📋 粘贴区域", self._paste_region)
            menu.addSeparator()
            menu.addAction("🔄 刷新视图", self._refresh_view)
        
        # 显示菜单
        menu.exec(event.globalPos())

    def _get_controller(self):
        """获取控制器实例"""
        # GraphicsView -> QSplitter -> EditorView
        # 需要向上查找两级才能找到 EditorView
        parent = self.parent()
        # 向上遍历查找 EditorView
        current = parent
        while current:
            if hasattr(current, 'controller'):
                return current.controller
            current = current.parent() if hasattr(current, 'parent') and callable(current.parent) else None
        return None

    def _ocr_selected_regions(self):
        """OCR识别选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller and selected_regions:
            controller.ocr_regions(selected_regions)

    def _translate_selected_regions(self):
        """翻译选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller and selected_regions:
            controller.translate_regions(selected_regions)

    def _copy_selected_region(self):
        """复制选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if len(selected_regions) == 1 and controller:
            controller.copy_region(selected_regions[0])

    def _paste_region_style(self):
        """粘贴区域样式"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if len(selected_regions) == 1 and controller:
            controller.paste_region_style(selected_regions[0])

    def _delete_selected_regions(self):
        """删除选中的区域"""
        selected_regions = self.model.get_selection()
        controller = self._get_controller()
        if controller:
            controller.delete_regions(selected_regions)

    def _add_text_box(self):
        """添加新的文本框"""
        controller = self._get_controller()
        if controller:
            controller.enter_drawing_mode()

    def _paste_region(self):
        """粘贴区域到鼠标位置"""
        controller = self._get_controller()
        if controller and self._image_item:
            # 获取鼠标在场景中的位置
            mouse_pos_scene = self.mapToScene(self.mapFromGlobal(QCursor.pos()))
            # 转换为图像坐标
            mouse_pos_image = self._image_item.mapFromScene(mouse_pos_scene)
            controller.paste_region(mouse_pos_image)

    def _refresh_view(self):
        """刷新视图"""
        self.scene.update()
        self.update()

    def _start_drawing_textbox(self, pos):
        """开始绘制新文本框"""
        if self._image_item is None:
            return

        self._is_drawing_textbox = True
        self._textbox_start_pos = self.mapToScene(pos)

        # 创建预览矩形
        if self._textbox_preview_item is None:
            pen = QPen(QColor(255, 0, 0, 200))  # 红色，半透明
            pen.setWidth(2)
            pen.setStyle(Qt.PenStyle.DashLine)
            brush = QColor(255, 0, 0, 50)  # 红色填充，更透明
            self._textbox_preview_item = self.scene.addRect(0, 0, 0, 0, pen, brush)
            self._textbox_preview_item.setZValue(200)

        # 重置预览矩形
        self._textbox_preview_item.setRect(0, 0, 0, 0)
        self._textbox_preview_item.setVisible(True)

    def _update_textbox_drawing(self, pos):
        """更新文本框绘制预览"""
        if not self._is_drawing_textbox or self._textbox_start_pos is None:
            return

        current_pos = self.mapToScene(pos)

        # 计算矩形
        x1, y1 = self._textbox_start_pos.x(), self._textbox_start_pos.y()
        x2, y2 = current_pos.x(), current_pos.y()

        # 确保矩形有正确的方向
        left = min(x1, x2)
        top = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 更新预览矩形
        if self._textbox_preview_item:
            self._textbox_preview_item.setRect(left, top, width, height)

    def _finish_textbox_drawing(self):
        """完成文本框绘制"""
        if not self._is_drawing_textbox or self._textbox_start_pos is None:
            return

        # 获取最终矩形
        rect = self._textbox_preview_item.rect()

        # 隐藏并重置预览
        if self._textbox_preview_item:
            self._textbox_preview_item.setVisible(False)
            self._textbox_preview_item.setRect(0, 0, 0, 0)

        # 检查矩形大小是否足够大
        min_size = 20
        if rect.width() < min_size or rect.height() < min_size:
            self._is_drawing_textbox = False
            self._textbox_start_pos = None
            return

        # 创建新的文本区域
        self._create_new_text_region(rect)

        # 重置状态
        self._is_drawing_textbox = False
        self._textbox_start_pos = None

        # 退出绘制模式
        self.model.set_active_tool('select')

    def _create_new_text_region(self, rect):
        """创建新的文本区域"""
        # 将场景坐标转换为图像坐标
        if not self._image_item:
            return
        
        # 获取图像的变换矩阵
        image_transform = self._image_item.transform()
        try:
            # 获取逆变换
            inverse_transform = image_transform.inverted()[0]
            
            # 获取模板角度（如果有选中区域的话）
            template_angle = 0
            controller = self._get_controller()
            if controller:
                selected_regions = self.model.get_selection()
                if selected_regions:
                    template_region = self.model.get_region_by_index(selected_regions[-1])
                    if template_region:
                        template_angle = template_region.get('angle', 0)
            
            # 创建矩形的四个角点（考虑模板角度）
            rect_center_x = (rect.left() + rect.right()) / 2
            rect_center_y = (rect.top() + rect.bottom()) / 2
            rect_width = rect.right() - rect.left()
            rect_height = rect.bottom() - rect.top()
            
            # 创建相对于中心的矩形点
            half_width = rect_width / 2
            half_height = rect_height / 2
            
            # 未旋转的矩形角点（相对于中心）
            relative_points = [
                (-half_width, -half_height),  # 左上
                (half_width, -half_height),   # 右上
                (half_width, half_height),    # 右下
                (-half_width, half_height)    # 左下
            ]
            
            # 如果有模板角度，旋转这些点
            if template_angle != 0:
                import math
                cos_a = math.cos(math.radians(template_angle))
                sin_a = math.sin(math.radians(template_angle))
                
                rotated_points = []
                for x, y in relative_points:
                    new_x = x * cos_a - y * sin_a
                    new_y = x * sin_a + y * cos_a
                    rotated_points.append((new_x, new_y))
                relative_points = rotated_points
            
            # 转换为场景坐标
            scene_points = []
            for x, y in relative_points:
                scene_points.append(QPointF(rect_center_x + x, rect_center_y + y))
            
            # 转换为图像坐标
            image_points = []
            for point in scene_points:
                image_point = inverse_transform.map(point)
                image_points.append([image_point.x(), image_point.y()])
            
            # 使用之前获取的模板区域数据
            template_data = {}
            if controller:
                selected_regions = self.model.get_selection()
                if selected_regions:
                    template_region = self.model.get_region_by_index(selected_regions[-1])
                    if template_region:
                        template_data = {
                            'font_family': template_region.get('font_family', 'Arial'),
                            'font_size': template_region.get('font_size', 24),
                            'font_color': template_region.get('font_color', '#000000'),
                            'alignment': template_region.get('alignment', 'center'),
                            'direction': template_region.get('direction', 'auto'),
                            'angle': template_region.get('angle', 0)
                        }
            
            # 计算中心点（在图像坐标系中）
            center_scene = QPointF(rect_center_x, rect_center_y)
            center_image = inverse_transform.map(center_scene)
            
            # 从配置服务获取默认渲染参数
            from services import get_config_service
            config_service = get_config_service()
            config = config_service.get_config()
            default_line_spacing = config.render.line_spacing if hasattr(config.render, 'line_spacing') else 1.0
            default_stroke_width = config.render.stroke_width if hasattr(config.render, 'stroke_width') else 0.07
            # 确保不为 None
            if default_line_spacing is None:
                default_line_spacing = 1.0
            if default_stroke_width is None:
                default_stroke_width = 0.07
            
            # 创建新区域数据，使用模板样式或默认值
            new_region_data = {
                'text': '',
                'texts': [''],  # 添加 texts 字段,避免 TextBlock 创建失败
                'translation': '',
                'polygons': [image_points],
                'lines': [image_points],
                'center': [center_image.x(), center_image.y()],  # 添加中心点
                'font_family': template_data.get('font_family', 'Arial'),
                'font_size': template_data.get('font_size', 24),
                'font_color': template_data.get('font_color', '#000000'),
                'alignment': template_data.get('alignment', 'center'),
                'direction': template_data.get('direction', 'auto'),
                'angle': template_data.get('angle', 0),
                'line_spacing': default_line_spacing,
                'stroke_width': default_stroke_width,
                'stroke_color_type': 'white',  # 默认白色描边
                'font_path': ''  # 空字符串，渲染时会自动使用主页配置的默认字体
            }
            
            # 通知控制器添加新区域 - 使用命令模式以支持撤销
            controller = self._get_controller()
            if controller:
                from editor.commands import AddRegionCommand

                # 重置 _last_edited_region_index,确保触发完全更新
                self._last_edited_region_index = None

                # 使用命令模式添加新区域
                command = AddRegionCommand(
                    model=self.model,
                    region_data=new_region_data,
                    description="Add New Text Box"
                )
                controller.execute_command(command)

                # 选中新创建的区域
                new_index = len(self.model.get_regions()) - 1
                self.model.set_selection([new_index])

                # 强制更新 view
                self.viewport().update()
                self.scene.update()

        except Exception as e:
            import traceback
            print(f"创建文本区域失败: {e}")
            traceback.print_exc()
