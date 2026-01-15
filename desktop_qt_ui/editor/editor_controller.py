import asyncio
import copy
import os
import sys
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, pyqtSlot

from editor.commands import UpdateRegionCommand
from services import (
    get_async_service,
    get_config_service,
    get_file_service,
    get_history_service,
    get_logger,
    get_ocr_service,
    get_resource_manager,
    get_translation_service,
)

from .editor_model import EditorModel
from .desktop_ui_geometry import get_polygon_center

# 添加项目根目录到路径以便导入path_manager
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from manga_translator.utils.path_manager import find_inpainted_path


class EditorController(QObject):
    """
    编辑器控制器 (Controller)

    负责处理编辑器的所有业务逻辑和用户交互。
    它响应来自视图(View)的信号，调用服务(Service)执行任务，并更新模型(Model)。
    """
    # Signal for thread-safe model updates
    _update_refined_mask = pyqtSignal(object)
    _update_display_mask_type = pyqtSignal(str)
    _regions_update_finished = pyqtSignal(list)
    _ocr_completed = pyqtSignal()
    _translation_completed = pyqtSignal()
    
    # Signal for thread-safe Toast notifications
    _show_toast_signal = pyqtSignal(str, int, bool, str)  # message, duration, success, clickable_path
    
    # Signal for thread-safe image loading
    _load_result_ready = pyqtSignal(dict)  # 加载结果信号

    def __init__(self, model: EditorModel, parent=None):
        super().__init__(parent)
        self.model = model
        self.view = None  # 将在 EditorView 中设置
        self.logger = get_logger(__name__)

        # 获取所需的服务
        self.ocr_service = get_ocr_service()
        self.translation_service = get_translation_service()
        self.async_service = get_async_service()
        self.history_service = get_history_service() # 用于撤销/重做
        self.file_service = get_file_service()
        self.config_service = get_config_service()
        self.resource_manager = get_resource_manager()  # 新的资源管理器

        # 缓存键常量
        self.CACHE_LAST_INPAINTED = "last_inpainted_image"
        self.CACHE_LAST_MASK = "last_processed_mask"
        
        # 用户透明度调整标志
        self._user_adjusted_alpha = False
        
        # 上次导出时的状态快照（用于检测是否有更改）
        self._last_export_snapshot = None

        # Connect internal signals for thread-safe updates
        self._update_refined_mask.connect(self.model.set_refined_mask)
        self._update_display_mask_type.connect(self.model.set_display_mask_type)
        self._regions_update_finished.connect(self.on_regions_update_finished)
        self._ocr_completed.connect(self._on_ocr_completed)
        self._translation_completed.connect(self._on_translation_completed)
        self._load_result_ready.connect(self._apply_load_result)  # 连接加载结果信号
        
        # 设置model的controller引用，用于命令模式
        self.model.controller = self

        self._connect_model_signals()
    
    # ========== Resource Access Helpers (新的资源访问辅助方法) ==========
    
    def _get_current_image(self) -> Optional[Image.Image]:
        """获取当前图片（PIL Image）
        
        优先从ResourceManager获取，如果失败则从Model获取（向后兼容）
        """
        resource = self.resource_manager.get_current_image()
        if resource:
            return resource.image
        # 向后兼容：如果ResourceManager没有，尝试从Model获取
        return self.model.get_image()
    
    def _get_current_mask(self, mask_type: str = "raw") -> Optional[np.ndarray]:
        """获取当前蒙版
        
        Args:
            mask_type: 蒙版类型，"raw" 或 "refined"
        
        Returns:
            Optional[np.ndarray]: 蒙版数据，如果不存在返回None
        """
        from desktop_qt_ui.editor.core.types import MaskType
        
        mask_type_enum = MaskType.RAW if mask_type == "raw" else MaskType.REFINED
        mask_resource = self.resource_manager.get_mask(mask_type_enum)
        
        if mask_resource:
            return mask_resource.data
        
        # 向后兼容
        if mask_type == "raw":
            return self.model.get_raw_mask()
        elif mask_type == "refined":
            return self.model.get_refined_mask()
        return None
    
    def _get_regions(self):
        """获取所有区域
        
        Returns:
            List[Dict]: 区域列表
        """
        # 从ResourceManager获取
        resources = self.resource_manager.get_all_regions()
        if resources:
            return [r.data for r in resources]
        # 向后兼容
        return self.model.get_regions()
    
    def _set_regions(self, regions: list):
        """设置所有区域
        
        Args:
            regions: 区域列表
        """
        # 清空现有区域
        self.resource_manager.clear_regions()
        # 添加新区域
        for region_data in regions:
            self.resource_manager.add_region(region_data)
        # 同步到Model（向后兼容）
        self.model.set_regions(regions)
    
    def _get_region_by_index(self, index: int):
        """根据索引获取区域
        
        Args:
            index: 区域索引
        
        Returns:
            Dict: 区域数据，如果不存在返回None
        """
        regions = self._get_regions()
        if 0 <= index < len(regions):
            return regions[index]
        return None
    
    def _update_region(self, index: int, updates: dict):
        """更新区域数据
        
        Args:
            index: 区域索引
            updates: 要更新的数据
        """
        regions = self._get_regions()
        if 0 <= index < len(regions):
            regions[index].update(updates)
            # 重新设置所有区域以同步
            self._set_regions(regions)
        # 同步到Model
        self.model.update_region_silent(index, updates)

    def set_view(self, view):
        """设置view引用，用于更新UI状态"""
        self.view = view
        # 初始化Toast管理器
        from desktop_qt_ui.widgets.toast_notification import ToastManager
        self.toast_manager = ToastManager(view)
        # 连接Toast信号到主线程槽函数
        self._show_toast_signal.connect(self._show_toast_in_main_thread)
        # 初始化撤销/重做按钮状态
        self._update_undo_redo_buttons()
    
    @pyqtSlot(str, int, bool, str)
    def _show_toast_in_main_thread(self, message: str, duration: int, success: bool, clickable_path: str):
        """在主线程显示Toast通知的槽函数"""
        try:
            # 先关闭"正在导出"Toast（在主线程中安全关闭）
            if hasattr(self, '_export_toast') and self._export_toast:
                try:
                    self._export_toast.close()
                    self._export_toast = None
                except Exception as e:
                    self.logger.warning(f"Failed to close export toast: {e}")
            
            # 显示新Toast
            if hasattr(self, 'toast_manager'):
                if success:
                    self.toast_manager.show_success(message, duration, clickable_path if clickable_path else None)
                else:
                    self.toast_manager.show_error(message, duration)
        except Exception as e:
            self.logger.error(f"Exception in _show_toast_in_main_thread: {e}", exc_info=True)

    def _connect_model_signals(self):
        """监听模型的变化，可能需要触发一些后续逻辑"""
        self.model.regions_changed.connect(self.on_regions_changed)
        # 监听蒙版编辑后触发 inpainting
        self.model.refined_mask_changed.connect(self.on_refined_mask_changed)

    def on_regions_changed(self, regions):
        """模型中的区域数据变化时的槽函数"""
        # print(f"Controller: Regions changed, {len(regions)} regions total.")
        # This is a placeholder for where you might trigger a repaint or update.
        # For example, if you have a graphics scene, you might update it here.
        pass

    def on_refined_mask_changed(self, mask):
        """refined mask 变化时的槽函数，触发增量 inpainting"""
        # 检查是否有必要的数据来进行 inpainting
        image = self._get_current_image()
        raw_mask = self.model.get_raw_mask()

        if image is not None and mask is not None and raw_mask is not None:
            # 计算差异蒙版（只包含用户编辑的部分）
            self.async_service.submit_task(self._async_incremental_inpaint(mask, raw_mask))

    @pyqtSlot(dict)
    def update_multiple_translations(self, translations: dict):
        """
        批量更新多个区域的译文。
        `translations` 是一个 {index: text} 格式的字典。
        """
        if not translations:
            return

        for index, text in translations.items():
            # 更新model中的数据
            self.model.update_region_data(index, 'translation', text)
            
            # 同步更新ResourceManager中的数据
            # ResourceManager使用region_id，需要通过索引获取对应的region_id
            try:
                all_regions = self.resource_manager.get_all_regions()
                if index < len(all_regions):
                    # 获取对应索引的region_id
                    region_resource = all_regions[index]
                    self.resource_manager.update_region(region_resource.region_id, {'translation': text})
            except Exception as e:
                # 静默失败，不影响主流程
                self.logger.warning(f"Failed to update region {index} in ResourceManager: {e}")

        # 一次性通知视图更新
        self.model.regions_changed.emit(self.model.get_regions())

    def _run_on_main_thread(self, func, *args):
        """确保一个函数在主GUI线程上运行"""
        def wrapper():
            try:
                result = func(*args)
                return result
            except Exception as e:
                self.logger.error(f"Error executing {func.__name__}: {e}")
                raise
        QTimer.singleShot(0, wrapper)

    # --- 公共槽函数 (Public Slots) ---

    def has_unsaved_changes(self) -> bool:
        """检查是否有未保存的编辑"""
        return self.history_service.can_undo()
    
    def _generate_export_snapshot(self) -> dict:
        """生成当前状态的快照，用于检测导出后是否有更改
        
        使用轻量级的特征值而不是完整哈希，避免阻塞主线程
        """
        regions = self._get_regions()
        
        # 提取关键数据生成哈希
        snapshot_data = []
        for region in regions:
            # 只关注会影响导出结果的字段
            region_key = {
                'translation': region.get('translation', ''),
                'font_size': region.get('font_size'),
                'font_color': region.get('font_color'),
                'alignment': region.get('alignment'),
                'direction': region.get('direction'),
                'xyxy': region.get('xyxy'),
                'lines': str(region.get('lines', [])),
            }
            snapshot_data.append(str(region_key))
        
        # 使用蒙版的轻量级特征（形状+总和+非零像素数）而不是完整哈希
        mask = self.model.get_refined_mask()
        if mask is None:
            mask = self.model.get_raw_mask()
        mask_signature = ""
        if mask is not None:
            # 使用形状、总和、非零像素数作为快速特征
            mask_signature = f"{mask.shape}_{mask.sum()}_{np.count_nonzero(mask)}"
        
        # 使用简单的字符串哈希
        regions_str = '|'.join(snapshot_data)
        
        return {
            'regions_hash': hash(regions_str),
            'mask_signature': mask_signature,
            'source_path': self.model.get_source_image_path(),
        }
    
    def _has_changes_since_last_export(self) -> bool:
        """检查自上次导出后是否有更改"""
        if self._last_export_snapshot is None:
            # 从未导出过，检查是否有撤销历史
            return self.history_service.can_undo()
        
        current_snapshot = self._generate_export_snapshot()
        
        # 比较快照
        if current_snapshot['source_path'] != self._last_export_snapshot['source_path']:
            # 不同的图片，不需要比较
            return self.history_service.can_undo()
        
        return (current_snapshot['regions_hash'] != self._last_export_snapshot['regions_hash'] or
                current_snapshot['mask_signature'] != self._last_export_snapshot['mask_signature'])
    
    def _save_export_snapshot(self):
        """保存当前状态快照（导出成功后调用）"""
        self._last_export_snapshot = self._generate_export_snapshot()
        self.logger.debug(f"Export snapshot saved: {self._last_export_snapshot}")

    def _clear_editor_state(self, release_image_cache: bool = False):
        """清空编辑器状态
        
        Args:
            release_image_cache: 是否同时释放图片缓存（切换文件时通常不需要）
        """
        import gc
        
        # 关闭加载提示（如果存在）
        if hasattr(self, '_loading_toast') and self._loading_toast:
            try:
                self._loading_toast.close()
                self._loading_toast = None
            except Exception:
                pass
        
        # 取消所有正在运行的后台任务
        self.async_service.cancel_all_tasks()

        # 使用ResourceManager卸载当前资源
        self.resource_manager.unload_image(release_from_cache=release_image_cache)

        # 清空模型数据（向后兼容，View仍然监听Model）
        self.model.set_regions([])
        self.model.set_raw_mask(None)
        self.model.set_refined_mask(None)
        self.model.set_inpainted_image(None)
        self.model.set_selection([])

        # 清空历史记录
        self.history_service.clear()
        
        # 清空导出快照（每张图片独立）
        self._last_export_snapshot = None

        # 清空缓存（使用ResourceManager）
        self.resource_manager.clear_cache()

        # 清空渲染参数缓存
        from services import get_render_parameter_service
        render_parameter_service = get_render_parameter_service()
        render_parameter_service.clear_cache()
        
        # 清空GraphicsView的缓存
        if self.view and hasattr(self.view, 'graphics_view'):
            gv = self.view.graphics_view
            if hasattr(gv, '_text_render_cache'):
                gv._text_render_cache.clear()
            if hasattr(gv, '_text_blocks_cache'):
                gv._text_blocks_cache = []
            if hasattr(gv, '_dst_points_cache'):
                gv._dst_points_cache = []
        
        # 关闭加载线程池（如果存在）
        if hasattr(self, '_load_executor'):
            try:
                self._load_executor.shutdown(wait=False)
                del self._load_executor
            except Exception:
                pass
        
        # 强制垃圾回收
        gc.collect()
        
        # 释放GPU显存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        self.logger.debug("Editor state cleared and memory released")

    def _is_translated_image(self, image_path: str) -> bool:
        """
        检查图片是否是翻译后的图片（通过translation_map.json）

        Args:
            image_path: 图片路径

        Returns:
            True if 是翻译后的图片, False otherwise
        """
        try:
            import json
            from pathlib import Path
            
            # 使用 pathlib 规范化路径
            path_obj = Path(image_path)
            norm_path = str(path_obj.resolve())
            output_dir = path_obj.parent
            map_path = output_dir / 'translation_map.json'

            self.logger.debug(f"Checking if translated image: {image_path}")
            self.logger.debug(f"Normalized path: {norm_path}")
            self.logger.debug(f"Looking for translation_map.json at: {map_path}")

            if map_path.exists():
                with open(map_path, 'r', encoding='utf-8') as f:
                    translation_map = json.load(f)
                self.logger.debug(f"Found translation_map.json with {len(translation_map)} entries")
                self.logger.debug(f"Translation map keys: {list(translation_map.keys())[:3]}...")  # 只显示前3个
                
                # 规范化 translation_map 中的所有键
                normalized_map = {str(Path(k).resolve()): v for k, v in translation_map.items()}
                
                # 如果当前路径是translation_map的key，说明是翻译后的图片
                if norm_path in normalized_map:
                    self.logger.debug(f"✓ Found translation mapping for: {image_path}")
                    return True
                else:
                    self.logger.debug(f"✗ No translation mapping found for: {norm_path}")
            else:
                self.logger.debug(f"✗ translation_map.json not found at: {map_path}")
        except Exception as e:
            self.logger.error(f"Error checking translation map: {e}")

        return False

    def load_image_and_regions(self, image_path: str):
        """加载图像及其关联的区域数据，并触发后台处理"""
        # 检查是否有未导出的更改（基于快照比较，而不仅仅是撤销历史）
        has_changes = self._has_changes_since_last_export()
        if has_changes:
            from PyQt6.QtWidgets import QMessageBox
            msg_box = QMessageBox(None)
            msg_box.setWindowTitle("未保存的编辑")
            msg_box.setText("当前图片有未保存的编辑")
            msg_box.setInformativeText("请选择保存方式：")
            
            # 添加三个按钮
            export_btn = msg_box.addButton("导出图片", QMessageBox.ButtonRole.YesRole)
            save_json_btn = msg_box.addButton("保存JSON", QMessageBox.ButtonRole.YesRole)
            cancel_btn = msg_box.addButton("取消", QMessageBox.ButtonRole.RejectRole)
            no_btn = msg_box.addButton("不保存", QMessageBox.ButtonRole.NoRole)
            
            msg_box.setDefaultButton(cancel_btn)
            msg_box.exec()
            
            clicked_button = msg_box.clickedButton()
            
            if clicked_button == cancel_btn:
                return
            elif clicked_button == export_btn:
                self.export_image()
                # 使用QTimer延迟加载，避免阻塞UI
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(500, lambda: self._do_load_image(image_path))
                return
            elif clicked_button == save_json_btn:
                self.save_json()
                # 使用QTimer延迟加载，避免阻塞UI
                from PyQt6.QtCore import QTimer
                QTimer.singleShot(500, lambda: self._do_load_image(image_path))
                return
            # 如果点击"不保存"，继续执行下面的加载逻辑

        self._do_load_image(image_path)
    
    def _do_load_image(self, image_path: str):
        """实际执行图片加载的内部方法 - 使用线程池避免阻塞UI"""
        import concurrent.futures
        
        # 清空旧状态
        self._clear_editor_state()
        
        # 显示加载提示
        if hasattr(self, 'toast_manager'):
            self._loading_toast = self.toast_manager.show_info("正在加载...", duration=0)
        
        # 使用线程池加载数据
        if not hasattr(self, '_load_executor'):
            self._load_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        
        def load_data():
            """在后台线程加载数据"""
            try:
                # 1. 加载图片
                image_resource = self.resource_manager.load_image(image_path)
                image = image_resource.image

                # 2. 检查是否是翻译后的图片
                is_translated_image = self._is_translated_image(image_path)

                if is_translated_image:
                    return {'type': 'translated', 'image_path': image_path, 'image': image}

                # 3. 加载JSON
                regions, raw_mask, original_size = self.file_service.load_translation_json(image_path)

                # 4. 查找和加载inpainted图片
                inpainted_path = find_inpainted_path(image_path)
                inpainted_image = None
                if inpainted_path:
                    try:
                        inpainted_image = Image.open(inpainted_path)
                        if inpainted_image.size != image.size:
                            inpainted_image = inpainted_image.resize(image.size, Image.LANCZOS)
                    except Exception as e:
                        self.logger.error(f"Error loading inpainted image: {e}")
                        inpainted_path = None
                        inpainted_image = None

                return {
                    'type': 'normal',
                    'image_path': image_path,
                    'image': image,
                    'regions': regions,
                    'raw_mask': raw_mask,
                    'inpainted_path': inpainted_path,
                    'inpainted_image': inpainted_image
                }
            except Exception as e:
                self.logger.error(f"Error loading image data: {e}", exc_info=True)
                return {'type': 'error', 'error': str(e)}
        
        def on_load_complete(future):
            """加载完成回调 - 使用信号确保在主线程更新UI"""
            try:
                result = future.result()
                self._load_result_ready.emit(result)
            except Exception as e:
                self.logger.error(f"Load failed: {e}", exc_info=True)
                self._load_result_ready.emit({'type': 'error', 'error': str(e)})
        
        future = self._load_executor.submit(load_data)
        future.add_done_callback(on_load_complete)
    
    @pyqtSlot(dict)
    def _apply_load_result(self, result: dict):
        """在主线程应用加载结果"""
        try:
            if result['type'] == 'error':
                self._handle_load_error(result['error'])
            elif result['type'] == 'translated':
                self._apply_translated_image_to_model(result['image_path'], result['image'])
            else:
                self._apply_loaded_data_to_model(
                    result['image_path'],
                    result['image'],
                    result['regions'],
                    result['raw_mask'],
                    result['inpainted_path'],
                    result['inpainted_image']
                )
        except Exception as e:
            self.logger.error(f"Exception in _apply_load_result: {e}", exc_info=True)
    
    def _apply_translated_image_to_model(self, image_path: str, image):
        """在主线程应用翻译后图片到Model"""
        try:
            # 关闭加载提示
            if hasattr(self, '_loading_toast') and self._loading_toast:
                self._loading_toast.close()
                self._loading_toast = None
            
            self.model.set_source_image_path(image_path)

            if not hasattr(self, '_user_adjusted_alpha') or not self._user_adjusted_alpha:
                self.model.set_original_image_alpha(1.0)

            self.model.set_image(image)
            self._set_regions([])
            self.model.set_raw_mask(None)
            self.model.set_refined_mask(None)
            self.model.set_inpainted_image_path(None)
        except Exception as e:
            self.logger.error(f"Error applying translated image to model: {e}")
    
    def _apply_loaded_data_to_model(self, image_path, image, regions, raw_mask, inpainted_path, inpainted_image):
        """在主线程应用加载的数据到Model"""
        try:
            # 关闭加载提示
            if hasattr(self, '_loading_toast') and self._loading_toast:
                self._loading_toast.close()
                self._loading_toast = None
            
            # 导入渲染参数
            if regions:
                from services import get_render_parameter_service
                render_parameter_service = get_render_parameter_service()
                for i, region_data in enumerate(regions):
                    render_parameter_service.import_parameters_from_json(i, region_data)

            self.model.set_source_image_path(image_path)

            if not hasattr(self, '_user_adjusted_alpha') or not self._user_adjusted_alpha:
                self.model.set_original_image_alpha(0.0)

            self.model.set_image(image)
            self._set_regions(regions)

            if raw_mask is not None:
                from desktop_qt_ui.editor.core.types import MaskType
                self.resource_manager.set_mask(MaskType.RAW, raw_mask)
                self.model.set_raw_mask(raw_mask)

            self.model.set_refined_mask(None)

            if inpainted_path:
                self.model.set_inpainted_image_path(inpainted_path)
                if inpainted_image:
                    self.model.set_inpainted_image(inpainted_image)
            else:
                self.model.set_inpainted_image_path(None)
                self.model.set_inpainted_image(None)

            # 触发后台处理
            if regions:
                self.async_service.submit_task(self._async_refine_and_inpaint())
                
        except Exception as e:
            self.logger.error(f"Error applying loaded data to model: {e}", exc_info=True)
    
    def _handle_load_error(self, error_msg: str):
        """处理加载错误"""
        # 关闭加载提示
        if hasattr(self, '_loading_toast') and self._loading_toast:
            self._loading_toast.close()
            self._loading_toast = None
        
        if hasattr(self, 'toast_manager'):
            self.toast_manager.show_error(f"加载失败: {error_msg}")
        
        self.model.set_image(None)
        self.model.set_regions([])
        self.model.set_raw_mask(None)
        self.model.set_refined_mask(None)

    async def _async_refine_and_inpaint(self):
        """Asynchronously refines the mask and generates an inpainting preview."""
        try:
            self.logger.debug("Starting async mask refinement and inpainting...")
            image = self._get_current_image()
            raw_mask = self.model.get_raw_mask() # Use the raw mask for refinement
            regions = self._get_regions()

            if image is None or raw_mask is None or not regions:
                self.logger.warning("Refinement/Inpainting skipped: image, mask, or regions not available.")
                return

            # 延迟导入后端模块
            try:
                from manga_translator.config import (
                    Inpainter,
                    InpainterConfig,
                    InpaintPrecision,
                )
                from manga_translator.inpainting import dispatch as inpaint_dispatch
                from manga_translator.mask_refinement import (
                    dispatch as refine_mask_dispatch,
                )
                from manga_translator.utils import TextBlock
            except ImportError as e:
                self.logger.error(f"Failed to import backend modules: {e}")
                return

            # 1. Refine Mask
            image_np = np.array(image.convert("RGB"))
            text_blocks = [TextBlock(**region_data) for region_data in regions]
            raw_mask_2d = cv2.cvtColor(raw_mask, cv2.COLOR_BGR2GRAY) if len(raw_mask.shape) == 3 else raw_mask
            raw_mask_contiguous = np.ascontiguousarray(raw_mask_2d, dtype=np.uint8)

            # 从配置服务获取mask参数
            config = self.config_service.get_config()
            dilation_offset = config.mask_dilation_offset
            kernel_size = config.kernel_size
            ignore_bubble = config.ocr.ignore_bubble

            refined_mask = await refine_mask_dispatch(
                text_blocks, image_np, raw_mask_contiguous, method='fit_text',
                dilation_offset=dilation_offset, ignore_bubble=ignore_bubble, kernel_size=kernel_size
            )

            if refined_mask is None:
                self.logger.error("Mask refinement failed.")
                return

            # Ensure refined_mask is a valid numpy array
            if not isinstance(refined_mask, np.ndarray):
                self.logger.error(f"Refined mask is not a numpy array: {type(refined_mask)}")
                return

            if refined_mask.size == 0:
                self.logger.error("Refined mask is empty")
                return

            # Since we're already in an async context that should be thread-safe for PyQt signals,
            # let's try direct calls
            self.model.set_refined_mask(refined_mask)

            # 不自动显示refined mask，让用户自己决定是否显示
            # self.model.set_display_mask_type('refined')

            # 2. Inpaint Image - 检查是否已有inpainted图片
            inpainted_path = self.model.get_inpainted_image_path()
            if inpainted_path and os.path.exists(inpainted_path):
                # 已有inpainted图片，直接加载
                try:
                    inpainted_image = Image.open(inpainted_path)
                    # 获取原图尺寸
                    original_image = self.model.get_image()
                    # 如果inpainted图尺寸与原图不同，缩放到原图尺寸
                    if original_image and inpainted_image.size != original_image.size:
                        inpainted_image = inpainted_image.resize(original_image.size, Image.LANCZOS)
                    inpainted_image_np = np.array(inpainted_image.convert("RGB"))

                    self.model.set_inpainted_image(inpainted_image)

                    # 缓存完整修复结果，用于后续增量修复
                    self.resource_manager.set_cache(self.CACHE_LAST_INPAINTED, inpainted_image_np.copy())
                    self.resource_manager.set_cache(self.CACHE_LAST_MASK, refined_mask.copy())

                except Exception as e:
                    self.logger.error(f"Failed to load existing inpainted image: {e}")
                    # 如果加载失败，继续执行inpainting
                    inpainted_path = None

            # 如果没有已有的inpainted图片，执行inpainting
            if not inpainted_path or not os.path.exists(inpainted_path):
                try:
                    # 从配置服务获取inpainter配置
                    config = self.config_service.get_config()
                    inpainter_config_model = config.inpainter

                    # 创建InpainterConfig实例并应用配置
                    inpainter_config = InpainterConfig()
                    inpainter_config.inpainting_precision = InpaintPrecision(inpainter_config_model.inpainting_precision)
                    inpainter_config.force_use_torch_inpainting = inpainter_config_model.force_use_torch_inpainting

                    # 从配置获取inpainter模型
                    inpainter_name = inpainter_config_model.inpainter
                    try:
                        inpainter_key = Inpainter(inpainter_name)
                    except ValueError:
                        self.logger.warning(f"Unknown inpainter model: {inpainter_name}, defaulting to lama_large")
                        inpainter_key = Inpainter.lama_large

                    # 从配置获取inpainting尺寸
                    inpainting_size = inpainter_config_model.inpainting_size

                    # 从配置获取GPU设置
                    cli_config = config.cli
                    use_gpu = cli_config.use_gpu
                    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

                    inpainted_image_np = await inpaint_dispatch(
                        inpainter_key=inpainter_key,
                        image=image_np,
                        mask=refined_mask,
                        config=inpainter_config,
                        inpainting_size=inpainting_size,
                        device=device
                    )

                    if inpainted_image_np is not None:
                        inpainted_image = Image.fromarray(inpainted_image_np)
                        self.model.set_inpainted_image(inpainted_image)

                        # 缓存完整修复结果，用于后续增量修复
                        self.resource_manager.set_cache(self.CACHE_LAST_INPAINTED, inpainted_image_np.copy())
                        self.resource_manager.set_cache(self.CACHE_LAST_MASK, refined_mask.copy())
                    else:
                        self.logger.error("Inpainting failed, returned None.")

                except Exception as e:
                    self.logger.error(f"Error during inpainting process: {e}", exc_info=True)

        except asyncio.CancelledError:
            raise  # 重新抛出，让任务正确取消
        except Exception as e:
            self.logger.error(f"Error during async refine and inpaint: {e}")

    async def _async_incremental_inpaint(self, current_mask, original_mask):
        """边界框局部修复 - 只修复变化区域的矩形边界框"""
        try:
            image = self._get_current_image()

            if image is None or current_mask is None:
                self.logger.warning("Incremental inpainting skipped: missing data.")
                return

            # 检查是否有缓存
            last_processed_mask = self.resource_manager.get_cache(self.CACHE_LAST_MASK)
            if last_processed_mask is None:
                await self._async_full_inpaint_with_cache(current_mask)
                return

            # 确保蒙版是2D灰度图
            if len(current_mask.shape) > 2:
                current_mask_2d = cv2.cvtColor(current_mask, cv2.COLOR_BGR2GRAY)
            else:
                current_mask_2d = current_mask.copy()

            if len(last_processed_mask.shape) > 2:
                last_mask_2d = cv2.cvtColor(last_processed_mask, cv2.COLOR_BGR2GRAY)
            else:
                last_mask_2d = last_processed_mask.copy()

            # 计算所有变化区域
            added_areas = cv2.subtract(current_mask_2d, last_mask_2d)
            removed_areas = cv2.subtract(last_mask_2d, current_mask_2d)
            all_changed_areas = cv2.bitwise_or(added_areas, removed_areas)

            if np.sum(all_changed_areas) == 0:
                return

            # 计算变化区域的边界框
            coords = np.where(all_changed_areas > 128)
            if len(coords[0]) == 0:
                return

            y_min, y_max = np.min(coords[0]), np.max(coords[0])
            x_min, x_max = np.min(coords[1]), np.max(coords[1])

            # 扩展边界框
            padding = 50
            h, w = current_mask_2d.shape
            y_min = max(0, y_min - padding)
            y_max = min(h, y_max + padding + 1)
            x_min = max(0, x_min - padding)
            x_max = min(w, x_max + padding + 1)

            # 裁剪原图和当前蒙版的对应区域
            image_np = np.array(image.convert("RGB"))
            bbox_image = image_np[y_min:y_max, x_min:x_max]
            bbox_mask = current_mask_2d[y_min:y_max, x_min:x_max]

            # 获取配置和执行局部inpainting
            config = self.config_service.get_config()
            inpainter_config_model = config.inpainter

            try:
                from manga_translator.config import (
                    Inpainter,
                    InpainterConfig,
                    InpaintPrecision,
                )
                from manga_translator.inpainting import dispatch as inpaint_dispatch
            except ImportError as e:
                self.logger.error(f"Failed to import backend modules: {e}")
                return

            inpainter_config = InpainterConfig()
            inpainter_config.inpainting_precision = InpaintPrecision(inpainter_config_model.inpainting_precision)
            inpainter_config.force_use_torch_inpainting = inpainter_config_model.force_use_torch_inpainting

            inpainter_name = inpainter_config_model.inpainter
            try:
                inpainter_key = Inpainter(inpainter_name)
            except ValueError:
                inpainter_key = Inpainter.lama_large

            inpainting_size = inpainter_config_model.inpainting_size
            cli_config = config.cli
            device = 'cuda' if cli_config.use_gpu and torch.cuda.is_available() else 'cpu'

            # 局部修复：原图矩形区域 + 对应蒙版
            bbox_result = await inpaint_dispatch(
                inpainter_key=inpainter_key,
                image=bbox_image,  # 裁剪的原图区域
                mask=bbox_mask,    # 裁剪的蒙版区域
                config=inpainter_config,
                inpainting_size=inpainting_size,
                device=device
            )

            if bbox_result is not None:
                # 将局部修复结果贴回完整图像
                last_inpainted = self.resource_manager.get_cache(self.CACHE_LAST_INPAINTED)
                if last_inpainted is None:
                    full_result = image_np.copy()
                else:
                    full_result = last_inpainted.copy()

                # 把修复结果贴回对应位置
                full_result[y_min:y_max, x_min:x_max] = bbox_result

                # 更新缓存
                self.resource_manager.set_cache(self.CACHE_LAST_INPAINTED, full_result.copy())
                self.resource_manager.set_cache(self.CACHE_LAST_MASK, current_mask_2d.copy())

                # 更新模型
                final_image = Image.fromarray(full_result)
                self.model.set_inpainted_image(final_image)

        except Exception as e:
            self.logger.error(f"Error during bounding box inpainting: {e}", exc_info=True)

    async def _async_full_inpaint_with_cache(self, mask):
        """执行完整修复并缓存结果"""
        try:
            image = self._get_current_image()

            if image is None or mask is None:
                return

            # 延迟导入后端模块
            try:
                from manga_translator.config import (
                    Inpainter,
                    InpainterConfig,
                    InpaintPrecision,
                )
                from manga_translator.inpainting import dispatch as inpaint_dispatch
            except ImportError as e:
                self.logger.error(f"Failed to import backend modules: {e}")
                return

            image_np = np.array(image.convert("RGB"))

            # 确保蒙版是2D灰度图
            if len(mask.shape) > 2:
                mask_2d = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            else:
                mask_2d = mask.copy()

            # 获取配置
            config = self.config_service.get_config()
            inpainter_config_model = config.inpainter

            inpainter_config = InpainterConfig()
            inpainter_config.inpainting_precision = InpaintPrecision(inpainter_config_model.inpainting_precision)
            inpainter_config.force_use_torch_inpainting = inpainter_config_model.force_use_torch_inpainting

            inpainter_name = inpainter_config_model.inpainter
            try:
                inpainter_key = Inpainter(inpainter_name)
            except ValueError:
                self.logger.warning(f"Unknown inpainter model: {inpainter_name}, defaulting to lama_large")
                inpainter_key = Inpainter.lama_large

            inpainting_size = inpainter_config_model.inpainting_size

            cli_config = config.cli
            use_gpu = cli_config.use_gpu
            device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

            inpainted_image_np = await inpaint_dispatch(
                inpainter_key=inpainter_key,
                image=image_np,
                mask=mask_2d,
                config=inpainter_config,
                inpainting_size=inpainting_size,
                device=device
            )

            if inpainted_image_np is not None:
                # 缓存结果
                self.resource_manager.set_cache(self.CACHE_LAST_INPAINTED, inpainted_image_np.copy())
                self.resource_manager.set_cache(self.CACHE_LAST_MASK, mask_2d.copy())

                # 更新模型
                inpainted_image = Image.fromarray(inpainted_image_np)
                self.model.set_inpainted_image(inpainted_image)

        except Exception as e:
            self.logger.error(f"Error during full inpainting with cache: {e}", exc_info=True)

    @pyqtSlot(list)
    def select_region(self, region_indices):
        self.model.set_selection(region_indices)

    @pyqtSlot(str, bool)
    def set_display_mask_type(self, mask_type: str, visible: bool):
        """Slot to control which mask is displayed ('raw' or 'refined') or if none is."""
        if visible:
            self.model.set_display_mask_type(mask_type)
        else:
            self.model.set_display_mask_type('none')

    @pyqtSlot(bool)
    def set_removed_mask_visible(self, visible: bool):
        """Slot to control visibility of removed mask parts."""
        self.model.set_removed_mask_visible(visible)

    @pyqtSlot(str)
    def set_active_tool(self, tool: str):
        """Sets the active tool in the model (e.g., 'pen', 'eraser')."""
        self.model.set_active_tool(tool)

    @pyqtSlot(bool)
    def set_geometry_edit_mode(self, enabled: bool):
        """Slot to enable or disable the geometry edit mode."""
        if enabled:
            selected_indices = self.model.get_selection()
            if len(selected_indices) != 1:
                self.logger.warning("Geometry edit mode requires a single region to be selected.")
                if self.view:
                    self.view.toolbar.edit_geometry_button.setChecked(False)
                return
            self.set_active_tool('geometry_edit')
        else:
            self.set_active_tool('select')

    @pyqtSlot(int)
    def set_brush_size(self, size: int):
        """Sets the brush size in the model."""
        self.model.set_brush_size(size)

    @pyqtSlot(dict)
    def update_mask_config(self, new_settings: dict):
        """Slot to update mask settings in the config service."""
        self.config_service.update_config(new_settings)



    @pyqtSlot(int, str)
    def update_translated_text(self, region_index: int, text: str):
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('translation') == text:
            return

        new_region_data = old_region_data.copy()
        new_region_data['translation'] = text
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Translation Region {region_index}"
        )
        self.execute_command(command)

    @pyqtSlot(int, str)
    def update_original_text(self, region_index: int, text: str):
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('text') == text:
            return

        new_region_data = old_region_data.copy()
        # 统一使用 text 字段，用户编辑和OCR识别都更新这个字段
        new_region_data['text'] = text
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Original Text Region {region_index}"
        )
        self.execute_command(command)

    @pyqtSlot(int, int)
    def update_font_size(self, region_index: int, size: int):
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('font_size') == size:
            return

        new_region_data = old_region_data.copy()
        new_region_data['font_size'] = size
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Font Size Region {region_index}"
        )
        self.execute_command(command)

    @pyqtSlot(int, str)
    def update_font_color(self, region_index: int, color: str):
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('font_color') == color:
            return

        new_region_data = old_region_data.copy()
        new_region_data['font_color'] = color
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Font Color Region {region_index}"
        )
        self.execute_command(command)
    
    @pyqtSlot(int, str)
    def update_font_family(self, region_index: int, font_filename: str):
        """Update the font family for a specific region.
        
        Args:
            region_index: Index of the region
            font_filename: Font filename (e.g., 'Arial.ttf') or empty string for default
        """
        import os
        from manga_translator.utils import BASE_PATH
        
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data:
            return
        
        # Convert filename to full path
        if font_filename:
            font_path = os.path.join(BASE_PATH, 'fonts', font_filename)
        else:
            font_path = ""
        
        # Check if font_path changed
        if old_region_data.get('font_path') == font_path:
            return

        new_region_data = old_region_data.copy()
        new_region_data['font_path'] = font_path
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Font Family Region {region_index}"
        )
        self.execute_command(command)

    @pyqtSlot(int, str)
    def update_alignment(self, region_index: int, alignment_text: str):
        """槽：响应UI中的对齐方式修改"""
        alignment_map = {"自动": "auto", "左对齐": "left", "居中": "center", "右对齐": "right"}
        alignment_value = alignment_map.get(alignment_text, "auto")

        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('alignment') == alignment_value:
            return

        new_region_data = old_region_data.copy()
        new_region_data['alignment'] = alignment_value
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Alignment to {alignment_value}"
        )
        self.execute_command(command)

    @pyqtSlot(int, dict)
    def update_region_geometry(self, region_index: int, new_region_data: dict):
        """处理来自视图的区域几���变化。"""
        # 现在RegionTextItem在调用callback之前不会修改self.region_data
        # 所以我们可以从模型中获取正确的旧数据
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data:
            return
            
        # 深拷贝以避免引用问题
        old_region_data = copy.deepcopy(old_region_data)
        
        

        # --- UNDO DEBUGGING ---
        self.logger.debug("--- Creating Undo Command ---")
        self.logger.debug(f"OLD_DATA (before command): angle={old_region_data.get('angle')}, polygons[0][0]={old_region_data.get('polygons')[0][0] if old_region_data.get('polygons') else 'N/A'}")
        self.logger.debug(f"NEW_DATA (from view): angle={new_region_data.get('angle')}, polygons[0][0]={new_region_data.get('polygons')[0][0] if new_region_data.get('polygons') else 'N/A'}")
        # --- END UNDO DEBUGGING ---

        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Resize/Move/Rotate Region {region_index}"
        )
        self.execute_command(command)

    @pyqtSlot(int, str)
    def update_direction(self, region_index: int, direction_text: str):
        """槽：响应UI中的方向修改"""
        direction_map = {"自动": "auto", "横排": "horizontal", "竖排": "vertical"}
        direction_value = direction_map.get(direction_text, "auto")

        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data or old_region_data.get('direction') == direction_value:
            return

        new_region_data = old_region_data.copy()
        new_region_data['direction'] = direction_value
        
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Update Direction to {direction_value}"
        )
        self.execute_command(command)

    def execute_command(self, command):
        """执行命令并更新UI - 使用 Qt 的 QUndoStack"""
        if command:
            # Qt 的 push() 会自动调用 command.redo()
            self.history_service.push_command(command)
            self._update_undo_redo_buttons()

    def undo(self):
        """撤销操作 - 使用 Qt 的 QUndoStack"""
        self.history_service.undo()
        self._update_undo_redo_buttons()

    def redo(self):
        """重做操作 - 使用 Qt 的 QUndoStack"""
        self.history_service.redo()
        self._update_undo_redo_buttons()

    @pyqtSlot(int, list)
    def add_geometry_to_region(self, region_index: int, new_polygon_coords: list):
        """Adds a new polygon (in image coordinates) to an existing region."""
        
        old_region_data = self._get_region_by_index(region_index)
        if not old_region_data:
            self.logger.error(f"Invalid region index {region_index} for adding geometry.")
            return

        new_region_data = old_region_data.copy()
        
        if 'lines' not in new_region_data or not new_region_data['lines']:
            new_region_data['lines'] = []
        
        # 获取原始数据
        old_angle = old_region_data.get('angle', 0)

        # 如果没有 center,从 lines 计算
        if 'center' in old_region_data:
            old_center = old_region_data['center']
        else:
            old_lines = old_region_data.get('lines', [])
            if old_lines:
                all_pts = [pt for ln in old_lines for pt in ln]
                old_cx, old_cy = get_polygon_center(all_pts)
                old_center = [old_cx, old_cy]
            else:
                old_center = [0, 0]

        # new_polygon_coords 是旋转后的世界坐标,需要反旋转回模型坐标
        from .desktop_ui_geometry import rotate_point

        # 反旋转: 使用 -angle 将世界坐标转换为模型坐标
        new_polygon_model = []
        for x, y in new_polygon_coords:
            # 反旋转: 围绕 old_center 旋转 -old_angle
            x_model, y_model = rotate_point(x, y, -old_angle, old_center[0], old_center[1])
            new_polygon_model.append([x_model, y_model])

        # 追加未旋转的模型坐标
        lines_old = list(new_region_data.get('lines', []))
        lines_new = lines_old + [new_polygon_model]

        # 重新计算 center
        all_pts = [pt for ln in lines_new for pt in ln]
        new_cx, new_cy = get_polygon_center(all_pts)

        # 将 lines 转换到新的模型坐标系(以新中心为旋转中心)
        # 这样可以保证视觉位置不变
        final_lines_model = []
        for poly_model in lines_new:
            # 先转换到世界坐标(旋转)
            poly_world = [rotate_point(x, y, old_angle, old_center[0], old_center[1]) for x, y in poly_model]
            # 再转换回新的模型坐标系(反旋转,使用新的 center)
            poly_new_model = [rotate_point(x, y, -old_angle, new_cx, new_cy) for x, y in poly_world]
            final_lines_model.append(poly_new_model)

        new_region_data['lines'] = final_lines_model
        new_region_data['center'] = [new_cx, new_cy]

        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Add Geometry to Region {region_index}"
        )
        self.execute_command(command)

    # --- 右键菜单相关方法 ---
    def ocr_regions(self, region_indices: list):
        """对指定区域进行OCR识别，使用与UI按钮相同的逻辑"""
        if not region_indices:
            return
        
        # 临时保存当前选择
        original_selection = self.model.get_selection()
        
        # 设置选择为要OCR的区域
        self.model.set_selection(region_indices)
        
        # 调用现有的OCR方法（这会使用UI配置的OCR模型）
        self.run_ocr_for_selection()
        
        # 恢复原始选择
        self.model.set_selection(original_selection)

    def translate_regions(self, region_indices: list):
        """翻译指定区域的文本，使用与UI按钮相同的逻辑"""
        if not region_indices:
            return
        
        # 临时保存当前选择
        original_selection = self.model.get_selection()
        
        # 设置选择为要翻译的区域
        self.model.set_selection(region_indices)
        
        # 调用现有的翻译方法（这会使用UI配置的翻译器和目标语言）
        self.run_translation_for_selection()
        
        # 恢复原始选择
        self.model.set_selection(original_selection)

    def copy_region(self, region_index: int):
        """复制指定区域的数据"""
        region_data = self.model.get_region_by_index(region_index)
        if not region_data:
            self.logger.error(f"区域 {region_index} 不存在")
            return

        # 将区域数据保存到历史服务的剪贴板
        self.history_service.copy_to_clipboard(copy.deepcopy(region_data))

    def paste_region_style(self, region_index: int):
        """将复制的样式粘贴到指定区域"""
        clipboard_data = self.history_service.paste_from_clipboard()
        if not clipboard_data:
            self.logger.warning("没有复制的区域数据")
            return
        
        region_data = self.model.get_region_by_index(region_index)
        if not region_data:
            self.logger.error(f"区域 {region_index} 不存在")
            return
        
        # 复制样式相关属性，但保留位置和文本
        old_region_data = region_data.copy()
        new_region_data = region_data.copy()
        
        # 复制样式属性
        style_keys = ['font_path', 'font_family', 'font_size', 'font_color', 'alignment', 'direction', 'bold', 'italic']
        for key in style_keys:
            if key in clipboard_data:
                new_region_data[key] = clipboard_data[key]
        
        command = UpdateRegionCommand(
            model=self.model,
            region_index=region_index,
            old_data=old_region_data,
            new_data=new_region_data,
            description=f"Paste Style to Region {region_index}"
        )
        self.execute_command(command)

    def delete_regions(self, region_indices: list):
        """删除指定的区域

        删除逻辑:
        - 如果区域有紫色多边形(active_polygon_index >= 0),只删除那个多边形
        - 如果区域没有紫色多边形(active_polygon_index == -1),删除整个区域
        """
        if not region_indices:
            return

        # 重置 _last_edited_region_index,确保删除操作触发完全更新
        if self.view and hasattr(self.view, 'graphics_view'):
            graphics_view = self.view.graphics_view
            if graphics_view and hasattr(graphics_view, '_last_edited_region_index'):
                graphics_view._last_edited_region_index = None

        # 按索引倒序处理，避免索引变化问题
        sorted_indices = sorted(region_indices, reverse=True)

        regions_to_delete = []  # 需要完全删除的区域索引

        for region_index in sorted_indices:
            if 0 <= region_index < len(self.model._regions):
                # 获取对应的 region_item,检查 active_polygon_index
                region_item = None
                if self.view and hasattr(self.view, 'graphics_view'):
                    graphics_view = self.view.graphics_view
                    if hasattr(graphics_view, '_region_items') and region_index < len(graphics_view._region_items):
                        region_item = graphics_view._region_items[region_index]

                active_polygon_index = -1
                if region_item and hasattr(region_item, 'active_polygon_index'):
                    active_polygon_index = region_item.active_polygon_index



                # 获取区域数据
                region_data = self.model._regions[region_index]
                lines = region_data.get('lines', [])

                if active_polygon_index >= 0 and active_polygon_index < len(lines):
                    # 有紫色多边形,只删除那个多边形
                    old_data = region_data.copy()
                    new_data = region_data.copy()

                    # 删除指定的多边形
                    new_lines_model = [line for i, line in enumerate(lines) if i != active_polygon_index]

                    if len(new_lines_model) == 0:
                        # 如果删除后没有多边形了,删除整个区域
                        regions_to_delete.append(region_index)
                    else:
                        # 获取旧的 center 和 angle
                        old_center = region_data.get('center', [0, 0])
                        old_angle = region_data.get('angle', 0)

                        # 重新计算新的 center (基于剩余的多边形)
                        from .desktop_ui_geometry import get_polygon_center, rotate_point
                        all_pts = [pt for ln in new_lines_model for pt in ln]
                        new_cx, new_cy = get_polygon_center(all_pts)

                        # 为了保持视觉位置不变,需要将 lines 转换到新的坐标系
                        # 步骤:
                        # 1. 将旧的 lines (模型坐标) 转换为世界坐标 (旋转)
                        # 2. 将世界坐标转换为新的模型坐标 (反旋转,使用新的 center)

                        final_lines_model = []
                        for poly_model in new_lines_model:
                            # 转换为世界坐标
                            poly_world = [rotate_point(x, y, old_angle, old_center[0], old_center[1]) for x, y in poly_model]
                            # 转换为新的模型坐标
                            poly_new_model = [rotate_point(x, y, -old_angle, new_cx, new_cy) for x, y in poly_world]
                            final_lines_model.append(poly_new_model)

                        # 更新数据
                        new_data['lines'] = final_lines_model
                        new_data['center'] = [new_cx, new_cy]

                        # 创建更新命令
                        command = UpdateRegionCommand(
                            model=self.model,
                            region_index=region_index,
                            old_data=old_data,
                            new_data=new_data,
                            description=f"Delete Polygon {active_polygon_index} from Region {region_index}"
                        )
                        self.execute_command(command)
                else:
                    # 没有紫色多边形,删除整个区域
                    regions_to_delete.append(region_index)

        # 删除需要完全删除的区域
        if regions_to_delete:
            # regions_to_delete 已经是倒序的,直接从后往前删除
            # 使用命令模式以支持撤销
            from editor.commands import DeleteRegionCommand

            for region_index in regions_to_delete:
                if 0 <= region_index < len(self.model._regions):
                    region_data = self.model._regions[region_index]
                    command = DeleteRegionCommand(
                        model=self.model,
                        region_index=region_index,
                        region_data=region_data,
                        description=f"Delete Region {region_index}"
                    )
                    self.execute_command(command)

        # 清除选择
        self.model.set_selection([])

    def enter_drawing_mode(self):
        """进入绘制模式以添加新文本框"""
        # 如果当前在编辑形状模式下,先退出
        if self.model.get_active_tool() == 'geometry_edit':
            self.set_active_tool('select')

        # 清除当前选择
        self.model.set_selection([])

        # 设置工具为绘制文本框
        self.model.set_active_tool('draw_textbox')

    def paste_region(self, mouse_pos=None):
        """粘贴复制的区域到新位置

        参数:
            mouse_pos: 鼠标位置 (scene coordinates),如果提供则在该位置粘贴
        """
        clipboard_data = self.history_service.paste_from_clipboard()
        if not clipboard_data:
            self.logger.warning("没有复制的区域数据")
            return

        # 创建新区域
        new_region_data = copy.deepcopy(clipboard_data)

        # 计算原区域的中心点
        if 'center' in new_region_data:
            old_center_x, old_center_y = new_region_data['center']
        elif 'lines' in new_region_data and new_region_data['lines']:
            # 从 lines 计算中心点
            all_points = [point for line in new_region_data['lines'] for point in line]
            if all_points:
                old_center_x = sum(p[0] for p in all_points) / len(all_points)
                old_center_y = sum(p[1] for p in all_points) / len(all_points)
            else:
                old_center_x, old_center_y = 0, 0
        else:
            old_center_x, old_center_y = 0, 0

        # 计算新的中心点
        if mouse_pos:
            # 如果提供了鼠标位置,在该位置粘贴
            new_center_x, new_center_y = mouse_pos.x(), mouse_pos.y()
        else:
            # 否则稍微偏移避免重叠
            new_center_x = old_center_x + 20
            new_center_y = old_center_y + 20

        # 计算偏移量
        offset_x = new_center_x - old_center_x
        offset_y = new_center_y - old_center_y

        # 应用偏移到所有坐标
        if 'center' in new_region_data:
            new_region_data['center'] = [new_center_x, new_center_y]

        if 'lines' in new_region_data and new_region_data['lines']:
            for line in new_region_data['lines']:
                for point in line:
                    point[0] += offset_x
                    point[1] += offset_y

        if 'polygons' in new_region_data and new_region_data['polygons']:
            for polygon in new_region_data['polygons']:
                for point in polygon:
                    point[0] += offset_x
                    point[1] += offset_y

        # 添加到模型 - 使用命令模式以支持撤销
        from editor.commands import AddRegionCommand

        command = AddRegionCommand(
            model=self.model,
            region_data=new_region_data,
            description="Paste Region"
        )
        self.execute_command(command)

        # 选中新粘贴的区域
        new_index = len(self.model._regions) - 1
        self.model.set_selection([new_index])

    def _update_undo_redo_buttons(self):
        """更新撤销/重做按钮的启用状态"""
        can_undo = self.history_service.can_undo()
        can_redo = self.history_service.can_redo()
        
        # 通过view更新工具栏按钮状态
        if hasattr(self, 'view'):
            if hasattr(self.view, 'toolbar'):
                self.view.toolbar.update_undo_redo_state(can_undo, can_redo)
        else:
            print("DEBUG: Controller has no view attribute")

    @pyqtSlot()
    def open_file_dialog_and_load(self):
        """Opens a file dialog and loads the selected image into the editor."""
        from PyQt6.QtWidgets import QFileDialog

        from services import ServiceManager

        config_service = get_config_service()
        last_dir = config_service.get_config().app.last_open_dir

        file_path, _ = QFileDialog.getOpenFileName(
            None,
            "打开图片文件",
            last_dir,
            "Image Files (*.png *.jpg *.jpeg *.bmp *.webp *.avif)"
        )
        if file_path:
            new_dir = os.path.dirname(file_path)
            app_logic = ServiceManager.get_service('app_logic')
            if app_logic:
                app_logic.set_last_open_dir(new_dir)
            self.load_image_and_regions(file_path)

    # --- Toolbar Slots ---

    @pyqtSlot()
    def go_back(self):
        pass  # This should likely signal the main window to switch views

    @pyqtSlot()
    def export_image(self):
        """导出基于编辑器当前数据的图片（使用编辑器的蒙版和样式设置）"""
        try:
            image = self._get_current_image()
            regions = self._get_regions()
            if not image:
                self.logger.warning("Cannot export: missing image data")
                if hasattr(self, 'toast_manager'):
                    self.toast_manager.show_error("导出失败：缺少图像数据")
                return
            
            # regions 可以为空列表，此时导出原图（可能经过上色/超分处理）
            if regions is None:
                regions = []

            mask = self.model.get_refined_mask()
            if mask is None:
                mask = self.model.get_raw_mask()
            # 如果没有区域，mask 可以为 None，后端会处理
            if mask is None and regions:
                self.logger.warning("Cannot export: no mask data available for regions")
                if hasattr(self, 'toast_manager'):
                    self.toast_manager.show_error("导出失败：没有可用的蒙版数据")
                return

            # 显示开始Toast，保存引用以便后续关闭
            self._export_toast = None
            if hasattr(self, 'toast_manager'):
                self._export_toast = self.toast_manager.show_info("正在导出...", duration=0)
            
            self.async_service.submit_task(self._async_export_with_desktop_ui_service(image, regions, mask))
        except Exception as e:
            self.logger.error(f"Error during export request: {e}", exc_info=True)
            if hasattr(self, 'toast_manager'):
                self.toast_manager.show_error("导出失败")

    async def _async_export_with_desktop_ui_service(self, image, regions, mask):
        """使用desktop-ui导出服务进行异步导出"""
        try:
            import os

            from PyQt6.QtCore import QTimer
            from PyQt6.QtWidgets import QMessageBox

            # 获取配置
            config = self.config_service.get_config()

            # 确定输出路径和文件名
            save_to_source_dir = getattr(config.cli, 'save_to_source_dir', False) if hasattr(config, 'cli') else False
            source_path = self.model.get_source_image_path()
            
            if save_to_source_dir and source_path:
                # 输出到原图所在目录的 manga_translator_work/result 子目录
                output_dir = os.path.join(os.path.dirname(source_path), 'manga_translator_work', 'result')
                os.makedirs(output_dir, exist_ok=True)
            else:
                # 原有逻辑：使用配置的输出目录
                output_dir = getattr(config.app, 'last_output_path', None) if hasattr(config, 'app') else None
                if not output_dir or not os.path.exists(output_dir):
                    if source_path:
                        output_dir = os.path.dirname(source_path)
                    else:
                        output_dir = os.getcwd()

            # 生成输出文件名（保持原文件名和格式）
            source_path = self.model.get_source_image_path()
            if source_path:
                base_name = os.path.splitext(os.path.basename(source_path))[0]
                # 获取输出格式
                output_format = getattr(config.cli, 'format', '') if hasattr(config, 'cli') else ''
                if output_format == "不指定":
                    output_format = None

                if output_format and output_format.strip():
                    output_filename = f"{base_name}.{output_format.lower()}"
                else:
                    original_ext = os.path.splitext(source_path)[1].lower()
                    output_filename = f"{base_name}{original_ext}" if original_ext else f"{base_name}.png"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_filename = f"exported_image_{timestamp}.png"

            output_path = os.path.join(output_dir, output_filename)


            # 使用本地desktop_qt_ui的导出服务
            import os
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from services.export_service import ExportService
            export_service = ExportService()

            def progress_callback(message):
                pass

            def success_callback(message):
                # 使用信号在主线程显示Toast
                self._show_toast_signal.emit(f"导出成功\n{output_path}", 5000, True, output_path)
                
                # 保存导出快照，用于检测后续是否有更改
                self._save_export_snapshot()
                
                # 导出成功后释放内存
                self.resource_manager.release_memory_after_export()

            def error_callback(message):
                self.logger.error(f"Export error: {message}")
                # 使用信号在主线程显示Toast
                self._show_toast_signal.emit(f"导出失败：{message}", 5000, False, "")

            # 转换配置为字典
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                config_dict = {}
            


            # 确保区域数据包含渲染所需的所有信息
            enhanced_regions = []
            for i, region in enumerate(regions):
                enhanced_region = region.copy()

                # 确保有翻译文本
                if not enhanced_region.get('translation'):
                    enhanced_region['translation'] = enhanced_region.get('text', '')

                # 确保有字体大小
                if not enhanced_region.get('font_size'):
                    enhanced_region['font_size'] = 16

                # 确保有对齐方式
                if not enhanced_region.get('alignment'):
                    enhanced_region['alignment'] = 'center'

                # 确保有方向
                if not enhanced_region.get('direction'):
                    enhanced_region['direction'] = 'auto'

                # 从渲染参数服务获取完整的渲染参数
                from services import get_render_parameter_service
                render_service = get_render_parameter_service()
                render_params = render_service.export_parameters_for_backend(i, enhanced_region)
                enhanced_region.update(render_params)

                enhanced_regions.append(enhanced_region)

            # 调用本地的导出服务
            export_service.export_rendered_image(
                image=image,
                regions_data=enhanced_regions,  # 使用增强的区域数据
                config=config_dict,
                output_path=output_path,
                mask=mask,
                progress_callback=progress_callback,
                success_callback=success_callback,
                error_callback=error_callback
            )

        except Exception as e:
            self.logger.error(f"Error during async export: {e}", exc_info=True)
            err_msg = str(e)
            QTimer.singleShot(0, lambda: QMessageBox.critical(None, "导出失败", f"导出过程中发生意外错误:\n{err_msg}"))

    @pyqtSlot()
    def save_json(self):
        """保存当前翻译数据到JSON文件"""
        try:
            source_path = self.model.get_source_image_path()
            if not source_path:
                self.logger.warning("Cannot save JSON: no image loaded")
                if hasattr(self, 'toast_manager'):
                    self.toast_manager.show_error("保存失败：没有加载图像")
                return
            
            regions = self._get_regions()
            if regions is None:
                regions = []
            
            # 查找现有的JSON文件路径
            from manga_translator.utils.path_manager import find_json_path, get_json_path
            json_path = find_json_path(source_path)
            
            # 如果找不到现有的JSON文件，使用默认路径
            if not json_path:
                json_path = get_json_path(source_path, create_dir=True)
                self.logger.info(f"No existing JSON found, will create new one at: {json_path}")
            else:
                self.logger.info(f"Found existing JSON, will replace: {json_path}")
            
            # 显示开始Toast
            if hasattr(self, 'toast_manager'):
                self.toast_manager.show_info("正在保存JSON...", duration=0)
            
            self.async_service.submit_task(self._async_save_json(source_path, regions, json_path))
        except Exception as e:
            self.logger.error(f"Error during save JSON request: {e}", exc_info=True)
            if hasattr(self, 'toast_manager'):
                self.toast_manager.show_error("保存JSON失败")

    async def _async_save_json(self, source_path, regions, json_path):
        """异步保存JSON文件"""
        try:
            import json
            import sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from services.export_service import ExportService
            
            export_service = ExportService()
            
            # 获取配置
            config = self.config_service.get_config()
            if hasattr(config, 'model_dump'):
                config_dict = config.model_dump()
            elif hasattr(config, 'dict'):
                config_dict = config.dict()
            else:
                config_dict = {}
            
            # 获取蒙版数据
            mask = self.model.get_refined_mask()
            if mask is None:
                mask = self.model.get_raw_mask()
            
            # 保存JSON文件 - 传入source_path用于生成正确的键
            export_service._save_regions_data_with_path(regions, json_path, source_path, mask, config_dict)
            
            # 保存快照，标记为已保存状态
            self._save_export_snapshot()
            
            # 关闭"正在保存"的toast，显示成功toast
            if hasattr(self, 'toast_manager'):
                # 直接调用close_all（在主线程中执行）
                self.toast_manager.close_all()
            
            # 使用信号在主线程显示Toast
            self._show_toast_signal.emit(f"JSON保存成功\n{json_path}", 3000, True, json_path)
            
        except Exception as e:
            self.logger.error(f"Error during async save JSON: {e}", exc_info=True)
            
            # 关闭"正在保存"的toast
            if hasattr(self, 'toast_manager'):
                self.toast_manager.close_all()
            
            # 使用信号在主线程显示Toast
            self._show_toast_signal.emit(f"保存JSON失败：{str(e)}", 3000, False, "")

    @pyqtSlot()
    def edit_source_file(self):
        """加载当前翻译后图片对应的原图进行编辑"""
        current_path = self.model.get_source_image_path()
        if not current_path:
            self.logger.warning("No image currently loaded")
            return

        # 检查当前是否是翻译后的图片
        if not self._is_translated_image(current_path):
            self.logger.info("Current image is already a source file, no need to switch")
            return

        # 从translation_map.json中查找原图路径
        try:
            import json
            norm_path = os.path.normpath(current_path)
            output_dir = os.path.dirname(norm_path)
            map_path = os.path.join(output_dir, 'translation_map.json')

            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    translation_map = json.load(f)

                source_path = translation_map.get(norm_path)
                if source_path and os.path.exists(source_path):
                    self.logger.debug(f"Loading source file for editing: {source_path}")
                    self.load_image_and_regions(source_path)
                else:
                    self.logger.warning(f"Source file not found: {source_path}")
            else:
                self.logger.warning(f"translation_map.json not found at: {map_path}")
        except Exception as e:
            self.logger.error(f"Error loading source file: {e}")

    @pyqtSlot(str)
    def set_display_mode(self, mode_text: str):
        """设置文本区域的显示模式"""
        mode_map = {
            "文字文本框显示": "full",
            "只显示文字": "text_only",
            "只显示框线": "box_only",
            "都不显示": "none"
        }
        mode = mode_map.get(mode_text, "full")
        self.logger.info(f"Toolbar: Display mode changed to '{mode_text}' -> '{mode}'.")
        self.model.set_region_display_mode(mode)
    
    def set_green_box_visible(self, visible: bool):
        """设置绿框（自动渲染区域）的可见性"""
        if self.view and hasattr(self.view.graphics_view, '_region_items'):
            for item in self.view.graphics_view._region_items:
                if hasattr(item, 'set_green_box_visible'):
                    item.set_green_box_visible(visible)
    
    def set_white_box_visible(self, visible: bool):
        """设置白框（手动调整边界）的可见性"""
        if self.view and hasattr(self.view.graphics_view, '_region_items'):
            for item in self.view.graphics_view._region_items:
                if hasattr(item, 'set_white_box_visible'):
                    item.set_white_box_visible(visible)

    @pyqtSlot(int)
    def set_original_image_alpha(self, alpha: int):
        """设置原图的不透明度 (0-100)，值越大越不透明（越显示原图）"""
        # slider = 0 -> alpha = 0.0（完全透明，显示inpainted）
        # slider = 100 -> alpha = 1.0（完全不透明，显示原图）
        alpha_float = alpha / 100.0
        self.model.set_original_image_alpha(alpha_float)
        # 标记用户已手动调整透明度
        self._user_adjusted_alpha = True

    @pyqtSlot(int)
    def set_preview_alpha(self, alpha: int):
        self.logger.info(f"Toolbar: Preview alpha changed to '{alpha}'. (Not Implemented)")
        # TODO: This should control the opacity of the inpainted image layer in the view

    def handle_global_render_setting_change(self):
        """Forces a re-render of all regions when a global render setting has changed."""

        # Clear the parameter service cache to ensure new global defaults are used
        from services import get_render_parameter_service
        render_parameter_service = get_render_parameter_service()
        render_parameter_service.clear_cache()

        # A heavy-handed but reliable way to force a full redraw of all regions with new global defaults
        self.model.set_regions(self.model.get_regions())

    @pyqtSlot()
    def render_inpaint(self):
        self.logger.info("Toolbar: 'Render Inpaint' requested.")
        # This can trigger the same async task as the one after loading a json
        self.async_service.submit_task(self._async_refine_and_inpaint())

    @pyqtSlot()
    def run_ocr_for_selection(self):
        selected_indices = self.model.get_selection()
        if not selected_indices:
            return
        image = self._get_current_image()
        if not image:
            return

        all_regions = self.model.get_regions()
        selected_regions_data = [all_regions[i] for i in selected_indices]
        
        # 显示开始Toast，保存引用以便后续关闭
        self._ocr_toast = None
        if hasattr(self, 'toast_manager'):
            self._ocr_toast = self.toast_manager.show_info("正在识别...", duration=0)
        
        self.async_service.submit_task(self._async_ocr_task(image, selected_regions_data, selected_indices))

    @pyqtSlot(list)
    def on_regions_update_finished(self, updated_regions: list):
        """Slot to safely update regions from the main thread."""
        self.model.set_regions(updated_regions)
        # 强制刷新属性栏（忽略焦点状态）
        if hasattr(self, 'view') and self.view and hasattr(self.view, 'property_panel'):
            self.view.property_panel.force_refresh_from_model()
    
    @pyqtSlot()
    def _on_ocr_completed(self):
        """OCR完成后在主线程处理Toast"""
        # 关闭"正在识别"Toast
        if hasattr(self, '_ocr_toast') and self._ocr_toast:
            try:
                self._ocr_toast.close()
                self._ocr_toast = None
            except:
                pass
        
        # 显示完成Toast
        if hasattr(self, 'toast_manager'):
            self.toast_manager.show_success("识别完成")
    
    @pyqtSlot()
    def _on_translation_completed(self):
        """翻译完成后在主线程处理Toast"""
        # 关闭"正在翻译"Toast
        if hasattr(self, '_translation_toast') and self._translation_toast:
            try:
                self._translation_toast.close()
                self._translation_toast = None
            except:
                pass
        
        # 显示完成Toast
        if hasattr(self, 'toast_manager'):
            self.toast_manager.show_success("翻译完成")

    async def _async_ocr_task(self, image, regions_to_process, indices):
        current_regions = self.model.get_regions()
        updated_regions = list(current_regions) # Create a shallow copy of the list

        # 从属性面板获取用户选择的OCR配置
        ocr_config = None
        if self.view and hasattr(self.view, 'property_panel'):
            selected_ocr = self.view.property_panel.get_selected_ocr_model()
            if selected_ocr:
                # 获取当前的OCR配置并更新ocr字段
                from manga_translator.config import OcrConfig, Ocr
                full_config = self.config_service.get_config()
                current_ocr_config = full_config.ocr if hasattr(full_config, 'ocr') else OcrConfig()
                try:
                    # 将字符串转换为Ocr枚举
                    ocr_enum = Ocr(selected_ocr) if selected_ocr else current_ocr_config.ocr
                    ocr_config = OcrConfig(
                        ocr=ocr_enum,
                        use_mocr_merge=current_ocr_config.use_mocr_merge,
                        use_hybrid_ocr=current_ocr_config.use_hybrid_ocr,
                        secondary_ocr=current_ocr_config.secondary_ocr,
                        min_text_length=current_ocr_config.min_text_length,
                        ignore_bubble=current_ocr_config.ignore_bubble,
                        prob=current_ocr_config.prob,
                        merge_gamma=current_ocr_config.merge_gamma,
                        merge_sigma=current_ocr_config.merge_sigma,
                        merge_edge_ratio_threshold=current_ocr_config.merge_edge_ratio_threshold
                    )
                    self.logger.info(f"Using OCR model from property panel: {selected_ocr}")
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"Invalid OCR selection '{selected_ocr}', using default: {e}")
                    ocr_config = None

        success_count = 0
        for i, region_data in enumerate(regions_to_process):
            region_idx = indices[i]
            try:
                ocr_result = await self.ocr_service.recognize_region(image, region_data, config=ocr_config)
                if ocr_result and ocr_result.text:
                    # Create a copy of the specific region dict to modify
                    new_region_data = updated_regions[region_idx].copy()
                    new_region_data['text'] = ocr_result.text
                    updated_regions[region_idx] = new_region_data # Replace the old dict with the new one
                    success_count += 1
            except Exception as e:
                self.logger.error(f"OCR识别失败: {e}")

        # Emit a signal to have the model updated on the main thread
        self._regions_update_finished.emit(updated_regions)
        
        # 发送OCR完成信号（在主线程处理Toast）
        self._ocr_completed.emit()
        

    @pyqtSlot()
    def run_translation_for_selection(self):
        selected_indices = self.model.get_selection()
        if not selected_indices:
            return
        image = self._get_current_image()
        if not image:
            return

        all_regions = self.model.get_regions()
        selected_regions_data = [all_regions[i] for i in selected_indices]
        texts_to_translate = [r.get('text', '') for r in selected_regions_data]
        
        # 显示开始Toast，保存引用以便后续关闭
        self._translation_toast = None
        if hasattr(self, 'toast_manager'):
            self._translation_toast = self.toast_manager.show_info("正在翻译...", duration=0)
        
        # 传递所有区域以提供上下文，但只翻译选中的文本
        self.async_service.submit_task(self._async_translation_task(texts_to_translate, selected_indices, image, all_regions))

    async def _async_translation_task(self, texts, indices, image, regions):
        # 从属性面板获取用户选择的翻译器配置
        translator_to_use = None
        target_lang_to_use = None
        
        if self.view and hasattr(self.view, 'property_panel'):
            selected_translator = self.view.property_panel.get_selected_translator()
            selected_target_lang = self.view.property_panel.get_selected_target_language()
            
            if selected_translator:
                from manga_translator.config import Translator
                try:
                    # 将字符串转换为Translator枚举
                    translator_to_use = Translator(selected_translator)
                    self.logger.info(f"Using translator from property panel: {selected_translator}")
                except (ValueError, AttributeError) as e:
                    self.logger.warning(f"Invalid translator selection '{selected_translator}', using default: {e}")
            
            if selected_target_lang:
                target_lang_to_use = selected_target_lang
                self.logger.info(f"Using target language from property panel: {selected_target_lang}")
        
        # 将image和所有regions信息传递给翻译服务以提供完整上下文
        success_count = 0
        try:
            results = await self.translation_service.translate_text_batch(
                texts, 
                translator=translator_to_use,
                target_lang=target_lang_to_use,
                image=image, 
                regions=regions
            )
            current_regions = self.model.get_regions()
            updated_regions = list(current_regions) # Create a shallow copy

            for i, result in enumerate(results):
                if result and result.translated_text:
                    region_idx = indices[i]
                    # Create a copy of the specific region dict to modify
                    new_region_data = updated_regions[region_idx].copy()
                    new_region_data['translation'] = result.translated_text
                    updated_regions[region_idx] = new_region_data # Replace the old dict
                    success_count += 1

            # Emit a signal to have the model updated on the main thread
            self._regions_update_finished.emit(updated_regions)
            
            # 发送翻译完成信号
            self._translation_completed.emit()
        except Exception as e:
            self.logger.error(f"翻译失败: {e}")
            # TODO: 添加翻译失败的信号处理

    @pyqtSlot(list)
    def set_selection_from_list(self, indices: list):
        """Slot to handle selection changes originating from the RegionListView."""
        self.model.set_selection(indices)

    @pyqtSlot(list)
    def on_backend_task_completed(self, results: list):
        """
        Slot to handle the completion of a backend task (like translation).
        Reloads the data for the currently active image to show the result.
        """
        self.logger.info("Backend task finished, received results. Refreshing current editor view.")
        current_source_path = self.model.get_source_image_path()
        if not current_source_path:
            self.logger.warning("No active image in editor to refresh.")
            return

        # Find the result for the current file
        norm_current_path = os.path.normpath(current_source_path)
        for result in results:
            if hasattr(result, 'image_path') and os.path.normpath(result.image_path) == norm_current_path:
                if result.success:
                    self.logger.info(f"Result for {os.path.basename(current_source_path)} is success. Reloading.")
                    # Re-load the image. The file service will now find the new translated file.
                    self.load_image_and_regions(current_source_path)
                else:
                    error_message = getattr(result, 'error_message', 'Unknown error')
                    self.logger.error(f"Backend task failed for {os.path.basename(current_source_path)}: {error_message}")
                break