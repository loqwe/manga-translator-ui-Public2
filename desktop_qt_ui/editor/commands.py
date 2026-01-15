
import copy
from typing import Any, Dict

import numpy as np
from PyQt6.QtGui import QUndoCommand


# Forward declaration for type hinting to avoid circular imports
class EditorModel:
    pass

class UpdateRegionCommand(QUndoCommand):
    """用于更新单个区域数据的通用命令。"""
    def __init__(self, model: EditorModel, region_index: int, old_data: Dict[str, Any], new_data: Dict[str, Any], description: str = "Update Region"):
        super().__init__(description)
        self._model = model
        self._index = region_index
        # 存储深拷贝以防止后续修改影响历史状态
        self._old_data = copy.deepcopy(old_data)
        self._new_data = copy.deepcopy(new_data)

    def _apply_data(self, data_to_apply: Dict[str, Any]):
        """将给定的数据字典应用到模型中的区域。"""
        if not (0 <= self._index < len(self._model._regions)):
            return

        # 检查 center 是否改变
        old_center = self._model._regions[self._index].get('center')
        new_center = data_to_apply.get('center')
        center_changed = old_center != new_center

        # 直接更新模型中的区域字典
        self._model._regions[self._index] = data_to_apply
        
        # 【修复】同时更新resource_manager中的regions
        # ResourceManager使用region_id，需要通过索引获取对应的region_id
        try:
            from services import get_resource_manager
            resource_manager = get_resource_manager()
            all_regions = resource_manager.get_all_regions()
            if self._index < len(all_regions):
                # 获取对应索引的region_id
                region_resource = all_regions[self._index]
                resource_manager.update_region(region_resource.region_id, data_to_apply)
        except Exception:
            # 静默失败，不影响主流程
            pass

        # 如果 center 改变了,需要触发完全更新,重新创建 item
        # 否则只触发单个 item 更新
        if center_changed:
            # 保存当前选择状态
            old_selection = self._model.get_selection()
            # 触发完全更新
            self._model.regions_changed.emit(self._model._regions)
            # 恢复选择状态(只有当选择的region还存在时)
            if old_selection:
                # 检查选择的region是否还在有效范围内
                valid_selection = [idx for idx in old_selection if 0 <= idx < len(self._model._regions)]
                if valid_selection:
                    self._model.set_selection(valid_selection)
        else:
            # 发出目标性强的信号，让UI只刷新这一个区域
            # region_style_updated 是一个理想的通用信号，因为它只传递索引
            self._model.region_style_updated.emit(self._index)

    def redo(self):
        """执行操作：应用新数据。"""
        self._apply_data(copy.deepcopy(self._new_data))

    def undo(self):
        """撤销操作：应用旧数据。"""
        self._apply_data(copy.deepcopy(self._old_data))

class AddRegionCommand(QUndoCommand):
    """用于添加新区域的命令。"""
    def __init__(self, model: EditorModel, region_data: Dict[str, Any], description: str = "Add Region"):
        super().__init__(description)
        self._model = model
        # 存储新区域的数据
        self._region_data = copy.deepcopy(region_data)
        # 记录添加的位置(索引)
        self._index = None

    def redo(self):
        """执行添加操作"""
        self._model._regions.append(copy.deepcopy(self._region_data))
        self._index = len(self._model._regions) - 1
        
        # 【修复】同时更新resource_manager中的regions
        try:
            from services import get_resource_manager
            resource_manager = get_resource_manager()
            resource_manager.add_region(self._region_data)
        except Exception:
            pass  # 静默失败
        
        # 添加操作触发完全更新
        self._model.regions_changed.emit(self._model._regions)

    def undo(self):
        """撤销添加操作:删除最后添加的区域"""
        if self._index is not None and 0 <= self._index < len(self._model._regions):
            self._model._regions.pop(self._index)
            
            # 【修复】同时更新resource_manager中的regions
            # 由于ResourceManager使用字典存储，需要重新同步整个列表
            try:
                from services import get_resource_manager
                resource_manager = get_resource_manager()
                resource_manager.clear_regions()
                for region_data in self._model._regions:
                    resource_manager.add_region(region_data)
            except Exception:
                pass  # 静默失败
            
            # 删除操作会改变后续区域的索引,必须触发完全更新
            self._model.regions_changed.emit(self._model._regions)
            # 清除选择
            self._model.set_selection([])

class DeleteRegionCommand(QUndoCommand):
    """用于删除区域的命令。"""
    def __init__(self, model: EditorModel, region_index: int, region_data: Dict[str, Any], description: str = "Delete Region"):
        super().__init__(description)
        self._model = model
        self._index = region_index
        # 存储被删除区域的数据,用于撤销
        self._deleted_data = copy.deepcopy(region_data)

    def redo(self):
        """执行删除操作"""
        if 0 <= self._index < len(self._model._regions):
            self._model._regions.pop(self._index)
            
            # 【修复】同时更新resource_manager中的regions
            # 由于ResourceManager使用字典存储，需要重新同步整个列表
            try:
                from services import get_resource_manager
                resource_manager = get_resource_manager()
                resource_manager.clear_regions()
                for region_data in self._model._regions:
                    resource_manager.add_region(region_data)
            except Exception:
                pass  # 静默失败
            
            # 删除操作会改变后续区域的索引,必须触发完全更新
            # 通过直接调用 regions_changed 信号(而不是 region_style_updated)来确保完全更新
            self._model.regions_changed.emit(self._model._regions)
            # 清除选择,因为被删除的区域可能被选中
            self._model.set_selection([])

    def undo(self):
        """撤销删除操作:在原位置插入回区域"""
        if 0 <= self._index <= len(self._model._regions):
            self._model._regions.insert(self._index, copy.deepcopy(self._deleted_data))
            
            # 【修复】同时更新resource_manager中的regions
            # 由于ResourceManager使用字典存储，需要重新同步整个列表
            try:
                from services import get_resource_manager
                resource_manager = get_resource_manager()
                resource_manager.clear_regions()
                for region_data in self._model._regions:
                    resource_manager.add_region(region_data)
            except Exception:
                pass  # 静默失败
            
            # 插入操作会改变后续区域的索引,必须触发完全更新
            self._model.regions_changed.emit(self._model._regions)
            # 恢复选择到被恢复的区域
            self._model.set_selection([self._index])

class MaskEditCommand(QUndoCommand):
    """用于处理蒙版编辑的命令。"""
    def __init__(self, model: EditorModel, old_mask: np.ndarray, new_mask: np.ndarray):
        super().__init__("Edit Mask")
        self._model = model
        self._old_mask = old_mask
        self._new_mask = new_mask

    def redo(self):
        self._model.set_refined_mask(self._new_mask)

    def undo(self):
        self._model.set_refined_mask(self._old_mask)
