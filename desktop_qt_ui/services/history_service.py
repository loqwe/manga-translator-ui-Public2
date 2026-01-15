
"""
编辑器历史管理器
现在使用 Qt 原生的 QUndoStack 处理撤销/重做，只保留剪贴板功能。
"""
import copy
import logging
from typing import Any, Optional

from PyQt6.QtGui import QUndoStack


class ClipboardManager:
    """剪贴板管理器，处理内部数据的复制粘贴。"""
    
    def __init__(self):
        self.clipboard_data = None
        self.logger = logging.getLogger(__name__)
    
    def copy_to_clipboard(self, data: Any):
        """复制数据到内部剪贴板。"""
        self.clipboard_data = copy.deepcopy(data)
        self.logger.debug("Data copied to internal clipboard")
    
    def paste_from_clipboard(self) -> Any:
        """从内部剪贴板粘贴数据。"""
        if self.clipboard_data is not None:
            return copy.deepcopy(self.clipboard_data)
        return None
    
    def has_data(self) -> bool:
        """检查剪贴板是否有数据。"""
        return self.clipboard_data is not None


class EditorStateManager:
    """
    编辑器状态管理器。
    
    现在使用 Qt 原生的 QUndoStack 处理撤销/重做，
    只保留剪贴板功能作为额外服务。
    """
    
    def __init__(self):
        # 使用 Qt 原生的 QUndoStack
        self.undo_stack = QUndoStack()
        self.undo_stack.setUndoLimit(50)  # 限制历史记录数量
        
        # 保留剪贴板功能
        self.clipboard = ClipboardManager()
        
        self.logger = logging.getLogger(__name__)
        
        # 连接信号用于调试
        self.undo_stack.canUndoChanged.connect(
            lambda can: self.logger.debug(f"Can undo changed: {can}")
        )
        self.undo_stack.canRedoChanged.connect(
            lambda can: self.logger.debug(f"Can redo changed: {can}")
        )
    
    def push_command(self, command):
        """
        推送命令到撤销栈。
        Qt 的 push() 会自动调用 command.redo()。
        """
        self.undo_stack.push(command)
    
    def undo(self):
        """撤销上一个操作。"""
        self.undo_stack.undo()
    
    def redo(self):
        """重做上一个被撤销的操作。"""
        self.undo_stack.redo()
    
    def can_undo(self) -> bool:
        """检查是否可以撤销。"""
        return self.undo_stack.canUndo()
    
    def can_redo(self) -> bool:
        """检查是否可以重做。"""
        return self.undo_stack.canRedo()
    
    def copy_to_clipboard(self, data: Any):
        """复制数据到内部剪贴板。"""
        self.clipboard.copy_to_clipboard(data)
    
    def paste_from_clipboard(self) -> Any:
        """从内部剪贴板粘贴数据。"""
        return self.clipboard.paste_from_clipboard()
    
    def clear(self):
        """清除历史记录。"""
        self.undo_stack.clear()
        self.logger.debug("Cleared undo stack")
    
    @property
    def undo_stack_size(self) -> int:
        """获取撤销栈的大小，用于检查是否有未保存的修改。"""
        return self.undo_stack.index()
    
    def create_undo_action(self, parent, text: str = "撤销"):
        """创建撤销动作（用于菜单/工具栏）。"""
        return self.undo_stack.createUndoAction(parent, text)
    
    def create_redo_action(self, parent, text: str = "重做"):
        """创建重做动作（用于菜单/工具栏）。"""
        return self.undo_stack.createRedoAction(parent, text)


# --- Singleton Pattern ---
_history_service_instance: Optional[EditorStateManager] = None

def get_history_service() -> EditorStateManager:
    """获取历史记录服务的单例。"""
    global _history_service_instance
    if _history_service_instance is None:
        _history_service_instance = EditorStateManager()
    return _history_service_instance
