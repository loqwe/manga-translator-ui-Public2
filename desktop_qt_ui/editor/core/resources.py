"""资源数据结构

定义编辑器中使用的所有资源类。
"""

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import numpy as np
from PIL import Image

from .types import JobState, MaskType


@dataclass
class ImageResource:
    """图片资源"""
    path: str
    image: Image.Image  # PIL Image
    width: int
    height: int
    json_data: Optional[Dict] = None  # 关联的JSON数据
    load_time: float = field(default_factory=time.time)
    
    def release(self) -> None:
        """释放资源"""
        if self.image:
            try:
                self.image.close()
            except Exception:
                pass
            self.image = None
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.release()


@dataclass
class MaskResource:
    """蒙版资源"""
    mask_type: MaskType
    data: np.ndarray
    width: int
    height: int
    create_time: float = field(default_factory=time.time)
    
    def release(self) -> None:
        """释放资源"""
        if self.data is not None:
            self.data = None
    
    def __del__(self):
        """析构函数，确保资源释放"""
        self.release()


@dataclass
class RegionResource:
    """文本区域资源"""
    region_id: int
    data: Dict  # 区域数据（包含坐标、文本、样式等）
    create_time: float = field(default_factory=time.time)
    update_time: float = field(default_factory=time.time)


@dataclass
class AsyncJob:
    """异步任务"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    state: JobState = JobState.PENDING
    priority: int = 5
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    result: Any = None
    error: Optional[Exception] = None
    on_complete: Optional[Callable[[Any], None]] = None
    on_error: Optional[Callable[[Exception], None]] = None
    on_cancelled: Optional[Callable[[], None]] = None
    
    def mark_running(self) -> None:
        """标记任务为运行中"""
        self.state = JobState.RUNNING
        self.started_at = time.time()
    
    def mark_completed(self, result: Any) -> None:
        """标记任务完成"""
        self.state = JobState.COMPLETED
        self.completed_at = time.time()
        self.result = result
        if self.on_complete:
            try:
                self.on_complete(result)
            except Exception as e:
                print(f"Error in on_complete callback: {e}")
    
    def mark_failed(self, error: Exception) -> None:
        """标记任务失败"""
        self.state = JobState.FAILED
        self.completed_at = time.time()
        self.error = error
        if self.on_error:
            try:
                self.on_error(error)
            except Exception as e:
                print(f"Error in on_error callback: {e}")
    
    def mark_cancelled(self) -> None:
        """标记任务取消"""
        self.state = JobState.CANCELLED
        self.completed_at = time.time()
        if self.on_cancelled:
            try:
                self.on_cancelled()
            except Exception as e:
                print(f"Error in on_cancelled callback: {e}")
    
    def is_active(self) -> bool:
        """检查任务是否活跃（运行中或等待中）"""
        return self.state in (JobState.PENDING, JobState.RUNNING)
    
    def is_finished(self) -> bool:
        """检查任务是否已结束"""
        return self.state in (JobState.COMPLETED, JobState.CANCELLED, JobState.FAILED)

