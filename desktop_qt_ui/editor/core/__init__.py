"""编辑器核心模块

此模块包含编辑器的核心数据结构、类型定义和管理器。
"""

from .types import (
    EditorState,
    JobState,
    JobPriority,
    MaskType,
    ResourceType,
)
from .resources import (
    ImageResource,
    MaskResource,
    RegionResource,
    AsyncJob,
)
from .async_job_manager import AsyncJobManager
from .resource_manager import ResourceManager

__all__ = [
    # Types
    "EditorState",
    "JobState",
    "JobPriority",
    "MaskType",
    "ResourceType",
    # Resources
    "ImageResource",
    "MaskResource",
    "RegionResource",
    "AsyncJob",
    # Managers
    "AsyncJobManager",
    "ResourceManager",
]

