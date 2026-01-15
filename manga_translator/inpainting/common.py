import numpy as np
from abc import abstractmethod

from ..config import InpainterConfig
from ..utils import InfererModule, ModelWrapper

class CommonInpainter(InfererModule):

    async def inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        return await self._inpaint(image, mask, config, inpainting_size, verbose)

    @abstractmethod
    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass

class OfflineInpainter(CommonInpainter, ModelWrapper):
    _MODEL_SUB_DIR = 'inpainting'

    async def _inpaint(self, *args, **kwargs):
        result = await self.infer(*args, **kwargs)
        # ✅ 统一Inpainting内存清理：在修复完成后立即清理
        self._cleanup_memory()
        return result
    
    def _cleanup_memory(self):
        """统一的Inpainting内存清理方法，在每次推理后自动调用"""
        import torch
        import gc
        
        # 清理CUDA缓存（多次确保彻底）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # 强制垃圾回收（3次确保彻底）
        for _ in range(3):
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @abstractmethod
    async def _infer(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        pass
