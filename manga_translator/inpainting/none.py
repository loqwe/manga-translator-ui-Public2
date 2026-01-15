import numpy as np

from .common import CommonInpainter
from ..config import InpainterConfig


class NoneInpainter(CommonInpainter):

    async def _inpaint(self, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
        import cv2
        img_inpainted = np.copy(image)
        
        # 确保蒙版是单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 二值化蒙版，避免灰色区域导致不完全覆盖
        _, mask_binary = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        
        # 将蒙版区域涂成纯白色
        img_inpainted[mask_binary > 0] = np.array([255, 255, 255], np.uint8)
        
        return img_inpainted
