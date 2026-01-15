from typing import Optional

import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
from .inpainting_sd import StableDiffusionInpainter
from .none import NoneInpainter
from .original import OriginalInpainter
from ..config import Inpainter, InpainterConfig

INPAINTERS = {
    Inpainter.default: AotInpainter,
    Inpainter.lama_large: LamaLargeInpainter,
    Inpainter.lama_mpe: LamaMPEInpainter,
    Inpainter.sd: StableDiffusionInpainter,
    Inpainter.none: NoneInpainter,
    Inpainter.original: OriginalInpainter,
}
inpainter_cache = {}

def get_inpainter(key: Inpainter, *args, **kwargs) -> CommonInpainter:
    if key not in INPAINTERS:
        raise ValueError(f'Could not find inpainter for: "{key}". Choose from the following: %s' % ','.join(INPAINTERS))
    if not inpainter_cache.get(key):
        inpainter = INPAINTERS[key]
        inpainter_cache[key] = inpainter(*args, **kwargs)
    return inpainter_cache[key]

async def prepare(inpainter_key: Inpainter, device: str = 'cpu', force_torch: bool = False):
    inpainter = get_inpainter(inpainter_key)
    if isinstance(inpainter, OfflineInpainter):
        await inpainter.download()
        await inpainter.load(device, force_torch=force_torch)

async def dispatch(inpainter_key: Inpainter, image: np.ndarray, mask: np.ndarray, config: Optional[InpainterConfig], inpainting_size: int = 1024, device: str = 'cpu', verbose: bool = False) -> np.ndarray:
    inpainter = get_inpainter(inpainter_key)
    config = config or InpainterConfig()
    if isinstance(inpainter, OfflineInpainter):
        force_torch = getattr(config, 'force_use_torch_inpainting', False)
        await inpainter.load(device, force_torch=force_torch)
    
    # 检查是否需要切割（极端长宽比）
    h, w = image.shape[:2]
    aspect_ratio = max(w / h, h / w)
    split_ratio = config.inpainting_split_ratio
    
    # 如果长宽比超过阈值，进行切割处理
    if split_ratio > 0 and aspect_ratio > split_ratio:
        return await _dispatch_with_split(inpainter, image, mask, config, inpainting_size, verbose)
    else:
        # 正常处理
        return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)

async def unload(inpainter_key: Inpainter):
    inpainter_cache.pop(inpainter_key, None)

async def _dispatch_with_split(inpainter: CommonInpainter, image: np.ndarray, mask: np.ndarray, config: InpainterConfig, inpainting_size: int, verbose: bool) -> np.ndarray:
    """
    对极端长宽比的图片进行切割修复
    
    Args:
        inpainter: 修复器实例
        image: 原始图像
        mask: 蒙版
        config: 修复配置
        inpainting_size: 修复大小
        verbose: 是否输出详细日志
    
    Returns:
        修复后的完整图像
    """
#     import cv2
    
    h, w = image.shape[:2]
    is_vertical = h > w  # 判断是竖长图还是横长图
    
    if is_vertical:
        # 竖长图：沿高度方向切割
        long_side = h
        short_side = w
    else:
        # 横长图：沿宽度方向切割
        long_side = w
        short_side = h
    
    # 计算切割块数：确保每块的长宽比接近 1:1 或不超过阈值
    target_ratio = config.inpainting_split_ratio
    num_splits = int(np.ceil(long_side / (short_side * target_ratio)))
    
    # 计算每块的大小（带重叠）
    overlap = int(short_side * 0.1)  # 10% 重叠以避免接缝
    tile_size = (long_side + overlap * (num_splits - 1)) // num_splits
    
    if verbose:
        print(f"[Inpainting Split] Image size: {w}x{h}, Aspect ratio: {max(w/h, h/w):.2f}")
        print(f"[Inpainting Split] Splitting into {num_splits} tiles along {'height' if is_vertical else 'width'}")
        print(f"[Inpainting Split] Tile size: {tile_size}, Overlap: {overlap}")
    
    # 切割并修复每一块
    tiles = []
    for i in range(num_splits):
        if is_vertical:
            # 计算切割位置
            start = max(0, i * tile_size - i * overlap)
            end = min(h, start + tile_size)
            
            # 切割图像和蒙版
            tile_img = image[start:end, :, :].copy()
            tile_mask = mask[start:end, :].copy()
            
            if verbose:
                print(f"[Inpainting Split] Processing tile {i+1}/{num_splits}: rows {start}-{end}")
        else:
            # 计算切割位置
            start = max(0, i * tile_size - i * overlap)
            end = min(w, start + tile_size)
            
            # 切割图像和蒙版
            tile_img = image[:, start:end, :].copy()
            tile_mask = mask[:, start:end].copy()
            
            if verbose:
                print(f"[Inpainting Split] Processing tile {i+1}/{num_splits}: cols {start}-{end}")
        
        # 修复当前块
        tile_inpainted = await inpainter.inpaint(tile_img, tile_mask, config, inpainting_size, verbose)
        tiles.append({
            'image': tile_inpainted,
            'start': start,
            'end': end
        })
    
    # 拼接修复后的块（使用羽化混合避免接缝）
    result = image.copy()
    blend_size = overlap // 2 if overlap > 0 else 0
    
    for i, tile_data in enumerate(tiles):
        tile_img = tile_data['image']
        start = tile_data['start']
        end = tile_data['end']
        
        if num_splits == 1:
            # 只有一块，直接使用
            if is_vertical:
                result[start:end, :, :] = tile_img
            else:
                result[:, start:end, :] = tile_img
        elif i == 0:
            # 第一块：前面部分直接使用，后面部分羽化
            if is_vertical:
                result[start:end - blend_size, :, :] = tile_img[:-blend_size, :, :]
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = 1 - (j / blend_size)
                        idx = end - blend_size + j
                        tile_idx = tile_img.shape[0] - blend_size + j
                        result[idx, :, :] = alpha * tile_img[tile_idx, :, :] + (1 - alpha) * result[idx, :, :]
            else:
                result[:, start:end - blend_size, :] = tile_img[:, :-blend_size, :]
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = 1 - (j / blend_size)
                        idx = end - blend_size + j
                        tile_idx = tile_img.shape[1] - blend_size + j
                        result[:, idx, :] = alpha * tile_img[:, tile_idx, :] + (1 - alpha) * result[:, idx, :]
        elif i == len(tiles) - 1:
            # 最后一块：前面部分羽化，后面部分直接使用
            if is_vertical:
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = j / blend_size
                        result[start + j, :, :] = (1 - alpha) * result[start + j, :, :] + alpha * tile_img[j, :, :]
                result[start + blend_size:end, :, :] = tile_img[blend_size:, :, :]
            else:
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = j / blend_size
                        result[:, start + j, :] = (1 - alpha) * result[:, start + j, :] + alpha * tile_img[:, j, :]
                result[:, start + blend_size:end, :] = tile_img[:, blend_size:, :]
        else:
            # 中间块：前后都羽化，中间直接使用
            if is_vertical:
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = j / blend_size
                        result[start + j, :, :] = (1 - alpha) * result[start + j, :, :] + alpha * tile_img[j, :, :]
                
                result[start + blend_size:end - blend_size, :, :] = tile_img[blend_size:-blend_size, :, :]
                
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = 1 - (j / blend_size)
                        idx = end - blend_size + j
                        tile_idx = tile_img.shape[0] - blend_size + j
                        result[idx, :, :] = alpha * tile_img[tile_idx, :, :] + (1 - alpha) * result[idx, :, :]
            else:
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = j / blend_size
                        result[:, start + j, :] = (1 - alpha) * result[:, start + j, :] + alpha * tile_img[:, j, :]
                
                result[:, start + blend_size:end - blend_size, :] = tile_img[:, blend_size:-blend_size, :]
                
                if blend_size > 0:
                    for j in range(blend_size):
                        alpha = 1 - (j / blend_size)
                        idx = end - blend_size + j
                        tile_idx = tile_img.shape[1] - blend_size + j
                        result[:, idx, :] = alpha * tile_img[:, tile_idx, :] + (1 - alpha) * result[:, idx, :]
    
    if verbose:
        print("[Inpainting Split] Tiles merged successfully")
    
    return result