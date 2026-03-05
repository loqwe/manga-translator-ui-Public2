from typing import Optional

import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
from .none import NoneInpainter
from .original import OriginalInpainter
from ..config import Inpainter, InpainterConfig

_SD_IMPORT_ERROR = None
try:
    from .inpainting_sd import StableDiffusionInpainter
except Exception as e:
    _SD_IMPORT_ERROR = e

    class StableDiffusionInpainter(OfflineInpainter):
        async def _load(self, device: str):
            raise RuntimeError(
                "Stable Diffusion inpainter is unavailable because optional dependencies are missing. "
                f"Original import error: {e!r}"
            )

        async def _infer(self, image: np.ndarray, mask: np.ndarray, inpainting_size: int = 1024, verbose: bool = False) -> np.ndarray:
            raise RuntimeError(
                "Stable Diffusion inpainter is unavailable because optional dependencies are missing. "
                f"Original import error: {e!r}"
            )

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

    # Priority 1: tile split based on inpainting_split_ratio
    h, w = image.shape[:2]
    aspect_ratio = max(w / h, h / w)
    split_ratio = config.inpainting_split_ratio

    if split_ratio > 0 and aspect_ratio > split_ratio:
        return await _dispatch_with_split(inpainter, image, mask, config, inpainting_size, verbose)

    return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)

async def unload(inpainter_key: Inpainter):
    inpainter_cache.pop(inpainter_key, None)

async def _dispatch_with_split(
    inpainter: CommonInpainter,
    image: np.ndarray,
    mask: np.ndarray,
    config: InpainterConfig,
    inpainting_size: int,
    verbose: bool,
) -> np.ndarray:
    """
    Split extreme-aspect-ratio images into tiles along the long edge,
    inpaint each tile independently, then reassemble with linear alpha
    feathering in the overlap zones to avoid visible seams.
    """
    h, w = image.shape[:2]
    is_vertical = h > w

    if is_vertical:
        long_side, short_side = h, w
    else:
        long_side, short_side = w, h

    # Number of tiles: keep each tile's aspect ratio <= target_ratio
    target_ratio = config.inpainting_split_ratio
    num_splits = int(np.ceil(long_side / (short_side * target_ratio)))

    # 10% overlap relative to the short side
    overlap = int(short_side * 0.1)
    tile_size = (long_side + overlap * (num_splits - 1)) // num_splits

    if verbose:
        print(f"[Inpainting Split] Image size: {w}x{h}, Aspect ratio: {max(w/h, h/w):.2f}")
        print(f"[Inpainting Split] Splitting into {num_splits} tiles along {'height' if is_vertical else 'width'}")
        print(f"[Inpainting Split] Tile size: {tile_size}, Overlap: {overlap}")

    # --- Cut & inpaint each tile ---
    tiles = []
    for i in range(num_splits):
        start = max(0, i * tile_size - i * overlap)
        if is_vertical:
            end = min(h, start + tile_size)
            tile_img = image[start:end, :, :].copy()
            tile_mask = mask[start:end, :].copy()
        else:
            end = min(w, start + tile_size)
            tile_img = image[:, start:end, :].copy()
            tile_mask = mask[:, start:end].copy()

        if verbose:
            axis = 'rows' if is_vertical else 'cols'
            print(f"[Inpainting Split] Processing tile {i+1}/{num_splits}: {axis} {start}-{end}")

        tile_inpainted = await inpainter.inpaint(tile_img, tile_mask, config, inpainting_size, verbose)
        tiles.append({'image': tile_inpainted, 'start': start, 'end': end})

    # --- Reassemble with linear alpha feathering ---
    result = image.astype(np.float32)
    blend_size = overlap // 2 if overlap > 0 else 0

    for i, td in enumerate(tiles):
        img = td['image'].astype(np.float32)
        s, e = td['start'], td['end']

        if num_splits == 1:
            if is_vertical:
                result[s:e, :, :] = img
            else:
                result[:, s:e, :] = img
            continue

        if i == 0:
            # First tile: write body, feather trailing edge
            if is_vertical:
                result[s:e - blend_size, :, :] = img[:-blend_size or None, :, :]
                for j in range(blend_size):
                    alpha = 1.0 - j / blend_size
                    ri = e - blend_size + j
                    ti = img.shape[0] - blend_size + j
                    result[ri, :, :] = alpha * img[ti, :, :] + (1 - alpha) * result[ri, :, :]
            else:
                result[:, s:e - blend_size, :] = img[:, :-blend_size or None, :]
                for j in range(blend_size):
                    alpha = 1.0 - j / blend_size
                    ri = e - blend_size + j
                    ti = img.shape[1] - blend_size + j
                    result[:, ri, :] = alpha * img[:, ti, :] + (1 - alpha) * result[:, ri, :]

        elif i == len(tiles) - 1:
            # Last tile: feather leading edge, write body
            if is_vertical:
                for j in range(blend_size):
                    alpha = j / blend_size
                    result[s + j, :, :] = (1 - alpha) * result[s + j, :, :] + alpha * img[j, :, :]
                result[s + blend_size:e, :, :] = img[blend_size:, :, :]
            else:
                for j in range(blend_size):
                    alpha = j / blend_size
                    result[:, s + j, :] = (1 - alpha) * result[:, s + j, :] + alpha * img[:, j, :]
                result[:, s + blend_size:e, :] = img[:, blend_size:, :]

        else:
            # Middle tiles: feather both edges
            if is_vertical:
                for j in range(blend_size):
                    alpha = j / blend_size
                    result[s + j, :, :] = (1 - alpha) * result[s + j, :, :] + alpha * img[j, :, :]
                result[s + blend_size:e - blend_size, :, :] = img[blend_size:-blend_size or None, :, :]
                for j in range(blend_size):
                    alpha = 1.0 - j / blend_size
                    ri = e - blend_size + j
                    ti = img.shape[0] - blend_size + j
                    result[ri, :, :] = alpha * img[ti, :, :] + (1 - alpha) * result[ri, :, :]
            else:
                for j in range(blend_size):
                    alpha = j / blend_size
                    result[:, s + j, :] = (1 - alpha) * result[:, s + j, :] + alpha * img[:, j, :]
                result[:, s + blend_size:e - blend_size, :] = img[:, blend_size:-blend_size or None, :]
                for j in range(blend_size):
                    alpha = 1.0 - j / blend_size
                    ri = e - blend_size + j
                    ti = img.shape[1] - blend_size + j
                    result[:, ri, :] = alpha * img[:, ti, :] + (1 - alpha) * result[:, ri, :]

    result = np.clip(np.rint(result), 0, 255).astype(np.uint8)
    if verbose:
        print("[Inpainting Split] Tiles merged successfully")
    return result
