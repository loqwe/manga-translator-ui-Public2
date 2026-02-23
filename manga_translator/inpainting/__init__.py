from typing import Optional

import numpy as np

from .common import CommonInpainter, OfflineInpainter
from .inpainting_aot import AotInpainter
from .inpainting_lama_mpe import LamaMPEInpainter, LamaLargeInpainter
from .inpainting_sd import StableDiffusionInpainter
from .none import NoneInpainter
from .original import OriginalInpainter
from ..config import Inpainter, InpainterConfig
from ..utils import (
    build_det_rearrange_plan,
    det_collect_plan_patches,
    det_rearrange_patch_array,
    det_unrearrange_patch_maps,
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

    rearrange_plan = build_det_rearrange_plan(image, tgt_size=inpainting_size)
    if rearrange_plan is None:
        return await inpainter.inpaint(image, mask, config, inpainting_size, verbose)
    return await _dispatch_with_det_rearrange(
        inpainter,
        image,
        mask,
        config,
        inpainting_size,
        verbose,
        rearrange_plan,
    )

async def unload(inpainter_key: Inpainter):
    inpainter_cache.pop(inpainter_key, None)

async def _dispatch_with_det_rearrange(
    inpainter: CommonInpainter,
    image: np.ndarray,
    mask: np.ndarray,
    config: InpainterConfig,
    inpainting_size: int,
    verbose: bool,
    rearrange_plan: dict,
) -> np.ndarray:
    """
    使用与检测器统一的切割/回拼逻辑进行修复。
    """
    if verbose:
        h, w = image.shape[:2]
        print(
            f"[Inpainting Rearrange] image={w}x{h}, "
            f"patch_size={rearrange_plan['patch_size']}, "
            f"ph_num={rearrange_plan['ph_num']}, pw_num={rearrange_plan['pw_num']}, "
            f"pad_num={rearrange_plan['pad_num']}, transpose={rearrange_plan['transpose']}"
        )

    image_patch_array = det_rearrange_patch_array(rearrange_plan)
    mask_plan = dict(rearrange_plan)
    mask_plan['patch_list'] = det_collect_plan_patches(mask, rearrange_plan)
    mask_patch_array = det_rearrange_patch_array(mask_plan)

    inpainted_patch_list = []
    for ii, (image_patch, mask_patch) in enumerate(zip(image_patch_array, mask_patch_array)):
        if image_patch.size == 0:
            inpainted_patch_list.append(image_patch.astype(np.float32))
            continue
        if np.max(mask_patch) == 0:
            inpainted_patch_list.append(image_patch.astype(np.float32))
            continue
        if verbose:
            print(f"[Inpainting Rearrange] processing patch {ii + 1}/{len(image_patch_array)}")
        inpainted_patch = await inpainter.inpaint(
            image_patch,
            mask_patch.astype(np.uint8),
            config,
            inpainting_size,
            verbose,
        )
        inpainted_patch_list.append(inpainted_patch.astype(np.float32))

    merged = det_unrearrange_patch_maps(inpainted_patch_list, rearrange_plan, data_format='hwc')
    merged = np.clip(np.rint(merged), 0, 255).astype(np.uint8)
    if verbose:
        print("[Inpainting Rearrange] patches merged successfully")
    return merged
