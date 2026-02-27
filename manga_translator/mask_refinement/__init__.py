from typing import Any, List, Sequence, Tuple
import cv2
import numpy as np

from .text_mask_utils import complete_mask_fill, complete_mask
from ..utils import (
    TextBlock,
    Quadrilateral,
    detect_bubbles_with_mangalens,
    build_bubble_mask_from_mangalens_result,
)
from ..utils.log import get_logger

logger = get_logger('mask_refinement')

# 气泡 mask 向内收缩像素，避免气泡边框被修复模型擦除
BUBBLE_MASK_ERODE_PX = 3


def _erode_bubble_mask(bubble_mask: np.ndarray) -> np.ndarray:
    """Erode the bubble mask inward by a fixed number of pixels."""
    if np.count_nonzero(bubble_mask) == 0:
        return bubble_mask
    h, w = bubble_mask.shape[:2]
    erode_px = max(int(BUBBLE_MASK_ERODE_PX), 0)
    if erode_px == 0:
        return bubble_mask
    kernel_size = 2 * erode_px + 1
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    logger.info(f"Bubble mask erosion: image={w}x{h}, erode_px={erode_px}")
    eroded = cv2.erode(bubble_mask, erode_kernel, iterations=1)
    if np.count_nonzero(eroded) == 0:
        logger.warning("Bubble mask fully eroded; falling back to original mask")
        return bubble_mask
    return eroded


def _build_model_bubble_mask(image_shape: Tuple[int, int], result: Any) -> Tuple[np.ndarray, str]:
    bubble_mask = build_bubble_mask_from_mangalens_result(result, image_shape)
    if np.count_nonzero(bubble_mask) == 0:
        return bubble_mask, 'none'

    raw_result = getattr(result, 'raw_result', None) if result is not None else None
    raw_masks = getattr(raw_result, 'masks', None) if raw_result is not None else None
    source = 'mask' if raw_masks is not None else 'box'
    return _erode_bubble_mask(bubble_mask), source


def _keep_bubble_components_intersecting_refined_mask(
    bubble_mask: np.ndarray,
    refined_mask: np.ndarray,
) -> Tuple[np.ndarray, int, int]:
    bubble_bin = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)
    refined_bin = np.where(refined_mask > 0, 255, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bubble_bin, connectivity=8)
    kept_mask = np.zeros_like(bubble_bin)

    total_components = max(num_labels - 1, 0)
    kept_components = 0

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area <= 0:
            continue

        label_view = labels[y:y + h, x:x + w]
        region = label_view == label_idx
        if np.any(refined_bin[y:y + h, x:x + w][region] > 0):
            dst = kept_mask[y:y + h, x:x + w]
            dst[region] = 255
            kept_mask[y:y + h, x:x + w] = dst
            kept_components += 1

    return kept_mask, total_components, kept_components


def _clip_refined_components_by_bubble_mask(
    refined_mask: np.ndarray,
    bubble_mask: np.ndarray,
) -> Tuple[np.ndarray, int, int, int]:
    """
    Clip refined-mask connected components by bubble mask:
    - If a refined component intersects bubble mask: keep only the intersection part.
    - If a refined component has no intersection: keep the whole component.
    """
    refined_bin = np.where(refined_mask > 0, 255, 0).astype(np.uint8)
    bubble_bin = np.where(bubble_mask > 0, 255, 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(refined_bin, connectivity=8)
    clipped_mask = np.zeros_like(refined_bin)

    total_components = max(num_labels - 1, 0)
    intersected_components = 0
    preserved_components = 0

    for label_idx in range(1, num_labels):
        x, y, w, h, area = stats[label_idx]
        if area <= 0:
            continue

        label_view = labels[y:y + h, x:x + w]
        region = label_view == label_idx
        bubble_region = bubble_bin[y:y + h, x:x + w] > 0
        intersection = region & bubble_region

        dst = clipped_mask[y:y + h, x:x + w]
        if np.any(intersection):
            dst[intersection] = 255
            intersected_components += 1
        else:
            dst[region] = 255
            preserved_components += 1
        clipped_mask[y:y + h, x:x + w] = dst

    return clipped_mask, total_components, intersected_components, preserved_components

async def dispatch(
    text_regions: List[TextBlock],
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    method: str = 'fit_text',
    dilation_offset: int = 0,
    verbose: bool = False,
    kernel_size: int = 3,
    use_model_bubble_repair_intersection: bool = False,
    limit_mask_dilation_to_bubble_mask: bool = False,
) -> np.ndarray:
    # Larger sized mask images will probably have crisper and thinner mask segments due to being able to fit the text pixels better
    # so we dont want to size them down as much to not lose information
    scale_factor = max(min((raw_mask.shape[0] - raw_image.shape[0] / 3) / raw_mask.shape[0], 1), 0.5)

    img_resized = cv2.resize(raw_image, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)
    mask_resized = cv2.resize(raw_mask, (int(raw_image.shape[1] * scale_factor), int(raw_image.shape[0] * scale_factor)), interpolation = cv2.INTER_LINEAR)

    mask_resized[mask_resized > 0] = 255
    textlines = []
    for region in text_regions:
        for l in region.lines:
            q = Quadrilateral(l * scale_factor, '', 0)
            textlines.append(q)

    final_mask = complete_mask(img_resized, mask_resized, textlines, dilation_offset=dilation_offset,kernel_size=kernel_size) if method == 'fit_text' else complete_mask_fill([txtln.aabb.xywh for txtln in textlines])
    if final_mask is None:
        final_mask = np.zeros((raw_image.shape[0], raw_image.shape[1]), dtype = np.uint8)
    else:
        final_mask = cv2.resize(final_mask, (raw_image.shape[1], raw_image.shape[0]), interpolation = cv2.INTER_LINEAR)
        final_mask[final_mask > 0] = 255

    if use_model_bubble_repair_intersection or limit_mask_dilation_to_bubble_mask:
        try:
            result = detect_bubbles_with_mangalens(raw_image, return_annotated=False, verbose=False)
            detections = result.detections if result is not None else []
            bubble_mask, bubble_source = _build_model_bubble_mask(final_mask.shape[:2], result)

            if np.count_nonzero(bubble_mask) == 0:
                logger.info(
                    "Model bubble mask post-process enabled, but no bubble detections found; keep refined mask unchanged"
                )
            elif use_model_bubble_repair_intersection:
                filtered_mask, total_components, kept_components = _keep_bubble_components_intersecting_refined_mask(
                    bubble_mask=bubble_mask,
                    refined_mask=final_mask,
                )
                merged_mask = cv2.bitwise_or(final_mask, filtered_mask)
                added_pixels = int(np.count_nonzero((filtered_mask > 0) & (final_mask == 0)))
                logger.info(
                    f"Bubble repair intersection: detections={len(detections)}, source={bubble_source}, "
                    f"bubble_components={total_components}, kept_components={kept_components}, "
                    f"refined_pixels={int(np.count_nonzero(final_mask))}, "
                    f"bubble_pixels={int(np.count_nonzero(filtered_mask))}, "
                    f"added_pixels={added_pixels}, output_pixels={int(np.count_nonzero(merged_mask))}"
                )
                final_mask = merged_mask

            if np.count_nonzero(bubble_mask) > 0 and limit_mask_dilation_to_bubble_mask:
                clipped_mask, total_components, intersected_components, preserved_components = _clip_refined_components_by_bubble_mask(
                    refined_mask=final_mask,
                    bubble_mask=bubble_mask,
                )
                removed_pixels = int(np.count_nonzero((final_mask > 0) & (clipped_mask == 0)))
                logger.info(
                    f"Bubble constrained dilation: detections={len(detections)}, source={bubble_source}, "
                    f"refined_components={total_components}, intersected_components={intersected_components}, "
                    f"preserved_components={preserved_components}, removed_pixels={removed_pixels}, "
                    f"output_pixels={int(np.count_nonzero(clipped_mask))}"
                )
                final_mask = clipped_mask
        except Exception as exc:
            logger.warning(f"Model bubble mask post-process failed, keep refined mask unchanged: {exc}")

    return final_mask
