from typing import Any, List, Sequence, Tuple
import cv2
import numpy as np

from .text_mask_utils import complete_mask_fill, complete_mask
from ..utils import TextBlock, Quadrilateral, detect_bubbles_with_mangalens
from ..utils.log import get_logger

logger = get_logger('mask_refinement')


def _build_model_bubble_mask(image_shape: Tuple[int, int], result: Any) -> Tuple[np.ndarray, str]:
    h, w = image_shape
    bubble_mask = np.zeros((h, w), dtype=np.uint8)

    raw_result = getattr(result, 'raw_result', None) if result is not None else None
    raw_masks = getattr(raw_result, 'masks', None) if raw_result is not None else None

    # Prefer segmentation masks from the model output.
    if raw_masks is not None:
        polygons = getattr(raw_masks, 'xy', None)
        if polygons is not None:
            for polygon in polygons:
                pts = np.asarray(polygon, dtype=np.int32)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
                pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
                cv2.fillPoly(bubble_mask, [pts], 255)
            if np.count_nonzero(bubble_mask) > 0:
                return bubble_mask, 'mask'

        mask_data = getattr(raw_masks, 'data', None)
        if mask_data is not None:
            try:
                if hasattr(mask_data, 'detach'):
                    mask_data = mask_data.detach().cpu().numpy()
                mask_data = np.asarray(mask_data)
                if mask_data.ndim == 3 and mask_data.shape[0] > 0:
                    merged_mask = (mask_data > 0.5).any(axis=0).astype(np.uint8) * 255
                    if merged_mask.shape != bubble_mask.shape:
                        merged_mask = cv2.resize(
                            merged_mask,
                            (w, h),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    bubble_mask = np.maximum(bubble_mask, merged_mask)
                    if np.count_nonzero(bubble_mask) > 0:
                        return bubble_mask, 'mask'
            except Exception:
                pass

    # Fallback for non-segmentation results.
    detections: Sequence = getattr(result, 'detections', []) if result is not None else []
    for det in detections:
        try:
            x1, y1, x2, y2 = det.xyxy
        except Exception:
            continue

        ix1 = max(0, min(w - 1, int(round(x1))))
        iy1 = max(0, min(h - 1, int(round(y1))))
        ix2 = max(0, min(w, int(round(x2))))
        iy2 = max(0, min(h, int(round(y2))))

        if ix2 <= ix1 or iy2 <= iy1:
            continue
        cv2.rectangle(bubble_mask, (ix1, iy1), (ix2, iy2), 255, -1)

    if np.count_nonzero(bubble_mask) > 0:
        return bubble_mask, 'box'
    return bubble_mask, 'none'


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

async def dispatch(
    text_regions: List[TextBlock],
    raw_image: np.ndarray,
    raw_mask: np.ndarray,
    method: str = 'fit_text',
    dilation_offset: int = 0,
    verbose: bool = False,
    kernel_size: int = 3,
    use_model_bubble_repair_intersection: bool = False,
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

    if use_model_bubble_repair_intersection:
        try:
            result = detect_bubbles_with_mangalens(raw_image, return_annotated=False, verbose=False)
            detections = result.detections if result is not None else []
            bubble_mask, bubble_source = _build_model_bubble_mask(final_mask.shape[:2], result)

            if np.count_nonzero(bubble_mask) == 0:
                logger.info("Bubble repair intersection enabled, but no bubble detections found; keep refined mask unchanged")
            else:
                filtered_mask, total_components, kept_components = _keep_bubble_components_intersecting_refined_mask(
                    bubble_mask=bubble_mask,
                    refined_mask=final_mask,
                )
                # Only use bubble boundary to constrain text mask dilation,
                # instead of merging the entire bubble interior.
                dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size * 2 + 1, kernel_size * 2 + 1))
                dilated_text_mask = cv2.dilate(final_mask, dilate_kernel, iterations=2)
                bubble_constrained = cv2.bitwise_and(dilated_text_mask, filtered_mask)
                merged_mask = cv2.bitwise_or(final_mask, bubble_constrained)
                added_pixels = int(np.count_nonzero((merged_mask > 0) & (final_mask == 0)))
                logger.info(
                    f"Bubble repair intersection: detections={len(detections)}, source={bubble_source}, "
                    f"bubble_components={total_components}, kept_components={kept_components}, "
                    f"refined_pixels={int(np.count_nonzero(final_mask))}, "
                    f"bubble_pixels={int(np.count_nonzero(filtered_mask))}, "
                    f"added_pixels={added_pixels}, output_pixels={int(np.count_nonzero(merged_mask))}"
                )
                final_mask = merged_mask
        except Exception as exc:
            logger.warning(f"Bubble repair intersection failed, keep refined mask unchanged: {exc}")

    return final_mask
