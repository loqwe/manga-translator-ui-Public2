import os
import numpy as np
from abc import abstractmethod
from typing import List, Tuple, Union
from collections import Counter
import networkx as nx
import itertools

from ..config import OcrConfig
from ..utils import InfererModule, TextBlock, ModelWrapper, Quadrilateral, detect_bubbles_with_mangalens, imwrite_unicode

class CommonOCR(InfererModule):
    @staticmethod
    def _calc_bbox_overlap_ratio(
        text_bbox: Tuple[int, int, int, int],
        bubble_bbox: Tuple[float, float, float, float],
    ) -> float:
        """
        Calculates intersection ratio using text box area as denominator.
        """
        tx, ty, tw, th = text_bbox
        bx1, by1, bx2, by2 = bubble_bbox

        tx1, ty1 = tx, ty
        tx2, ty2 = tx + tw, ty + th

        inter_x1 = max(tx1, bx1)
        inter_y1 = max(ty1, by1)
        inter_x2 = min(tx2, bx2)
        inter_y2 = min(ty2, by2)

        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        text_area = max(float(tw * th), 1.0)
        return inter_area / text_area

    def _get_model_bubble_boxes(self, full_image: np.ndarray) -> List[Tuple[float, float, float, float]]:
        """
        Runs the bubble detector once per image and caches the bounding boxes.
        When MANGA_OCR_RESULT_DIR is set (verbose mode), saves an annotated image.
        """
        cache_key = (id(full_image), full_image.shape[0], full_image.shape[1])
        if getattr(self, '_model_bubble_cache_key', None) == cache_key:
            return getattr(self, '_model_bubble_cache_boxes', [])

        try:
            ocr_result_dir = os.environ.get('MANGA_OCR_RESULT_DIR', '')
            want_annotated = bool(ocr_result_dir)
            result = detect_bubbles_with_mangalens(full_image, return_annotated=want_annotated, verbose=False)
            boxes = [det.xyxy for det in result.detections]
            self.logger.info(f"Model bubble filter: detected {len(boxes)} bubbles")

            # Save annotated debug image when verbose
            if want_annotated and result.annotated_image is not None:
                try:
                    import cv2
                    save_path = os.path.join(ocr_result_dir, 'model_bubble_detection.png')
                    annotated_bgr = cv2.cvtColor(result.annotated_image, cv2.COLOR_RGB2BGR)
                    imwrite_unicode(save_path, annotated_bgr, self.logger)
                    self.logger.info(f"Saved bubble detection debug image: {save_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to save bubble detection debug image: {e}")
        except Exception as e:
            self.logger.warning(f"Model bubble filter failed, fallback to heuristic filter: {e}")
            boxes = []

        self._model_bubble_cache_key = cache_key
        self._model_bubble_cache_boxes = boxes
        return boxes

    def _generate_text_direction(self, bboxes: List[Union[Quadrilateral, TextBlock]]):
        if len(bboxes) > 0:
            if isinstance(bboxes[0], TextBlock):
                for blk in bboxes:
                    for line_idx in range(len(blk.lines)):
                        yield blk, line_idx
            else:
                from ..utils import quadrilateral_can_merge_region

                G = nx.Graph()
                for i, box in enumerate(bboxes):
                    G.add_node(i, box = box)
                for ((u, ubox), (v, vbox)) in itertools.combinations(enumerate(bboxes), 2):
                    if quadrilateral_can_merge_region(ubox, vbox, aspect_ratio_tol=1):
                        G.add_edge(u, v)
                for node_set in nx.algorithms.components.connected_components(G):
                    nodes = list(node_set)
                    # majority vote for direction
                    dirs = [box.direction for box in [bboxes[i] for i in nodes]]
                    majority_dir = Counter(dirs).most_common(1)[0][0]
                    # sort
                    if majority_dir == 'h':
                        nodes = sorted(nodes, key = lambda x: bboxes[x].aabb.y + bboxes[x].aabb.h // 2)
                    elif majority_dir == 'v':
                        nodes = sorted(nodes, key = lambda x: -(bboxes[x].aabb.x + bboxes[x].aabb.w))
                    # yield overall bbox and sorted indices
                    for node in nodes:
                        yield bboxes[node], majority_dir

    def _should_ignore_region(self, region_img: np.ndarray, ignore_bubble: float,
                              full_image: np.ndarray = None, textline: Quadrilateral = None,
                              ocr_config: OcrConfig = None) -> bool:
        """
        通用的气泡过滤方法，判断文本区域是否应该被忽略

        Args:
            region_img: 裁剪后的文本区域图像（用于简单方法）
            ignore_bubble: 忽略气泡阈值 (0-1)
            full_image: 完整图像（可选，用于高级方法）
            textline: 文本行对象（可选，用于获取坐标）
            ocr_config: OCR配置（可选，用于模型气泡过滤）

        Returns:
            True: 应该忽略（非气泡区域）
            False: 应该保留（气泡区域）
        """
        from ..utils.bubble import is_ignore

        # 模型气泡过滤（与 ignore_bubble 同阶段，但基于检测模型）
        use_model_filter = bool(getattr(ocr_config, 'use_model_bubble_filter', False)) if ocr_config is not None else False
        if use_model_filter and full_image is not None and textline is not None:
            bbox = textline.aabb
            text_bbox = (int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h))

            bubble_boxes = self._get_model_bubble_boxes(full_image)
            if not bubble_boxes:
                if not getattr(self, '_model_bubble_no_boxes_logged', False):
                    self.logger.info("Model bubble filter: no bubble boxes detected, skip model gating for this image")
                    self._model_bubble_no_boxes_logged = True
            else:
                overlap_threshold = float(getattr(ocr_config, 'model_bubble_overlap_threshold', 0.1))
                overlap_threshold = max(0.0, min(1.0, overlap_threshold))

                max_overlap_ratio = 0.0
                for bubble_bbox in bubble_boxes:
                    ratio = self._calc_bbox_overlap_ratio(text_bbox, bubble_bbox)
                    if ratio > max_overlap_ratio:
                        max_overlap_ratio = ratio

                if max_overlap_ratio < overlap_threshold:
                    self.logger.info(
                        f"Model bubble filter: overlap={max_overlap_ratio:.3f} < threshold={overlap_threshold:.3f}, filtering region"
                    )
                    return True

        # 如果提供了完整图像和文本行，使用高级方法
        if full_image is not None and textline is not None:
            # 获取文本行的边界框
            bbox = textline.aabb
            x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.w), int(bbox.h)
            return is_ignore(region_img, ignore_bubble, full_image, [x, y, w, h])

        # 否则使用简单方法
        return is_ignore(region_img, ignore_bubble)

    async def recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        '''
        Performs the optical character recognition, using the `textlines` as areas of interests.
        Returns a `textlines` list with the `textline.text` property set to the detected text string.
        '''
        # Reset per-image model bubble cache
        self._model_bubble_cache_key = None
        self._model_bubble_cache_boxes = []
        self._model_bubble_no_boxes_logged = False
        if bool(getattr(config, 'use_model_bubble_filter', False)):
            threshold = float(getattr(config, 'model_bubble_overlap_threshold', 0.1))
            self.logger.info(f"Model bubble filter enabled (overlap_threshold={threshold:.3f})")
        return await self._recognize(image, textlines, config, verbose)

    @abstractmethod
    async def _recognize(self, image: np.ndarray, textlines: List[Quadrilateral], config: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        pass


class OfflineOCR(CommonOCR, ModelWrapper):
    _MODEL_SUB_DIR = 'ocr'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gpu = False  # 子类应该在 _load 中设置这个标志

    async def _recognize(self, *args, **kwargs):
        result = await self.infer(*args, **kwargs)
        return result

    @abstractmethod
    async def _infer(self, image: np.ndarray, textlines: List[Quadrilateral], args: OcrConfig, verbose: bool = False) -> List[Quadrilateral]:
        pass

    def _cleanup_ocr_memory(self, *objects, force_gpu_cleanup: bool = False):
        """
        OCR 模块统一的内存清理方法
        
        Args:
            *objects: 要删除的对象（变量名或对象引用）
            force_gpu_cleanup: 是否强制清理 GPU 显存
            
        Example:
            # 清理单个对象
            self._cleanup_ocr_memory(region)
            
            # 清理多个对象
            self._cleanup_ocr_memory(region, image_tensor, ret)
            
            # 清理并强制 GPU 清理
            self._cleanup_ocr_memory(region, image_tensor, force_gpu_cleanup=True)
        """
#         import gc
        
        # 删除传入的对象
        for obj in objects:
            try:
                del obj
            except:
                pass
        
        # 如果使用 GPU 或强制清理，清理 GPU 显存
        if force_gpu_cleanup or (hasattr(self, 'use_gpu') and self.use_gpu):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
        
        # 轻量级垃圾回收（不强制完整 GC，避免性能影响）
        # 主进程会在批次结束时进行完整的 gc.collect()
        
    def _cleanup_batch_data(self, *data_lists, force_gpu_cleanup: bool = False):
        """
        清理批量数据（列表、字典等容器）
        
        Args:
            *data_lists: 要清理的数据容器（list, dict 等）
            force_gpu_cleanup: 是否强制清理 GPU 显存
            
        Example:
            # 清理列表
            self._cleanup_batch_data(region_imgs, quadrilaterals)
            
            # 清理字典
            self._cleanup_batch_data(out_regions, texts)
        """
#         import gc
        
        for data in data_lists:
            if data is None:
                continue
                
            try:
                if isinstance(data, list):
                    data.clear()
                elif isinstance(data, dict):
                    data.clear()
                del data
            except:
                pass
        
        # GPU 清理
        if force_gpu_cleanup or (hasattr(self, 'use_gpu') and self.use_gpu):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass
