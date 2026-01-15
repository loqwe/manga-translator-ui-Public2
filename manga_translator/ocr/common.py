import numpy as np
from abc import abstractmethod
from typing import List, Union
from collections import Counter
import networkx as nx
import itertools

from ..config import OcrConfig
from ..utils import InfererModule, TextBlock, ModelWrapper, Quadrilateral

class CommonOCR(InfererModule):
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
                              full_image: np.ndarray = None, textline: Quadrilateral = None) -> bool:
        """
        通用的气泡过滤方法，判断文本区域是否应该被忽略
        
        Args:
            region_img: 裁剪后的文本区域图像（用于简单方法）
            ignore_bubble: 忽略气泡阈值 (0-1)
            full_image: 完整图像（可选，用于高级方法）
            textline: 文本行对象（可选，用于获取坐标）
            
        Returns:
            True: 应该忽略（非气泡区域）
            False: 应该保留（气泡区域）
        """
        from ..utils.bubble import is_ignore
        
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
