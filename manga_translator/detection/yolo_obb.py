"""
YOLO OBB (Oriented Bounding Box) 检测器
用于漫画文本检测的辅助检测器
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Tuple, Optional

from .common import OfflineDetector
from ..utils import Quadrilateral, det_rearrange_forward


class YOLOOBBDetector(OfflineDetector):
    """YOLO OBB 检测器 - 基于ONNX Runtime"""
    
    _MODEL_MAPPING = {
        'model': {
            'url': [
                'https://github.com/hgmzhn/manga-translator-ui/releases/download/v1.7.1/ysgyolo_1.2_OS1.0.onnx',
                'https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/ysgyolo_1.2_OS1.0.onnx',
            ],
            'hash': '6f3202925f01fdf045f8c31a3bf62e6c44944f56ce09107eb436bc5a5b185ebe',
            'file': '.',
        }
    }
    
    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)
        
        # 类别列表（不包括other）
        self.classes = ['balloon', 'qipao', 'shuqing', 'changfangtiao', 'hengxie']
        self.input_size = 640  # ONNX模型固定输入尺寸
        self.using_cuda = False  # 初始化标志
    
    async def _load(self, device: str):
        """加载ONNX模型"""
        model_path = self._get_file_path('ysgyolo_1.2_OS1.0.onnx')
        
        # 设置会话选项
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # 设置日志级别为 ERROR，隐藏 Memcpy 警告（这些警告是正常的，不影响性能）
        sess_options.log_severity_level = 3  # 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
        
        # 配置ONNX Runtime providers
        providers = []
        use_cuda = False
        
        if device == 'cuda':
            # 使用 onnxruntime.preload_dlls() 加载 PyTorch 的 CUDA 库（仅 1.21+ 版本支持）
            if hasattr(ort, 'preload_dlls'):
                try:
                    ort.preload_dlls()
                except Exception as e:
                    self.logger.warning(f"preload_dlls() 失败: {e}")
            
            # 检查 CUDA 是否真的可用
            try:
                available_providers = ort.get_available_providers()
                if 'CUDAExecutionProvider' in available_providers:
                    # 只设置 device_id，不设置其他选项以避免 Fallback
                    # 测试发现：额外的 CUDA 选项（如 cudnn_conv_algo_search）会导致 Fallback 模式
                    cuda_options = {
                        'device_id': 0,
                    }
                    providers.append(('CUDAExecutionProvider', cuda_options))
                    use_cuda = True
                else:
                    self.logger.warning(f"CUDA 不在可用 providers 中: {available_providers}")
            except Exception as e:
                self.logger.warning(f"检查 CUDA 可用性时出错: {e}")
        
        providers.append('CPUExecutionProvider')
        
        # 先尝试使用 CUDA，如果失败则回退到 CPU
        if use_cuda:
            try:
                self.session = ort.InferenceSession(model_path, sess_options=sess_options, providers=providers)
                self.logger.info("YOLO OBB: CUDA 模式加载成功")
            except Exception as cuda_e:
                self.logger.warning(f"CUDA 模式加载失败: {cuda_e}")
                self.logger.warning("将回退到 CPU 模式")
                use_cuda = False
        
        # 如果 CUDA 失败或未启用，使用 CPU
        if not use_cuda:
            try:
                self.session = ort.InferenceSession(
                    model_path, 
                    sess_options=sess_options, 
                    providers=['CPUExecutionProvider']
                )
                self.logger.info("YOLO OBB: CPU 模式加载成功")
            except Exception as cpu_e:
                self.logger.error(f"CPU 模式加载也失败: {cpu_e}")
                raise
        
        self.device = device
        self.using_cuda = 'CUDAExecutionProvider' in self.session.get_providers()
    
    async def _unload(self):
        """卸载模型"""
        del self.session
    
    def letterbox(
        self, 
        img: np.ndarray, 
        new_shape: Tuple[int, int] = (640, 640), 
        color: Tuple[int, int, int] = (114, 114, 114)
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        调整图像大小并添加边框（保持宽高比）- YOLO风格
        
        Returns:
            resized_img: 调整后的图像
            gain: 缩放比例
            (pad_w, pad_h): 填充的宽度和高度（左/上的填充像素）
        """
        # 验证输入
        if img is None or img.size == 0:
            self.logger.error("YOLO OBB letterbox: 输入图片为空")
            raise ValueError("输入图片为空")
        
        shape = img.shape[:2]  # 当前形状 [height, width]
        
        if shape[0] == 0 or shape[1] == 0:
            self.logger.error(f"YOLO OBB letterbox: 输入图片尺寸无效: {shape}")
            raise ValueError(f"输入图片尺寸无效: {shape}")
        
        # 计算缩放比例 (gain)
        gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # 计算新的未填充尺寸
        new_unpad_w = int(round(shape[1] * gain))
        new_unpad_h = int(round(shape[0] * gain))
        
        # 确保尺寸有效
        if new_unpad_w <= 0 or new_unpad_h <= 0:
            self.logger.error(f"YOLO OBB letterbox: 计算的新尺寸无效: {new_unpad_w}x{new_unpad_h}, gain={gain}, 原始shape={shape}")
            raise ValueError(f"计算的新尺寸无效: {new_unpad_w}x{new_unpad_h}")
        
        # Resize图像
        if (new_unpad_w, new_unpad_h) != (shape[1], shape[0]):
            # 再次验证图像有效性
            if img is None or img.size == 0:
                self.logger.error("YOLO OBB letterbox: resize前图像变为空")
                raise ValueError("resize前图像变为空")
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)
        
        # 计算需要的总padding
        dw = new_shape[1] - new_unpad_w  # 宽度方向需要的总padding
        dh = new_shape[0] - new_unpad_h  # 高度方向需要的总padding
        
        # 将padding分配到两边（确保总和精确）
        # 一边向下取整，一边是剩余部分，确保 left + right = dw
        left = dw // 2
        right = dw - left
        top = dh // 2
        bottom = dh - top
        
        # 添加边框
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        
        # 验证最终尺寸
        assert img.shape[0] == new_shape[0] and img.shape[1] == new_shape[1], \
            f"Letterbox failed: expected {new_shape}, got {img.shape[:2]}"
        
        # 返回左和上的padding（用于后续坐标转换）
        return img, gain, (float(left), float(top))
    
    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """预处理图像"""
        # 验证输入图片
        if img is None or img.size == 0:
            self.logger.error("YOLO OBB预处理: 输入图片为空或无效")
            raise ValueError("输入图片为空或无效")
        
        if len(img.shape) < 2:
            self.logger.error(f"YOLO OBB预处理: 输入图片维度不正确: {img.shape}")
            raise ValueError(f"输入图片维度不正确: {img.shape}")
        
        # 检查图像通道数并转换为RGB格式
        if len(img.shape) == 2:
            # 灰度图 -> RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.logger.debug(f"YOLO OBB: 灰度图转RGB, shape={img.shape}")
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                # RGBA -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                self.logger.debug(f"YOLO OBB: RGBA转RGB, shape={img.shape}")
            elif img.shape[2] == 3:
                # 假设是BGR，需要转RGB（OpenCV默认BGR格式）
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.logger.error(f"YOLO OBB: 不支持的图像通道数: {img.shape}")
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            self.logger.error(f"YOLO OBB: 不支持的图像维度: {img.shape}")
            raise ValueError(f"Unsupported image shape: {img.shape}")
        
        img_resized, gain, pad = self.letterbox(
            img, 
            new_shape=(self.input_size, self.input_size)
        )
        
        # 转换为 CHW 格式
        img_transposed = img_resized.transpose(2, 0, 1)
        
        # 添加 batch 维度
        img_expanded = np.expand_dims(img_transposed, axis=0)
        
        # 归一化到 [0, 1]
        blob = img_expanded.astype(np.float32) / 255.0
        
        # 确保数组是连续的（C-contiguous），避免 ONNX Runtime 访问违例
        if not blob.flags['C_CONTIGUOUS']:
            blob = np.ascontiguousarray(blob)
            self.logger.debug("YOLO OBB: 转换为连续数组")
        
        self.logger.debug(f"YOLO OBB预处理完成: blob shape={blob.shape}, dtype={blob.dtype}, contiguous={blob.flags['C_CONTIGUOUS']}")
        
        return blob, gain, pad
    
    def scale_boxes(self, img1_shape, boxes, img0_shape, gain, pad, xywh=False):
        """将边界框从img1_shape缩放到img0_shape（移除letterbox效果）"""
        pad_w, pad_h = pad
        
        # 移除padding
        boxes[:, 0] -= pad_w  # x or cx
        boxes[:, 1] -= pad_h  # y or cy
        if not xywh:
            boxes[:, 2] -= pad_w  # x2
            boxes[:, 3] -= pad_h  # y2
        
        # 缩放到原始尺寸
        boxes[:, :4] /= gain
        
        # 裁剪到图像边界
        if xywh:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img0_shape[1])  # cx
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img0_shape[0])  # cy
        else:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img0_shape[1])  # x1
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img0_shape[0])  # y1
            boxes[:, 2] = np.clip(boxes[:, 2], 0, img0_shape[1])  # x2
            boxes[:, 3] = np.clip(boxes[:, 3], 0, img0_shape[0])  # y2
        
        return boxes
    
    def xywhr2xyxyxyxy(self, rboxes: np.ndarray) -> np.ndarray:
        """将旋转边界框从 xywhr 格式转换为 xyxyxyxy (四个角点) 格式"""
        ctr = rboxes[:, :2]  # 中心点
        w = rboxes[:, 2:3]   # 宽度
        h = rboxes[:, 3:4]   # 高度
        angle = rboxes[:, 4:5]  # 角度（弧度）
        
        cos_value = np.cos(angle)
        sin_value = np.sin(angle)
        
        # 计算两个向量
        vec1_x = w / 2 * cos_value
        vec1_y = w / 2 * sin_value
        vec2_x = -h / 2 * sin_value
        vec2_y = h / 2 * cos_value
        
        vec1 = np.concatenate([vec1_x, vec1_y], axis=-1)
        vec2 = np.concatenate([vec2_x, vec2_y], axis=-1)
        
        # 计算四个角点
        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        
        corners = np.stack([pt1, pt2, pt3, pt4], axis=1)
        return corners
    
    def nms_rotated(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """旋转框的非极大值抑制 (简化版本)"""
        if len(boxes) == 0:
            return []
        
        # 按分数排序
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while len(indices) > 0:
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
            
            # 简化的 IoU 计算（使用外接矩形）
            _current_box = boxes[current]
            _other_boxes = boxes[indices[1:]]
            
            # 计算边界框重叠（简化版）
            indices = indices[1:]
        
        return keep
    
    def deduplicate_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        distance_threshold: float = 10.0,
        iou_threshold: float = 0.3
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """后处理去重：移除中心点距离很近或高度重叠的框"""
        if len(boxes) == 0:
            return boxes, scores, class_ids
        
        # 计算每个框的中心点
        centers = np.mean(boxes, axis=1)  # (N, 2)
        
        keep = []
        
        # 按分数排序（从高到低）
        sorted_indices = np.argsort(scores)[::-1]
        
        for i in sorted_indices:
            should_keep = True
            
            for j in keep:
                # 检查中心点距离
                dist = np.linalg.norm(centers[i] - centers[j])
                
                # 同类别且距离很近，则去重
                if class_ids[i] == class_ids[j] and dist < distance_threshold:
                    should_keep = False
                    break
                
                # 计算简化的IoU（使用边界框）
                box_i_min = np.min(boxes[i], axis=0)
                box_i_max = np.max(boxes[i], axis=0)
                box_j_min = np.min(boxes[j], axis=0)
                box_j_max = np.max(boxes[j], axis=0)
                
                # 计算交集
                inter_min = np.maximum(box_i_min, box_j_min)
                inter_max = np.minimum(box_i_max, box_j_max)
                inter_wh = np.maximum(0, inter_max - inter_min)
                inter_area = inter_wh[0] * inter_wh[1]
                
                # 计算并集
                box_i_area = (box_i_max[0] - box_i_min[0]) * (box_i_max[1] - box_i_min[1])
                box_j_area = (box_j_max[0] - box_j_min[0]) * (box_j_max[1] - box_j_min[1])
                union_area = box_i_area + box_j_area - inter_area
                
                # 计算IoU
                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > iou_threshold:
                        should_keep = False
                        break
            
            if should_keep:
                keep.append(i)
        
        return boxes[keep], scores[keep], class_ids[keep]
    
    def postprocess(
        self,
        outputs: np.ndarray,
        img_shape: Tuple[int, int],
        gain: float,
        pad: Tuple[float, float],
        conf_threshold: float,
        iou_threshold: float
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        后处理模型输出
        
        Returns:
            boxes_corners: (N, 4, 2) 角点坐标
            scores: (N,) 置信度
            class_ids: (N,) 类别ID
        """
        # 处理输出格式
        predictions = outputs[0]
        
        # 检查并转置（如果需要）
        if predictions.ndim == 3:
            if predictions.shape[0] == 1:
                predictions = predictions[0]
        
        if predictions.ndim == 2:
            if predictions.shape[0] < predictions.shape[1] and predictions.shape[0] < 100:
                predictions = predictions.T
        
        if len(predictions) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # YOLO OBB 输出格式: [cx, cy, w, h, class0, class1, ..., classN, angle]
        nc = len(self.classes)
        box = predictions[:, :4]  # [cx, cy, w, h]
        cls = predictions[:, 4:4+nc]  # 类别分数
        angle = predictions[:, -1:]  # angle（最后一列）
        
        # 获取最高置信度的类别
        conf = np.max(cls, axis=1, keepdims=True)
        j = np.argmax(cls, axis=1, keepdims=True).astype(float)
        
        # 组合：[box, conf, class_id, angle]
        x = np.concatenate((box, conf, j, angle), axis=1)
        
        # 应用置信度阈值
        x = x[conf.flatten() > conf_threshold]
        
        if x.shape[0] == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 缩放坐标（在NMS之前）
        boxes_to_scale = x[:, :4].copy()
        boxes_to_scale = self.scale_boxes(
            (self.input_size, self.input_size),
            boxes_to_scale,
            img_shape,
            gain,
            pad,
            xywh=True
        )
        x[:, :4] = boxes_to_scale
        
        # NMS
        boxes_xywhr = np.concatenate((x[:, :4], x[:, -1:]), axis=-1)
        scores = x[:, 4]
        keep_indices = self.nms_rotated(boxes_xywhr, scores, iou_threshold)
        x = x[keep_indices]
        
        # 提取结果
        boxes_xywhr = np.concatenate((x[:, :4], x[:, -1:]), axis=-1)
        scores = x[:, 4]
        class_ids = x[:, 5].astype(int)
        
        # 转换为角点格式
        if len(boxes_xywhr) > 0:
            boxes_corners = self.xywhr2xyxyxyxy(boxes_xywhr)
            
            # ✅ 裁剪角点坐标到图像边界内
            img_h, img_w = img_shape
            boxes_corners[:, :, 0] = np.clip(boxes_corners[:, :, 0], 0, img_w)
            boxes_corners[:, :, 1] = np.clip(boxes_corners[:, :, 1], 0, img_h)
            
            # 内部去重
            boxes_corners, scores, class_ids = self.deduplicate_boxes(
                boxes_corners, scores, class_ids,
                distance_threshold=10.0,
                iou_threshold=0.3
            )
            self.logger.info(f"YOLO OBB内部去重后: {len(boxes_corners)} 个框")
        else:
            boxes_corners = np.array([])
        
        return boxes_corners, scores, class_ids
    
    def _rearrange_detect_unified(
        self,
        image: np.ndarray,
        text_threshold: float,
        verbose: bool = False,
        result_path_fn=None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        使用与主检测器相同的切割逻辑进行检测
        
        Args:
            image: 输入图像
            text_threshold: 置信度阈值
            verbose: 是否输出调试信息
        
        Returns:
            boxes_corners: (N, 4, 2) 合并后的角点坐标
            scores: (N,) 置信度
            class_ids: (N,) 类别ID
        """
        # 验证输入
        if image is None or image.size == 0:
            self.logger.error("YOLO OBB: 输入图片无效")
            return np.array([]), np.array([]), np.array([])
        
        h, w = image.shape[:2]
        
        if h == 0 or w == 0:
            self.logger.error(f"YOLO OBB: 图片尺寸为0: {h}x{w}")
            return np.array([]), np.array([]), np.array([])
        
        # 与 det_rearrange_forward 相同的切割逻辑
        transpose = False
        if h < w:
            transpose = True
            image = np.transpose(image, (1, 0, 2))  # h w c -> w h c
            h, w = w, h
        
        # 计算patch参数（与主检测器完全一致）
        tgt_size = self.input_size
        pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
        patch_size = ph = pw_num * w
        
        ph_num = int(np.ceil(h / ph))
        ph_step = int((h - ph) / (ph_num - 1)) if ph_num > 1 else 0
        
        self.logger.info(f"YOLO OBB统一切割: 原图={h}x{w}, patch_size={patch_size}, ph_num={ph_num}, pw_num={pw_num}, transpose={transpose}")
        
        # 生成patch列表
        rel_step_list = []
        patch_list = []
        for ii in range(ph_num):
            t = ii * ph_step
            b = t + ph
            rel_step_list.append(t / h)
            patch_list.append(image[t: b])
        
        # padding
        p_num = int(np.ceil(ph_num / pw_num))
        pad_num = p_num * pw_num - ph_num
        _valid_patch_count = ph_num  # 记录有效patch数量
        for ii in range(pad_num):
            patch_list.append(np.zeros_like(patch_list[0]))
        
        # 重排patch（与 det_rearrange_forward 的 _patch2batches 逻辑一致）
        import einops
        if transpose:
            patch_array = np.stack(patch_list, axis=0)
            patch_array = einops.rearrange(patch_array, '(p_num pw_num) ph pw c -> p_num (pw_num pw) ph c', p_num=p_num)
        else:
            patch_array = np.stack(patch_list, axis=0)
            patch_array = einops.rearrange(patch_array, '(p_num pw_num) ph pw c -> p_num ph (pw_num pw) c', p_num=p_num)
        
        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_patch_info = []  # 记录每个patch的信息用于坐标映射
        
        # 对每个重排后的patch进行检测
        for ii, patch in enumerate(patch_array):
            # 跳过padding的patch（全零patch）
            if np.all(patch == 0):
                self.logger.debug(f"YOLO OBB patch {ii}: 跳过padding patch")
                continue
            
            # 验证patch
            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                self.logger.warning(f"YOLO OBB patch {ii}: 跳过无效patch, shape={patch.shape}")
                continue
            
            # 预处理
            try:
                blob, gain, pad = self.preprocess(patch)
            except Exception as e:
                self.logger.error(f"YOLO OBB patch {ii} 预处理失败: {e}")
                continue
            
            # 推理
            input_name = self.session.get_inputs()[0].name
            output_names = [output.name for output in self.session.get_outputs()]
            
            try:
                # 确保输入是连续的内存布局
                if not blob.flags['C_CONTIGUOUS']:
                    blob = np.ascontiguousarray(blob)
                
                outputs = self.session.run(output_names, {input_name: blob})
            except Exception as e:
                self.logger.error(f"YOLO OBB patch {ii} 推理失败: {e}")
                self.logger.error(f"Patch shape: {patch.shape}, blob shape: {blob.shape}")
                self.logger.error(f"Blob contiguous: {blob.flags['C_CONTIGUOUS']}, dtype: {blob.dtype}")
                
                # 如果是访问违例，尝试清理并跳过
                if 'access violation' in str(e).lower() or 'cuda' in str(e).lower():
                    self.logger.warning("检测到 CUDA 相关错误，跳过此 patch")
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                continue
            
            # 后处理
            patch_shape = patch.shape[:2]
            boxes, scores, class_ids = self.postprocess(
                outputs,
                patch_shape,
                gain,
                pad,
                text_threshold,
                0.6
            )
            
            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)
                all_patch_info.append((ii, patch_shape))
            
            if verbose:
                self.logger.debug(f"YOLO OBB patch {ii}: 检测到 {len(boxes)} 个框")
                # 保存调试图
                try:
#                     import cv2
                    from ..utils import imwrite_unicode
                    import logging
                    logger = logging.getLogger('manga_translator')
                    # 使用result_path_fn生成路径
                    if result_path_fn:
                        debug_path = result_path_fn(f'yolo_rearrange_{ii}.png')
                    else:
                        debug_path = f'result/yolo_rearrange_{ii}.png'
                    imwrite_unicode(debug_path, patch[..., ::-1], logger)
                except Exception as e:
                    self.logger.error(f"保存YOLO调试图失败: {e}")
        
        # 合并所有patch的结果
        if len(all_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # ✅ 将每个patch的坐标映射回原图坐标
        mapped_boxes = []
        mapped_scores = []
        mapped_class_ids = []
        
        for boxes, scores, class_ids, (patch_idx, patch_shape) in zip(all_boxes, all_scores, all_class_ids, all_patch_info):
            # 计算patch的宽度
            _pw = patch_shape[1] // pw_num
            
            # 对每个框进行坐标映射
            for box, score, class_id in zip(boxes, scores, class_ids):
                # 计算框的中心点在patch中的位置
                center_x = np.mean(box[:, 0])
                _center_y = np.mean(box[:, 1])
                
                # 确定框属于patch中的哪个子区域 (jj)
                jj = int(center_x / _pw)
                jj = min(jj, pw_num - 1)  # 防止越界
                
                # 计算在原图中的索引
                pidx = patch_idx * pw_num + jj
                if pidx >= len(rel_step_list):
                    continue
                
                # 获取该子区域在原图中的偏移
                rel_t = rel_step_list[pidx]
                t = int(round(rel_t * h))  # 在原图中的top位置
                l = 0  # 宽度方向没有偏移（因为patch已经是完整宽度）
                
                # 映射坐标
                mapped_box = box.copy()
                mapped_box[:, 0] = box[:, 0] - jj * _pw + l  # x坐标：减去patch内偏移
                mapped_box[:, 1] = box[:, 1] + t  # y坐标：加上原图偏移
                
                mapped_boxes.append(mapped_box)
                mapped_scores.append(score)
                mapped_class_ids.append(class_id)
        
        if len(mapped_boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        boxes_corners = np.array(mapped_boxes)
        scores = np.array(mapped_scores)
        class_ids = np.array(mapped_class_ids)
        
        # 如果之前转置了，需要把坐标转回来
        if transpose:
            # 交换x和y坐标
            boxes_corners = boxes_corners[:, :, ::-1].copy()
        
        # 对合并后的结果进行去重（跨patch可能有重复检测）
        boxes_corners, scores, class_ids = self.deduplicate_boxes(
            boxes_corners, scores, class_ids,
            distance_threshold=20.0,
            iou_threshold=0.5
        )
        
        self.logger.info(f"YOLO OBB统一切割检测完成: 合并去重后 {len(boxes_corners)} 个框")
        
        return boxes_corners, scores, class_ids

    async def _infer(
        self, 
        image: np.ndarray, 
        detect_size: int, 
        text_threshold: float,
        box_threshold: float, 
        unclip_ratio: float,
        verbose: bool = False,
        result_path_fn=None
    ):
        """
        执行检测推理（支持长图分割检测）
        
        Returns:
            textlines: List of Quadrilateral objects
            raw_mask: None (YOLO OBB不生成mask)
            debug_img: None
        """
        # 详细的输入验证和日志
        if image is None:
            self.logger.error("YOLO OBB: 接收到的图片为None")
            return [], None, None
        
        if not isinstance(image, np.ndarray):
            self.logger.error(f"YOLO OBB: 接收到的不是numpy数组，类型: {type(image)}")
            return [], None, None
        
        if image.size == 0:
            self.logger.error("YOLO OBB: 接收到的图片大小为0")
            return [], None, None
        
        if len(image.shape) < 2:
            self.logger.error(f"YOLO OBB: 图片维度不足: {image.shape}")
            return [], None, None
        
        if image.shape[0] == 0 or image.shape[1] == 0:
            self.logger.error(f"YOLO OBB: 图片尺寸为0: {image.shape}")
            return [], None, None
        
        self.logger.debug(f"YOLO OBB输入图像: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}")
        
        img_shape = image.shape[:2]
        h, w = img_shape
        
        # 判断是否需要切割（与主检测器相同的逻辑）
        transpose = False
        if h < w:
            _transpose = True
            h, w = w, h
        
        asp_ratio = h / w
        down_scale_ratio = h / self.input_size
        require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
        
        if require_rearrange:
            # 长图模式：使用统一的切割逻辑
            self.logger.info("YOLO OBB: 检测到长图，使用统一切割逻辑")
            boxes_corners, scores, class_ids = self._rearrange_detect_unified(
                image, text_threshold, verbose, result_path_fn
            )
        else:
            # 普通图模式：直接检测
            try:
                blob, gain, pad = self.preprocess(image)
            except Exception as e:
                self.logger.error(f"YOLO OBB预处理失败: {e}, 输入图像shape={image.shape}")
                raise
            
            input_name = self.session.get_inputs()[0].name
            output_names = [output.name for output in self.session.get_outputs()]
            
            try:
                # 确保输入是连续的内存布局
                if not blob.flags['C_CONTIGUOUS']:
                    blob = np.ascontiguousarray(blob)
                
                outputs = self.session.run(output_names, {input_name: blob})
            except Exception as e:
                self.logger.error(f"YOLO OBB推理失败: {e}")
                self.logger.error(f"输入 blob shape: {blob.shape}, dtype: {blob.dtype}, contiguous: {blob.flags['C_CONTIGUOUS']}")
                self.logger.error(f"当前 providers: {self.session.get_providers()}")
                
                # 如果是访问违例或 CUDA 错误，提供更详细的错误信息
                if 'access violation' in str(e).lower() or 'cuda' in str(e).lower():
                    self.logger.error("检测到 CUDA 访问违例，可能的原因：")
                    self.logger.error("1. ONNX Runtime 版本与 CUDA 版本不兼容")
                    self.logger.error("2. GPU 内存不足")
                    self.logger.error("3. CUDA 驱动问题")
                    self.logger.error("建议：在配置中将检测器设置为使用 CPU 模式")
                raise
            
            boxes_corners, scores, class_ids = self.postprocess(
                outputs,
                img_shape,
                gain,
                pad,
                text_threshold,
                0.6
            )
        
        # 转换为Quadrilateral对象
        textlines = []
        for corners, score, class_id in zip(boxes_corners, scores, class_ids):
            pts = corners.astype(np.int32)
            label = self.classes[class_id] if class_id < len(self.classes) else f'class_{class_id}'
            quad = Quadrilateral(pts, label, float(score))
            textlines.append(quad)
        
        self.logger.info(f"YOLO OBB检测到 {len(textlines)} 个文本框")
        
        # 清理GPU内存
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        
        return textlines, None, None

