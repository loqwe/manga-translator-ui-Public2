"""
YOLO 辅助检测器
使用原生 PyTorch checkpoint 进行推理。
"""

import os
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np
import torch

from .common import OfflineDetector
from ..utils import Quadrilateral, build_det_rearrange_plan, det_rearrange_patch_array


class YOLOOBBDetector(OfflineDetector):
    """YOLO 辅助检测器 - 基于原生 PyTorch checkpoint"""

    _MODEL_FILENAME = "ysgyolo_yolo26_2.0.pt"
    _SOURCE_CHECKPOINT_FILENAME = "ysgyolo_yolo26_2.0.pt"
    _MODEL_MAPPING = {
        "model": {
            "url": [
                "https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/ysgyolo_yolo26_2.0.pt",
            ],
            "hash": "889347d65c8636dd188a8ed4f312b29658543faaa69016b5958ddf0559980e22",
            "file": "ysgyolo_yolo26_2.0.pt",
        },
    }

    _DEFAULT_CLASS_ID_TO_LABEL = {
        0: "balloon",
        1: "qipao",
        2: "fangkuai",
        3: "changfangtiao",
        4: "kuangwai",
        5: "other",
    }

    def __init__(self, *args, **kwargs):
        os.makedirs(self.model_dir, exist_ok=True)
        super().__init__(*args, **kwargs)

        self.class_id_to_label = dict(self._DEFAULT_CLASS_ID_TO_LABEL)
        self.classes = [label for idx, label in sorted(self.class_id_to_label.items()) if label != "other"]
        self.input_size = 1600
        self.using_cuda = False
        self.device = "cpu"
        self.torch_device = torch.device("cpu")
        self.model: Optional[torch.nn.Module] = None

    def _check_downloaded(self) -> bool:
        return os.path.exists(self._get_file_path(self._MODEL_FILENAME))

    @staticmethod
    def _empty_results() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            np.empty((0, 4, 2), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    def _resolve_device(self, device: str) -> torch.device:
        requested = (device or "cpu").lower()
        if requested.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(device)
            self.logger.warning("YOLO OBB: 请求 CUDA，但当前不可用，回退到 CPU")
        elif requested.startswith("mps"):
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return torch.device("mps")
            self.logger.warning("YOLO OBB: 请求 MPS，但当前不可用，回退到 CPU")
        return torch.device("cpu")

    async def _load(self, device: str):
        model_path = self._get_file_path(self._MODEL_FILENAME)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YOLO PyTorch checkpoint 不存在: {model_path}")

        self.torch_device = self._resolve_device(device)
        self.device = str(self.torch_device)
        self.using_cuda = self.torch_device.type == "cuda"

        load_path = os.path.relpath(model_path, os.getcwd())
        if not os.path.exists(load_path):
            load_path = model_path

        checkpoint = torch.load(load_path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            model = checkpoint.get("ema") or checkpoint.get("model")
            if model is None:
                raise TypeError(f"YOLO OBB checkpoint 结构不支持: {type(checkpoint)}")
        else:
            model = checkpoint
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"YOLO OBB checkpoint 未解析出 torch.nn.Module: {type(model)}")
        model = model.float().eval()
        for param in model.parameters():
            param.requires_grad_(False)
        model.to(self.torch_device)
        self.model = model

        self.logger.info(f"YOLO OBB: {self.torch_device.type.upper()} 模式加载成功")

    async def _unload(self):
        self.model = None

    def letterbox(
        self,
        img: np.ndarray,
        new_shape: Tuple[int, int] = (1600, 1600),
        color: Tuple[int, int, int] = (0, 0, 0),
    ) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """
        调整图像大小并添加边框（保持宽高比）

        Returns:
            resized_img: 调整后的图像
            gain: 缩放比例
            (pad_w, pad_h): 左/上 padding
        """
        if img is None or img.size == 0:
            self.logger.error("YOLO OBB letterbox: 输入图片为空")
            raise ValueError("输入图片为空")

        shape = img.shape[:2]
        if shape[0] == 0 or shape[1] == 0:
            self.logger.error(f"YOLO OBB letterbox: 输入图片尺寸无效: {shape}")
            raise ValueError(f"输入图片尺寸无效: {shape}")

        gain = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad_w = int(round(shape[1] * gain))
        new_unpad_h = int(round(shape[0] * gain))

        if new_unpad_w <= 0 or new_unpad_h <= 0:
            self.logger.error(
                f"YOLO OBB letterbox: 计算的新尺寸无效: {new_unpad_w}x{new_unpad_h}, "
                f"gain={gain}, 原始shape={shape}"
            )
            raise ValueError(f"计算的新尺寸无效: {new_unpad_w}x{new_unpad_h}")

        if (new_unpad_w, new_unpad_h) != (shape[1], shape[0]):
            img = cv2.resize(img, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        dw = new_shape[1] - new_unpad_w
        dh = new_shape[0] - new_unpad_h
        left = 0
        top = 0
        right = dw
        bottom = dh

        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )

        assert img.shape[0] == new_shape[0] and img.shape[1] == new_shape[1], (
            f"Letterbox failed: expected {new_shape}, got {img.shape[:2]}"
        )
        return img, gain, (float(left), float(top))

    def preprocess(self, img: np.ndarray) -> Tuple[torch.Tensor, float, Tuple[float, float]]:
        """预处理图像"""
        if img is None or img.size == 0:
            self.logger.error("YOLO OBB预处理: 输入图片为空或无效")
            raise ValueError("输入图片为空或无效")

        if len(img.shape) < 2:
            self.logger.error(f"YOLO OBB预处理: 输入图片维度不正确: {img.shape}")
            raise ValueError(f"输入图片维度不正确: {img.shape}")

        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            self.logger.debug(f"YOLO OBB: 灰度图转RGB, shape={img.shape}")
        elif len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                self.logger.debug(f"YOLO OBB: RGBA转RGB, shape={img.shape}")
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                self.logger.error(f"YOLO OBB: 不支持的图像通道数: {img.shape}")
                raise ValueError(f"Unsupported image shape: {img.shape}")
        else:
            self.logger.error(f"YOLO OBB: 不支持的图像维度: {img.shape}")
            raise ValueError(f"Unsupported image shape: {img.shape}")

        img_resized, gain, pad = self.letterbox(img, new_shape=(self.input_size, self.input_size))
        blob = img_resized.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)
        tensor = torch.from_numpy(blob)

        self.logger.debug(
            f"YOLO OBB预处理完成: blob shape={tuple(tensor.shape)}, dtype={tensor.dtype}"
        )
        return tensor, gain, pad

    def scale_boxes(self, img1_shape, boxes, img0_shape, gain, pad, xywh=False):
        """将边界框从 img1_shape 缩放到 img0_shape（移除 letterbox 影响）"""
        pad_w, pad_h = pad

        boxes[:, 0] -= pad_w
        boxes[:, 1] -= pad_h
        if not xywh:
            boxes[:, 2] -= pad_w
            boxes[:, 3] -= pad_h

        boxes[:, :4] /= gain

        if xywh:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img0_shape[1])
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img0_shape[0])
        else:
            boxes[:, 0] = np.clip(boxes[:, 0], 0, img0_shape[1])
            boxes[:, 1] = np.clip(boxes[:, 1], 0, img0_shape[0])
            boxes[:, 2] = np.clip(boxes[:, 2], 0, img0_shape[1])
            boxes[:, 3] = np.clip(boxes[:, 3], 0, img0_shape[0])
        return boxes

    def xywhr2xyxyxyxy(self, rboxes: np.ndarray) -> np.ndarray:
        """将旋转边界框从 xywhr 转换为四角点"""
        ctr = rboxes[:, :2]
        w = rboxes[:, 2:3]
        h = rboxes[:, 3:4]
        angle = rboxes[:, 4:5]

        cos_value = np.cos(angle)
        sin_value = np.sin(angle)

        vec1_x = w / 2 * cos_value
        vec1_y = w / 2 * sin_value
        vec2_x = -h / 2 * sin_value
        vec2_y = h / 2 * cos_value

        vec1 = np.concatenate([vec1_x, vec1_y], axis=-1)
        vec2 = np.concatenate([vec2_x, vec2_y], axis=-1)

        pt1 = ctr + vec1 + vec2
        pt2 = ctr + vec1 - vec2
        pt3 = ctr - vec1 - vec2
        pt4 = ctr - vec1 + vec2
        return np.stack([pt1, pt2, pt3, pt4], axis=1)

    def xyxy2xyxyxyxy(self, boxes: np.ndarray) -> np.ndarray:
        """将轴对齐框从 xyxy 转换为四角点"""
        x1 = boxes[:, 0:1]
        y1 = boxes[:, 1:2]
        x2 = boxes[:, 2:3]
        y2 = boxes[:, 3:4]
        pt1 = np.concatenate([x1, y1], axis=1)
        pt2 = np.concatenate([x2, y1], axis=1)
        pt3 = np.concatenate([x2, y2], axis=1)
        pt4 = np.concatenate([x1, y2], axis=1)
        return np.stack([pt1, pt2, pt3, pt4], axis=1)

    def deduplicate_boxes(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        distance_threshold: float = 10.0,
        iou_threshold: float = 0.3,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """后处理去重：移除中心点距离很近或高度重叠的框"""
        if len(boxes) == 0:
            return boxes, scores, class_ids

        centers = np.mean(boxes, axis=1)
        keep = []
        sorted_indices = np.argsort(scores)[::-1]

        for i in sorted_indices:
            should_keep = True
            for j in keep:
                dist = np.linalg.norm(centers[i] - centers[j])
                if class_ids[i] == class_ids[j] and dist < distance_threshold:
                    should_keep = False
                    break

                box_i_min = np.min(boxes[i], axis=0)
                box_i_max = np.max(boxes[i], axis=0)
                box_j_min = np.min(boxes[j], axis=0)
                box_j_max = np.max(boxes[j], axis=0)

                inter_min = np.maximum(box_i_min, box_j_min)
                inter_max = np.minimum(box_i_max, box_j_max)
                inter_wh = np.maximum(0, inter_max - inter_min)
                inter_area = inter_wh[0] * inter_wh[1]

                box_i_area = (box_i_max[0] - box_i_min[0]) * (box_i_max[1] - box_i_min[1])
                box_j_area = (box_j_max[0] - box_j_min[0]) * (box_j_max[1] - box_j_min[1])
                union_area = box_i_area + box_j_area - inter_area

                if union_area > 0:
                    iou = inter_area / union_area
                    if iou > iou_threshold:
                        should_keep = False
                        break

            if should_keep:
                keep.append(i)

        return boxes[keep], scores[keep], class_ids[keep]

    def _normalize_outputs(self, outputs: Any) -> np.ndarray:
        if isinstance(outputs, (list, tuple)):
            if not outputs:
                return np.empty((0, 6), dtype=np.float32)
            outputs = outputs[0]

        if isinstance(outputs, torch.Tensor):
            outputs = outputs.detach().cpu().numpy()
        elif not isinstance(outputs, np.ndarray):
            raise TypeError(f"YOLO OBB输出类型不支持: {type(outputs)}")

        if outputs.ndim == 3 and outputs.shape[0] == 1:
            outputs = outputs[0]
        if outputs.ndim != 2:
            raise ValueError(f"YOLO OBB输出形状异常: {outputs.shape}")
        return outputs

    def postprocess(
        self,
        outputs: Any,
        img_shape: Tuple[int, int],
        gain: float,
        pad: Tuple[float, float],
        conf_threshold: float,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        后处理模型输出

        Returns:
            boxes_corners: (N, 4, 2)
            scores: (N,)
            class_ids: (N,)
        """
        predictions = self._normalize_outputs(outputs)
        if len(predictions) == 0:
            return self._empty_results()

        scores: np.ndarray
        class_ids: np.ndarray

        if predictions.shape[1] == 6:
            x = predictions.copy()
            x = x[x[:, 4] > conf_threshold]
            if x.shape[0] == 0:
                return self._empty_results()

            boxes_xyxy = self.scale_boxes(
                (self.input_size, self.input_size),
                x[:, :4].copy(),
                img_shape,
                gain,
                pad,
                xywh=False,
            )
            scores = x[:, 4].astype(np.float32)
            class_ids = np.rint(x[:, 5]).astype(np.int32)
            boxes_corners = self.xyxy2xyxyxyxy(boxes_xyxy)
        else:
            if predictions.shape[1] == 7:
                box = predictions[:, :4]
                conf = predictions[:, 4:5]
                j = predictions[:, 5:6]
                angle = predictions[:, 6:7]
                x = np.concatenate((box, conf, j, angle), axis=1)
            else:
                nc = len(self.class_id_to_label)
                if predictions.shape[1] < 4 + nc + 1:
                    self.logger.warning(f"YOLO OBB输出格式异常: {predictions.shape}")
                    return self._empty_results()
                box = predictions[:, :4]
                cls = predictions[:, 4 : 4 + nc]
                angle = predictions[:, -1:]
                conf = np.max(cls, axis=1, keepdims=True)
                j = np.argmax(cls, axis=1, keepdims=True).astype(np.float32)
                x = np.concatenate((box, conf, j, angle), axis=1)

            x = x[x[:, 4] > conf_threshold]
            if x.shape[0] == 0:
                return self._empty_results()

            boxes_to_scale = self.scale_boxes(
                (self.input_size, self.input_size),
                x[:, :4].copy(),
                img_shape,
                gain,
                pad,
                xywh=True,
            )
            boxes_xywhr = np.concatenate((boxes_to_scale, x[:, -1:]), axis=-1)
            scores = x[:, 4].astype(np.float32)
            class_ids = np.rint(x[:, 5]).astype(np.int32)
            boxes_corners = self.xywhr2xyxyxyxy(boxes_xywhr)

        valid_class_ids = np.array(list(self.class_id_to_label.keys()), dtype=np.int32)
        valid_cls_mask = np.isin(class_ids, valid_class_ids)
        if not np.all(valid_cls_mask):
            drop_count = int(np.size(valid_cls_mask) - np.sum(valid_cls_mask))
            self.logger.info(f"YOLO OBB过滤无效类别: 移除 {drop_count} 个框")
            boxes_corners = boxes_corners[valid_cls_mask]
            scores = scores[valid_cls_mask]
            class_ids = class_ids[valid_cls_mask]

        if len(boxes_corners) == 0:
            return self._empty_results()

        img_h, img_w = img_shape
        boxes_corners[:, :, 0] = np.clip(boxes_corners[:, :, 0], 0, img_w)
        boxes_corners[:, :, 1] = np.clip(boxes_corners[:, :, 1], 0, img_h)
        return boxes_corners.astype(np.float32), scores.astype(np.float32), class_ids.astype(np.int32)

    def _run_model(self, blob: torch.Tensor) -> Any:
        if self.model is None:
            raise RuntimeError("YOLO OBB 模型未加载")

        blob = blob.to(self.torch_device)
        with torch.inference_mode():
            return self.model(blob)

    def _rearrange_detect_unified(
        self,
        image: np.ndarray,
        text_threshold: float,
        verbose: bool = False,
        result_path_fn=None,
        rearrange_plan: Optional[dict] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用与主检测器相同的切割逻辑进行检测"""
        if image is None or image.size == 0:
            self.logger.error("YOLO OBB: 输入图片无效")
            return self._empty_results()

        h, w = image.shape[:2]
        if h == 0 or w == 0:
            self.logger.error(f"YOLO OBB: 图片尺寸为0: {h}x{w}")
            return self._empty_results()

        if rearrange_plan is None:
            rearrange_plan = build_det_rearrange_plan(image, tgt_size=self.input_size)
        if rearrange_plan is None:
            self.logger.warning("YOLO OBB统一切割: 当前图像不满足切割条件")
            return self._empty_results()

        transpose = rearrange_plan["transpose"]
        h = rearrange_plan["h"]
        w = rearrange_plan["w"]
        pw_num = rearrange_plan["pw_num"]
        patch_size = rearrange_plan["patch_size"]
        ph_num = rearrange_plan["ph_num"]
        rel_step_list = rearrange_plan["rel_step_list"]
        pad_num = rearrange_plan["pad_num"]

        self.logger.info(
            f"YOLO OBB统一切割: 原图={h}x{w}, patch_size={patch_size}, "
            f"ph_num={ph_num}, pw_num={pw_num}, pad_num={pad_num}, transpose={transpose}"
        )

        patch_array = det_rearrange_patch_array(rearrange_plan)

        all_boxes = []
        all_scores = []
        all_class_ids = []
        all_patch_info = []

        for ii, patch in enumerate(patch_array):
            if np.all(patch == 0):
                self.logger.debug(f"YOLO OBB patch {ii}: 跳过padding patch")
                continue

            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                self.logger.warning(f"YOLO OBB patch {ii}: 跳过无效patch, shape={patch.shape}")
                continue

            try:
                blob, gain, pad = self.preprocess(patch)
            except Exception as e:
                self.logger.error(f"YOLO OBB patch {ii} 预处理失败: {e}")
                continue

            try:
                outputs = self._run_model(blob)
            except Exception as e:
                self.logger.error(f"YOLO OBB patch {ii} 推理失败: {e}")
                self.logger.error(f"Patch shape: {patch.shape}, blob shape: {tuple(blob.shape)}")
                continue

            patch_shape = patch.shape[:2]
            boxes, scores, class_ids = self.postprocess(
                outputs,
                patch_shape,
                gain,
                pad,
                text_threshold,
            )

            if len(boxes) > 0:
                all_boxes.append(boxes)
                all_scores.append(scores)
                all_class_ids.append(class_ids)
                all_patch_info.append((ii, patch_shape))

            if verbose:
                self.logger.debug(f"YOLO OBB patch {ii}: 检测到 {len(boxes)} 个框")
                try:
                    from ..utils import imwrite_unicode
                    import logging

                    logger = logging.getLogger("manga_translator")
                    debug_path = (
                        result_path_fn(f"yolo_rearrange_{ii}.png")
                        if result_path_fn
                        else f"result/yolo_rearrange_{ii}.png"
                    )
                    imwrite_unicode(debug_path, patch[..., ::-1], logger)
                except Exception as e:
                    self.logger.error(f"保存YOLO调试图失败: {e}")

        if len(all_boxes) == 0:
            return self._empty_results()

        mapped_boxes = []
        mapped_scores = []
        mapped_class_ids = []

        for boxes, scores, class_ids, (patch_idx, patch_shape) in zip(
            all_boxes, all_scores, all_class_ids, all_patch_info
        ):
            _pw = patch_shape[1] // pw_num
            if _pw <= 0:
                continue

            for box, score, class_id in zip(boxes, scores, class_ids):
                x_min = float(np.min(box[:, 0]))
                x_max = float(np.max(box[:, 0]))

                jj_start = max(0, int(np.floor(x_min / _pw)))
                jj_end = min(pw_num - 1, int(np.floor(max(x_max - 1e-6, x_min) / _pw)))

                for jj in range(jj_start, jj_end + 1):
                    pidx = patch_idx * pw_num + jj
                    if pidx >= len(rel_step_list):
                        continue

                    stripe_l = jj * _pw
                    stripe_r = (jj + 1) * _pw
                    if x_max <= stripe_l or x_min >= stripe_r:
                        continue

                    rel_t = rel_step_list[pidx]
                    t = int(round(rel_t * h))

                    mapped_box = box.copy()
                    mapped_box[:, 0] = np.clip(mapped_box[:, 0], stripe_l, stripe_r) - stripe_l
                    mapped_box[:, 1] = np.clip(mapped_box[:, 1] + t, 0, h)

                    mapped_w = float(np.max(mapped_box[:, 0]) - np.min(mapped_box[:, 0]))
                    mapped_h = float(np.max(mapped_box[:, 1]) - np.min(mapped_box[:, 1]))
                    if mapped_w < 1.0 or mapped_h < 1.0:
                        continue

                    mapped_boxes.append(mapped_box)
                    mapped_scores.append(score)
                    mapped_class_ids.append(class_id)

        if len(mapped_boxes) == 0:
            return self._empty_results()

        boxes_corners = np.array(mapped_boxes, dtype=np.float32)
        scores = np.array(mapped_scores, dtype=np.float32)
        class_ids = np.array(mapped_class_ids, dtype=np.int32)

        if transpose:
            boxes_corners = boxes_corners[:, :, ::-1].copy()

        boxes_corners, scores, class_ids = self.deduplicate_boxes(
            boxes_corners,
            scores,
            class_ids,
            distance_threshold=20.0,
            iou_threshold=0.5,
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
        result_path_fn=None,
    ):
        """
        执行检测推理（支持长图分割检测）

        Returns:
            textlines: List[Quadrilateral]
            raw_mask: None
            debug_img: None
        """
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

        self.logger.debug(
            f"YOLO OBB输入图像: shape={image.shape}, dtype={image.dtype}, min={image.min()}, max={image.max()}"
        )

        img_shape = image.shape[:2]
        rearrange_plan = build_det_rearrange_plan(image, tgt_size=self.input_size)

        if rearrange_plan is not None:
            self.logger.info("YOLO OBB: 检测到长图，使用统一切割逻辑")
            boxes_corners, scores, class_ids = self._rearrange_detect_unified(
                image, text_threshold, verbose, result_path_fn, rearrange_plan=rearrange_plan
            )
        else:
            try:
                blob, gain, pad = self.preprocess(image)
            except Exception as e:
                self.logger.error(f"YOLO OBB预处理失败: {e}, 输入图像shape={image.shape}")
                raise

            try:
                outputs = self._run_model(blob)
            except Exception as e:
                self.logger.error(f"YOLO OBB推理失败: {e}")
                self.logger.error(f"输入 blob shape: {tuple(blob.shape)}, dtype: {blob.dtype}")
                self.logger.error(f"当前 device: {self.device}")
                raise

            boxes_corners, scores, class_ids = self.postprocess(
                outputs,
                img_shape,
                gain,
                pad,
                text_threshold,
            )

        textlines = []
        for corners, score, class_id in zip(boxes_corners, scores, class_ids):
            pts = corners.astype(np.int32)
            label = self.class_id_to_label.get(int(class_id))
            if not label:
                label = self.classes[class_id] if class_id < len(self.classes) else f"class_{class_id}"
            quad = Quadrilateral(pts, label, float(score))
            quad.det_label = label
            quad.yolo_label = label
            quad.is_yolo_box = True
            textlines.append(quad)

        self.logger.info(f"YOLO OBB检测到 {len(textlines)} 个文本框")
        return textlines, None, None
