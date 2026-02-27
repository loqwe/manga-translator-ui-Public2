from __future__ import annotations

import asyncio
from dataclasses import dataclass
import os
import threading
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import cv2
import numpy as np

from .generic import BASE_PATH
from .inference import ModelWrapper
from .log import get_logger

ImageInput = Union[str, Path, np.ndarray]


@dataclass
class BubbleDetection:
    xyxy: tuple[float, float, float, float]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class BubbleDetectionResult:
    detections: list[BubbleDetection]
    image_shape: tuple[int, int]
    annotated_image: Optional[np.ndarray] = None
    raw_result: Any = None


@dataclass
class _SimpleBoxes:
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray


@dataclass
class _SimpleMasks:
    data: np.ndarray
    xy: list[np.ndarray]


@dataclass
class _SimpleRawResult:
    boxes: _SimpleBoxes
    names: dict[int, str]
    orig_img: Optional[np.ndarray] = None
    masks: Optional[_SimpleMasks] = None


class MangaLensBubbleDetector(ModelWrapper):
    """
    Lightweight backend detector for `models/detection/mangalens.onnx`.
    Uses pure ONNX Runtime (no ultralytics dependency).
    """

    _MODEL_SUB_DIR = "detection"
    _DEFAULT_MODEL_URLS = [
        # Keep consistent with other models (same ModelScope repo as OCR, detector, etc.).
        "https://www.modelscope.cn/models/hgmzhn/manga-translator-ui/resolve/master/mangalens.onnx",
    ]
    _MODEL_MAPPING = {
        "model": {
            "url": _DEFAULT_MODEL_URLS,
            "file": "mangalens.onnx",
        }
    }
    DEFAULT_MODEL_PATH = Path(BASE_PATH) / "models" / "detection" / "mangalens.onnx"

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        imgsz: int = 1600,
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[str] = None,
        auto_download: bool = True,
        auto_load: bool = True,
    ):
        self.logger = get_logger(self.__class__.__name__)
        self._custom_model_path = Path(model_path) if model_path else None
        self.auto_download = auto_download
        self.default_imgsz = imgsz
        self.default_conf = conf
        self.default_iou = iou
        self.default_device = device or self._auto_select_device()

        self.model = None
        self._input_name: Optional[str] = None
        self._output_names: Optional[list[str]] = None
        self._session_device = "cpu"
        self._provider_logged = False
        self._device_logged = False

        model_url_override = os.getenv("MANGALENS_MODEL_URL", "").strip()
        model_urls: list[str] = []
        if model_url_override:
            model_urls.append(model_url_override)
        model_urls.extend(self._DEFAULT_MODEL_URLS)
        model_urls = list(dict.fromkeys(model_urls))
        self._MODEL_MAPPING = {
            "model": {
                "url": model_urls,
                "file": "mangalens.onnx",
            }
        }

        super().__init__()
        self.model_path = self._custom_model_path if self._custom_model_path else Path(self._get_file_path("mangalens.onnx"))

        if auto_load:
            self.load_model()

    @staticmethod
    def _run_coro_sync(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)

        result = {}
        error = {}

        def _runner():
            try:
                result["value"] = asyncio.run(coro)
            except Exception as exc:  # pragma: no cover
                error["value"] = exc

        thread = threading.Thread(target=_runner, daemon=True)
        thread.start()
        thread.join()

        if "value" in error:
            raise error["value"]
        return result.get("value")

    @staticmethod
    def _auto_select_device() -> str:
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _normalize_device(device: Optional[str]) -> str:
        dev = (device or "cpu").lower()
        if dev.startswith("cuda"):
            return "cuda"
        return "cpu"

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if hasattr(x, "detach"):
            x = x.detach()
        if hasattr(x, "cpu"):
            x = x.cpu()
        if hasattr(x, "numpy"):
            return x.numpy()
        return np.asarray(x)

    def _resolve_runtime_device(self, requested_device: Optional[str]) -> str:
        if requested_device:
            return requested_device
        if self.default_device:
            return self.default_device
        return self._auto_select_device()

    def _create_ort_session(self, runtime_device: str):
        try:
            import onnxruntime as ort
        except Exception as exc:
            raise ImportError(
                "onnxruntime is required for MangaLens ONNX inference. "
                "Install with: pip install onnxruntime-gpu (or onnxruntime)"
            ) from exc

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        wanted = self._normalize_device(runtime_device)
        providers: list[Any] = ["CPUExecutionProvider"]

        if wanted == "cuda":
            if hasattr(ort, "preload_dlls"):
                try:
                    ort.preload_dlls()
                except Exception as exc:
                    self.logger.warning(f"onnxruntime.preload_dlls() failed: {exc}")
            available = ort.get_available_providers()
            if not self._provider_logged:
                self.logger.debug(f"MangaLens ONNX available providers: {available}")
                self._provider_logged = True
            if "CUDAExecutionProvider" in available:
                providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
            else:
                self.logger.warning("CUDAExecutionProvider not available, fallback to CPU")

        session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        active = "cuda" if "CUDAExecutionProvider" in session.get_providers() else "cpu"
        self.logger.debug(f"MangaLens ONNX session providers: {session.get_providers()}")
        return session, active

    def load_model(self, device: Optional[str] = None):
        runtime_device = self._resolve_runtime_device(device)
        wanted = self._normalize_device(runtime_device)

        if self.model is not None and self._session_device == wanted:
            return self.model

        if not self.model_path.exists():
            if self._custom_model_path is None and self.auto_download:
                self.logger.info("MangaLens model missing, start download...")
                self._run_coro_sync(self.download())
            if not self.model_path.exists():
                raise FileNotFoundError(f"MangaLens model not found: {self.model_path}")

        session, active = self._create_ort_session(runtime_device)
        self.model = session
        self._input_name = session.get_inputs()[0].name
        self._output_names = [o.name for o in session.get_outputs()]
        self._session_device = active
        self.logger.debug(f"MangaLens model loaded: {self.model_path} (device={self._session_device})")
        return self.model

    def unload_model(self):
        self.model = None
        self._input_name = None
        self._output_names = None
        self._session_device = "cpu"

    async def _load(self, device: str, *args, **kwargs):
        self.load_model(device=device)

    async def _unload(self):
        self.unload_model()

    async def _infer(self, *args, **kwargs):
        raise NotImplementedError("Use detect() for MangaLensBubbleDetector.")

    @staticmethod
    def _resolve_class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, list) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _extract_image_shape(image: ImageInput, raw_result: Any) -> tuple[int, int]:
        if isinstance(image, np.ndarray):
            return image.shape[:2]
        orig_img = getattr(raw_result, "orig_img", None)
        if isinstance(orig_img, np.ndarray):
            return orig_img.shape[:2]
        return (0, 0)

    @staticmethod
    def _parse_detections(raw_result: Any) -> list[BubbleDetection]:
        boxes = getattr(raw_result, "boxes", None)
        if boxes is None:
            return []
        xyxy = MangaLensBubbleDetector._to_numpy(getattr(boxes, "xyxy", np.empty((0, 4), dtype=np.float32)))
        if xyxy.size == 0:
            return []
        conf = MangaLensBubbleDetector._to_numpy(getattr(boxes, "conf", np.zeros((len(xyxy),), dtype=np.float32)))
        cls = MangaLensBubbleDetector._to_numpy(getattr(boxes, "cls", np.zeros((len(xyxy),), dtype=np.int32))).astype(int)
        names = getattr(raw_result, "names", {})

        detections: list[BubbleDetection] = []
        for idx, box in enumerate(xyxy):
            class_id = int(cls[idx]) if idx < len(cls) else -1
            confidence = float(conf[idx]) if idx < len(conf) else 0.0
            detections.append(
                BubbleDetection(
                    xyxy=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    confidence=confidence,
                    class_id=class_id,
                    class_name=MangaLensBubbleDetector._resolve_class_name(names, class_id),
                )
            )
        return detections

    @staticmethod
    def _load_input_image(image: ImageInput) -> np.ndarray:
        if isinstance(image, np.ndarray):
            img = image
        else:
            img = cv2.imread(str(image), cv2.IMREAD_COLOR)
            if img is None:
                raise FileNotFoundError(f"Failed to read image: {image}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.ndim == 3 and img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        if img.ndim != 3 or img.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {img.shape}")
        return img

    @staticmethod
    def _letterbox(
        img: np.ndarray,
        new_shape: Union[int, tuple[int, int]],
        color: tuple[int, int, int] = (114, 114, 114),
    ) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        h0, w0 = img.shape[:2]
        gain = min(new_shape[0] / h0, new_shape[1] / w0)
        new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
        pad_w = (new_shape[1] - new_w) / 2
        pad_h = (new_shape[0] - new_h) / 2

        if (w0, h0) != (new_w, new_h):
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        top = int(round(pad_h - 0.1))
        bottom = int(round(pad_h + 0.1))
        left = int(round(pad_w - 0.1))
        right = int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return img, gain, (pad_w, pad_h), (new_h, new_w)

    @staticmethod
    def _xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
        out = boxes.copy()
        out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return out

    @staticmethod
    def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> np.ndarray:
        if len(boxes) == 0:
            return np.empty((0,), dtype=np.int32)
        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = scores.argsort()[::-1]
        keep: list[int] = []

        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            union = areas[i] + areas[order[1:]] - inter + 1e-6
            iou = inter / union
            inds = np.where(iou <= iou_thres)[0]
            order = order[inds + 1]
        return np.asarray(keep, dtype=np.int32)

    @staticmethod
    def _scale_boxes_to_original(
        boxes_xyxy: np.ndarray,
        gain: float,
        pad: tuple[float, float],
        orig_shape: tuple[int, int],
    ) -> np.ndarray:
        out = boxes_xyxy.copy()
        out[:, [0, 2]] -= pad[0]
        out[:, [1, 3]] -= pad[1]
        out[:, :4] /= max(gain, 1e-6)
        h, w = orig_shape
        out[:, 0] = np.clip(out[:, 0], 0, w)
        out[:, 1] = np.clip(out[:, 1], 0, h)
        out[:, 2] = np.clip(out[:, 2], 0, w)
        out[:, 3] = np.clip(out[:, 3], 0, h)
        return out

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))

    def _decode_masks(
        self,
        mask_coeff: np.ndarray,
        protos: np.ndarray,
        input_boxes_xyxy: np.ndarray,
        orig_boxes_xyxy: np.ndarray,
        input_shape: tuple[int, int],
        unpad_shape: tuple[int, int],
        pad: tuple[float, float],
        orig_shape: tuple[int, int],
    ) -> Optional[np.ndarray]:
        if mask_coeff.size == 0:
            return None

        nm, mh, mw = protos.shape
        proto_flat = protos.reshape(nm, -1)
        masks = self._sigmoid(mask_coeff @ proto_flat).reshape((-1, mh, mw))

        input_h, input_w = input_shape
        unpad_h, unpad_w = unpad_shape
        pad_w, pad_h = pad
        top = int(round(pad_h - 0.1))
        left = int(round(pad_w - 0.1))
        orig_h, orig_w = orig_shape

        out_masks: list[np.ndarray] = []
        for i, m in enumerate(masks):
            m = cv2.resize(m, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
            y1 = max(0, min(input_h, top))
            x1 = max(0, min(input_w, left))
            y2 = max(y1, min(input_h, y1 + unpad_h))
            x2 = max(x1, min(input_w, x1 + unpad_w))
            m = m[y1:y2, x1:x2]
            if m.size == 0:
                out_masks.append(np.zeros((orig_h, orig_w), dtype=np.uint8))
                continue
            m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
            m = (m > 0.5).astype(np.uint8)

            bx1, by1, bx2, by2 = orig_boxes_xyxy[i]
            bx1 = max(0, min(orig_w, int(round(bx1))))
            by1 = max(0, min(orig_h, int(round(by1))))
            bx2 = max(bx1, min(orig_w, int(round(bx2))))
            by2 = max(by1, min(orig_h, int(round(by2))))
            cropped = np.zeros_like(m)
            if bx2 > bx1 and by2 > by1:
                cropped[by1:by2, bx1:bx2] = m[by1:by2, bx1:bx2]
            out_masks.append(cropped)

        if not out_masks:
            return None
        return np.stack(out_masks, axis=0)

    @staticmethod
    def _masks_to_polygons(mask_data: Optional[np.ndarray]) -> list[np.ndarray]:
        if mask_data is None or mask_data.size == 0:
            return []
        polygons: list[np.ndarray] = []
        for m in mask_data:
            poly = np.empty((0, 2), dtype=np.float32)
            mask_uint8 = (m > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) >= 4:
                    candidate = contour.reshape(-1, 2).astype(np.float32)
                    if candidate.shape[0] >= 3:
                        poly = candidate
            polygons.append(poly)
        return polygons

    @staticmethod
    def _requires_long_rearrange(image_shape: tuple[int, int], target_size: int) -> bool:
        h, w = image_shape
        if h <= 0 or w <= 0 or target_size <= 0:
            return False
        if h < w:
            h, w = w, h
        asp_ratio = h / float(max(w, 1))
        down_scale_ratio = h / float(max(target_size, 1))
        return down_scale_ratio > 2.5 and asp_ratio > 3.0

    def _infer_single(
        self,
        session: Any,
        img: np.ndarray,
        target_size: int,
        conf_thres: float,
        iou_thres: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        img_lb, gain, pad, unpad_shape = self._letterbox(img, target_size)
        input_h, input_w = img_lb.shape[:2]

        blob = img_lb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob)

        assert self._input_name is not None and self._output_names is not None
        outputs = session.run(self._output_names, {self._input_name: blob})
        return self._postprocess(
            outputs=outputs,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            gain=gain,
            pad=pad,
            input_shape=(input_h, input_w),
            unpad_shape=unpad_shape,
            orig_shape=(h, w),
        )

    def _infer_long_image(
        self,
        session: Any,
        img: np.ndarray,
        target_size: int,
        conf_thres: float,
        iou_thres: float,
        verbose: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        transpose = False
        work_img = img
        if work_img.shape[0] < work_img.shape[1]:
            transpose = True
            work_img = np.transpose(work_img, (1, 0, 2)).copy()

        work_h, work_w = work_img.shape[:2]
        pw_num = max(int(np.floor(2 * target_size / max(work_w, 1))), 2)
        patch_h = min(work_h, max(work_w, pw_num * work_w))
        patch_num = int(np.ceil(work_h / max(patch_h, 1)))
        patch_step = int((work_h - patch_h) / (patch_num - 1)) if patch_num > 1 else 0

        self.logger.info(
            "MangaLens long-image slicing enabled: "
            f"transpose={transpose}, work_shape={work_h}x{work_w}, "
            f"patch_h={patch_h}, patch_num={patch_num}, step={patch_step}"
        )

        all_boxes: list[np.ndarray] = []
        all_scores: list[np.ndarray] = []
        all_cls: list[np.ndarray] = []
        all_masks: list[Optional[np.ndarray]] = []

        for patch_idx in range(patch_num):
            top = patch_idx * patch_step if patch_num > 1 else 0
            bottom = min(work_h, top + patch_h)
            patch = work_img[top:bottom, :]
            if patch.size == 0 or patch.shape[0] == 0 or patch.shape[1] == 0:
                continue

            boxes, scores, cls_ids, mask_data = self._infer_single(
                session=session,
                img=patch,
                target_size=target_size,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )

            if boxes.size == 0:
                if verbose:
                    self.logger.info(f"MangaLens long-image patch {patch_idx}: detections=0")
                continue

            mapped_boxes = boxes.copy()
            mapped_boxes[:, [1, 3]] += float(top)
            mapped_boxes[:, 0] = np.clip(mapped_boxes[:, 0], 0, work_w)
            mapped_boxes[:, 1] = np.clip(mapped_boxes[:, 1], 0, work_h)
            mapped_boxes[:, 2] = np.clip(mapped_boxes[:, 2], 0, work_w)
            mapped_boxes[:, 3] = np.clip(mapped_boxes[:, 3], 0, work_h)

            mapped_masks = None
            if mask_data is not None and mask_data.ndim == 3 and mask_data.shape[0] == mapped_boxes.shape[0]:
                n = mask_data.shape[0]
                mapped_masks = np.zeros((n, work_h, work_w), dtype=np.uint8)
                ph = patch.shape[0]
                pw = patch.shape[1]
                h_use = min(ph, mask_data.shape[1], work_h - top)
                w_use = min(pw, mask_data.shape[2], work_w)
                if h_use > 0 and w_use > 0:
                    mapped_masks[:, top : top + h_use, :w_use] = mask_data[:, :h_use, :w_use]

            all_boxes.append(mapped_boxes.astype(np.float32))
            all_scores.append(scores.astype(np.float32))
            all_cls.append(cls_ids.astype(np.int32))
            all_masks.append(mapped_masks)

            if verbose:
                self.logger.info(f"MangaLens long-image patch {patch_idx}: detections={len(mapped_boxes)}")

        if not all_boxes:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        boxes_xyxy = np.concatenate(all_boxes, axis=0)
        scores = np.concatenate(all_scores, axis=0)
        cls_ids = np.concatenate(all_cls, axis=0)

        mask_data = None
        if any(m is not None for m in all_masks):
            filled_masks: list[np.ndarray] = []
            for patch_boxes, patch_masks in zip(all_boxes, all_masks):
                n = patch_boxes.shape[0]
                if patch_masks is None:
                    filled_masks.append(np.zeros((n, work_h, work_w), dtype=np.uint8))
                else:
                    filled_masks.append(patch_masks.astype(np.uint8))
            mask_data = np.concatenate(filled_masks, axis=0) if filled_masks else None

        keep_idx = self._nms_xyxy(boxes_xyxy, scores, iou_thres)
        if keep_idx.size > 0:
            boxes_xyxy = boxes_xyxy[keep_idx]
            scores = scores[keep_idx]
            cls_ids = cls_ids[keep_idx]
            if mask_data is not None:
                mask_data = mask_data[keep_idx]
        else:
            boxes_xyxy = np.empty((0, 4), dtype=np.float32)
            scores = np.empty((0,), dtype=np.float32)
            cls_ids = np.empty((0,), dtype=np.int32)
            mask_data = None

        if transpose:
            if boxes_xyxy.size > 0:
                boxes_xyxy = boxes_xyxy[:, [1, 0, 3, 2]]
            if mask_data is not None:
                mask_data = np.transpose(mask_data, (0, 2, 1))

        orig_h, orig_w = img.shape[:2]
        if boxes_xyxy.size > 0:
            boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, orig_w)
            boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, orig_h)
            boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, orig_w)
            boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, orig_h)

        return (
            boxes_xyxy.astype(np.float32),
            scores.astype(np.float32),
            cls_ids.astype(np.int32),
            mask_data.astype(np.uint8) if mask_data is not None else None,
        )

    def _postprocess(
        self,
        outputs: list[np.ndarray],
        conf_thres: float,
        iou_thres: float,
        gain: float,
        pad: tuple[float, float],
        input_shape: tuple[int, int],
        unpad_shape: tuple[int, int],
        orig_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if not outputs:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        pred_out = outputs[0]
        proto_out = outputs[1] if len(outputs) > 1 else None
        pred = pred_out[0]
        if pred.ndim != 2:
            pred = np.squeeze(pred)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T

        if pred.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        protos = None
        nm = 0
        if proto_out is not None:
            protos = proto_out[0] if proto_out.ndim == 4 else proto_out
            nm = int(protos.shape[0]) if protos is not None and protos.ndim == 3 else 0

        no = int(pred.shape[1])
        nc = max(no - 4 - nm, 1)

        boxes_xywh = pred[:, :4]
        cls_scores = pred[:, 4 : 4 + nc]
        mask_coeff = pred[:, 4 + nc : 4 + nc + nm] if nm > 0 else np.empty((pred.shape[0], 0), dtype=np.float32)

        conf = cls_scores.max(axis=1)
        cls = cls_scores.argmax(axis=1).astype(np.int32)
        keep = conf >= conf_thres
        if not np.any(keep):
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        boxes_xywh = boxes_xywh[keep]
        conf = conf[keep]
        cls = cls[keep]
        mask_coeff = mask_coeff[keep] if mask_coeff.size else mask_coeff

        input_boxes = self._xywh_to_xyxy(boxes_xywh)
        keep_idx = self._nms_xyxy(input_boxes, conf, iou_thres)
        if keep_idx.size == 0:
            return np.empty((0, 4), dtype=np.float32), np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int32), None

        input_boxes = input_boxes[keep_idx]
        conf = conf[keep_idx]
        cls = cls[keep_idx]
        mask_coeff = mask_coeff[keep_idx] if mask_coeff.size else mask_coeff

        orig_boxes = self._scale_boxes_to_original(input_boxes, gain=gain, pad=pad, orig_shape=orig_shape)
        mask_data = None
        if protos is not None and mask_coeff.size:
            mask_data = self._decode_masks(
                mask_coeff=mask_coeff,
                protos=protos,
                input_boxes_xyxy=input_boxes,
                orig_boxes_xyxy=orig_boxes,
                input_shape=input_shape,
                unpad_shape=unpad_shape,
                pad=pad,
                orig_shape=orig_shape,
            )
        return orig_boxes, conf, cls, mask_data

    def detect(
        self,
        image: ImageInput,
        imgsz: Optional[int] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        device: Optional[str] = None,
        classes: Optional[Sequence[int]] = None,
        return_annotated: bool = False,
        verbose: bool = False,
    ) -> BubbleDetectionResult:
        active_device = self._resolve_runtime_device(device)
        session = self.load_model(device=active_device)

        if not self._device_logged:
            self.logger.info(f"MangaLens detect device request: {active_device}, active_session={self._session_device}")
            self._device_logged = True

        img = self._load_input_image(image)
        orig_h, orig_w = img.shape[:2]
        target_size = imgsz if imgsz is not None else self.default_imgsz
        conf_thres = float(conf if conf is not None else self.default_conf)
        iou_thres = float(iou if iou is not None else self.default_iou)

        use_long_rearrange = self._requires_long_rearrange((orig_h, orig_w), target_size)
        if use_long_rearrange:
            boxes_xyxy, scores, cls_ids, mask_data = self._infer_long_image(
                session=session,
                img=img,
                target_size=target_size,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
                verbose=verbose,
            )
        else:
            boxes_xyxy, scores, cls_ids, mask_data = self._infer_single(
                session=session,
                img=img,
                target_size=target_size,
                conf_thres=conf_thres,
                iou_thres=iou_thres,
            )

        if classes is not None and len(classes) > 0 and len(cls_ids) > 0:
            class_set = set(int(c) for c in classes)
            idx = np.array([i for i, cid in enumerate(cls_ids) if int(cid) in class_set], dtype=np.int32)
            boxes_xyxy = boxes_xyxy[idx] if idx.size else np.empty((0, 4), dtype=np.float32)
            scores = scores[idx] if idx.size else np.empty((0,), dtype=np.float32)
            cls_ids = cls_ids[idx] if idx.size else np.empty((0,), dtype=np.int32)
            if mask_data is not None:
                mask_data = mask_data[idx] if idx.size else np.empty((0, orig_h, orig_w), dtype=np.uint8)

        names = {0: "bubble"}
        simple_boxes = _SimpleBoxes(
            xyxy=boxes_xyxy.astype(np.float32),
            conf=scores.astype(np.float32),
            cls=cls_ids.astype(np.int32),
        )
        simple_masks = None
        if mask_data is not None:
            polys = self._masks_to_polygons(mask_data)
            simple_masks = _SimpleMasks(data=mask_data.astype(np.uint8), xy=polys)

        raw_result = _SimpleRawResult(
            boxes=simple_boxes,
            names=names,
            orig_img=img,
            masks=simple_masks,
        )

        detections = self._parse_detections(raw_result)
        image_shape = self._extract_image_shape(image, raw_result)

        annotated_image = None
        if return_annotated:
            annotated_image = img.copy()
            # Draw filled semi-transparent bubble masks if available
            if simple_masks is not None and simple_masks.xy:
                overlay = annotated_image.copy()
                for poly in simple_masks.xy:
                    if poly.shape[0] >= 3:
                        pts = poly.astype(np.int32).reshape((-1, 1, 2))
                        cv2.fillPoly(overlay, [pts], (100, 120, 255))
                cv2.addWeighted(overlay, 0.45, annotated_image, 0.55, 0, annotated_image)
            # Draw bounding boxes and labels on top
            for det in detections:
                x1, y1, x2, y2 = [int(round(v)) for v in det.xyxy]
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 64, 64), 2)
                cv2.putText(
                    annotated_image,
                    f"{det.class_name}:{det.confidence:.2f}",
                    (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 64, 64),
                    1,
                    cv2.LINE_AA,
                )

        if verbose:
            self.logger.info(f"MangaLens detections: {len(detections)}")

        return BubbleDetectionResult(
            detections=detections,
            image_shape=image_shape,
            annotated_image=annotated_image,
            raw_result=raw_result,
        )


_default_detector: Optional[MangaLensBubbleDetector] = None


def get_mangalens_detector(model_path: Optional[Union[str, Path]] = None) -> MangaLensBubbleDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = MangaLensBubbleDetector(model_path=model_path)
    return _default_detector


def detect_bubbles_with_mangalens(
    image: ImageInput,
    model_path: Optional[Union[str, Path]] = None,
    **kwargs,
) -> BubbleDetectionResult:
    detector = get_mangalens_detector(model_path=model_path)
    return detector.detect(image, **kwargs)


def build_bubble_mask_from_mangalens_result(
    result: Optional[BubbleDetectionResult],
    image_shape: tuple[int, int],
) -> np.ndarray:
    h, w = int(image_shape[0]), int(image_shape[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    if result is None or h <= 0 or w <= 0:
        return mask

    raw_result = getattr(result, "raw_result", None)
    raw_masks = getattr(raw_result, "masks", None) if raw_result is not None else None

    # Prefer full segmentation masks from model output.
    if raw_masks is not None:
        mask_data = getattr(raw_masks, "data", None)
        if mask_data is not None:
            try:
                if hasattr(mask_data, "detach"):
                    mask_data = mask_data.detach().cpu().numpy()
                mask_data = np.asarray(mask_data)
                if mask_data.ndim == 3 and mask_data.shape[0] > 0:
                    merged = (mask_data > 0).any(axis=0).astype(np.uint8) * 255
                    if merged.shape != (h, w):
                        merged = cv2.resize(merged, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask = np.maximum(mask, merged)
            except Exception:
                pass

        polygons = getattr(raw_masks, "xy", None)
        if polygons is not None:
            for polygon in polygons:
                pts = np.asarray(polygon, dtype=np.int32)
                if pts.ndim != 2 or pts.shape[0] < 3:
                    continue
                pts[:, 0] = np.clip(pts[:, 0], 0, max(w - 1, 0))
                pts[:, 1] = np.clip(pts[:, 1], 0, max(h - 1, 0))
                cv2.fillPoly(mask, [pts], 255)

    # Fallback to boxes if segmentation is unavailable.
    if np.count_nonzero(mask) == 0:
        for det in getattr(result, "detections", []):
            try:
                x1, y1, x2, y2 = det.xyxy
            except Exception:
                continue
            ix1 = max(0, min(w - 1, int(round(x1))))
            iy1 = max(0, min(h - 1, int(round(y1))))
            ix2 = max(0, min(w, int(round(x2))))
            iy2 = max(0, min(h, int(round(y2))))
            if ix2 > ix1 and iy2 > iy1:
                cv2.rectangle(mask, (ix1, iy1), (ix2, iy2), 255, -1)

    return mask
