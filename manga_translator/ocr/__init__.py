import numpy as np
from typing import List, Optional
from .common import CommonOCR, OfflineOCR
from .model_32px import Model32pxOCR
from .model_48px import Model48pxOCR
from .model_48px_ctc import Model48pxCTCOCR
# ModelMangaOCR 延迟导入，避免未使用时下载模型
from .model_paddleocr import ModelPaddleOCR, ModelPaddleOCRKorean, ModelPaddleOCRLatin
from ..config import Ocr, OcrConfig
from ..utils import Quadrilateral


def _get_manga_ocr_class():
    """延迟导入 ModelMangaOCR，只有在真正使用 mocr 时才导入"""
    from .model_manga_ocr import ModelMangaOCR
    return ModelMangaOCR


def _get_paddleocr_vl_class():
    """延迟导入 ModelPaddleOCRVL，只有在真正使用 paddleocr_vl 时才导入"""
    from .model_paddleocr_vl import ModelPaddleOCRVL
    return ModelPaddleOCRVL


OCRS = {
    Ocr.ocr32px: Model32pxOCR,
    Ocr.ocr48px: Model48pxOCR,
    Ocr.ocr48px_ctc: Model48pxCTCOCR,
    Ocr.mocr: _get_manga_ocr_class,  # 延迟导入
    Ocr.paddleocr: ModelPaddleOCR,
    Ocr.paddleocr_korean: ModelPaddleOCRKorean,
    Ocr.paddleocr_latin: ModelPaddleOCRLatin,
    Ocr.paddleocr_vl: _get_paddleocr_vl_class,  # 延迟导入 PaddleOCR-VL
}
ocr_cache = {}

def get_ocr(key: Ocr, *args, **kwargs) -> CommonOCR:
    if key not in OCRS:
        raise ValueError(f'Could not find OCR for: "{key}". Choose from the following: %s' % ','.join(OCRS))
    # Use cache to avoid reloading models in the same translation session
    if key not in ocr_cache:
        ocr_class = OCRS[key]
        # 处理延迟导入的情况（mocr, paddleocr_vl）
        if callable(ocr_class) and key in (Ocr.mocr, Ocr.paddleocr_vl):
            ocr_class = ocr_class()  # 调用函数获取真正的类
        ocr_cache[key] = ocr_class(*args, **kwargs)
    return ocr_cache[key]

async def prepare(ocr_key: Ocr, device: str = 'cpu'):
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.download()
        await ocr.load(device)

async def dispatch(ocr_key: Ocr, image: np.ndarray, regions: List[Quadrilateral], config:Optional[OcrConfig] = None, device: str = 'cpu', verbose: bool = False) -> List[Quadrilateral]:
    ocr = get_ocr(ocr_key)
    if isinstance(ocr, OfflineOCR):
        await ocr.load(device)
    config = config or OcrConfig()
    return await ocr.recognize(image, regions, config, verbose)

async def unload(ocr_key: Ocr):
    ocr_cache.pop(ocr_key, None)