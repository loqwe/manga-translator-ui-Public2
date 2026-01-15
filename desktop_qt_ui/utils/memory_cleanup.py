"""
内存清理工具模块

提供统一的内存清理功能，用于翻译完成后释放模型占用的内存。
特别针对CPU模式进行优化，因为CPU模式下模型权重直接占用RAM。
"""
import gc
import logging

logger = logging.getLogger(__name__)


def cleanup_all_model_caches() -> int:
    """
    清理所有模块的模型缓存
    
    Returns:
        清理的缓存数量
    """
    cleanup_count = 0
    
    # 清理翻译器缓存
    try:
        from manga_translator.translators import translator_cache
        if translator_cache:
            cleanup_count += len(translator_cache)
            translator_cache.clear()
    except Exception:
        pass
    
    # 清理OCR缓存
    try:
        from manga_translator.ocr import ocr_cache
        if ocr_cache:
            cleanup_count += len(ocr_cache)
            ocr_cache.clear()
    except Exception:
        pass
    
    # 清理检测器缓存
    try:
        from manga_translator.detection import detector_cache
        if detector_cache:
            cleanup_count += len(detector_cache)
            detector_cache.clear()
    except Exception:
        pass
    
    # 清理修复器缓存
    try:
        from manga_translator.inpainting import inpainter_cache
        if inpainter_cache:
            cleanup_count += len(inpainter_cache)
            inpainter_cache.clear()
    except Exception:
        pass
    
    # 清理超分缓存
    try:
        from manga_translator.upscaling import upscaler_cache
        if upscaler_cache:
            cleanup_count += len(upscaler_cache)
            upscaler_cache.clear()
    except Exception:
        pass
    
    # 清理着色器缓存
    try:
        from manga_translator.colorization import colorizer_cache
        if colorizer_cache:
            cleanup_count += len(colorizer_cache)
            colorizer_cache.clear()
    except Exception:
        pass
    
    return cleanup_count


def cleanup_gpu_memory():
    """
    清理GPU显存
    """
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
    except Exception:
        pass
    return False


def cleanup_physical_memory():
    """
    释放物理内存（Windows特定）
    
    在Windows上调用SetProcessWorkingSetSize强制释放物理内存。
    对CPU模式特别重要。
    """
    try:
        import ctypes
        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        return True
    except Exception:
        pass  # 非Windows系统忽略
    return False


def full_memory_cleanup(log_callback=None):
    """
    执行完整的内存清理
    
    Args:
        log_callback: 日志回调函数，接收字符串消息
    
    Returns:
        dict: 包含清理结果的字典
    """
    def log(msg):
        if log_callback:
            log_callback(msg)
        logger.info(msg)
    
    result = {
        'caches_cleared': 0,
        'gpu_cleared': False,
        'physical_memory_released': False
    }
    
    log("--- [CLEANUP] 开始完整内存清理...")
    
    # 1. 清理模型缓存
    result['caches_cleared'] = cleanup_all_model_caches()
    if result['caches_cleared'] > 0:
        log(f"--- [CLEANUP] 已清理 {result['caches_cleared']} 个模型缓存")
    
    # 2. 强制垃圾回收（多次执行确保彻底清理）
    gc.collect()
    gc.collect()
    gc.collect()
    
    # 3. 清理GPU显存
    result['gpu_cleared'] = cleanup_gpu_memory()
    if result['gpu_cleared']:
        log("--- [CLEANUP] GPU显存已清理")
    
    # 4. 释放物理内存
    result['physical_memory_released'] = cleanup_physical_memory()
    if result['physical_memory_released']:
        log("--- [CLEANUP] 物理内存已释放")
    
    log("--- [CLEANUP] 内存清理完成")
    
    return result
