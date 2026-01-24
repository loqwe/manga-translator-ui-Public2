#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
失败记录管理模块
保存翻译失败的图片信息和上下文，支持后续重试
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from .path_manager import get_work_dir, WORK_DIR_NAME

logger = logging.getLogger('manga_translator')

# 失败记录文件名
FAILURE_RECORD_FILE = "failure_records.json"


def _get_failure_record_path(image_path: str) -> str:
    """
    获取失败记录文件路径
    
    Args:
        image_path: 任意一张图片的路径（用于定位工作目录）
        
    Returns:
        失败记录文件的绝对路径
    """
    work_dir = get_work_dir(image_path)
    os.makedirs(work_dir, exist_ok=True)
    return os.path.join(work_dir, FAILURE_RECORD_FILE)


def _serialize_context(ctx) -> Dict[str, Any]:
    """
    序列化上下文对象为可JSON存储的字典
    只保留重试所需的关键信息
    """
    result = {}
    
    # 保存图片路径
    if hasattr(ctx, 'image_name'):
        result['image_name'] = ctx.image_name
    
    # 保存原文文本（用于重试时的上下文）
    if hasattr(ctx, 'text_regions') and ctx.text_regions:
        original_texts = []
        for region in ctx.text_regions:
            text = getattr(region, 'text_raw', None) or getattr(region, 'text', None)
            if text:
                original_texts.append(str(text).strip())
        if original_texts:
            result['original_texts'] = original_texts
    
    # 保存同批次局部上下文
    if hasattr(ctx, 'local_prev_context') and ctx.local_prev_context:
        result['local_prev_context'] = ctx.local_prev_context
    
    return result


def _serialize_config(config) -> Dict[str, Any]:
    """
    序列化配置对象为可JSON存储的字典
    只保留翻译相关的关键配置
    """
    result = {}
    
    if hasattr(config, 'translator'):
        translator_config = config.translator
        result['translator'] = {
            'translator': str(getattr(translator_config, 'translator', 'unknown')),
            'target_lang': getattr(translator_config, 'target_lang', 'CHS'),
            'attempts': getattr(translator_config, 'attempts', -1),
        }
        
        # 保存自定义提示词路径（如果有）
        if hasattr(translator_config, 'custom_prompt_path') and translator_config.custom_prompt_path:
            result['translator']['custom_prompt_path'] = translator_config.custom_prompt_path
    
    return result


def save_failure_record(
    image_path: str,
    error_type: str,
    error_message: str,
    ctx: Any,
    config: Any,
    historical_context: Optional[Dict[str, Any]] = None
) -> bool:
    """
    保存失败记录
    
    Args:
        image_path: 失败图片的路径
        error_type: 错误类型 (network_error, safety_limit, timeout_error 等)
        error_message: 错误详细信息
        ctx: 图片上下文对象
        config: 配置对象
        historical_context: 历史上下文 {all_page_translations, _original_page_texts, context_size}
        
    Returns:
        bool: 是否保存成功
    """
    try:
        record_path = _get_failure_record_path(image_path)
        
        # 读取现有记录
        existing_records = load_failure_records(image_path)
        
        # 检查是否已存在相同图片的记录（避免重复）
        for i, record in enumerate(existing_records):
            if record.get('image_path') == image_path:
                # 更新现有记录
                existing_records.pop(i)
                break
        
        # 创建新记录
        new_record = {
            'image_path': image_path,
            'image_name': os.path.basename(image_path),
            'error_type': error_type,
            'error_message': error_message,
            'timestamp': datetime.now().isoformat(),
            'context': _serialize_context(ctx),
            'config_snapshot': _serialize_config(config),
        }
        
        # 保存历史上下文
        if historical_context:
            # 只保存最近几页的上下文（避免文件过大）
            context_size = historical_context.get('context_size', 3)
            all_trans = historical_context.get('all_page_translations', [])
            all_orig = historical_context.get('_original_page_texts', [])
            
            # 取最后 context_size 页
            new_record['historical_context'] = {
                'context_size': context_size,
                'all_page_translations': all_trans[-context_size:] if all_trans else [],
                '_original_page_texts': all_orig[-context_size:] if all_orig else [],
            }
        
        # 添加到记录列表
        existing_records.append(new_record)
        
        # 写入文件
        data = {
            'version': 1,
            'updated_at': datetime.now().isoformat(),
            'failed_images': existing_records
        }
        
        with open(record_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"[失败记录] 已保存: {os.path.basename(image_path)} (错误类型: {error_type})")
        return True
        
    except Exception as e:
        logger.error(f"[失败记录] 保存失败: {e}")
        return False


def load_failure_records(image_path: str) -> List[Dict[str, Any]]:
    """
    加载失败记录
    
    Args:
        image_path: 任意一张图片的路径（用于定位工作目录）
        
    Returns:
        失败记录列表
    """
    try:
        record_path = _get_failure_record_path(image_path)
        
        if not os.path.exists(record_path):
            return []
        
        with open(record_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('failed_images', [])
        
    except Exception as e:
        logger.error(f"[失败记录] 加载失败: {e}")
        return []


def remove_failure_record(image_path: str) -> bool:
    """
    移除指定图片的失败记录（重试成功后调用）
    
    Args:
        image_path: 图片路径
        
    Returns:
        bool: 是否移除成功
    """
    try:
        record_path = _get_failure_record_path(image_path)
        
        if not os.path.exists(record_path):
            return True
        
        records = load_failure_records(image_path)
        
        # 过滤掉指定图片
        new_records = [r for r in records if r.get('image_path') != image_path]
        
        if len(new_records) == len(records):
            # 没有找到要移除的记录
            return True
        
        # 写回文件
        if new_records:
            data = {
                'version': 1,
                'updated_at': datetime.now().isoformat(),
                'failed_images': new_records
            }
            with open(record_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        else:
            # 没有剩余记录，删除文件
            os.remove(record_path)
        
        logger.debug(f"[失败记录] 已移除: {os.path.basename(image_path)}")
        return True
        
    except Exception as e:
        logger.error(f"[失败记录] 移除失败: {e}")
        return False


def clear_failure_records(image_path: str) -> bool:
    """
    清除所有失败记录
    
    Args:
        image_path: 任意一张图片的路径（用于定位工作目录）
        
    Returns:
        bool: 是否清除成功
    """
    try:
        record_path = _get_failure_record_path(image_path)
        
        if os.path.exists(record_path):
            os.remove(record_path)
            logger.info(f"[失败记录] 已清除所有记录")
        
        return True
        
    except Exception as e:
        logger.error(f"[失败记录] 清除失败: {e}")
        return False


def get_failure_count(image_path: str) -> int:
    """
    获取失败记录数量
    
    Args:
        image_path: 任意一张图片的路径（用于定位工作目录）
        
    Returns:
        失败记录数量
    """
    records = load_failure_records(image_path)
    return len(records)


def get_failed_image_paths(image_path: str) -> List[str]:
    """
    获取所有失败图片的路径列表
    
    Args:
        image_path: 任意一张图片的路径（用于定位工作目录）
        
    Returns:
        失败图片路径列表
    """
    records = load_failure_records(image_path)
    return [r.get('image_path') for r in records if r.get('image_path')]


def prepare_retry_context(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    从失败记录中准备重试所需的上下文
    
    Args:
        record: 失败记录字典
        
    Returns:
        重试上下文字典，包含 local_prev_context 和 historical_context
    """
    result = {}
    
    # 提取同批次局部上下文
    ctx_data = record.get('context', {})
    if 'local_prev_context' in ctx_data:
        result['local_prev_context'] = ctx_data['local_prev_context']
    
    # 提取历史上下文
    hist_ctx = record.get('historical_context', {})
    if hist_ctx:
        result['historical_context'] = {
            'all_page_translations': hist_ctx.get('all_page_translations', []),
            '_original_page_texts': hist_ctx.get('_original_page_texts', []),
            'context_size': hist_ctx.get('context_size', 3),
        }
    
    return result
