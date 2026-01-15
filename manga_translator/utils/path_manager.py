#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
路径管理模块
提供统一的文件路径生成和查找功能，支持新的目录结构和向后兼容
"""

import hashlib
import os
import re
from typing import Dict, Iterable, Optional, Tuple


# 工作目录名称常量
WORK_DIR_NAME = "manga_translator_work"
JSON_SUBDIR = "json"
TRANSLATIONS_SUBDIR = "translations"
ORIGINALS_SUBDIR = "originals"
INPAINTED_SUBDIR = "inpainted"
TRANSLATED_IMAGES_SUBDIR = "translated_images"  # 已翻译图片目录（替换翻译模式使用）


def build_unique_folder_aliases(folders: Iterable[str], max_depth: int = 4) -> Dict[str, str]:
    """为输入文件夹生成唯一的输出子目录别名，避免同名目录导出合并。

    典型问题：用户同时导入两个不同作品的 "Chapter 31" 文件夹时，如果仅用 basename 作为输出子目录，
    会导致输出目录冲突并把两章内容混在一起。

    规则：
    - 默认使用末级目录名（depth=1）
    - 若发生冲突，则逐步增加使用的尾部路径层级（depth=2..max_depth），直到别名唯一
    - 若仍无法唯一（极少见），则在最后一级目录名后追加短 hash

    Args:
        folders: 输入文件夹路径集合（可为绝对或相对路径）
        max_depth: 最大尾部层级数（避免路径过长，默认 4）

    Returns:
        dict: {规范化后的输入文件夹路径: 输出子目录相对路径}
    """
    # 规范化并去重
    norm_folders = [os.path.normpath(f) for f in folders if f]
    unique_folders = sorted(set(norm_folders))
    if not unique_folders:
        return {}

    parts_by_folder: Dict[str, list] = {}
    for folder in unique_folders:
        # 去掉盘符，按 / 或 \ 切分，得到稳定的路径段
        _, tail = os.path.splitdrive(folder)
        tail = tail.strip('\\/').strip()
        parts = [p for p in re.split(r"[\\/]+", tail) if p]
        parts_by_folder[folder] = parts

    depth_by_folder: Dict[str, int] = {f: 1 for f in unique_folders}

    def _make_alias(folder: str) -> str:
        parts = parts_by_folder.get(folder) or []
        if not parts:
            return os.path.basename(folder)
        depth = min(depth_by_folder.get(folder, 1), len(parts))
        suffix = parts[-depth:]
        return os.path.join(*suffix) if suffix else os.path.basename(folder)

    # 逐步提升 depth，直到别名唯一或无法继续
    while True:
        alias_map = {f: _make_alias(f) for f in unique_folders}
        groups: Dict[str, list] = {}
        for f, a in alias_map.items():
            groups.setdefault(a, []).append(f)

        dup_groups = [fs for fs in groups.values() if len(fs) > 1]
        if not dup_groups:
            break

        progressed = False
        for fs in dup_groups:
            for f in fs:
                parts_len = len(parts_by_folder.get(f) or [])
                max_for_f = min(max_depth, parts_len) if parts_len > 0 else 1
                if depth_by_folder[f] < max_for_f:
                    depth_by_folder[f] += 1
                    progressed = True

        if not progressed:
            # 已到达 max_depth 仍有冲突，退出并使用 hash 兆底
            break

    alias_map = {f: _make_alias(f) for f in unique_folders}

    # 冲突兆底：追加 hash
    groups: Dict[str, list] = {}
    for f, a in alias_map.items():
        groups.setdefault(a, []).append(f)

    for a, fs in groups.items():
        if len(fs) <= 1:
            continue
        for f in fs:
            h = hashlib.md5(f.encode('utf-8')).hexdigest()[:8]
            head, last = os.path.split(a)
            last = last or os.path.basename(f)
            new_last = f"{last}__{h}"
            alias_map[f] = os.path.join(head, new_last) if head else new_last

    return alias_map


def get_work_dir(image_path: str) -> str:
    """
    获取图片对应的工作目录路径
    
    Args:
        image_path: 原图片路径
        
    Returns:
        工作目录的绝对路径
    """
    image_dir = os.path.dirname(os.path.abspath(image_path))
    return os.path.join(image_dir, WORK_DIR_NAME)


def get_json_path(image_path: str, create_dir: bool = True) -> str:
    """
    获取JSON配置文件的路径
    
    Args:
        image_path: 原图片路径
        create_dir: 是否自动创建目录
        
    Returns:
        JSON文件的绝对路径
    """
    work_dir = get_work_dir(image_path)
    json_dir = os.path.join(work_dir, JSON_SUBDIR)
    
    if create_dir:
        os.makedirs(json_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(json_dir, f"{base_name}_translations.json")


def get_original_txt_path(image_path: str, create_dir: bool = True) -> str:
    """
    获取原文TXT文件的路径
    
    Args:
        image_path: 原图片路径
        create_dir: 是否自动创建目录
        
    Returns:
        原文TXT文件的绝对路径
    """
    work_dir = get_work_dir(image_path)
    originals_dir = os.path.join(work_dir, ORIGINALS_SUBDIR)
    
    if create_dir:
        os.makedirs(originals_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(originals_dir, f"{base_name}_original.txt")


def get_translated_txt_path(image_path: str, create_dir: bool = True) -> str:
    """
    获取翻译TXT文件的路径
    
    Args:
        image_path: 原图片路径
        create_dir: 是否自动创建目录
        
    Returns:
        翻译TXT文件的绝对路径
    """
    work_dir = get_work_dir(image_path)
    translations_dir = os.path.join(work_dir, TRANSLATIONS_SUBDIR)
    
    if create_dir:
        os.makedirs(translations_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join(translations_dir, f"{base_name}_translated.txt")


def get_inpainted_path(image_path: str, create_dir: bool = True) -> str:
    """
    获取修复后图片的路径
    
    Args:
        image_path: 原图片路径
        create_dir: 是否自动创建目录
        
    Returns:
        修复后图片的绝对路径
    """
    work_dir = get_work_dir(image_path)
    inpainted_dir = os.path.join(work_dir, INPAINTED_SUBDIR)
    
    if create_dir:
        os.makedirs(inpainted_dir, exist_ok=True)
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    ext = os.path.splitext(image_path)[1]
    return os.path.join(inpainted_dir, f"{base_name}_inpainted{ext}")


def get_translated_images_dir(image_path: str, create_dir: bool = True) -> str:
    """
    获取已翻译图片目录的路径
    
    Args:
        image_path: 原图片路径
        create_dir: 是否自动创建目录
        
    Returns:
        已翻译图片目录的绝对路径
    """
    work_dir = get_work_dir(image_path)
    translated_dir = os.path.join(work_dir, TRANSLATED_IMAGES_SUBDIR)
    
    if create_dir:
        os.makedirs(translated_dir, exist_ok=True)
    
    return translated_dir


def find_translated_source_json(target_image_path: str, translated_dir: str) -> Optional[str]:
    """
    在已翻译图片目录中查找与目标图同名的翻译数据JSON
    
    用于替换翻译模式：根据生肉图的文件名，在已翻译目录中查找同名图片的JSON
    
    Args:
        target_image_path: 目标图片（生肉）的路径
        translated_dir: 已翻译图片所在目录
        
    Returns:
        找到的JSON文件路径，如果不存在返回None
    """
    if not translated_dir or not os.path.isdir(translated_dir):
        return None
    
    # 获取目标图的基础文件名（不含扩展名）
    target_basename = os.path.splitext(os.path.basename(target_image_path))[0]
    
    # 在已翻译目录中查找同名图片
    # 尝试查找 manga_translator_work/json/文件名_translations.json
    translated_work_dir = os.path.join(translated_dir, WORK_DIR_NAME, JSON_SUBDIR)
    if os.path.isdir(translated_work_dir):
        json_path = os.path.join(translated_work_dir, f"{target_basename}_translations.json")
        if os.path.exists(json_path):
            return json_path
    
    # 向后兼容：查找 已翻译目录/文件名_translations.json
    old_json_path = os.path.join(translated_dir, f"{target_basename}_translations.json")
    if os.path.exists(old_json_path):
        return old_json_path
    
    # 尝试匹配任意图片扩展名
    for ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp', '.gif']:
        # 构造可能的已翻译图片路径
        possible_translated_image = os.path.join(translated_dir, f"{target_basename}{ext}")
        if os.path.exists(possible_translated_image):
            # 查找该图片对应的JSON
            json_path = find_json_path(possible_translated_image)
            if json_path:
                return json_path
    
    return None


def find_json_path(image_path: str) -> Optional[str]:
    """
    查找JSON配置文件，优先查找新位置，支持向后兼容
    
    Args:
        image_path: 原图片路径
        
    Returns:
        找到的JSON文件路径，如果不存在返回None
    """
    # 1. 优先查找新位置
    new_json_path = get_json_path(image_path, create_dir=False)
    if os.path.exists(new_json_path):
        return new_json_path
    
    # 2. 向后兼容：查找旧位置（图片同目录）
    old_json_path = os.path.splitext(image_path)[0] + '_translations.json'
    if os.path.exists(old_json_path):
        return old_json_path
    
    return None


def find_inpainted_path(image_path: str) -> Optional[str]:
    """
    查找修复后的图片文件
    
    Args:
        image_path: 原图片路径
        
    Returns:
        找到的修复后图片路径，如果不存在返回None
    """
    inpainted_path = get_inpainted_path(image_path, create_dir=False)
    if os.path.exists(inpainted_path):
        return inpainted_path
    
    return None


def find_txt_files(image_path: str) -> Tuple[Optional[str], Optional[str]]:
    """
    查找原文和翻译TXT文件
    
    Args:
        image_path: 原图片路径
        
    Returns:
        (原文TXT路径, 翻译TXT路径)，不存在的返回None
    """
    original_path = get_original_txt_path(image_path, create_dir=False)
    translated_path = get_translated_txt_path(image_path, create_dir=False)
    
    original_exists = original_path if os.path.exists(original_path) else None
    translated_exists = translated_path if os.path.exists(translated_path) else None
    
    # 向后兼容：查找旧格式的TXT文件
    if not translated_exists:
        old_txt_path = os.path.splitext(image_path)[0] + '_translations.txt'
        if os.path.exists(old_txt_path):
            translated_exists = old_txt_path
    
    return original_exists, translated_exists


def get_legacy_json_path(image_path: str) -> str:
    """
    获取旧版JSON文件路径（图片同目录）
    用于向后兼容
    
    Args:
        image_path: 原图片路径
        
    Returns:
        旧版JSON文件路径
    """
    return os.path.splitext(image_path)[0] + '_translations.json'


def migrate_legacy_files(image_path: str, move_files: bool = False) -> dict:
    """
    迁移旧版文件到新目录结构
    
    Args:
        image_path: 原图片路径
        move_files: 是否移动文件（True）还是复制文件（False）
        
    Returns:
        迁移结果字典，包含成功和失败的文件列表
    """
    import shutil
    
    result = {
        'success': [],
        'failed': [],
        'skipped': []
    }
    
    # 检查并迁移JSON文件
    old_json = get_legacy_json_path(image_path)
    if os.path.exists(old_json):
        new_json = get_json_path(image_path, create_dir=True)
        if not os.path.exists(new_json):
            try:
                if move_files:
                    shutil.move(old_json, new_json)
                else:
                    shutil.copy2(old_json, new_json)
                result['success'].append(('json', old_json, new_json))
            except Exception as e:
                result['failed'].append(('json', old_json, str(e)))
        else:
            result['skipped'].append(('json', old_json, 'target exists'))
    
    # 检查并迁移旧版TXT文件
    old_txt = os.path.splitext(image_path)[0] + '_translations.txt'
    if os.path.exists(old_txt):
        new_txt = get_translated_txt_path(image_path, create_dir=True)
        if not os.path.exists(new_txt):
            try:
                if move_files:
                    shutil.move(old_txt, new_txt)
                else:
                    shutil.copy2(old_txt, new_txt)
                result['success'].append(('txt', old_txt, new_txt))
            except Exception as e:
                result['failed'].append(('txt', old_txt, str(e)))
        else:
            result['skipped'].append(('txt', old_txt, 'target exists'))
    
    return result

