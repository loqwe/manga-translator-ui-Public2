# -*- coding: utf-8 -*-
"""manga_translator.utils.work_resolver

运行时作品识别与术语筛选工具。

用途：
- 根据“原图路径”推断作品名（通常为：.../<作品名>/<Chapter X>/xxx.jpg）
- 通过 name_mapping.json 将生肉名映射为熟肉名
- 在自定义 HQ prompt JSON 中，如果 glossary 采用 works 分作品结构，则按作品筛选出当前作品的 glossary

说明：
- 这里的“作品名”指熟肉名（名称映射后的名字），用于匹配 glossary["works"][work_name]。
- 若无法识别作品名，返回空术语表（不读取任何术语），但术语提取时写回 unassigned（未确定作品名区）。
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from .generic import BASE_PATH


_STANDARD_GLOSSARY_KEYS = ("Person", "Location", "Org", "Item", "Skill", "Creature")

# 这些目录名通常不是作品名（用于跳过/回退）
_INVALID_DIR_NAMES = {
    "manga_translator_work",
    "result",
    "json",
    "translations",
    "originals",
    "inpainted",
    "output",
    "input",
    "src",
    "source",
    "raw",
    "temp",
    "tmp",
    "cache",
    "images",
    "image",
    "imgs",
    "img",
    "",  # 保险：空名
    # Chinese invalid directory names (sync with term_manager.py)
    "未翻译",
    "原始",
    "输入",
    "输出",
    # Common top-level directory names (should not be recognized as work names)
    "漫画",
    "manga",
    "comic",
    "comics",
    "download",
    "downloads",
    "下载",
    "desktop",
    "桌面",
    "documents",
    "文档",
    "pictures",
    "图片",
}


def _is_chapter_like_dir(name: str) -> bool:
    """判断目录名是否更像“章节/卷/话”而非作品名。"""
    n = (name or "").strip()
    if not n:
        return False

    lower = n.lower()

    # 英文常见
    if re.search(r"\b(chapter|chap|ch\.?\s*\d+|episode|ep\.?\s*\d+|vol\.?\s*\d+|volume)\b", lower):
        return True

    # 中文/日文常见
    if re.search(r"^(第\s*)?\d+\s*(话|話|章|卷|回|集)$", n):
        return True

    # 纯数字目录（不少人用 001/002 表示章节）
    if re.fullmatch(r"\d{1,4}", n):
        return True

    # 包含明显的章节标志 + 数字
    if re.search(r"(话|話|章|卷|回|集)", n) and re.search(r"\d", n):
        return True

    return False


# --- name_mapping.json 读取缓存 ---
_NAME_MAPPING_PATH: Optional[Path] = None
_NAME_MAPPING_MTIME: Optional[float] = None
_NAME_MAPPING_DATA: Dict[str, str] = {}
# normalized index cache: norm_name -> canonical translated name
_NAME_MAPPING_INDEX: Dict[str, str] = {}


def _normalize_work_name_for_index(name: str) -> str:
    """Normalize work name for tolerant matching.

    Steps (kept conservative to avoid false positives):
    - lowercase
    - replace '_' and '-' with space
    - strip simple brackets
    - drop trailing token 'raw'
    - collapse spaces
    """
    if not name:
        return ""
    n = str(name).lower()
    n = n.replace("_", " ").replace("-", " ")
    n = re.sub(r"[\[\](){}`]+", " ", n)
    n = re.sub(r"\s*(?:raw)\s*$", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _build_name_mapping_index(mapping: Dict[str, str]) -> Dict[str, str]:
    index: Dict[str, str] = {}

    def add(alias: str, canonical: str):
        norm = _normalize_work_name_for_index(alias)
        if not norm:
            return
        if norm not in index:
            index[norm] = canonical

    for key, value in mapping.items():
        canonical = (str(value) if value is not None else "").strip()
        if not canonical:
            continue

        # canonical itself
        add(canonical, canonical)

        variants = [v.strip() for v in str(key).split("|") if v.strip()] or [str(key)]
        for alias in variants:
            add(alias, canonical)
            add(f"{alias} raw", canonical)
            add(f"{canonical} raw", canonical)

    return index


def _get_name_mapping_path() -> Optional[Path]:
    """定位 name_mapping.json（优先项目根目录，其次 examples/config）。"""
    base = Path(BASE_PATH)
    cand1 = base / "name_mapping.json"
    if cand1.exists():
        return cand1

    cand2 = base / "examples" / "config" / "name_mapping.json"
    if cand2.exists():
        return cand2

    return None


def load_name_mapping() -> Dict[str, str]:
    """Load name_mapping.json with mtime cache and build normalized index."""
    global _NAME_MAPPING_PATH, _NAME_MAPPING_MTIME, _NAME_MAPPING_DATA, _NAME_MAPPING_INDEX

    path = _get_name_mapping_path()
    if not path:
        _NAME_MAPPING_PATH = None
        _NAME_MAPPING_MTIME = None
        _NAME_MAPPING_DATA = {}
        _NAME_MAPPING_INDEX = {}
        return {}

    try:
        mtime = path.stat().st_mtime
        if _NAME_MAPPING_PATH == path and _NAME_MAPPING_MTIME == mtime and _NAME_MAPPING_DATA:
            return _NAME_MAPPING_DATA

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, dict):
            data = {}

        mapping = {str(k): str(v) for k, v in data.items() if k is not None and v is not None}

        _NAME_MAPPING_PATH = path
        _NAME_MAPPING_MTIME = mtime
        _NAME_MAPPING_DATA = mapping
        _NAME_MAPPING_INDEX = _build_name_mapping_index(mapping)
        return _NAME_MAPPING_DATA

    except Exception:
        _NAME_MAPPING_PATH = path
        _NAME_MAPPING_MTIME = None
        _NAME_MAPPING_DATA = {}
        _NAME_MAPPING_INDEX = {}
        return {}


def _ensure_index_loaded():
    global _NAME_MAPPING_INDEX
    if _NAME_MAPPING_INDEX:
        return
    load_name_mapping()


def load_name_mapping_index() -> Dict[str, str]:
    _ensure_index_loaded()
    return _NAME_MAPPING_INDEX


def map_translated_work_name(raw_name: str, mapping: Dict[str, str]) -> str:
    """Map raw work name to translated name with normalized index support."""
    raw = (raw_name or "").strip()
    if not raw:
        return raw_name

    idx = load_name_mapping_index()
    norm = _normalize_work_name_for_index(raw)
    if norm and norm in idx:
        return idx[norm]

    if raw in mapping:
        return mapping[raw]

    for key, value in mapping.items():
        if "|" in key:
            variants = [v.strip() for v in key.split("|") if v.strip()]
            if raw in variants:
                return value

    return raw


def try_map_translated_work_name(raw_name: str, mapping: Dict[str, str]) -> Optional[str]:
    """Try mapping raw work name to translated name using index; return None on miss."""
    raw = (raw_name or "").strip()
    if not raw:
        return None

    idx = load_name_mapping_index()
    norm = _normalize_work_name_for_index(raw)
    if norm and norm in idx:
        mapped = idx.get(norm, "").strip()
        return mapped or None

    if raw in mapping:
        v = (mapping.get(raw) or "").strip()
        return v or None

    for key, value in mapping.items():
        if "|" not in str(key):
            continue
        variants = [v.strip() for v in str(key).split("|") if v.strip()]
        if raw in variants:
            mapped = (str(value) if value is not None else "").strip()
            return mapped or None

    return None


def resolve_translated_work_name_from_raw(raw_name: str) -> Optional[str]:
    """Resolve translated work name from raw name, preferring normalized index."""
    raw = (raw_name or "").strip()
    if not raw:
        return None

    mapping = load_name_mapping()

    idx = load_name_mapping_index()
    norm = _normalize_work_name_for_index(raw)
    if norm and norm in idx:
        return idx[norm]

    mapped = try_map_translated_work_name(raw, mapping)
    if mapped:
        return mapped

    try:
        if raw in set(mapping.values()):
            return raw
    except Exception:
        pass

    return None


def infer_raw_work_name_from_image_path(image_path: str) -> Optional[str]:
    """从原图路径推断“生肉作品名”。

    常见结构：.../<作品名>/<Chapter X>/xxx.jpg
    - 若图片目录像章节目录，则返回其父目录名
    - 否则返回图片目录名

    同时会跳过一些明显不可能是作品名的目录（manga_translator_work/result 等）。
    """
    if not image_path:
        return None

    try:
        p = Path(image_path)
        # strict=False 避免文件不存在时报错
        p = p.resolve(strict=False)
    except Exception:
        p = Path(image_path)

    cur = p.parent

    # 先跳过一些明显的处理目录
    for _ in range(8):
        name = (cur.name or "").strip()
        if not name:
            break
        if name.lower() in _INVALID_DIR_NAMES:
            if cur.parent == cur:
                break
            cur = cur.parent
            continue
        break

    # 如果当前目录更像章节目录，则使用父目录作为作品名
    if _is_chapter_like_dir(cur.name) and cur.parent != cur:
        cur = cur.parent

    # 再次清理（避免 parent 也是无效目录）
    for _ in range(3):
        name = (cur.name or "").strip()
        if not name:
            break
        if name.lower() in _INVALID_DIR_NAMES or _is_chapter_like_dir(name):
            if cur.parent == cur:
                break
            cur = cur.parent
            continue
        break

    return (cur.name or "").strip() or None


def get_source_path_from_translation_map(image_path: str) -> Optional[str]:
    """从 translation_map.json 获取原图路径。

    当打开的是翻译后的图片时，可以通过此函数追溯原图路径，
    从而正确识别作品名。

    Args:
        image_path: 图片路径（可能是翻译后的图片）

    Returns:
        原图路径，如果找不到则返回 None
    """
    if not image_path:
        return None

    try:
        p = Path(image_path).resolve(strict=False)
        image_dir = p.parent

        # 在图片所在目录查找 translation_map.json
        map_path = image_dir / "translation_map.json"
        if not map_path.exists():
            return None

        with map_path.open("r", encoding="utf-8") as f:
            translation_map = json.load(f)

        if not isinstance(translation_map, dict):
            return None

        # 规范化当前图片路径
        image_path_norm = os.path.normpath(str(p))

        # 查找映射（translated_path -> source_path）
        source_path = translation_map.get(image_path_norm)
        if source_path:
            return source_path

        # 尝试不同的路径格式匹配
        for translated_path, src_path in translation_map.items():
            if os.path.normpath(translated_path) == image_path_norm:
                return src_path

    except Exception:
        pass

    return None


def get_source_path_from_source_path_txt(image_path: str, max_depth: int = 5) -> Optional[str]:
    """从 _source_path.txt 获取原图目录或原图路径（作为 translation_map 的降级方案）。

    约定：
    - _source_path.txt 放在输出目录（通常是 Chapter 目录）
    - 文件内容建议写原图所在目录（例如：.../<作品名>/Chapter 33）

    Args:
        image_path: 当前图片路径（通常是翻译后图片）
        max_depth: 向上查找层级

    Returns:
        _source_path.txt 中记录的路径（可能是目录或文件路径），找不到则返回 None
    """
    if not image_path:
        return None

    try:
        p = Path(image_path).resolve(strict=False)
    except Exception:
        p = Path(image_path)

    start_dir = p.parent
    parents = [start_dir] + list(start_dir.parents)[:max_depth]

    for parent in parents:
        fp = parent / "_source_path.txt"
        if not fp.exists():
            continue
        try:
            content = fp.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not content:
            continue

        # 只取第一行，避免用户写了多行说明
        line = (content.splitlines()[0] or "").strip().strip('"').strip("'")
        return line or None

    return None


def _normalize_source_hint_to_image_path(source_hint: str, image_path: str) -> Optional[str]:
    """把 _source_path.txt 里记录的“目录/文件”归一成“像图片文件的路径”。

    - 如果 source_hint 是目录（或无扩展名），拼接当前图片的文件名
    - 如果 source_hint 是文件路径（有扩展名），原样返回
    """
    hint = (source_hint or "").strip()
    if not hint:
        return None

    try:
        hint_path = Path(hint)
    except Exception:
        return hint

    # 没有扩展名：按目录处理，拼接当前图片文件名
    if not hint_path.suffix:
        try:
            name = Path(image_path).name
        except Exception:
            name = ""
        if not name:
            name = "__source__.png"
        return str(hint_path / name)

    return str(hint_path)


def resolve_source_image_path(image_path: str) -> Optional[str]:
    """尽最大努力把“当前图片路径”解析为“原图路径”。

    优先级：translation_map.json > _source_path.txt
    """
    if not image_path:
        return None

    src = get_source_path_from_translation_map(image_path)
    if src:
        return src

    hint = get_source_path_from_source_path_txt(image_path)
    if hint:
        return _normalize_source_hint_to_image_path(hint, image_path)

    return None


# Fallback value when raw work name cannot be determined
NO_RAW_WORK_NAME = "无生肉名"


def resolve_raw_work_name(image_path: str) -> Optional[str]:
    """解析生肉作品名。

    对翻译后图片：优先通过 translation_map.json 或 _source_path.txt 追溯原图路径，再提取生肉名。
    若无法识别，返回 "无生肉名"。
    """
    if not image_path:
        return NO_RAW_WORK_NAME

    src_path = resolve_source_image_path(image_path) or image_path
    result = infer_raw_work_name_from_image_path(src_path)
    return result if result else NO_RAW_WORK_NAME


def resolve_work_name(image_path: str) -> Optional[str]:
    """从图片路径解析“熟肉作品名”。

    策略：
    1. 先尝试从 translation_map.json 或 _source_path.txt 追溯原图路径（适用于翻译后的图片）
    2. 再从路径推断生肉作品名
    3. 最后通过 name_mapping.json 映射为熟肉名
    """
    if not image_path:
        return None

    source_path = resolve_source_image_path(image_path)
    if source_path:
        image_path = source_path

    raw = infer_raw_work_name_from_image_path(image_path)
    if not raw:
        return None

    mapping = load_name_mapping()
    return map_translated_work_name(raw, mapping) or raw


def filter_custom_prompt_json_by_work(
    custom_prompt_json: Any,
    work_name: Optional[str],
    raw_work_name: Optional[str] = None
) -> Tuple[Any, Optional[str]]:
    """如果 custom_prompt_json 中 glossary 采用 works 分作品结构，则按作品筛选。

    策略：
    1. 若能识别作品名且存在于 works 中，则返回该作品的术语
    2. 若无法识别作品，则从 unassigned 中按 raw_work_name 匹配术语
    3. 若仍无匹配，返回空术语表

    Args:
        custom_prompt_json: 提示词 JSON
        work_name: 熟肉作品名（经过 name_mapping 映射后）
        raw_work_name: 生肉作品名（从路径直接提取，用于 unassigned 匹配）

    Returns:
        (new_prompt_json, selected_work_name)
        - 若不满足 works 结构，返回原对象和 None
        - 若无法识别作品，返回空术语表和 None
    """
    if not isinstance(custom_prompt_json, dict):
        return custom_prompt_json, None

    glossary = custom_prompt_json.get("glossary")
    if not isinstance(glossary, dict):
        return custom_prompt_json, None

    works = glossary.get("works")
    if not isinstance(works, dict):
        return custom_prompt_json, None

    # 1. 尝试从 works 中查找熟肉名匹配
    selected_work = None
    if work_name and work_name in works:
        selected_work = work_name

    # 2. 若 works 中找不到，尝试从 unassigned 中按 raw_work_name 匹配
    # Skip if raw_work_name is the fallback value (no valid raw name)
    if not selected_work and raw_work_name and raw_work_name != NO_RAW_WORK_NAME:
        unassigned = glossary.get("unassigned")
        if isinstance(unassigned, dict):
            # 从 unassigned 中筛选 raw_work_name 匹配的术语
            filtered_glossary = {k: [] for k in _STANDARD_GLOSSARY_KEYS}
            has_match = False
            for cat in _STANDARD_GLOSSARY_KEYS:
                terms = unassigned.get(cat, [])
                if not isinstance(terms, list):
                    continue
                for term in terms:
                    if not isinstance(term, dict):
                        continue
                    term_raw_name = (term.get("raw_work_name") or "").strip()
                    # 匹配 raw_work_name
                    if term_raw_name and term_raw_name == raw_work_name:
                        filtered_glossary[cat].append(term)
                        has_match = True
            
            if has_match:
                new_prompt = dict(custom_prompt_json)
                new_prompt["glossary"] = filtered_glossary
                # 记录来源为 unassigned
                tg = new_prompt.get("terminology_guide")
                if not isinstance(tg, dict):
                    tg = {}
                tg["work_name"] = f"[unassigned:{raw_work_name}]"
                new_prompt["terminology_guide"] = tg
                return new_prompt, None  # selected_work 仍为 None，表示未确定作品

    # 3. 无法识别作品时，返回空术语表
    if not selected_work:
        empty_glossary = {k: [] for k in _STANDARD_GLOSSARY_KEYS}
        new_prompt = dict(custom_prompt_json)
        new_prompt["glossary"] = empty_glossary
        return new_prompt, None

    work_glossary = works.get(selected_work)
    if not isinstance(work_glossary, dict):
        work_glossary = {}

    filtered_glossary = {k: work_glossary.get(k, []) for k in _STANDARD_GLOSSARY_KEYS}

    # 只在运行时“瘦身” glossary，避免把所有作品术语都塞进提示词
    new_prompt = dict(custom_prompt_json)
    new_prompt["glossary"] = filtered_glossary

    # 额外提供作品名给模型（帮助其理解当前术语表适用范围）
    tg = new_prompt.get("terminology_guide")
    if not isinstance(tg, dict):
        tg = {}
    tg["work_name"] = selected_work
    new_prompt["terminology_guide"] = tg

    return new_prompt, selected_work


def merge_custom_prompt_json_by_works(
    custom_prompt_json: Any,
    work_names: list[str],
    include_default: bool = True,
) -> Tuple[Any, Optional[list[str]]]:
    """混作品场景：把多个作品的术语合并成一份扁平 glossary。

    注意：
    - 这里的合并只发生在运行时，不会修改原始 JSON 文件。
    - 去重粒度为 (category, original, translation)。
    - 若同一 original 在不同作品下有不同 translation，会同时保留（可能产生冲突，这是“混作品合并术语”的固有代价）。

    Returns:
        (new_prompt_json, selected_work_names)
        - 若不满足 works 结构或无可合并作品，返回原对象和 None
    """
    if not isinstance(custom_prompt_json, dict):
        return custom_prompt_json, None

    glossary = custom_prompt_json.get("glossary")
    if not isinstance(glossary, dict):
        return custom_prompt_json, None

    works = glossary.get("works")
    if not isinstance(works, dict):
        return custom_prompt_json, None

    selected: list[str] = []

    # 默认作品先合并（如果需要）
    if include_default:
        if "默认" in works:
            selected.append("默认")
        elif "default" in works:
            selected.append("default")

    # 再合并本批次识别到的作品（保持稳定顺序）
    for wn in sorted({(w or "").strip() for w in (work_names or []) if isinstance(w, str)}):
        if wn and wn in works and wn not in selected:
            selected.append(wn)

    if not selected:
        return custom_prompt_json, None

    merged_glossary = {k: [] for k in _STANDARD_GLOSSARY_KEYS}
    seen = set()

    for wn in selected:
        entry = works.get(wn)
        if not isinstance(entry, dict):
            continue

        for cat in _STANDARD_GLOSSARY_KEYS:
            term_list = entry.get(cat)
            if not isinstance(term_list, list):
                continue

            for term in term_list:
                if not isinstance(term, dict):
                    continue
                original = (term.get("original") or "").strip()
                translation = (term.get("translation") or "").strip()
                if not original or not translation:
                    continue

                key = (cat, original, translation)
                if key in seen:
                    continue
                seen.add(key)

                merged_glossary[cat].append({
                    "original": original,
                    "translation": translation,
                })

    # 输出扁平 glossary，避免把 works 整个塞进提示词
    new_prompt = dict(custom_prompt_json)
    new_prompt["glossary"] = merged_glossary

    # terminology_guide: 记录这是“混合作品”术语（纯提示用途）
    tg = new_prompt.get("terminology_guide")
    if not isinstance(tg, dict):
        tg = {}
    tg["work_name"] = " + ".join(selected)
    new_prompt["terminology_guide"] = tg

    return new_prompt, selected
