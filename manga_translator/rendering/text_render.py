import os
import re
import cv2
import numpy as np
import freetype
import functools
import logging
from pathlib import Path
from typing import Tuple, Optional, List
from hyphen import Hyphenator
from hyphen.dictools import LANGUAGES as HYPHENATOR_LANGUAGES
from langcodes import standardize_tag

from ..utils import BASE_PATH, is_punctuation, is_whitespace, imwrite_unicode

try:
    HYPHENATOR_LANGUAGES.remove('fr')
    HYPHENATOR_LANGUAGES.append('fr_FR')
except Exception:
    pass

CJK_H2V = {
    "‥": "︰",
    "—": "︱",
    "―": "|",
    "–": "︲",
    "_": "︳",
    "_": "︴",
    "(": "︵",
    ")": "︶",
    "（": "︵",
    "）": "︶",
    "{": "︷",
    "}": "︸",
    "〔": "︹",
    "〕": "︺",
    "【": "︻",
    "】": "︼",
    "《": "︽",
    "》": "︾",
    "〈": "︿",
    "〉": "﹀",
    "⟨": "︿",   
    "⟩": "﹀",   
    "⟪": "︿",   
    "⟫": "﹀",       
    "「": "﹁",
    "」": "﹂",
    "『": "﹃",
    "』": "﹄",
    "﹑": "﹅",
    "﹆": "﹆",
    "[": "﹇",
    "]": "﹈",
    "⦅": "︵",   
    "⦆": "︶",   
    "❨": "︵",          
    "❩": "︶",   
    "❪": "︷",   
    "❫": "︸",   
    "❬": "﹇",   
    "❭": "﹈",   
    "❮": "︿",   
    "❯": "﹀",    
    "﹉": "﹉",
    "﹊": "﹊",
    "﹋": "﹋",
    "﹌": "﹌",
    "﹍": "﹍",
    "﹎": "﹎",
    "﹏": "﹏",
    "…": "⋮",
    "⋯": "︙", 
    "⋰": "⋮",    
    "⋱": "⋮",           
    "\"": "﹁",   
    "\"": "﹂",   
    "'": "﹁",   
    "'": "﹂",   
    "″": "﹂",   
    "‴": "﹂",   
    "‶": "﹁",   
    "ⷷ": "﹁",   
    "~": "︴",   
    "〜": "︴",   
    "～": "︴",   
    "~": "≀",
    "〰": "︴",
    "!": "︕",    
    "?": "︖",    
    "؟": "︖",    
    "¿": "︖",    
    "¡": "︕",    
    ".": "︒",    
    "。": "︒",   
    ";": "︔",    
    "；": "︔",   
    ":": "︓",    
    "：": "︓",  
    ",": "︐",    
    "，": "︐",   
    # "､": "︐",    
    "‚": "︐",    
    "„": "︐",    
    # "、": "︑",    
    "-": "︲",    
    "−": "︲",
    "・": "·",
}

CJK_V2H = {
    **{v: k for k, v in CJK_H2V.items()},
}

logger = logging.getLogger(__name__)  
logger.addHandler(logging.NullHandler())  

DEFAULT_FONT = os.path.join(BASE_PATH, 'fonts', 'Arial-Unicode-Regular.ttf')
try:
    FONT = freetype.Face(Path(DEFAULT_FONT).open('rb'))
except Exception as e:
    logger.error(f"Failed to initialize default font: {e}")
    FONT = None  

def CJK_Compatibility_Forms_translate(cdpt: str, direction: int):
    """direction: 0 - horizontal, 1 - vertical"""
    
    if cdpt == 'ー' and direction == 1:
        return 'ー', 90
    if cdpt in CJK_V2H:
        if direction == 0:
            # translate
            return CJK_V2H[cdpt], 0
        else:
            return cdpt, 0
    elif cdpt in CJK_H2V:
        if direction == 1:
            # translate
            return CJK_H2V[cdpt], 0
        else:
            return cdpt, 0
    return cdpt, 0

def compact_special_symbols(text: str) -> str:
    # 替换半角省略号
    text = text.replace('...', '…')
    text = text.replace('..', '…')
    # 仅合并 3 个及以上连续省略号，保留标准六点省略号（两个“…“）
    text = re.sub(r'…{3,}', '……', text)
    # 将西文省略号(U+2026,贴底)替换为居中省略号(U+22EF)，解决横排省略号位置偏下的问题
    text = text.replace('…', '⋯')
    # Remove half-width and full-width spaces after each punctuation mark
    # 只删除标点符号后的空格，不删除字母/数字后的空格
    # 匹配常见的标点符号：。，、！？；：…等
    pattern = r'([。，、！？；：…—～「」『』【】（）《》〈〉.,!?;:\-])[ 　]+'
    text = re.sub(pattern, r'\1', text)
    return text

def should_force_wrap_for_mismatched_direction(config, logger) -> bool:
    """
    检测是否应该强制打开换行：
    如果自动算法检测到是竖排（检测框宽高比 < 1）但被强制设置成横排，返回 True
    如果自动算法检测到是横排（检测框宽高比 >= 1）但被强制设置成竖排，返回 True
    
    Args:
        config: 配置对象，需要包含 _current_region 属性
        logger: 日志对象
    
    Returns:
        bool: 是否应该强制打开换行
    """
    if not config or not hasattr(config, '_current_region'):
        return False
    
    region = config._current_region
    
    # 检查是否被强制设置了方向
    if not hasattr(region, '_direction') or region._direction not in ['h', 'horizontal', 'v', 'vertical']:
        return False
    
    forced_horizontal = region._direction in ['h', 'horizontal']
    
    # 检查自动检测的方向
    if not hasattr(region, 'lines') or len(region.lines) == 0:
        return False
    
    # 计算面积最大的检测框的宽高比
    max_area = 0
    largest_box_aspect_ratio = 1
    
    try:
        from shapely.geometry import Polygon
        for line in region.lines:
            line_polygon = Polygon(line)
            area = line_polygon.area
            if area > max_area:
                max_area = area
                x_coords = line[:, 0]
                y_coords = line[:, 1]
                line_width = np.max(x_coords) - np.min(x_coords)
                line_height = np.max(y_coords) - np.min(y_coords)
                largest_box_aspect_ratio = line_width / line_height if line_height > 0 else 1
    except Exception as e:
        logger.warning(f"检测方向不匹配时出错: {e}")
        return False
    
    auto_detected_horizontal = largest_box_aspect_ratio >= 1
    
    # 如果强制方向与自动检测方向不一致，返回 True
    if forced_horizontal != auto_detected_horizontal:
        direction_name = "横排" if forced_horizontal else "竖排"
        auto_direction_name = "横排" if auto_detected_horizontal else "竖排"
        logger.debug(f"[方向不匹配] 自动检测为{auto_direction_name}(宽高比={largest_box_aspect_ratio:.2f})但被强制设置为{direction_name}，打开换行")
        return True
    
    return False

def auto_add_horizontal_tags(text: str) -> str:
    """自动为竖排文本中的短英文单词或连续符号添加<H>标签，使其横向显示。

    处理规则：
    - 整段文本统一处理，不按 [BR]/\n 分段
    - 多词英文词组（如 "Tek Tok"）：整体横排显示
    - 独立的短英文单词：添加<H>标签
    - 符号（!?）2-3个：横排显示，4个以上不包裹

    渲染规则：
    - 仅字母/数字且长度>=3：旋转90度显示
    - 其他情况（含符号）：保持横排显示
    """
    # 如果文本中已有<H>标签，则不进行处理，以尊重手动设置
    if '<H>' in text or '<h>' in text.lower():
        return text

    # 先保护 BR 标记与换行，避免被步骤1/2直接匹配吞掉。
    br_pattern = re.compile(r'(\[BR\]|<br\s*/?>|【BR】|\r\n|\r|\n)', flags=re.IGNORECASE)
    br_tokens = []

    def _mask_br(match):
        idx = len(br_tokens)
        br_tokens.append(match.group(0))
        # 使用不会被英文/数字正则命中的私有区字符占位
        return chr(0xE000 + idx)

    seg = br_pattern.sub(_mask_br, text)

    # 步骤1：为多词英文词组添加<H>标签（至少2个单词，用空格分隔）
    multi_word_pattern = r'[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+(?:\s+[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+)+'
    seg = re.sub(multi_word_pattern, r'<H>\g<0></H>', seg)

    # 步骤2：对剩余的独立英文单词添加<H>标签
    word_pattern = r'(?<![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]{2,})(?![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])'

    # 用闭包捕获当前 seg 的值
    def _make_replace_word(current_seg):
        def replace_word(match):
            start_pos = match.start()
            text_before = current_seg[:start_pos]
            last_open = text_before.rfind('<H>')
            last_close = text_before.rfind('</H>')
            if last_open > last_close:
                return match.group(0)
            return f'<H>{match.group(1)}</H>'
        return replace_word

    seg = re.sub(word_pattern, _make_replace_word(seg), seg)

    # 步骤3：匹配符号（2-4个，同时支持半角和全角，5个以上不包裹）
    symbol_pattern = r'[!?！？]{2,4}'
    seg = re.sub(symbol_pattern, r'<H>\g<0></H>', seg)

    # 还原 BR/换行 标记
    for idx, original in enumerate(br_tokens):
        seg = seg.replace(chr(0xE000 + idx), original)

    # 跨 BR/换行的纯符号块一律不打标（两边都去掉 <H>）
    # 例：
    # - <H>!?</H>[BR]<H>!?</H> -> !?[BR]!?
    # - <H>!!!</H>[BR]<H>!!</H> -> !!![BR]!!
    symbol_pair_on_br_pattern = re.compile(
        r'<H>([!?！？]{2,4})</H>\s*(\r\n|\r|\n|\[BR\]|<br\s*/?>|【BR】)\s*<H>([!?！？]{2,4})</H>',
        flags=re.IGNORECASE
    )

    def _unwrap_symbol_pair(match):
        left, sep, right = match.group(1), match.group(2), match.group(3)
        return f"{left}{sep}{right}"

    while True:
        updated = symbol_pair_on_br_pattern.sub(_unwrap_symbol_pair, seg)
        if updated == seg:
            break
        seg = updated

    # 将被 BR/换行分隔的纯字母数字横排块合并为一个块：
    # <H>abc</H>[BR]<H>def</H> -> <H>abc[BR]def</H>
    merge_pattern = re.compile(
        r'<H>([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19]+)</H>\s*(?:\r\n|\r|\n|\[BR\]|<br\s*/?>|【BR】)\s*<H>([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19]+)</H>',
        flags=re.IGNORECASE
    )
    while True:
        merged = merge_pattern.sub(r'<H>\1[BR]\2</H>', seg)
        if merged == seg:
            break
        seg = merged

    # 跨 BR/换行的单字母/数字对也打成一个横排块：
    # a[BR]c -> <H>a[BR]c</H>
    single_pair_pattern = re.compile(
        r'(?<![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19])'
        r'([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19])'
        r'\s*(?:\r\n|\r|\n|\[BR\]|<br\s*/?>|【BR】)\s*'
        r'([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19])'
        r'(?![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19])',
        flags=re.IGNORECASE
    )
    seg = single_pair_pattern.sub(r'<H>\1[BR]\2</H>', seg)

    # 横排块内部若有换行，统一规范为 [BR]
    seg = re.sub(
        r'<H>(.*?)</H>',
        lambda m: (
            "<H>"
            + m.group(1).replace('\r\n', '[BR]').replace('\r', '[BR]').replace('\n', '[BR]')
            + "</H>"
        ),
        seg,
        flags=re.IGNORECASE | re.DOTALL
    )

    return seg


def _normalize_horizontal_block_content(content: str) -> str:
    """Normalize content inside <H>...</H> for rotation/measurement/rendering."""
    if not content:
        return ""
    content = content.replace('__ROT90__', '')
    content = re.sub(r'\s*(\[BR\]|<br\s*/?>|【BR】)\s*', '', content, flags=re.IGNORECASE)
    content = content.replace('\r', '').replace('\n', '')
    return content

def _convert_br_outside_h_tags(text: str) -> str:
    """Convert BR markers to '\\n' outside <H>, and split BR inside <H> into per-line <H> blocks."""
    parts = re.split(r'(<H>.*?</H>)', text, flags=re.IGNORECASE | re.DOTALL)
    converted = []
    for part in parts:
        if not part:
            continue
        is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
        if is_horizontal_block:
            content = part[3:-4]
            # BR inside one <H> block: split into multiple lines
            chunks = re.split(r'\s*(?:\[BR\]|<br\s*/?>|【BR】|\r\n|\r|\n)\s*', content, flags=re.IGNORECASE)
            chunks = [c for c in chunks if c]
            if len(chunks) <= 1:
                converted.append(part)
                continue

            full_content = _normalize_horizontal_block_content(content)
            # If combined alnum length across BR reaches 2, force each split line to rotate.
            force_rotate = bool(re.fullmatch(r'[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19]+', full_content)) and len(full_content) >= 2

            split_blocks = []
            for chunk in chunks:
                chunk_clean = _normalize_horizontal_block_content(chunk)
                if not chunk_clean:
                    continue
                if force_rotate:
                    chunk_clean = '__ROT90__' + chunk_clean
                split_blocks.append(f'<H>{chunk_clean}</H>')
            converted.append('\n'.join(split_blocks))
        else:
            converted.append(re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', part, flags=re.IGNORECASE))
    return ''.join(converted)

def should_rotate_horizontal_block_90(content: str) -> bool:
    """Only rotate 90 degrees for alphanumeric blocks in vertical layout."""
    if not content:
        return False
    force_rotate = '__ROT90__' in content
    content = _normalize_horizontal_block_content(content)
    if not content:
        return False
    if force_rotate:
        return True
    alnum_only = re.fullmatch(r'[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19]+', content)
    return bool(alnum_only) and len(content) >= 3
    
def rotate_image(image, angle):
    if angle == 0:
        return image, (0, 0)
    image_exp = np.zeros((round(image.shape[0] * 1.5), round(image.shape[1] * 1.5), image.shape[2]), dtype = np.uint8)
    diff_i = (image_exp.shape[0] - image.shape[0]) // 2
    diff_j = (image_exp.shape[1] - image.shape[1]) // 2
    image_exp[diff_i:diff_i+image.shape[0], diff_j:diff_j+image.shape[1]] = image
    # from https://stackoverflow.com/questions/9041681/opencv-python-rotate-image-by-x-degrees-around-specific-point
    image_center = tuple(np.array(image_exp.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image_exp, rot_mat, image_exp.shape[1::-1], flags=cv2.INTER_LINEAR)
    if angle == 90:
        return result, (0, 0)
    return result, (diff_i, diff_j)

def add_color(bw_char_map, color, stroke_char_map, stroke_color):
    if bw_char_map.size == 0:
        fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype = np.uint8)
        return fg
    
    if stroke_color is None :
        x, y, w, h = cv2.boundingRect(bw_char_map)
    else :
        x, y, w, h = cv2.boundingRect(stroke_char_map)

    # 检查 boundingRect 返回的尺寸是否有效
    if w == 0 or h == 0:
        fg = np.zeros((bw_char_map.shape[0], bw_char_map.shape[1], 4), dtype = np.uint8)
        return fg

    fg = np.zeros((h, w, 4), dtype = np.uint8)
    fg[:,:,0] = color[0]
    fg[:,:,1] = color[1]
    fg[:,:,2] = color[2]
    fg[:,:,3] = bw_char_map[y:y+h, x:x+w]

    if stroke_color is None :
        stroke_color = color
    bg = np.zeros((stroke_char_map.shape[0], stroke_char_map.shape[1], 4), dtype = np.uint8)
    bg[:,:,0] = stroke_color[0]
    bg[:,:,1] = stroke_color[1]
    bg[:,:,2] = stroke_color[2]
    bg[:,:,3] = stroke_char_map

    fg_alpha = fg[:, :, 3] / 255.0
    bg_alpha = 1.0 - fg_alpha
    bg[y:y+h, x:x+w, :] = (fg_alpha[:, :, np.newaxis] * fg[:, :, :] + bg_alpha[:, :, np.newaxis] * bg[y:y+h, x:x+w, :])

    return bg

FALLBACK_FONTS = [
    os.path.join(BASE_PATH, 'fonts/Arial-Unicode-Regular.ttf'),
    os.path.join(BASE_PATH, 'fonts/msyh.ttc'),
    os.path.join(BASE_PATH, 'fonts/msgothic.ttc'),
    r'C:\Windows\Fonts\seguisym.ttf',
]
FONT_SELECTION: List[freetype.Face] = []
font_cache = {}
_font_file_handles = {}  # 保存文件句柄，防止被垃圾回收

def get_cached_font(path: str) -> freetype.Face:
    path = path.replace('\\', '/')
    if not font_cache.get(path):
        # 保存文件句柄引用，防止被关闭
        file_handle = Path(path).open('rb')
        _font_file_handles[path] = file_handle
        font_cache[path] = freetype.Face(file_handle)
    return font_cache[path]

def update_font_selection():
    global FONT_SELECTION
    FONT_SELECTION = []
    if FONT:
        FONT_SELECTION.append(FONT)
    for font_path in FALLBACK_FONTS:
        try:
            face = get_cached_font(font_path)
            if face and face not in FONT_SELECTION:
                FONT_SELECTION.append(face)
        except Exception as e:
            logger.error(f"Failed to load fallback font: {font_path} - {e}")


def set_font(path: str):
    global FONT
    
    # 处理相对路径：尝试在 BASE_PATH 下查找
    resolved_path = path
    if path and not os.path.isabs(path) and not os.path.exists(path):
        # 尝试 BASE_PATH/fonts/filename 或 BASE_PATH/path
        possible_paths = [
            os.path.join(BASE_PATH, 'fonts', os.path.basename(path)),
            os.path.join(BASE_PATH, path),
        ]
        for p in possible_paths:
            if os.path.exists(p):
                resolved_path = p
                break
    
    if not resolved_path or not os.path.exists(resolved_path):
        if path:
            logger.error(f'Could not load font: {path}')
        try:
            FONT = get_cached_font(DEFAULT_FONT)
        except Exception:
            logger.critical("Default font could not be loaded. Please check your installation.")
            FONT = None
        update_font_selection()
        get_char_glyph.cache_clear()
        return

    try:
        FONT = get_cached_font(resolved_path)
    except Exception:
        logger.error(f'Could not load font: {resolved_path}')
        try:
            FONT = get_cached_font(DEFAULT_FONT)
        except Exception:
            logger.critical("Default font could not be loaded. Please check your installation.")
            FONT = None
    update_font_selection()
    get_char_glyph.cache_clear()

class namespace:
    pass

class Glyph:
    def __init__(self, glyph):
        self.bitmap = namespace()
        self.bitmap.buffer = glyph.bitmap.buffer
        self.bitmap.rows = glyph.bitmap.rows
        self.bitmap.width = glyph.bitmap.width
        self.advance = namespace()
        self.advance.x = glyph.advance.x
        self.advance.y = glyph.advance.y
        self.bitmap_left = glyph.bitmap_left
        self.bitmap_top = glyph.bitmap_top
        self.metrics = namespace()
        self.metrics.vertBearingX = glyph.metrics.vertBearingX
        self.metrics.vertBearingY = glyph.metrics.vertBearingY
        self.metrics.horiBearingX = glyph.metrics.horiBearingX
        self.metrics.horiBearingY = glyph.metrics.horiBearingY
        self.metrics.horiAdvance = glyph.metrics.horiAdvance
        self.metrics.vertAdvance = glyph.metrics.vertAdvance

@functools.lru_cache(maxsize = 1024, typed = True)
def get_char_glyph(cdpt: str, font_size: int, direction: int) -> Glyph:
    global FONT_SELECTION
    for i, face in enumerate(FONT_SELECTION):
        char_index = face.get_char_index(cdpt)
        if char_index != 0:
            # Character found, load and return glyph
            if direction == 0:
                face.set_pixel_sizes(0, font_size)
            elif direction == 1:
                face.set_pixel_sizes(font_size, 0)
            face.load_char(cdpt)
            return Glyph(face.glyph)
        
        # Log fallback attempt only on the primary font for clarity
        if i == 0:
            try:
                font_name = face.family_name.decode('utf-8') if face.family_name else 'Unknown'
                logger.debug(f"Character '{cdpt}' not found in primary font '{font_name}'. Trying fallbacks.")
            except Exception:
                pass # Avoid logging errors within logging

    # If the loop completes, the character was not found in any font.
    logger.error(f"FATAL: Character '{cdpt}' (U+{ord(cdpt):04X}) not found in any of the available fonts. Substituting with a placeholder.")
    
    # To prevent a crash, recursively call with a placeholder that is guaranteed to exist.
    # Use '?' as placeholder instead of space - it's visible and indicates missing character
    # Avoid infinite recursion if placeholder itself is not found
    if cdpt in (' ', '?', '□'):
        # This should never happen with valid fonts, but as a safeguard:
        # We can't return a glyph, so we must raise an exception.
        raise RuntimeError(f"Catastrophic failure: Placeholder character '{cdpt}' not found in any font.")
    
    # Try '?' first (visible placeholder), then '□' (replacement character), then space
    for placeholder in ('?', '□', ' '):
        if placeholder != cdpt:
            try:
                return get_char_glyph(placeholder, font_size, direction)
            except RuntimeError:
                continue
    
    # Last resort - should never reach here
    raise RuntimeError("Catastrophic failure: No placeholder character found in any font.")

# 缓存 glyph border 对象，避免重复创建导致内存泄漏
# 注意：由于 stroke() 会修改 glyph，这里不缓存 glyph，而是直接返回新对象
# 真正的优化在 _stroke_border_cache 中缓存最终的 bitmap 结果
#@functools.lru_cache(maxsize = 1024, typed = True)
def get_char_border(cdpt: str, font_size: int, direction: int):
    global FONT_SELECTION
    for i, face in enumerate(FONT_SELECTION):
        if face.get_char_index(cdpt) == 0 and i != len(FONT_SELECTION) - 1:
            continue
        if direction == 0:
            face.set_pixel_sizes(0, font_size)
        elif direction == 1:
            face.set_pixel_sizes(font_size, 0)
        face.load_char(cdpt, freetype.FT_LOAD_DEFAULT | freetype.FT_LOAD_NO_BITMAP)
        slot_border = face.glyph
        return slot_border.get_glyph()

def _is_vertical_ellipsis_char(cdpt: str) -> bool:
    # Support all common ellipsis forms that may appear in vertical flow.
    return cdpt in ('︙', '⋯', '…')

def _estimate_ellipsis_center_gap(bitmap_char: np.ndarray) -> Optional[float]:
    """Estimate center-to-center vertical gap of the 3 dots inside one vertical ellipsis glyph."""
    if bitmap_char is None or bitmap_char.size == 0:
        return None
    mask = (bitmap_char > 0).astype(np.uint8)
    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return None
    # Skip label 0 (background), keep visible components.
    ys = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area <= 0:
            continue
        ys.append(float(centroids[i][1]))
    if len(ys) < 3:
        return None
    ys.sort()
    # Use the first 3 dots (for standard vertical ellipsis).
    y1, y2, y3 = ys[0], ys[1], ys[2]
    g1 = y2 - y1
    g2 = y3 - y2
    if g1 <= 0 or g2 <= 0:
        return None
    return (g1 + g2) / 2.0

def _smart_vertical_ellipsis_advance(slot, font_size: int, char_bitmap_rows: int, bitmap_char: Optional[np.ndarray] = None) -> int:
    """Make the seam between two vertical ellipsis glyphs close to internal dot spacing."""
    vert_bearing_y = 0
    vert_advance = 0
    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'vertBearingY') and slot.metrics.vertBearingY:
            vert_bearing_y = slot.metrics.vertBearingY >> 6
        if hasattr(slot.metrics, 'vertAdvance') and slot.metrics.vertAdvance:
            vert_advance = slot.metrics.vertAdvance >> 6

    # Prefer ink-height + bearing, then fallback to font advance.
    raw = 0
    if char_bitmap_rows > 0:
        raw = char_bitmap_rows + vert_bearing_y
    if raw <= 0:
        raw = vert_advance
    if raw <= 0 and hasattr(slot, 'advance') and slot.advance and hasattr(slot.advance, 'y') and slot.advance.y:
        raw = slot.advance.y >> 6
    if raw <= 0:
        raw = font_size

    # If we can read the 3-dot centers, compute advance so seam gap ~= internal gap.
    # For centers [c1, c2, c3] with gap g, seam-equal advance is about 3g.
    g = _estimate_ellipsis_center_gap(bitmap_char)
    if g is not None and g > 0:
        raw = int(round(3.0 * g))

    return max(1, raw)

def _measure_vertical_char_metrics(font_size: int, cdpt: str):
    if _is_newline_control_char(cdpt):
        return 0, 0, 0, False

    """Return (advance_y, top_offset, bitmap_rows, has_visible_ink) for vertical rendering."""
    cdpt_trans, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    slot = get_char_glyph(cdpt_trans, font_size, 1)
    bitmap = slot.bitmap
    char_bitmap_rows = bitmap.rows
    char_bitmap_width = bitmap.width

    char_offset_y = font_size
    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'vertAdvance') and slot.metrics.vertAdvance:
            char_offset_y = slot.metrics.vertAdvance >> 6
        elif hasattr(slot.metrics, 'height') and slot.metrics.height:
            char_offset_y = slot.metrics.height >> 6

        if _is_vertical_ellipsis_char(cdpt_trans):
            bitmap_char = None
            if char_bitmap_rows * char_bitmap_width > 0 and len(bitmap.buffer) == char_bitmap_rows * char_bitmap_width:
                bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((char_bitmap_rows, char_bitmap_width))
            char_offset_y = _smart_vertical_ellipsis_advance(slot, font_size, char_bitmap_rows, bitmap_char)

    if char_offset_y == font_size and hasattr(slot, 'advance') and slot.advance:
        if hasattr(slot.advance, 'y') and slot.advance.y:
            char_offset_y = slot.advance.y >> 6

    has_bitmap = (
        char_bitmap_rows * char_bitmap_width > 0 and
        len(bitmap.buffer) == char_bitmap_rows * char_bitmap_width
    )
    if has_bitmap:
        # Keep vertical alignment in font-metrics space to avoid glyph-specific top snapping.
        top_offset = (slot.metrics.vertBearingY >> 6) if (hasattr(slot, 'metrics') and slot.metrics and hasattr(slot.metrics, 'vertBearingY')) else 0
        return char_offset_y, top_offset, char_bitmap_rows, True
    return char_offset_y, 0, 0, False

def _get_vertical_column_char_width(font_size: int, cdpt: str) -> int:
    if _is_newline_control_char(cdpt):
        return 0

    """Column frame width for vertical layout (Ballons-style: frame + ink), in pixels."""
    cdpt_trans, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    slot = get_char_glyph(cdpt_trans, font_size, 1)
    width = font_size
    if hasattr(slot, 'metrics') and slot.metrics and hasattr(slot.metrics, 'horiAdvance') and slot.metrics.horiAdvance:
        width = max(width, slot.metrics.horiAdvance >> 6)
    bitmap = slot.bitmap
    if bitmap.rows * bitmap.width > 0 and len(bitmap.buffer) == bitmap.rows * bitmap.width:
        bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
        nz = cv2.findNonZero(bitmap_char)
        if nz is not None:
            _, _, ink_w, _ = cv2.boundingRect(nz)
            width = max(width, ink_w)
        else:
            width = max(width, bitmap.width)
    return max(1, int(width))

def _measure_horizontal_block_render_height(font_size: int, content: str, border_size: int, config=None, stroke_ratio: float = 0.07, rotate_90: bool = False) -> int:
    """Measure the actual rendered pixel height of a horizontal block used in vertical text."""
    # preserve force-rotate marker by deciding rotation before normalization
    if not rotate_90:
        rotate_90 = should_rotate_horizontal_block_90(content)
    content = _normalize_horizontal_block_content(content)
    if not content:
        return 0

    h_width = get_string_width(font_size, content) + font_size
    h_height = font_size * 2
    temp_canvas_text = np.zeros((h_height, h_width), dtype=np.uint8)
    temp_canvas_border = np.zeros((h_height, h_width), dtype=np.uint8)
    pen_h = [font_size // 2, font_size]

    for char_h in content:
        if _is_newline_control_char(char_h):
            continue
        if char_h == '！':
            char_h = '!'
        elif char_h == '？':
            char_h = '?'
        offset_x = put_char_horizontal(
            font_size,
            char_h,
            pen_h,
            temp_canvas_text,
            temp_canvas_border,
            border_size=border_size,
            config=config,
            stroke_width=stroke_ratio
        )
        pen_h[0] += offset_x

    if rotate_90:
        temp_canvas_text = cv2.rotate(temp_canvas_text, cv2.ROTATE_90_CLOCKWISE)
        temp_canvas_border = cv2.rotate(temp_canvas_border, cv2.ROTATE_90_CLOCKWISE)

    combined_temp = cv2.add(temp_canvas_text, temp_canvas_border)
    x, y, w, h_crop = cv2.boundingRect(combined_temp)
    if w == 0 or h_crop == 0:
        return 0
    return h_crop

def _measure_vertical_line_visual_extents(line_text: str, font_size: int, border_size: int, config=None, stroke_ratio: float = 0.07):
    """Measure visual ink extents (top, bottom) for one vertical line, relative to line start pen y."""
    pen_y = 0
    min_y = None
    max_y = None

    parts = re.split(r'(<H>.*?</H>)', line_text, flags=re.IGNORECASE | re.DOTALL)
    for part in parts:
        if not part:
            continue

        is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
        if is_horizontal_block:
            content = part[3:-4]
            if not content:
                continue
            rh = _measure_horizontal_block_render_height(
                font_size,
                content,
                border_size=border_size,
                config=config,
                stroke_ratio=stroke_ratio,
                rotate_90=should_rotate_horizontal_block_90(content)
            )
            if rh > 0:
                block_top = pen_y
                block_bottom = pen_y + rh
                min_y = block_top if min_y is None else min(min_y, block_top)
                max_y = block_bottom if max_y is None else max(max_y, block_bottom)
            pen_y += rh
            continue

        for c in part:
            if not c:
                continue
            char_offset_y, top_offset, char_rows, has_ink = _measure_vertical_char_metrics(font_size, c)
            if has_ink and char_rows > 0:
                char_top = pen_y + top_offset
                char_bottom = char_top + char_rows
                min_y = char_top if min_y is None else min(min_y, char_top)
                max_y = char_bottom if max_y is None else max(max_y, char_bottom)
            pen_y += char_offset_y

    if min_y is None or max_y is None:
        return 0, 0
    return min_y, max_y

def _measure_horizontal_line_visual_extents(line_text: str, font_size: int, border_size: int, config=None, stroke_ratio: float = 0.07, reversed_direction: bool = False):
    """Measure visual ink extents (left, right) for one horizontal line, relative to line start pen x."""
    if not line_text:
        return 0, 0

    total_advance = get_string_width(font_size, line_text)
    temp_w = max(font_size * 4, total_advance + (font_size + border_size) * 4)
    temp_h = max(font_size * 3, (font_size + border_size) * 4)
    canvas_text = np.zeros((temp_h, temp_w), dtype=np.uint8)
    canvas_border = np.zeros((temp_h, temp_w), dtype=np.uint8)

    origin_x = (font_size + border_size) if not reversed_direction else (temp_w - (font_size + border_size))
    pen = [origin_x, font_size + border_size]

    for c in line_text:
        if _is_newline_control_char(c):
            continue
        if reversed_direction:
            cdpt, _ = CJK_Compatibility_Forms_translate(c, 0)
            glyph = get_char_glyph(cdpt, font_size, 0)
            offset_x = glyph.metrics.horiAdvance >> 6
            pen[0] -= offset_x
        offset_x = put_char_horizontal(
            font_size,
            c,
            pen,
            canvas_text,
            canvas_border,
            border_size=border_size,
            config=config,
            stroke_width=stroke_ratio
        )
        if not reversed_direction:
            pen[0] += offset_x

    combined = cv2.add(canvas_text, canvas_border)
    x, _, w, _ = cv2.boundingRect(combined)
    if w == 0:
        return 0, 0
    left_rel = x - origin_x
    right_rel = left_rel + w
    return left_rel, right_rel

def calc_horizontal_block_height(font_size: int, content: str) -> int:
    """
    预先计算横排块在竖排文本中的实际渲染高度
    用于准确计算竖排文本的总高度，特别是在智能缩放模式下

    注意：需要与 put_text_vertical 中的渲染逻辑保持一致
    - 字母/数字块且长度>=3：旋转90度，返回旋转后的实际高度
    - 其他情况（含符号）：横排显示，返回横排块的实际高度
    """
    rotate_90 = should_rotate_horizontal_block_90(content)
    content = _normalize_horizontal_block_content(content)
    if not content:
        return font_size

    # 仅对字母/数字块启用90度旋转
    if rotate_90:
        # --- 计算竖排旋转块的高度 ---
        # 使用与渲染相同的方式：横排渲染后旋转90度
        # 旋转后，原来的宽度变成高度
        total_width = get_string_width(font_size, content)
        # 旋转90度后，宽度变成高度
        return total_width
    else:
        # --- 计算横排块的高度（所有非旋转块） ---
        h_font_size = font_size

        # 创建临时画布来渲染横排内容
        h_height = h_font_size * 2
        h_width = get_string_width(h_font_size, content) + h_font_size

        temp_canvas = np.zeros((h_height, h_width), dtype=np.uint8)
        pen_h = [h_font_size // 2, h_font_size]

        # 渲染每个字符
        for char_h in content:
            if char_h == '！':
                char_h = '!'
            elif char_h == '？':
                char_h = '?'
            try:
                offset_x = get_char_offset_x(h_font_size, char_h)
                cdpt, _ = CJK_Compatibility_Forms_translate(char_h, 0)
                slot = get_char_glyph(cdpt, h_font_size, 0)
                bitmap = slot.bitmap

                if bitmap.rows * bitmap.width > 0 and len(bitmap.buffer) == bitmap.rows * bitmap.width:
                    bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
                    char_place_x = pen_h[0] + slot.bitmap_left
                    char_place_y = pen_h[1] - slot.bitmap_top

                    paste_y_start = max(0, char_place_y)
                    paste_x_start = max(0, char_place_x)
                    paste_y_end = min(temp_canvas.shape[0], char_place_y + bitmap.rows)
                    paste_x_end = min(temp_canvas.shape[1], char_place_x + bitmap.width)

                    if paste_y_start < paste_y_end and paste_x_start < paste_x_end:
                        temp_canvas[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = np.maximum(
                            temp_canvas[paste_y_start:paste_y_end, paste_x_start:paste_x_end],
                            bitmap_char[:(paste_y_end-paste_y_start), :(paste_x_end-paste_x_start)]
                        )

                pen_h[0] += offset_x
            except Exception:
                pass

        # 裁剪空白并返回实际高度
        if np.any(temp_canvas):
            _, _, _, h_crop = cv2.boundingRect(temp_canvas)
            return h_crop

        # 如果没有内容，返回默认高度
        return font_size

def calc_vertical(font_size: int, text: str, max_height: int, config=None):
    """
    Line breaking logic for vertical text.
    Handles forced newlines (\\n) and is aware of <H> horizontal blocks.
    """
    # 统一处理所有类型的AI换行符（仅转换 <H> 外部）
    text = _convert_br_outside_h_tags(text)
    # 与 put_text_vertical 保持一致：竖排计算前先将省略号转换为竖排符号
    text = text.replace('…', '︙')

    line_text_list = []
    line_height_list = []

    paragraphs = text.split('\n')

    for paragraph in paragraphs:
        if not paragraph:
            line_text_list.append('')
            line_height_list.append(0)
            continue

        current_line_text = ""
        current_line_height = 0

        parts = re.split(r'(<H>.*?</H>)', paragraph, flags=re.IGNORECASE | re.DOTALL)

        for part in parts:
            if not part:
                continue

            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')

            if is_horizontal_block:
                content = part[3:-4]
                if not content:
                    continue
                # 使用实际渲染高度而不是固定的 font_size
                # 这对于智能缩放模式下准确计算文本框拉伸比例至关重要
                block_height = calc_horizontal_block_height(font_size, content)

                if current_line_height + block_height > max_height and current_line_text:
                    line_text_list.append(current_line_text)
                    line_height_list.append(current_line_height)
                    current_line_text = part
                    current_line_height = block_height
                else:
                    current_line_text += part
                    current_line_height += block_height
            else:  # It's a vertical part, process character by character
                for cdpt in part:
                    if not cdpt or _is_newline_control_char(cdpt):
                        continue
                    
                    cdpt_trans, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
                    ckpt = get_char_glyph(cdpt_trans, font_size, 1)
                    bitmap = ckpt.bitmap

                    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
                        char_offset_y = ckpt.metrics.vertAdvance >> 6 if hasattr(ckpt.metrics, 'vertAdvance') and ckpt.metrics.vertAdvance != 0 else font_size
                    else:
                        char_offset_y = ckpt.metrics.vertAdvance >> 6 if hasattr(ckpt.metrics, 'vertAdvance') and ckpt.metrics.vertAdvance != 0 else font_size

                    # Keep vertical ellipsis compact and stable across fonts.
                    if _is_vertical_ellipsis_char(cdpt_trans):
                        bitmap_char = None
                        if bitmap.rows * bitmap.width > 0 and len(bitmap.buffer) == bitmap.rows * bitmap.width:
                            bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
                        char_offset_y = _smart_vertical_ellipsis_advance(ckpt, font_size, bitmap.rows, bitmap_char)

                    should_wrap = current_line_height + char_offset_y > max_height

                    if should_wrap and current_line_text:
                        line_text_list.append(current_line_text)
                        line_height_list.append(current_line_height)
                        current_line_text = cdpt
                        current_line_height = char_offset_y
                    else:
                        current_line_text += cdpt
                        current_line_height += char_offset_y

        if current_line_text:
            line_text_list.append(current_line_text)
            line_height_list.append(current_line_height)

    if not line_text_list:
        line_text_list.append("")
        line_height_list.append(0)

    return line_text_list, line_height_list

def put_char_vertical(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int, config=None, line_width: int = 0, force_rotate_90: bool = False, stroke_width: float = None):
    if _is_newline_control_char(cdpt):
        return 0

    if cdpt == '＿':
        # For the placeholder, just advance the pen vertically and do nothing else.
        return font_size

    pen = pen_l.copy()
    _is_pun = is_punctuation(cdpt)

    # 如果 force_rotate_90=True，强制旋转90度（用于英文数字）
    if force_rotate_90:
        rot_degree = 90
    else:
        cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)

    slot = get_char_glyph(cdpt, font_size, 1)
    bitmap = slot.bitmap
    char_bitmap_rows = bitmap.rows
    char_bitmap_width = bitmap.width
    # 统一的竖排字体度量获取逻辑
    char_offset_y = font_size  # 默认值
    
    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'vertAdvance') and slot.metrics.vertAdvance:
            char_offset_y = slot.metrics.vertAdvance >> 6
        elif hasattr(slot.metrics, 'height') and slot.metrics.height:
            # 使用字符高度作为备选
            char_offset_y = slot.metrics.height >> 6

        # 连续省略号场景：使用该字体自身的点间距估计行进量，改善两组省略号中缝不等距
        if _is_vertical_ellipsis_char(cdpt):
            bitmap_for_ellipsis = None
            if char_bitmap_rows * char_bitmap_width > 0 and len(bitmap.buffer) == char_bitmap_rows * char_bitmap_width:
                bitmap_for_ellipsis = np.array(bitmap.buffer, dtype=np.uint8).reshape((char_bitmap_rows, char_bitmap_width))
            char_offset_y = _smart_vertical_ellipsis_advance(slot, font_size, char_bitmap_rows, bitmap_for_ellipsis)
    
    # 如果metrics不可用，尝试使用advance
    if char_offset_y == font_size and hasattr(slot, 'advance') and slot.advance:
        if hasattr(slot.advance, 'y') and slot.advance.y:
            char_offset_y = slot.advance.y >> 6
    
    # 如果bitmap为空，直接返回计算好的offset
    if char_bitmap_rows * char_bitmap_width == 0 or len(bitmap.buffer) != char_bitmap_rows * char_bitmap_width:
        return char_offset_y
    bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((char_bitmap_rows, char_bitmap_width))

    # 保存原始尺寸用于位置补偿计算
    _original_bitmap_rows = char_bitmap_rows
    original_bitmap_width = char_bitmap_width
    
    # 如果需要旋转90度
    if force_rotate_90:
        # 顺时针旋转90度 (相当于逆时针旋转270度或使用 cv2.ROTATE_90_CLOCKWISE)
        bitmap_char = cv2.rotate(bitmap_char, cv2.ROTATE_90_CLOCKWISE)
        # 旋转后更新尺寸
        char_bitmap_rows, char_bitmap_width = bitmap_char.shape

    # --- ALIGNMENT FIX ---
    if line_width <= 0:
        line_width = font_size
    # Center by the real non-zero ink box, not raw bitmap width.
    ink_x = 0
    ink_w = char_bitmap_width
    if char_bitmap_rows > 0 and char_bitmap_width > 0:
        nz = cv2.findNonZero(bitmap_char)
        if nz is not None:
            ink_x, _, ink_w, _ = cv2.boundingRect(nz)
    line_start_x = pen[0] - line_width
    char_place_x = line_start_x + round((line_width - ink_w) / 2.0) - ink_x
    # --- END FIX ---

    # 计算Y位置，考虑旋转后的位置补偿
    if force_rotate_90:
        # 旋转后需要调整Y位置：原来的宽度变成了高度
        # 使用原始宽度的一半作为垂直偏移的基准
        char_place_y = pen[1] + round((original_bitmap_width - char_bitmap_rows) / 2.0)
    else:
        char_place_y = pen[1] + (slot.metrics.vertBearingY >> 6)
    paste_y_start = max(0, char_place_y)
    paste_x_start = max(0, char_place_x)
    paste_y_end = min(canvas_text.shape[0], char_place_y + char_bitmap_rows)
    paste_x_end = min(canvas_text.shape[1], char_place_x + char_bitmap_width)
    if paste_y_start >= paste_y_end or paste_x_start >= paste_x_end:
        logger.warning(f"Char '{cdpt}' is completely outside the canvas or on the boundary, skipped. Position: x={char_place_x}, y={char_place_y}, Canvas size: {canvas_text.shape}")
    else:
        bitmap_char_slice = bitmap_char[paste_y_start-char_place_y : paste_y_end-char_place_y,
                                        paste_x_start-char_place_x : paste_x_end-char_place_x]
        if bitmap_char_slice.size > 0:
            canvas_text[paste_y_start:paste_y_end, paste_x_start:paste_x_end] = bitmap_char_slice
    if border_size > 0:
        glyph_border = get_char_border(cdpt, font_size, 1)
        stroker = freetype.Stroker()
        # Use passed stroke_width, fallback to config or default
        if stroke_width is None:
            stroke_ratio = config.render.stroke_width if (config and hasattr(config.render, 'stroke_width')) else 0.07
        else:
            stroke_ratio = stroke_width
        stroke_radius = 64 * max(int(stroke_ratio * font_size), 1)
        stroker.set(stroke_radius, freetype.FT_STROKER_LINEJOIN_ROUND, freetype.FT_STROKER_LINECAP_ROUND, 0)
        glyph_border.stroke(stroker, destroy=True)
        blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, freetype.Vector(0, 0), True)
        bitmap_b = blyph.bitmap
        border_bitmap_rows = bitmap_b.rows
        border_bitmap_width = bitmap_b.width
        if border_bitmap_rows * border_bitmap_width > 0 and len(bitmap_b.buffer) == border_bitmap_rows * border_bitmap_width:
            bitmap_border = np.array(bitmap_b.buffer, dtype=np.uint8).reshape((border_bitmap_rows, border_bitmap_width))

            # 如果需要旋转90度，边框也要旋转
            if force_rotate_90:
                bitmap_border = cv2.rotate(bitmap_border, cv2.ROTATE_90_CLOCKWISE)
                border_bitmap_rows, border_bitmap_width = bitmap_border.shape

            # 改进的边框位置计算：直接基于字符位置和尺寸差异
            # 避免浮点累积误差
            size_diff_x = border_bitmap_width - char_bitmap_width
            size_diff_y = border_bitmap_rows - char_bitmap_rows
            
            # 边框应该围绕字符居中，所以偏移是尺寸差的一半
            pen_border_x = char_place_x - round(size_diff_x / 2.0)
            pen_border_y = char_place_y - round(size_diff_y / 2.0)
            pen_border = (max(0, pen_border_x), max(0, pen_border_y))
            paste_border_y_start = pen_border[1]
            paste_border_x_start = pen_border[0]
            paste_border_y_end = min(canvas_border.shape[0], pen_border[1] + border_bitmap_rows)
            paste_border_x_end = min(canvas_border.shape[1], pen_border[0] + border_bitmap_width)
            if paste_border_y_start >= paste_border_y_end or paste_border_x_start >= paste_border_x_end:
                logger.warning(f"The border of char '{cdpt}' is completely outside the canvas or on the boundary, skipped. Position: x={pen_border[0]}, y={pen_border[1]}, Canvas size: {canvas_border.shape}")
            else:
                bitmap_border_slice = bitmap_border[0 : paste_border_y_end-paste_border_y_start,
                                                    0 : paste_border_x_end-paste_border_x_start]
                if bitmap_border_slice.size > 0:
                    target_slice = canvas_border[paste_border_y_start:paste_border_y_end,
                                                 paste_border_x_start:paste_border_x_end]
                    if target_slice.shape == bitmap_border_slice.shape:
                        canvas_border[paste_border_y_start:paste_border_y_end,
                                      paste_border_x_start:paste_border_x_end] = cv2.add(target_slice, bitmap_border_slice)
                    else:
                        logger.warning("Shape mismatch: target={{target_slice.shape}}, source={{bitmap_border_slice.shape}}")
    return char_offset_y  

def put_text_vertical(font_size: int, text: str, h: int, alignment: str, fg: Tuple[int, int, int], bg: Optional[Tuple[int, int, int]], line_spacing: int, config=None, region_count: int = 1, stroke_width: float = None):

    # 应用最大字体限制
    if config and hasattr(config.render, 'max_font_size') and config.render.max_font_size > 0:
        font_size = min(font_size, config.render.max_font_size)

    text = compact_special_symbols(text)
    # 在竖排文本中，将省略号替换为单个竖排省略号符号（两个小点竖排）
    text = text.replace('…', '︙')
    if not text:
        return

    # 初始化 stroke_ratio，避免 UnboundLocalError
    if stroke_width is None:
        stroke_ratio = config.render.stroke_width if (config and hasattr(config.render, 'stroke_width')) else 0.07
    else:
        stroke_ratio = stroke_width
    bg_size = int(max(font_size * stroke_ratio, 1)) if bg is not None else 0
    # line_spacing 是基本间距的倍率
    # 竖排基本间距: 0.2
    spacing_x = int(font_size * 0.2 * (line_spacing or 1.0))

    # 固定为大高度以禁用自动换行（仅由显式换行符控制分段）
    effective_max_height = 99999
    logger.debug(f"[VERTICAL DEBUG] effective_max_height={effective_max_height}")

    # Use original font size for line breaking calculation
    line_text_list, line_height_list = calc_vertical(font_size, text, effective_max_height, config=config)
    if not line_height_list:
        return

    max_line_height = max(line_height_list) if line_height_list else 0

    # 预先计算每一列的实际最大字符宽度
    line_widths = []
    for line_text in line_text_list:
        max_width = font_size  # 默认使用font_size
        parts = re.split(r'(<H>.*?</H>)', line_text, flags=re.IGNORECASE | re.DOTALL)
        for part in parts:
            if not part:
                continue
            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            if not is_horizontal_block:
                # 计算竖排字符的实际宽度
                for c in part:
                    max_width = max(max_width, _get_vertical_column_char_width(font_size, c))
        line_widths.append(max_width)

    # 使用实际列宽计算画布大小
    canvas_x = sum(line_widths) + spacing_x * (len(line_text_list) - 1) + (font_size + bg_size) * 2
    canvas_y = max_line_height + (font_size + bg_size) * 2

    canvas_text = np.zeros((canvas_y, canvas_x), dtype=np.uint8)
    canvas_border = canvas_text.copy()
    pen_orig = [canvas_text.shape[1] - (font_size + bg_size), (font_size + bg_size)]

    # 预先计算每一列的最大字符宽度，防止字符超出列边界
    line_max_widths = []
    for line_text in line_text_list:
        max_char_width = font_size  # 默认使用font_size
        parts = re.split(r'(<H>.*?</H>)', line_text, flags=re.IGNORECASE | re.DOTALL)
        for part in parts:
            if not part:
                continue
            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')
            if not is_horizontal_block:
                # 只计算竖排字符的宽度
                for c in part:
                    max_char_width = max(max_char_width, _get_vertical_column_char_width(font_size, c))
        line_max_widths.append(max_char_width)

    for line_idx, (line_text, line_height) in enumerate(zip(line_text_list, line_height_list)):
        pen_line = pen_orig.copy()
        # 使用该列的实际最大字符宽度作为列宽
        line_width = line_max_widths[line_idx]
        if alignment == 'center':
            pen_line[1] += round((max_line_height - line_height) / 2.0)
        elif alignment == 'right': # In vertical, right means bottom
            pen_line[1] += max_line_height - line_height

        parts = re.split(r'(<H>.*?</H>)', line_text, flags=re.IGNORECASE | re.DOTALL)

        for part in parts:
            if not part:
                continue

            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')

            if is_horizontal_block:
                raw_content = part[3:-4]
                if not raw_content:
                    continue
                rotate_90 = should_rotate_horizontal_block_90(raw_content)
                content = _normalize_horizontal_block_content(raw_content)
                if not content:
                    continue

                # 仅字母/数字块（长度>=3）旋转90度；符号块保持横排
                if rotate_90:
                    # --- RENDER ROTATED BLOCK (字母/数字块旋转90度) ---
                    # 使用与2个字符横排相同的渲染方式，确保字符间距一致
                    r_font_size = font_size
                    
                    # 先在临时画布上横排渲染
                    r_width = get_string_width(r_font_size, content) + r_font_size
                    r_height = r_font_size * 2
                    
                    temp_canvas_text = np.zeros((r_height, r_width), dtype=np.uint8)
                    temp_canvas_border = np.zeros((r_height, r_width), dtype=np.uint8)
                    pen_r = [r_font_size // 2, r_font_size]
                    
                    for char_r in content:
                        if char_r == '！': char_r = '!'
                        elif char_r == '？': char_r = '?'
                        offset_x = put_char_horizontal(r_font_size, char_r, pen_r, temp_canvas_text, temp_canvas_border, border_size=bg_size, config=config, stroke_width=stroke_ratio)
                        pen_r[0] += offset_x
                    
                    # 旋转90度（顺时针）
                    rotated_text = cv2.rotate(temp_canvas_text, cv2.ROTATE_90_CLOCKWISE)
                    rotated_border = cv2.rotate(temp_canvas_border, cv2.ROTATE_90_CLOCKWISE)
                    
                    # 裁剪空白
                    combined_temp = cv2.add(rotated_text, rotated_border)
                    x, y, w, h_crop = cv2.boundingRect(combined_temp)
                    if w == 0 or h_crop == 0:
                        logger.warning(f"[RENDER SKIPPED] Rotated block has zero dimensions. Width: {w}, Height: {h_crop}")
                        continue
                    
                    rotated_block_text = rotated_text[y:y+h_crop, x:x+w]
                    rotated_block_border = rotated_border[y:y+h_crop, x:x+w]
                    
                    rh, rw = rotated_block_text.shape
                    
                    # 在竖排行的中心对齐旋转块
                    line_start_x = pen_line[0] - line_width
                    paste_x = line_start_x + round((line_width - rw) / 2.0)
                    paste_y = pen_line[1]
                    
                    # 边界检查和调整
                    canvas_h, canvas_w = canvas_text.shape
                    if paste_y + rh > canvas_h or paste_x + rw > canvas_w or paste_x < 0 or paste_y < 0:
                        # 向中心调整
                        if paste_x < 0:
                            paste_x = 0
                        elif paste_x + rw > canvas_w:
                            paste_x = canvas_w - rw
                        if paste_y < 0:
                            paste_y = 0
                        elif paste_y + rh > canvas_h:
                            paste_y = canvas_h - rh
                        
                        paste_x = max(0, min(paste_x, canvas_w - rw))
                        paste_y = max(0, min(paste_y, canvas_h - rh))
                        
                        if paste_x < 0 or paste_y < 0 or paste_x + rw > canvas_w or paste_y + rh > canvas_h:
                            logger.warning(f"Rotated block too large for canvas, skipping. Size: {rw}x{rh}, Canvas: {canvas_w}x{canvas_h}")
                            continue
                    
                    # 粘贴到主画布
                    target_text_roi = canvas_text[paste_y:paste_y+rh, paste_x:paste_x+rw]
                    canvas_text[paste_y:paste_y+rh, paste_x:paste_x+rw] = np.maximum(target_text_roi, rotated_block_text)
                    
                    target_border_roi = canvas_border[paste_y:paste_y+rh, paste_x:paste_x+rw]
                    canvas_border[paste_y:paste_y+rh, paste_x:paste_x+rw] = np.maximum(target_border_roi, rotated_block_border)
                    
                    # 使用旋转后的实际高度
                    pen_line[1] += rh
                    # --- END ROTATED BLOCK RENDER ---
                else:
                    # --- RENDER HORIZONTAL BLOCK (2个字符横排) ---
                    h_font_size = font_size

                    h_width = get_string_width(h_font_size, content) + h_font_size
                    h_height = h_font_size * 2

                    temp_canvas_text = np.zeros((h_height, h_width), dtype=np.uint8)
                    temp_canvas_border = np.zeros((h_height, h_width), dtype=np.uint8)
                    pen_h = [h_font_size // 2, h_font_size]

                    for char_h in content:
                        if char_h == '！': char_h = '!'
                        elif char_h == '？': char_h = '?'
                        offset_x = put_char_horizontal(h_font_size, char_h, pen_h, temp_canvas_text, temp_canvas_border, border_size=bg_size, config=config, stroke_width=stroke_ratio)
                        pen_h[0] += offset_x

                    combined_temp = cv2.add(temp_canvas_text, temp_canvas_border)
                    x, y, w, h_crop = cv2.boundingRect(combined_temp)
                    if w == 0 or h_crop == 0:
                        logger.warning(f"[RENDER SKIPPED] Horizontal block in vertical text has zero dimensions. Width: {w}, Height: {h_crop}")
                        continue

                    horizontal_block_text = temp_canvas_text[y:y+h_crop, x:x+w]
                    horizontal_block_border = temp_canvas_border[y:y+h_crop, x:x+w]

                    rh, rw = horizontal_block_text.shape

                    # 修复：在竖排行的中心正确地对齐横排块
                    # pen_line[0] 是右边界，font_size 是行宽
                    line_start_x = pen_line[0] - line_width
                    paste_x = line_start_x + round((line_width - rw) / 2.0)

                    # 横排块的Y位置：直接从pen_line[1]开始，不做baseline对齐
                    # pen_line[1]是当前渲染位置（前一个字符的底部）
                    paste_y = pen_line[1]

                    # 智能边界调整：向中心方向移动而不是跳过
                    canvas_h, canvas_w = canvas_text.shape
                    adjusted = False
                    if paste_y + rh > canvas_h or paste_x + rw > canvas_w or paste_x < 0 or paste_y < 0:
                        _adjusted = True
                        # 向中心调整位置
                        _center_x = canvas_w // 2
                        _center_y = canvas_h // 2

                        # X 方向调整
                        if paste_x < 0:
                            paste_x = 0
                        elif paste_x + rw > canvas_w:
                            paste_x = canvas_w - rw

                        # Y 方向调整
                        if paste_y < 0:
                            paste_y = 0
                        elif paste_y + rh > canvas_h:
                            paste_y = canvas_h - rh

                        # 确保调整后仍在边界内
                        paste_x = max(0, min(paste_x, canvas_w - rw))
                        paste_y = max(0, min(paste_y, canvas_h - rh))

                        if paste_x < 0 or paste_y < 0 or paste_x + rw > canvas_w or paste_y + rh > canvas_h:
                            logger.warning(f"Text block too large for canvas, skipping. Size: {rw}x{rh}, Canvas: {canvas_w}x{canvas_h}")
                            continue

                        logger.info(f"Adjusted text position to fit canvas bounds: ({line_start_x + round((font_size - rw) / 2.0)}, {pen_line[1]}) -> ({paste_x}, {paste_y})")

                    target_text_roi = canvas_text[paste_y:paste_y+rh, paste_x:paste_x+rw]
                    canvas_text[paste_y:paste_y+rh, paste_x:paste_x+rw] = np.maximum(target_text_roi, horizontal_block_text)

                    target_border_roi = canvas_border[paste_y:paste_y+rh, paste_x:paste_x+rw]
                    canvas_border[paste_y:paste_y+rh, paste_x:paste_x+rw] = np.maximum(target_border_roi, horizontal_block_border)

                    # 使用实际渲染高度而不是固定的 font_size
                    # 这确保了渲染输出与 calc_vertical 的高度计算一致
                    pen_line[1] += rh
                    # --- END HORIZONTAL BLOCK RENDER ---

            else: # It's a vertical part
                for char_idx, c in enumerate(part):
                    if _is_newline_control_char(c):
                        continue
                    offset_y = put_char_vertical(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size, config=config, line_width=line_width, stroke_width=stroke_ratio)
                    pen_line[1] += offset_y
        
        # 使用实际列宽而不是固定的font_size来计算下一列的位置
        pen_orig[0] -= spacing_x + line_width

    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)
    combined_canvas = cv2.add(canvas_text, canvas_border)
    x, y, w, h = cv2.boundingRect(combined_canvas)

    if w == 0 or h == 0:
        logger.warning(f"[RENDER SKIPPED] Vertical text rendered with zero width or height. Width: {w}, Height: {h}, Text: {text[:50]}...")
        return None

    result = line_box[y:y+h, x:x+w]

    return result

_hyphenator_cache = {}

def select_hyphenator(lang: str):
    # 处理空字符串或None的情况，使用英文作为默认值
    if not lang or not lang.strip():
        lang = 'en_US'
    
    lang = standardize_tag(lang)
    if lang not in HYPHENATOR_LANGUAGES:
        for avail_lang in reversed(HYPHENATOR_LANGUAGES):
            if avail_lang.startswith(lang):
                lang = avail_lang
                break
        else:
            return None
    
    if lang in _hyphenator_cache:
        return _hyphenator_cache[lang]
    
    try:
        h = Hyphenator(lang)
        _hyphenator_cache[lang] = h
        return h
    except Exception:
        _hyphenator_cache[lang] = None
        return None

def _is_newline_control_char(cdpt: str) -> bool:
    return cdpt in ('\n', '\r')

def get_char_offset_x(font_size: int, cdpt: str):
    if _is_newline_control_char(cdpt):
        return 0

    if cdpt == '＿':
        # Return the width of a full-width space for the placeholder
        return get_char_offset_x(font_size, '　')

    c, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 0)
    glyph = get_char_glyph(c, font_size, 0)
    bitmap = glyph.bitmap
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        char_offset_x = glyph.advance.x >> 6
    else:
        char_offset_x = glyph.metrics.horiAdvance >> 6
    return char_offset_x

def get_string_width(font_size: int, text: str):
    return sum(get_char_offset_x(font_size, c) for c in text if not _is_newline_control_char(c))

def get_char_offset_y(font_size: int, cdpt: str):
    if _is_newline_control_char(cdpt):
        return 0

    """获取单个字符的竖排高度（像素）"""
    if cdpt == '＿':
        return get_char_offset_y(font_size, '　')

    cdpt_trans, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    slot = get_char_glyph(cdpt_trans, font_size, 1)

    char_offset_y = font_size  # 默认值

    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'vertAdvance') and slot.metrics.vertAdvance:
            char_offset_y = slot.metrics.vertAdvance >> 6
        elif hasattr(slot.metrics, 'height') and slot.metrics.height:
            char_offset_y = slot.metrics.height >> 6

    if _is_vertical_ellipsis_char(cdpt_trans):
        bitmap_rows = slot.bitmap.rows if hasattr(slot, 'bitmap') and slot.bitmap else 0
        bitmap_char = None
        if slot.bitmap and bitmap_rows > 0 and slot.bitmap.width > 0 and len(slot.bitmap.buffer) == bitmap_rows * slot.bitmap.width:
            bitmap_char = np.array(slot.bitmap.buffer, dtype=np.uint8).reshape((bitmap_rows, slot.bitmap.width))
        char_offset_y = _smart_vertical_ellipsis_advance(slot, font_size, bitmap_rows, bitmap_char)

    if char_offset_y == font_size and hasattr(slot, 'advance') and slot.advance:
        if hasattr(slot.advance, 'y') and slot.advance.y:
            char_offset_y = slot.advance.y >> 6

    return char_offset_y

def get_string_height(font_size: int, text: str):
    """获取字符串的竖排总高度（像素），考虑<H>横排块"""
    # 处理 BR 标记（不换行时当作连续文本）
    text_clean = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '', text, flags=re.IGNORECASE)

    total_height = 0

    # 分割 <H> 块
    parts = re.split(r'(<H>.*?</H>)', text_clean, flags=re.IGNORECASE | re.DOTALL)

    for part in parts:
        if not part:
            continue

        is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')

        if is_horizontal_block:
            content = part[3:-4]
            if content:
                # 用精确的横排块高度计算
                total_height += calc_horizontal_block_height(font_size, content)
        else:
            # 普通竖排字符
            total_height += sum([get_char_offset_y(font_size, c) for c in part])

    return total_height

def calc_horizontal_cjk(font_size: int, text: str, max_width: int) -> Tuple[List[str], List[int]]:
    """
    Line breaking logic for CJK languages with punctuation rules.
    Handles forced newlines (\n) and invisible placeholders (＿).
    """
    # 统一处理所有类型的AI换行符
    text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)

    lines = []
    no_start_chars = "》，。．」』】）！；：？"
    no_end_chars = "《「『【（"

    paragraphs = text.split('\n')

    for para_idx, paragraph in enumerate(paragraphs):
        if not paragraph:
            lines.append(("", 0))
            continue

        current_line = ""
        current_width = 0
        for char_idx, char in enumerate(paragraph):
            char_width = get_char_offset_x(font_size, char)

            if current_width + char_width > max_width and current_line:
                if current_line and current_line[-1] in no_end_chars:
                    last_char = current_line[-1]
                    current_line = current_line[:-1]
                    lines.append((current_line, get_string_width(font_size, current_line)))
                    current_line = last_char + char
                else:
                    lines.append((current_line, current_width))
                    current_line = char
                current_width = get_string_width(font_size, current_line)
            elif not current_line and char in no_start_chars:
                if lines:
                    prev_line_text, prev_line_width = lines[-1]
                    lines[-1] = (prev_line_text + char, prev_line_width + char_width)
                else:
                    current_line += char
                    current_width += char_width
            else:
                current_line += char
                current_width += char_width

        if current_line:
            lines.append((current_line, current_width))

    line_text_list = [line[0] for line in lines]
    line_width_list = [line[1] for line in lines]

    return line_text_list, line_width_list

def calc_horizontal(font_size: int, text: str, max_width: int, max_height: int, language: str = 'en_US', hyphenate: bool = True) -> Tuple[List[str], List[int]]:

    # 统一处理所有类型的AI换行符
    text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)

    max_width = max(max_width, 2 * font_size)

    whitespace_offset_x = get_char_offset_x(font_size, ' ')
    hyphen_offset_x = get_char_offset_x(font_size, '-')

    # 先按换行符分割段落，然后对每段分割单词
    # 使用特殊标记来保留换行位置
    paragraphs = text.split('\n')
    words = []
    newline_positions = set()  # 记录哪些位置是段落结束（需要强制换行）

    for para_idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():  # 非空段落
            para_words = re.split(r'[ \t]+', paragraph)  # 只按空格和制表符分割，不包括 \n
            words.extend(para_words)
            if para_idx < len(paragraphs) - 1:  # 不是最后一段
                newline_positions.add(len(words) - 1)  # 标记这个单词后面需要换行
        elif para_idx < len(paragraphs) - 1:  # 空段落但不是最后一个
            # 空行也需要保留
            words.append('')
            newline_positions.add(len(words) - 1)

    # 如果没有单词，返回空结果
    if not words:
        return [], []

    word_widths = []
    for i, word in enumerate(words):
        width = get_string_width(font_size, word)
        word_widths.append(width)

    while True:
        max_lines = max_height // font_size + 1
        expected_size = sum(word_widths) + max((len(word_widths) - 1) * whitespace_offset_x - (max_lines - 1) * hyphen_offset_x, 0)
        max_size = max_width * max_lines

        if max_size < expected_size:
            multiplier = np.sqrt(expected_size / max_size)
            max_width *= max(multiplier, 1.05)
            max_height *= multiplier
        else:
            break

    syllables = []
    hyphenator = select_hyphenator(language)

    for i, word in enumerate(words):
        new_syls = []
        if hyphenator and len(word) <= 100:
            try:
                new_syls = hyphenator.syllables(word)
            except Exception as _e:
                new_syls = []

        if len(new_syls) == 0:
            if len(word) <= 3:
                new_syls = [word]
            else:
                new_syls = list(word)

        normalized_syls = []
        for syl in new_syls:
            syl_width = get_string_width(font_size, syl)
            if syl_width > max_width:
                normalized_syls.extend(list(syl))
            else:
                normalized_syls.append(syl)
        syllables.append(normalized_syls)

    line_words_list = []
    line_width_list = []
    hyphenation_idx_list = []
    line_words = []
    line_width = 0
    hyphenation_idx = 0

    def break_line():
        nonlocal line_words, line_width, hyphenation_idx
        line_words_list.append(line_words)
        line_width_list.append(line_width)
        hyphenation_idx_list.append(hyphenation_idx)
        line_words = []
        line_width = 0
        hyphenation_idx = 0

    def get_present_syllables_range(line_idx, word_pos):
        while word_pos < 0:
            word_pos += len(line_words_list[line_idx])
        word_idx = line_words_list[line_idx][word_pos]
        syl_start_idx = 0
        syl_end_idx = len(syllables[word_idx])
        if line_idx > 0 and word_pos == 0 and line_words_list[line_idx - 1][-1] == word_idx:
            syl_start_idx = hyphenation_idx_list[line_idx - 1]
        if line_idx < len(line_words_list) - 1 and word_pos == len(line_words_list[line_idx]) - 1 and line_words_list[line_idx + 1][0] == word_idx:
            syl_end_idx = hyphenation_idx_list[line_idx]
        return syl_start_idx, syl_end_idx

    def get_present_syllables(line_idx, word_pos):
        syl_start_idx, syl_end_idx = get_present_syllables_range(line_idx, word_pos)
        return syllables[line_words_list[line_idx][word_pos]][syl_start_idx:syl_end_idx]

    i = 0
    while True:
        if i >= len(words):
            if line_width > 0:
                break_line()
            break

        current_width = whitespace_offset_x if line_width > 0 else 0

        if line_width + current_width + word_widths[i] <= max_width + hyphen_offset_x:
            line_words.append(i)
            line_width += current_width + word_widths[i]
            i += 1
            # 检查是否需要强制换行（AI 断句）
            if (i - 1) in newline_positions:
                break_line()
        elif word_widths[i] > max_width:
            j = 0
            hyphenation_idx = 0
            while j < len(syllables[i]):
                syl = syllables[i][j]
                syl_width = get_string_width(font_size, syl)

                if line_width + current_width + syl_width <= max_width:
                    current_width += syl_width
                    j += 1
                    hyphenation_idx = j
                else:
                    if hyphenation_idx > 0:
                        line_words.append(i)
                        line_width += current_width
                    current_width = 0
                    break_line()
            line_words.append(i)
            line_width += current_width
            i += 1
            # 检查是否需要强制换行（AI 断句）
            if (i - 1) in newline_positions:
                break_line()
        else:
            if hyphenate:
                break_line()
            else:
                line_words.append(i)
                line_width += current_width + word_widths[i]
                i += 1
                if (i - 1) in newline_positions:
                    break_line()


    # 连字符优化阶段
    if hyphenate and len(line_words_list) > max_lines:
        line_idx = 0
        while line_idx < len(line_words_list) - 1:
            line_words1 = line_words_list[line_idx]
            line_words2 = line_words_list[line_idx + 1]
            left_space = max_width - line_width_list[line_idx]
            first_word = True
            while len(line_words2) != 0:
                word_idx = line_words2[0]
                if first_word and word_idx == line_words1[-1]:
                    syl_start_idx = hyphenation_idx_list[line_idx]
                    if line_idx < len(line_width_list) - 2 and word_idx == line_words_list[line_idx + 2][0]:
                        syl_end_idx = hyphenation_idx_list[line_idx + 1]
                    else:
                        syl_end_idx = len(syllables[word_idx])
                else:
                    left_space -= whitespace_offset_x
                    syl_start_idx = 0
                    syl_end_idx = len(syllables[word_idx]) if len(line_words2) > 1 else hyphenation_idx_list[line_idx + 1]
                first_word = False
                current_width = 0
                for i in range(syl_start_idx, syl_end_idx):
                    syl = syllables[word_idx][i]
                    syl_width = get_string_width(font_size, syl)
                    if left_space > current_width + syl_width:
                        current_width += syl_width
                    else:
                        if current_width > 0:
                            left_space -= current_width
                            line_width_list[line_idx] = max_width - left_space
                            hyphenation_idx_list[line_idx] = i
                            line_words1.append(word_idx)
                        break
                else:
                    left_space -= current_width
                    line_width_list[line_idx] = max_width - left_space
                    line_words1.append(word_idx)
                    line_words2.pop(0)
                    continue
                break
            if len(line_words2) == 0:
                line_words_list.pop(line_idx + 1)
                line_width_list.pop(line_idx + 1)
                hyphenation_idx_list.pop(line_idx)
            else:
                line_idx += 1

    # 行合并优化阶段
    line_idx = 0
    while line_idx < len(line_words_list) - 1:
        line_words1 = line_words_list[line_idx]
        line_words2 = line_words_list[line_idx + 1]
        merged_word_idx = -1
        if line_words1[-1] == line_words2[0]:
            word1_text = ''.join(get_present_syllables(line_idx, -1))
            word2_text = ''.join(get_present_syllables(line_idx + 1, 0))
            word1_width = get_string_width(font_size, word1_text)
            word2_width = get_string_width(font_size, word2_text)
            if len(word2_text) == 1 or word2_width < font_size:
                merged_word_idx = line_words1[-1]
                line_words2.pop(0)
                line_width_list[line_idx] += word2_width
                line_width_list[line_idx + 1] -= word2_width + whitespace_offset_x
            elif len(word1_text) == 1 or word1_width < font_size:
                merged_word_idx = line_words1[-1]
                line_words1.pop(-1)
                line_width_list[line_idx] -= word1_width + whitespace_offset_x
                line_width_list[line_idx + 1] += word1_width
        if len(line_words1) == 0:
            line_words_list.pop(line_idx)
            line_width_list.pop(line_idx)
            hyphenation_idx_list.pop(line_idx)
        elif len(line_words2) == 0:
            line_words_list.pop(line_idx + 1)
            line_width_list.pop(line_idx + 1)
            hyphenation_idx_list.pop(line_idx)
        elif line_idx >= len(line_words_list) - 1 or line_words_list[line_idx + 1] != merged_word_idx:
            line_idx += 1

    use_hyphen_chars = hyphenate and hyphenator and max_width > 1.5 * font_size and len(words) > 1

    line_text_list = []
    for i, line in enumerate(line_words_list):
        line_text = ''
        for j, word_idx in enumerate(line):
            syl_start_idx, syl_end_idx = get_present_syllables_range(i, j)
            current_syllables = syllables[word_idx][syl_start_idx:syl_end_idx]
            line_text += ''.join(current_syllables)
            if len(line_text) == 0:
                continue
            if j == 0 and i > 0 and len(line_text_list[-1]) > 0 and line_text_list[-1][-1] == '-' and line_text[0] == '-':
                line_text = line_text[1:]
                line_width_list[i] -= hyphen_offset_x
            if j < len(line) - 1 and len(line_text) > 0:
                line_text += ' '
            elif use_hyphen_chars and syl_end_idx != len(syllables[word_idx]) and len(words[word_idx]) > 3 and line_text[-1] != '-' and not (syl_end_idx < len(syllables[word_idx]) and not re.search(r'\w', syllables[word_idx][syl_end_idx][0])):
                line_text += '-'
                line_width_list[i] += hyphen_offset_x
        line_width_list[i] = get_string_width(font_size, line_text)
        line_text_list.append(line_text)

    return line_text_list, line_width_list

def put_char_horizontal(font_size: int, cdpt: str, pen_l: Tuple[int, int], canvas_text: np.ndarray, canvas_border: np.ndarray, border_size: int, config=None, stroke_width: float = None):
    if _is_newline_control_char(cdpt):
        return 0

    if cdpt == '＿':
        # For the placeholder, just advance the pen and do nothing else.
        return get_char_offset_x(font_size, '＿')

    pen = list(pen_l)
    cdpt, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 0)
    slot = get_char_glyph(cdpt, font_size, 0)
    bitmap = slot.bitmap
    
    # 统一的字体度量获取逻辑，优先使用metrics.horiAdvance
    char_offset_x = font_size  # 默认值
    
    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'horiAdvance') and slot.metrics.horiAdvance:
            char_offset_x = slot.metrics.horiAdvance >> 6
        elif hasattr(slot.metrics, 'width') and slot.metrics.width:
            # 使用字符宽度作为备选
            char_offset_x = slot.metrics.width >> 6
    
    # 如果metrics不可用，尝试使用advance
    if char_offset_x == font_size and hasattr(slot, 'advance') and slot.advance:
        if hasattr(slot.advance, 'x') and slot.advance.x:
            char_offset_x = slot.advance.x >> 6
    
    # 最后的fallback：使用bitmap宽度
    if char_offset_x == font_size and bitmap.width > 0:
        if hasattr(slot, 'bitmap_left'):
            char_offset_x = slot.bitmap_left + bitmap.width
        else:
            char_offset_x = bitmap.width
    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
        return char_offset_x
    bitmap_char = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
    char_place_x = pen[0] + slot.bitmap_left
    char_place_y = pen[1] - slot.bitmap_top
    paste_y_start = max(0, char_place_y)
    paste_x_start = max(0, char_place_x)
    paste_y_end = min(canvas_text.shape[0], char_place_y + bitmap.rows)
    paste_x_end = min(canvas_text.shape[1], char_place_x + bitmap.width)
    bitmap_slice_y_start = paste_y_start - char_place_y
    bitmap_slice_x_start = paste_x_start - char_place_x
    bitmap_slice_y_end = bitmap_slice_y_start + (paste_y_end - paste_y_start)
    bitmap_slice_x_end = bitmap_slice_x_start + (paste_x_end - paste_x_start)
    bitmap_char_slice = bitmap_char[
        bitmap_slice_y_start:bitmap_slice_y_end,
        bitmap_slice_x_start:bitmap_slice_x_end
    ]
    if (bitmap_char_slice.size > 0 and 
        bitmap_char_slice.shape == (paste_y_end - paste_y_start,
                                   paste_x_end - paste_x_start)):
        canvas_text[paste_y_start:paste_y_end, 
                    paste_x_start:paste_x_end] = bitmap_char_slice
    if border_size > 0:
        glyph_border = get_char_border(cdpt, font_size, 0)
        stroker = freetype.Stroker()
        # Use passed stroke_width, fallback to config or default
        if stroke_width is None:
            stroke_ratio = config.render.stroke_width if (config and hasattr(config.render, 'stroke_width')) else 0.07
        else:
            stroke_ratio = stroke_width
        stroke_radius = 64 * max(int(stroke_ratio * font_size), 1)
        stroker.set(stroke_radius, 
                   freetype.FT_STROKER_LINEJOIN_ROUND,
                   freetype.FT_STROKER_LINECAP_ROUND,
                   0)
        glyph_border.stroke(stroker, destroy=True)
        blyph = glyph_border.to_bitmap(freetype.FT_RENDER_MODE_NORMAL, 
                                      freetype.Vector(0, 0), True)
        bitmap_b = blyph.bitmap
        border_bitmap_rows = bitmap_b.rows
        border_bitmap_width = bitmap_b.width
        if (border_bitmap_rows * border_bitmap_width > 0 and 
            len(bitmap_b.buffer) == border_bitmap_rows * border_bitmap_width):
            bitmap_border = np.array(bitmap_b.buffer, dtype=np.uint8
                                   ).reshape((border_bitmap_rows, border_bitmap_width))
            char_bitmap_rows = bitmap.rows
            char_bitmap_width = bitmap.width
            
            # 改进的边框位置计算：直接基于字符位置和尺寸差异
            # 避免浮点累积误差
            size_diff_x = border_bitmap_width - char_bitmap_width
            size_diff_y = border_bitmap_rows - char_bitmap_rows
            
            # 边框应该围绕字符居中，所以偏移是尺寸差的一半
            pen_border_x = char_place_x - round(size_diff_x / 2.0)
            pen_border_y = char_place_y - round(size_diff_y / 2.0)
            paste_border_y_start = max(0, pen_border_y)
            paste_border_x_start = max(0, pen_border_x)
            paste_border_y_end = min(canvas_border.shape[0], pen_border_y + border_bitmap_rows)
            paste_border_x_end = min(canvas_border.shape[1], pen_border_x + border_bitmap_width)
            border_slice_y_start = paste_border_y_start - pen_border_y
            border_slice_x_start = paste_border_x_start - pen_border_x
            border_slice_y_end = border_slice_y_start + (paste_border_y_end - paste_border_y_start)
            border_slice_x_end = border_slice_x_start + (paste_border_x_end - paste_border_x_start)
            bitmap_border_slice = bitmap_border[
                border_slice_y_start:border_slice_y_end,
                border_slice_x_start:border_slice_x_end
            ]
            if (bitmap_border_slice.size > 0 and 
                bitmap_border_slice.shape == (paste_border_y_end - paste_border_y_start,
                                            paste_border_x_end - paste_border_x_start)):
                target_slice = canvas_border[
                    paste_border_y_start:paste_border_y_end,
                    paste_border_x_start:paste_border_x_end
                ]
                if target_slice.shape == bitmap_border_slice.shape:
                    canvas_border[paste_border_y_start:paste_border_y_end,
                                paste_border_x_start:paste_border_x_end] = cv2.add(
                        target_slice, bitmap_border_slice)
                else:
                    print("[Error] Shape mismatch during border paste: "
                         f"target={{target_slice.shape}}, source={{bitmap_border_slice.shape}}")
    return char_offset_x

def is_cjk_lang(lang: str):
    lang = lang.lower()
    # Check for common language codes for Chinese, Japanese, Korean
    return lang in ['chs', 'cht', 'jpn', 'kor', 'zh', 'ja', 'ko']

def put_text_horizontal(font_size: int, text: str, width: int, height: int, alignment: str,
                        reversed_direction: bool, fg: Tuple[int, int, int], bg: Tuple[int, int, int],
                        lang: str = 'en_US', hyphenate: bool = True, line_spacing: int = 0, config=None, region_count: int = 1, stroke_width: float = None):

    # 应用最大字体限制
    if config and hasattr(config.render, 'max_font_size') and config.render.max_font_size > 0:
        font_size = min(font_size, config.render.max_font_size)

    text = compact_special_symbols(text)
    if not text :
        logger.warning("[RENDER SKIPPED] Horizontal text is empty after processing")
        return

    layout_mode = 'default'
    if config:
        layout_mode = config.render.layout_mode

    # 初始化 stroke_ratio，避免 UnboundLocalError
    if stroke_width is None:
        stroke_ratio = config.render.stroke_width if (config and hasattr(config.render, 'stroke_width')) else 0.07
    else:
        stroke_ratio = stroke_width

    # 简化逻辑：统一处理BR标记，有BR就按它换行，没有就不换行
    text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)

    # 统一使用无限宽度：换行完全由BR/\n决定，不自动断行
    # 有BR时按BR换行，无BR时不换行（都是单行or多行由BR控制）
    width = 99999
    has_newline = '\n' in text
    logger.debug(f"[HORIZONTAL DEBUG] width=99999, has_newline={has_newline}")

    bg_size = int(max(font_size * stroke_ratio, 1)) if bg is not None else 0
    # line_spacing 是基本间距的倍率
    # 横排基本间距: 0.01
    spacing_y = int(font_size * 0.01 * (line_spacing or 1.0))
    if layout_mode != 'default' and is_cjk_lang(lang):
        line_text_list, line_width_list = calc_horizontal_cjk(font_size, text, width)
    else:
        line_text_list, line_width_list = calc_horizontal(font_size, text, width, height, lang, hyphenate)

    line_visual_extents_x = []
    line_visual_widths = []
    for line_text in line_text_list:
        line_left, line_right = _measure_horizontal_line_visual_extents(
            line_text,
            font_size,
            border_size=bg_size,
            config=config,
            stroke_ratio=stroke_ratio,
            reversed_direction=reversed_direction
        )
        line_visual_extents_x.append((line_left, line_right))
        line_visual_widths.append(max(0, line_right - line_left))
    max_visual_width = max(line_visual_widths) if line_visual_widths else 0

    canvas_w = max(max(line_width_list), max_visual_width) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2

    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()
    pen_orig = [font_size + bg_size, font_size + bg_size]

    if reversed_direction:
        pen_orig[0] = canvas_w - bg_size - 10

    for line_idx, (line_text, line_width) in enumerate(zip(line_text_list, line_width_list)):
        pen_line = pen_orig.copy()
        line_left, line_right = line_visual_extents_x[line_idx]
        line_visual_width = max(0, line_right - line_left)

        if not reversed_direction:
            slot_left = pen_orig[0]
            if alignment == 'center':
                target_left = slot_left + round((max_visual_width - line_visual_width) / 2.0)
            elif alignment == 'right':
                target_left = slot_left + (max_visual_width - line_visual_width)
            else:
                target_left = slot_left
            pen_line[0] = round(target_left - line_left)
        else:
            slot_right = pen_orig[0]
            slot_left = slot_right - max_visual_width
            if alignment == 'left':
                target_left = slot_left
            elif alignment == 'center':
                target_left = slot_left + round((max_visual_width - line_visual_width) / 2.0)
            else:
                target_left = slot_right - line_visual_width
            target_right = target_left + line_visual_width
            pen_line[0] = round(target_right - line_right)

        for char_idx, c in enumerate(line_text):
            if _is_newline_control_char(c):
                continue
            if reversed_direction:
                cdpt, rot_degree = CJK_Compatibility_Forms_translate(c, 0)
                glyph = get_char_glyph(cdpt, font_size, 0)
                offset_x = glyph.metrics.horiAdvance >> 6
                pen_line[0] -= offset_x
            offset_x = put_char_horizontal(font_size, c, pen_line, canvas_text, canvas_border, border_size=bg_size, config=config, stroke_width=stroke_ratio)
            if not reversed_direction:
                pen_line[0] += offset_x
        pen_orig[1] += spacing_y + font_size

    canvas_border = np.clip(canvas_border, 0, 255)
    line_box = add_color(canvas_text, fg, canvas_border, bg)
    combined_canvas = cv2.add(canvas_text, canvas_border)
    x, y, w, h = cv2.boundingRect(combined_canvas)
    
    if w == 0 or h == 0:
        logger.warning(f"[RENDER SKIPPED] Horizontal text rendered with zero width or height. Width: {w}, Height: {h}, Text: {text[:50]}...")
        return None
    
    result = line_box[y:y+h, x:x+w]
    return result

def test():
    import logging
    logger = logging.getLogger('manga_translator')
    canvas = put_text_horizontal(64, 1.0, '因为不同‼ [这"真的是普]通的》肉！那个"姑娘"的恶作剧！是吗？咲夜⁉', 400, (0, 0, 0), (255, 128, 128))
    imwrite_unicode('text_render_combined.png', canvas, logger)

# Initialize font selection on module load
update_font_selection()

if __name__ == '__main__':
    test()
