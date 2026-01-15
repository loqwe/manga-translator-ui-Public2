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
    **dict(zip(CJK_H2V.items(), CJK_H2V.keys())),
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
    # 合并连续的省略号为一个
    text = re.sub(r'…+', '…', text)
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
    - 多词英文词组（如 "Tek Tok"）：整体横排显示
    - 独立的短英文单词：添加<H>标签
    - 符号（!?）2-4个：横排显示

    渲染规则（在渲染时根据长度决定）：
    - 2个字符：横排显示
    - 3个及以上字符：竖排显示但每个字符旋转90度
    """
    # 如果文本中已有<H>标签，则不进行处理，以尊重手动设置
    if '<H>' in text or '<h>' in text.lower():
        return text

    # 步骤1：为多词英文词组添加<H>标签（至少2个单词，用空格分隔）
    # 匹配：字母/数字 + 空格 + 字母/数字（可以重复多次）
    # 注意：移除了点号(.)以避免匹配省略号
    multi_word_pattern = r'[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+(?:\s+[a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]+)+'
    text = re.sub(multi_word_pattern, r'<H>\g<0></H>', text)

    # 步骤2：对剩余的独立英文单词添加<H>标签
    # 匹配2个及以上字符，排除已经在<H>标签内的内容
    # 注意：移除了点号(.)以避免匹配省略号
    word_pattern = r'(?<![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])([a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-]{2,})(?![a-zA-Z0-9\uff21-\uff3a\uff41-\uff5a\uff10-\uff19_-])'

    # 只替换不在<H>标签内的匹配
    def replace_word(match):
        # 检查匹配位置是否在<H>...</H>之间
        start_pos = match.start()
        # 简单检查：查找前面最近的<H>和</H>
        text_before = text[:start_pos]
        last_open = text_before.rfind('<H>')
        last_close = text_before.rfind('</H>')
        if last_open > last_close:
            # 在<H>标签内，不替换
            return match.group(0)
        return f'<H>{match.group(1)}</H>'

    text = re.sub(word_pattern, replace_word, text)

    # 步骤3：匹配符号（2-4个，同时支持半角和全角）
    symbol_pattern = r'[!?！？]{2,4}'
    text = re.sub(symbol_pattern, r'<H>\g<0></H>', text)

    return text
    
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
            FONT = freetype.Face(Path(DEFAULT_FONT).open('rb'))
        except (freetype.ft_errors.FT_Exception, FileNotFoundError):
            logger.critical("Default font could not be loaded. Please check your installation.")
            FONT = None
        update_font_selection()
        get_char_glyph.cache_clear()
        return

    try:
        FONT = freetype.Face(Path(resolved_path).open('rb'))
    except (freetype.ft_errors.FT_Exception, FileNotFoundError):
        logger.error(f'Could not load font: {resolved_path}')
        try:
            FONT = freetype.Face(Path(DEFAULT_FONT).open('rb'))
        except (freetype.ft_errors.FT_Exception, FileNotFoundError):
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

def calc_horizontal_block_height(font_size: int, content: str) -> int:
    """
    预先计算横排块在竖排文本中的实际渲染高度
    用于准确计算竖排文本的总高度，特别是在智能缩放模式下

    注意：需要与 put_text_vertical 中的渲染逻辑保持一致
    - 2个字符：横排显示，返回横排块的实际高度
    - 3个及以上字符：竖排显示但每个字符旋转90度，返回所有字符的累计高度
    """
    if not content:
        return font_size

    # 判断：2个字符横排，3个及以上字符竖排但旋转90度
    if len(content) >= 3:
        # --- 计算竖排旋转块的高度 ---
        # 使用与渲染相同的方式：横排渲染后旋转90度
        # 旋转后，原来的宽度变成高度
        total_width = get_string_width(font_size, content)
        # 旋转90度后，宽度变成高度
        return total_width
    else:
        # --- 计算横排块的高度（2个字符） ---
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
    # 统一处理所有类型的AI换行符，确保后续逻辑的正确性
    text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)

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
                    if not cdpt:
                        continue
                    
                    cdpt_trans, rot_degree = CJK_Compatibility_Forms_translate(cdpt, 1)
                    ckpt = get_char_glyph(cdpt_trans, font_size, 1)
                    bitmap = ckpt.bitmap

                    if bitmap.rows * bitmap.width == 0 or len(bitmap.buffer) != bitmap.rows * bitmap.width:
                        char_offset_y = ckpt.metrics.vertAdvance >> 6 if hasattr(ckpt.metrics, 'vertAdvance') and ckpt.metrics.vertAdvance != 0 else font_size
                    else:
                        char_offset_y = ckpt.metrics.vertAdvance >> 6 if hasattr(ckpt.metrics, 'vertAdvance') and ckpt.metrics.vertAdvance != 0 else font_size

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
    # The pen's x-coordinate is the right boundary of the line. Center the character in the column.
    # 使用round()而不是整数除法，提高居中精度
    char_place_x = (pen[0] - line_width) + round((line_width - char_bitmap_width) / 2.0)
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

    # Conditional wrapping logic based on disable_auto_wrap and region_count
    # 检测文本中是否有BR标记或换行符
    has_br = bool(re.search(r'(\[BR\]|【BR】|<br>|\n)', text, flags=re.IGNORECASE))
    
    effective_max_height = h
    if config and config.render.disable_auto_wrap:
        if has_br:
            # 当AI断句开启且有BR标记时，使用无限高度，让文本按AI断句标记换行
            effective_max_height = 99999
            logger.debug(f"[VERTICAL DEBUG] AI断句开启且有BR标记，effective_max_height=99999, region_count={region_count}")
        else:
            # AI断句开启但没有BR标记，根据 layout_mode 回退
            # 严格模式(strict)和智能缩放(smart_scaling)在单行时都强制不换行
            if (config.render.layout_mode in ['smart_scaling', 'strict']) and region_count <= 1:
                effective_max_height = 99999
                logger.debug(f"[VERTICAL DEBUG] AI断句开启但无BR标记，单行Strict/SmartScaling，effective_max_height=99999, region_count={region_count}")
            else:
                # 回退到自动换行模式
                effective_max_height = h
                logger.debug(f"[VERTICAL DEBUG] AI断句开启但无BR标记，回退自动换行，effective_max_height={h}, region_count={region_count}")
    elif config and config.render.layout_mode == 'smart_scaling':
        # smart_scaling 模式下，检查是否有换行符
        # 只在非AI断句模式下检测方向不匹配
        force_wrap = should_force_wrap_for_mismatched_direction(config, logger)
        
        if has_br:
            # 有换行符，使用传入的框高度
            effective_max_height = h
            logger.debug(f"[VERTICAL DEBUG] Smart scaling有换行符，effective_max_height={h}, region_count={region_count}")
        else:
            # 无换行符，单区域使用无限高度
            if region_count <= 1 and not force_wrap:
                effective_max_height = 99999
                logger.debug(f"[VERTICAL DEBUG] Smart scaling单区域无换行符，effective_max_height=99999, region_count={region_count}")
            else:
                effective_max_height = h
                if force_wrap:
                    logger.debug(f"[VERTICAL DEBUG] Smart scaling强制换行模式，effective_max_height={h}, region_count={region_count}")
                else:
                    logger.debug(f"[VERTICAL DEBUG] Smart scaling多区域无换行符，effective_max_height={h}, region_count={region_count}")
    else:
        logger.debug(f"[VERTICAL DEBUG] 默认模式，effective_max_height={h}, region_count={region_count}")

    # Use original font size for line breaking calculation
    line_text_list, line_height_list = calc_vertical(font_size, text, effective_max_height, config=config)
    if not line_height_list:
        return

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
                    cdpt, _ = CJK_Compatibility_Forms_translate(c, 1)
                    slot = get_char_glyph(cdpt, font_size, 1)
                    if slot.bitmap.width > max_width:
                        max_width = slot.bitmap.width
        line_widths.append(max_width)

    # 使用实际列宽计算画布大小
    canvas_x = sum(line_widths) + spacing_x * (len(line_text_list) - 1) + (font_size + bg_size) * 2
    canvas_y = max(line_height_list) + (font_size + bg_size) * 2

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
                    cdpt, _ = CJK_Compatibility_Forms_translate(c, 1)
                    slot = get_char_glyph(cdpt, font_size, 1)
                    bitmap = slot.bitmap
                    # 使用实际bitmap宽度，确保不会超出列边界
                    if bitmap.width > max_char_width:
                        max_char_width = bitmap.width
        line_max_widths.append(max_char_width)

    for line_idx, (line_text, line_height) in enumerate(zip(line_text_list, line_height_list)):
        pen_line = pen_orig.copy()
        # 使用该列的实际最大字符宽度作为列宽
        line_width = line_max_widths[line_idx]

        if alignment == 'center':
            # 使用round()提高居中精度
            pen_line[1] += round((max(line_height_list) - line_height) / 2.0)
        elif alignment == 'right': # In vertical, right means bottom
            pen_line[1] += max(line_height_list) - line_height

        parts = re.split(r'(<H>.*?</H>)', line_text, flags=re.IGNORECASE | re.DOTALL)

        for part in parts:
            if not part:
                continue

            is_horizontal_block = part.lower().startswith('<h>') and part.lower().endswith('</h>')

            if is_horizontal_block:
                content = part[3:-4]
                if not content:
                    continue

                # 判断：2个字符横排，3个及以上字符竖排但旋转90度
                if len(content) >= 3:
                    # --- RENDER ROTATED BLOCK (竖排但每个字符旋转90度) ---
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
                    line_start_x = pen_line[0] - font_size
                    paste_x = line_start_x + round((font_size - rw) / 2.0)
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
                    line_start_x = pen_line[0] - font_size
                    paste_x = line_start_x + round((font_size - rw) / 2.0)

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
    try:
        return Hyphenator(lang)
    except Exception:
        return None

def get_char_offset_x(font_size: int, cdpt: str):
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
    return sum([get_char_offset_x(font_size, c) for c in text])

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
            if j == 0 and i > 0 and line_text_list[-1][-1] == '-' and line_text[0] == '-':
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
    
    # 当AI断句开启时，统一处理换行符并使用无限宽度
    if config and config.render.disable_auto_wrap:
        # 统一处理所有类型的AI换行符
        text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)
        
        has_newline = '\n' in text
        if has_newline:
            # 有AI换行符，使用无限宽度
            width = 99999
            logger.debug(f"[HORIZONTAL DEBUG] AI断句开启且有换行符，width=99999, region_count={region_count}")
        else:
            # 无换行符，检查是否单行严格/智能模式
            if (config.render.layout_mode in ['smart_scaling', 'strict']) and region_count <= 1:
                width = 99999
                logger.debug(f"[HORIZONTAL DEBUG] AI断句开启但无换行符，单行Strict/SmartScaling，width=99999, region_count={region_count}")
            else:
                # 保持原宽度（允许自动换行）
                logger.debug(f"[HORIZONTAL DEBUG] AI断句开启但无换行符，回退自动换行，width={width}, region_count={region_count}")
                
    elif layout_mode == 'smart_scaling':
        # In smart_scaling mode, wrapping is conditional.
        # It wraps only if manual line breaks ([BR] or \n) are present.
        # Otherwise, it expands without wrapping.
        # 统一处理所有类型的AI换行符
        text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text, flags=re.IGNORECASE)
        
        # 只在非AI断句模式下检测方向不匹配
        force_wrap = should_force_wrap_for_mismatched_direction(config, logger)
        
        if '\n' not in text and not force_wrap:
            # No manual breaks found, so disable wrapping by setting a large width.
            if region_count <= 1:
                width = 99999
                logger.debug(f"[HORIZONTAL DEBUG] Smart scaling单区域无换行符，width=99999, region_count={region_count}")
            else:
                logger.debug(f"[HORIZONTAL DEBUG] Smart scaling多区域无换行符，width={width}, region_count={region_count}")
        else:
            if force_wrap:
                logger.debug(f"[HORIZONTAL DEBUG] Smart scaling强制换行模式，width={width}, region_count={region_count}")
            else:
                logger.debug(f"[HORIZONTAL DEBUG] Smart scaling有换行符，width={width}, region_count={region_count}, line_count={text.count(chr(10))+1}")
    bg_size = int(max(font_size * stroke_ratio, 1)) if bg is not None else 0
    # line_spacing 是基本间距的倍率
    # 横排基本间距: 0.01
    spacing_y = int(font_size * 0.01 * (line_spacing or 1.0))
    if layout_mode != 'default' and is_cjk_lang(lang):
        line_text_list, line_width_list = calc_horizontal_cjk(font_size, text, width)
    else:
        line_text_list, line_width_list = calc_horizontal(font_size, text, width, height, lang, hyphenate)

    canvas_w = max(line_width_list) + (font_size + bg_size) * 2
    canvas_h = font_size * len(line_width_list) + spacing_y * (len(line_width_list) - 1) + (font_size + bg_size) * 2

    canvas_text = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
    canvas_border = canvas_text.copy()
    pen_orig = [font_size + bg_size, font_size + bg_size]

    if reversed_direction:
        pen_orig[0] = canvas_w - bg_size - 10

    for line_idx, (line_text, line_width) in enumerate(zip(line_text_list, line_width_list)):
        pen_line = pen_orig.copy()

        if alignment == 'center':
            # 使用round()提高居中精度
            pen_line[0] += round((max(line_width_list) - line_width) / 2.0) * (-1 if reversed_direction else 1)
        elif alignment == 'right' and not reversed_direction:
            pen_line[0] += max(line_width_list) - line_width
        elif alignment == 'left' and reversed_direction:
            pen_line[0] -= max(line_width_list) - line_width
            pen_line[0] = max(line_width, pen_line[0])

        for char_idx, c in enumerate(line_text):
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
