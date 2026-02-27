# auto_linebreak v2.1.0
# 完全自包含的换行引擎：竖排 <H> 块、CJK 标点禁则、英文连字符均内嵌在布局决策阶段
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, List, Tuple

from . import text_render
from .text_render import (
    CJK_Compatibility_Forms_translate,
    get_char_glyph,
    auto_add_horizontal_tags,
    should_rotate_horizontal_block_90,
    get_string_width,
    get_char_offset_x,
    get_char_offset_y,
    select_hyphenator,
    calc_horizontal_block_height,
    compact_special_symbols,
)


@dataclass
class NoBrLayoutResult:
    text_with_br: str
    font_size: int
    n_segments: int
    required_width: float
    required_height: float


def _normalize_no_br_text(text: str) -> str:
    text = compact_special_symbols(text or "")
    return re.sub(r"\s*(\[BR\]|<br>|【BR】)\s*", "", text, flags=re.IGNORECASE)


def _calculate_uniformity(values: List[float]) -> float:
    if not values or len(values) <= 1:
        return 0.0
    mean_v = sum(values) / len(values)
    if mean_v <= 0:
        return float("inf")
    variance = sum((v - mean_v) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean_v


def _hyphenate_enabled(config: Any) -> bool:
    return not (config and hasattr(config, "render") and getattr(config.render, "no_hyphenation", False))


# ---------------------------------------------------------------------------
# 竖排换行引擎（完全内嵌，不依赖 text_render.calc_vertical）
# ---------------------------------------------------------------------------

_H_BLOCK_RE = re.compile(r'(<H>.*?</H>)', re.IGNORECASE | re.DOTALL)
_BR_RE = re.compile(r'\s*(\[BR\]|<br>|【BR】)\s*', re.IGNORECASE)


def _h_block_height(font_size: int, content: str) -> int:
    """计算 <H> 横排块在竖排列中占用的高度，直接复用 text_render 的精确实现。"""
    return calc_horizontal_block_height(font_size, content)


def _vert_char_advance(font_size: int, cdpt: str) -> int:
    """单个字符的竖排进量（像素），与 text_render.get_char_offset_y 逻辑一致。"""
    cdpt_trans, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    try:
        slot = get_char_glyph(cdpt_trans, font_size, 1)
    except Exception:
        return font_size
    adv = font_size
    if hasattr(slot, 'metrics') and slot.metrics:
        if hasattr(slot.metrics, 'vertAdvance') and slot.metrics.vertAdvance:
            adv = slot.metrics.vertAdvance >> 6
        elif hasattr(slot.metrics, 'height') and slot.metrics.height:
            adv = slot.metrics.height >> 6
    if adv == font_size and hasattr(slot, 'advance') and slot.advance:
        if hasattr(slot.advance, 'y') and slot.advance.y:
            adv = slot.advance.y >> 6
    return adv


def _vert_char_bitmap_width(font_size: int, cdpt: str) -> int:
    """单个字符的竖排字形实际宽度。"""
    cdpt_trans, _ = CJK_Compatibility_Forms_translate(cdpt, 1)
    try:
        slot = get_char_glyph(cdpt_trans, font_size, 1)
        return slot.bitmap.width or font_size
    except Exception:
        return font_size


def _layout_vertical(font_size: int, text: str, max_height: int) -> Tuple[List[str], List[int]]:
    """
    竖排换行引擎，完全自包含。

    特性：
    1. auto_add_horizontal_tags 先把英文/数字词包成 <H> 块
    2. <H> 块用 _h_block_height 计算高度（和渲染一致）
    3. 普通 CJK 字符用 vertAdvance 逐字累积
    4. CJK_H2V 字形替换（通过 CJK_Compatibility_Forms_translate）
    5. [BR]/<br> 等统一预处理为 \\n
    6. 输出的 line 文本保留 <H> 标签供渲染侧使用

    返回 (line_text_list, line_height_list)
    """
    # 先加 <H> 标签，使英文词被识别为横排块
    text = auto_add_horizontal_tags(text)
    # 统一 BR 标记
    text = _BR_RE.sub('\n', text)

    line_text_list: List[str] = []
    line_height_list: List[int] = []

    for paragraph in text.split('\n'):
        if not paragraph:
            line_text_list.append('')
            line_height_list.append(0)
            continue

        current_line_text = ""
        current_line_height = 0

        for part in _H_BLOCK_RE.split(paragraph):
            if not part:
                continue

            is_h = part.lower().startswith('<h>') and part.lower().endswith('</h>')

            if is_h:
                content = part[3:-4]
                if not content:
                    continue
                block_h = _h_block_height(font_size, content)
                if current_line_height + block_h > max_height and current_line_text:
                    line_text_list.append(current_line_text)
                    line_height_list.append(current_line_height)
                    current_line_text = part
                    current_line_height = block_h
                else:
                    current_line_text += part
                    current_line_height += block_h
            else:
                for cdpt in part:
                    if not cdpt:
                        continue
                    adv = _vert_char_advance(font_size, cdpt)
                    if current_line_height + adv > max_height and current_line_text:
                        line_text_list.append(current_line_text)
                        line_height_list.append(current_line_height)
                        current_line_text = cdpt
                        current_line_height = adv
                    else:
                        current_line_text += cdpt
                        current_line_height += adv

        if current_line_text:
            line_text_list.append(current_line_text)
            line_height_list.append(current_line_height)

    if not line_text_list:
        line_text_list.append("")
        line_height_list.append(0)

    return line_text_list, line_height_list


def _vert_line_width(line_text: str, font_size: int) -> int:
    """竖排单列的实际最大字形宽度，与 put_text_vertical 的 line_widths 逻辑一致。"""
    max_width = font_size
    for part in _H_BLOCK_RE.split(line_text):
        if not part:
            continue
        is_h = part.lower().startswith('<h>') and part.lower().endswith('</h>')
        if is_h:
            # <H> 块居中置于列内，列宽取 font_size
            pass
        else:
            for c in part:
                w = _vert_char_bitmap_width(font_size, c)
                if w > max_width:
                    max_width = w
    return max_width


def _vert_total_height(text: str, font_size: int) -> int:
    """不换行时竖排文本的总高度，考虑 <H> 块。"""
    text = auto_add_horizontal_tags(_BR_RE.sub('', text))
    total = 0
    for part in _H_BLOCK_RE.split(text):
        if not part:
            continue
        is_h = part.lower().startswith('<h>') and part.lower().endswith('</h>')
        if is_h:
            content = part[3:-4]
            if content:
                total += _h_block_height(font_size, content)
        else:
            for c in part:
                total += _vert_char_advance(font_size, c)
    return total


# ---------------------------------------------------------------------------
# 横排 CJK 换行引擎（完全内嵌，含标点禁则）
# ---------------------------------------------------------------------------

_NO_START_CHARS = "》，。．」』】）！；：？"
_NO_END_CHARS = "《「『【（"


def _layout_horizontal_cjk(font_size: int, text: str, max_width: int) -> Tuple[List[str], List[int]]:
    """
    横排 CJK 换行，完全自包含。

    特性：
    1. [BR] 等统一为 \\n
    2. 标点禁则：行首禁则字符追到上一行；行尾禁则字符推到下一行
    """
    text = _BR_RE.sub('\n', text)
    lines: List[Tuple[str, int]] = []

    for paragraph in text.split('\n'):
        if not paragraph:
            lines.append(("", 0))
            continue

        current_line = ""
        current_width = 0

        for char in paragraph:
            char_width = get_char_offset_x(font_size, char)

            if current_width + char_width > max_width and current_line:
                # 行尾禁则：行尾不能以 no_end_chars 结尾 → 把末字推到下一行
                if current_line and current_line[-1] in _NO_END_CHARS:
                    last_char = current_line[-1]
                    current_line = current_line[:-1]
                    lines.append((current_line, get_string_width(font_size, current_line)))
                    current_line = last_char + char
                else:
                    lines.append((current_line, current_width))
                    current_line = char
                current_width = get_string_width(font_size, current_line)
            elif not current_line and char in _NO_START_CHARS:
                # 行首禁则：把它追加到上一行
                if lines:
                    prev_text, prev_w = lines[-1]
                    lines[-1] = (prev_text + char, prev_w + char_width)
                else:
                    current_line += char
                    current_width += char_width
            else:
                current_line += char
                current_width += char_width

        if current_line:
            lines.append((current_line, current_width))

    return [l[0] for l in lines], [l[1] for l in lines]


# ---------------------------------------------------------------------------
# 横排英文换行引擎（完全内嵌，含连字符断字 + 超宽自扩 + 优化 pass）
# ---------------------------------------------------------------------------

def _layout_horizontal_eng(
    font_size: int,
    text: str,
    max_width: int,
    language: str = 'en_US',
    hyphenate: bool = True,
) -> Tuple[List[str], List[int]]:
    """
    横排英文换行，完全自包含。

    特性：
    1. [BR] 等统一为 \\n，保留强制换行
    2. 超宽时自动扩大 max_width（防止死循环）
    3. Hyphenator 音节断字（语言敏感）
    4. 连字符优化 pass：把下一行音节塞到当前行
    5. 行合并 pass：相邻行合并节省行数
    """
    text = _BR_RE.sub('\n', text)
    max_width = max(max_width, 2 * font_size)

    space_w = get_char_offset_x(font_size, ' ')
    hyphen_w = get_char_offset_x(font_size, '-')

    paragraphs = text.split('\n')
    words: List[str] = []
    newline_positions: set = set()

    for para_idx, paragraph in enumerate(paragraphs):
        if paragraph.strip():
            para_words = re.split(r'[ \t]+', paragraph)
            words.extend(para_words)
            if para_idx < len(paragraphs) - 1:
                newline_positions.add(len(words) - 1)
        elif para_idx < len(paragraphs) - 1:
            words.append('')
            newline_positions.add(len(words) - 1)

    if not words:
        return [], []

    word_widths = [get_string_width(font_size, w) for w in words]

    # 超宽自动扩 max_width
    max_height = 99999
    while True:
        max_lines = max_height // font_size + 1
        expected_size = sum(word_widths) + max((len(word_widths) - 1) * space_w - (max_lines - 1) * hyphen_w, 0)
        max_size = max_width * max_lines
        if max_size < expected_size:
            multiplier = math.sqrt(expected_size / max_size)
            max_width = int(max_width * max(multiplier, 1.05))
            max_height *= multiplier
        else:
            break

    hyphenator = select_hyphenator(language) if hyphenate else None

    # 切音节
    syllables: List[List[str]] = []
    for word in words:
        new_syls: List[str] = []
        if hyphenator and len(word) <= 100:
            try:
                new_syls = hyphenator.syllables(word)
            except Exception:
                new_syls = []
        if not new_syls:
            new_syls = [word] if len(word) <= 3 else list(word)
        normalized: List[str] = []
        for syl in new_syls:
            if get_string_width(font_size, syl) > max_width:
                normalized.extend(list(syl))
            else:
                normalized.append(syl)
        syllables.append(normalized)

    # 主换行 pass
    line_words_list: List[List[int]] = []
    line_width_list: List[int] = []
    hyphenation_idx_list: List[int] = []
    line_words: List[int] = []
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

    def get_syllables_range(line_idx, word_pos):
        while word_pos < 0:
            word_pos += len(line_words_list[line_idx])
        word_idx = line_words_list[line_idx][word_pos]
        syl_start = 0
        syl_end = len(syllables[word_idx])
        if line_idx > 0 and word_pos == 0 and line_words_list[line_idx - 1][-1] == word_idx:
            syl_start = hyphenation_idx_list[line_idx - 1]
        if line_idx < len(line_words_list) - 1 and word_pos == len(line_words_list[line_idx]) - 1 and line_words_list[line_idx + 1][0] == word_idx:
            syl_end = hyphenation_idx_list[line_idx]
        return syl_start, syl_end

    i = 0
    while True:
        if i >= len(words):
            if line_width > 0:
                break_line()
            break
        cur_w = space_w if line_width > 0 else 0
        if line_width + cur_w + word_widths[i] <= max_width + hyphen_w:
            line_words.append(i)
            line_width += cur_w + word_widths[i]
            i += 1
            if (i - 1) in newline_positions:
                break_line()
        elif word_widths[i] > max_width:
            j = 0
            hyphenation_idx = 0
            while j < len(syllables[i]):
                syl = syllables[i][j]
                sw = get_string_width(font_size, syl)
                if line_width + cur_w + sw <= max_width:
                    cur_w += sw
                    j += 1
                    hyphenation_idx = j
                else:
                    if hyphenation_idx > 0:
                        line_words.append(i)
                        line_width += cur_w
                    cur_w = 0
                    break_line()
            line_words.append(i)
            line_width += cur_w
            i += 1
            if (i - 1) in newline_positions:
                break_line()
        else:
            if hyphenate:
                break_line()
            else:
                line_words.append(i)
                line_width += cur_w + word_widths[i]
                i += 1
                if (i - 1) in newline_positions:
                    break_line()

    # 连字符优化 pass
    max_lines = max_height // font_size + 1
    if hyphenate and len(line_words_list) > max_lines:
        li = 0
        while li < len(line_words_list) - 1:
            lw1 = line_words_list[li]
            lw2 = line_words_list[li + 1]
            left_space = max_width - line_width_list[li]
            first_word = True
            while lw2:
                widx = lw2[0]
                if first_word and widx == lw1[-1]:
                    ss = hyphenation_idx_list[li]
                    se = hyphenation_idx_list[li + 1] if li < len(line_width_list) - 2 and widx == line_words_list[li + 2][0] else len(syllables[widx])
                else:
                    left_space -= space_w
                    ss = 0
                    se = len(syllables[widx]) if len(lw2) > 1 else hyphenation_idx_list[li + 1]
                first_word = False
                cur_w = 0
                for si in range(ss, se):
                    sw = get_string_width(font_size, syllables[widx][si])
                    if left_space > cur_w + sw:
                        cur_w += sw
                    else:
                        if cur_w > 0:
                            left_space -= cur_w
                            line_width_list[li] = max_width - left_space
                            hyphenation_idx_list[li] = si
                            lw1.append(widx)
                        break
                else:
                    left_space -= cur_w
                    line_width_list[li] = max_width - left_space
                    lw1.append(widx)
                    lw2.pop(0)
                    continue
                break
            if not lw2:
                line_words_list.pop(li + 1)
                line_width_list.pop(li + 1)
                hyphenation_idx_list.pop(li)
            else:
                li += 1

    # 行合并 pass
    li = 0
    while li < len(line_words_list) - 1:
        lw1 = line_words_list[li]
        lw2 = line_words_list[li + 1]
        merged_widx = -1
        if lw1[-1] == lw2[0]:
            s1, e1 = get_syllables_range(li, -1)
            s2, e2 = get_syllables_range(li + 1, 0)
            w1_text = ''.join(syllables[lw1[-1]][s1:e1])
            w2_text = ''.join(syllables[lw2[0]][s2:e2])
            w1_w = get_string_width(font_size, w1_text)
            w2_w = get_string_width(font_size, w2_text)
            if len(w2_text) == 1 or w2_w < font_size:
                merged_widx = lw1[-1]
                lw2.pop(0)
                line_width_list[li] += w2_w
                line_width_list[li + 1] -= w2_w + space_w
            elif len(w1_text) == 1 or w1_w < font_size:
                merged_widx = lw1[-1]
                lw1.pop(-1)
                line_width_list[li] -= w1_w + space_w
                line_width_list[li + 1] += w1_w
        if not lw1:
            line_words_list.pop(li)
            line_width_list.pop(li)
            hyphenation_idx_list.pop(li)
        elif not lw2:
            line_words_list.pop(li + 1)
            line_width_list.pop(li + 1)
            hyphenation_idx_list.pop(li)
        elif li >= len(line_words_list) - 1 or line_words_list[li + 1] != merged_widx:
            li += 1

    use_hyphen_chars = hyphenate and hyphenator and max_width > 1.5 * font_size and len(words) > 1

    line_text_list: List[str] = []
    for li, line in enumerate(line_words_list):
        line_text = ''
        for j, widx in enumerate(line):
            s, e = get_syllables_range(li, j)
            line_text += ''.join(syllables[widx][s:e])
            if not line_text:
                continue
            if j == 0 and li > 0 and line_text_list and line_text_list[-1].endswith('-') and line_text.startswith('-'):
                line_text = line_text[1:]
                line_width_list[li] -= hyphen_w
            if j < len(line) - 1 and line_text:
                line_text += ' '
            elif use_hyphen_chars and e != len(syllables[widx]) and len(words[widx]) > 3 and not line_text.endswith('-') and not (e < len(syllables[widx]) and not re.search(r'\w', syllables[widx][e][0])):
                line_text += '-'
                line_width_list[li] += hyphen_w
        line_width_list[li] = get_string_width(font_size, line_text)
        line_text_list.append(line_text)

    return line_text_list, line_width_list


# ---------------------------------------------------------------------------
# 统一的布局调度函数
# ---------------------------------------------------------------------------

def _is_cjk_lang(lang: str) -> bool:
    lang = (lang or '').lower()
    return any(lang.startswith(p) for p in ('zh', 'ja', 'ko'))


def _calc_horizontal_layout(
    font_size: int,
    text: str,
    max_width: int,
    target_lang: str,
    hyphenate: bool,
) -> Tuple[List[str], List[int]]:
    width = max(1, int(max_width))
    if _is_cjk_lang(target_lang or 'en_US'):
        return _layout_horizontal_cjk(font_size, text, width)
    return _layout_horizontal_eng(font_size, text, width, language=target_lang or 'en_US', hyphenate=hyphenate)


def _calc_vertical_layout(
    font_size: int,
    text: str,
    max_height: int,
    config: Any,
) -> Tuple[List[str], List[int]]:
    height = max(1, int(max_height))
    return _layout_vertical(font_size, text, height)


# ---------------------------------------------------------------------------
# fallback: 像素预算均匀插 [BR]
# ---------------------------------------------------------------------------

def _insert_br_by_pixel_budget(text: str, n_segments: int, font_size: int, horizontal: bool) -> str:
    if not text or n_segments <= 1:
        return text

    text_len = len(text)
    if text_len <= 1:
        return text

    n_segments = max(1, min(n_segments, text_len))
    n_breaks = n_segments - 1
    if n_breaks <= 0:
        return text

    if horizontal:
        advances = [max(0, get_char_offset_x(font_size, c)) for c in text]
    else:
        advances = [max(0, get_char_offset_y(font_size, c)) for c in text]

    prefix: List[int] = []
    total = 0
    for adv in advances:
        total += adv
        prefix.append(total)

    if total <= 0:
        step = text_len / n_segments
        break_positions = []
        prev = 0
        for k in range(1, n_segments):
            pos = int(round(step * k))
            pos = max(prev + 1, min(pos, text_len - (n_segments - k)))
            break_positions.append(pos)
            prev = pos
    else:
        break_positions = []
        prev = 0
        for k in range(1, n_segments):
            target = total * (k / n_segments)
            min_pos = prev + 1
            max_pos = text_len - (n_segments - k)
            if min_pos > max_pos:
                break
            idx = bisect_left(prefix, target)
            candidates = []
            for ci in (idx - 1, idx):
                pos = ci + 1
                if min_pos <= pos <= max_pos:
                    candidates.append(pos)
            if candidates:
                pos = min(candidates, key=lambda p: abs(prefix[p - 1] - target))
            else:
                pos = min(max(idx + 1, min_pos), max_pos)
            break_positions.append(pos)
            prev = pos

    if not break_positions:
        return text

    break_set = set(break_positions)
    out = []
    for i, ch in enumerate(text, start=1):
        out.append(ch)
        if i in break_set and i < text_len:
            out.append("[BR]")
    return "".join(out)


# ---------------------------------------------------------------------------
# 最优换行搜索
# ---------------------------------------------------------------------------

def _find_best_lines_for_target_segments(
    clean_text: str,
    font_size: int,
    horizontal: bool,
    target_segments: int,
    target_lang: str,
    config: Any,
) -> List[str]:
    if not clean_text:
        return []

    hyphenate = _hyphenate_enabled(config)

    if horizontal:
        base_lines, base_metrics = _calc_horizontal_layout(font_size, clean_text, 99999, target_lang, hyphenate)
        total_budget = max(1, int(max(base_metrics))) if base_metrics else max(1, get_string_width(font_size, clean_text))
    else:
        base_lines, base_metrics = _calc_vertical_layout(font_size, clean_text, 99999, config)
        total_budget = max(1, int(max(base_metrics))) if base_metrics else max(1, _vert_total_height(clean_text, font_size))

    _ = base_lines
    min_budget = max(1, int(font_size))
    max_budget = max(min_budget, total_budget)
    target_segments = max(1, target_segments)

    evaluated = {}

    def evaluate(budget: int):
        budget = max(min_budget, min(int(budget), max_budget))
        if budget in evaluated:
            return evaluated[budget]

        if horizontal:
            lines, metrics = _calc_horizontal_layout(font_size, clean_text, budget, target_lang, hyphenate)
        else:
            lines, metrics = _calc_vertical_layout(font_size, clean_text, budget, config)

        if not lines:
            evaluated[budget] = None
            return None

        line_count = len(lines)
        uniformity = _calculate_uniformity(metrics if metrics else [len(line) for line in lines])
        score = (abs(line_count - target_segments), 1 if line_count > target_segments else 0, uniformity)
        evaluated[budget] = (score, lines, line_count)
        return evaluated[budget]

    low, high = min_budget, max_budget
    for _ in range(24):
        if low > high:
            break
        mid = (low + high) // 2
        result = evaluate(mid)
        if result is None:
            break
        _, _, line_count = result
        if line_count > target_segments:
            low = mid + 1
        else:
            high = mid - 1

    anchors = {min_budget, max_budget, low, high, low - 1, low + 1, high - 1, high + 1}
    base = max_budget / max(1, target_segments)
    for factor in (0.75, 0.9, 1.0, 1.1, 1.25):
        anchors.add(int(round(base * factor)))
    for anchor in anchors:
        evaluate(anchor)

    candidates = [v for v in evaluated.values() if v is not None]
    if not candidates:
        return []
    _, best_lines, _ = min(candidates, key=lambda item: item[0])
    # 保留 <H> 标签，渲染侧用于竖排内嵌横排
    return best_lines


# ---------------------------------------------------------------------------
# 尺寸度量
# ---------------------------------------------------------------------------

def _measure_required_size(
    text_with_br: str,
    font_size: int,
    horizontal: bool,
    line_spacing_multiplier: float,
    target_lang: str,
    config: Any,
) -> Tuple[int, float, float]:
    hyphenate = _hyphenate_enabled(config)

    if horizontal:
        lines, widths = _calc_horizontal_layout(font_size, text_with_br, 99999, target_lang, hyphenate)
        n = max(1, len(lines))
        spacing_y = int(font_size * 0.01 * line_spacing_multiplier)
        required_width = max(widths) if widths else get_string_width(font_size, _normalize_no_br_text(text_with_br))
        required_height = font_size * n + spacing_y * max(0, n - 1)
        return n, float(required_width), float(required_height)

    lines, heights = _calc_vertical_layout(font_size, text_with_br, 99999, config)
    n = max(1, len(lines))
    spacing_x = int(font_size * 0.2 * line_spacing_multiplier)
    required_height = max(heights) if heights else _vert_total_height(_normalize_no_br_text(text_with_br), font_size)
    # 精确计算各列实际字形宽度之和，与 put_text_vertical 的 line_widths 逻辑一致
    line_widths = [_vert_line_width(line, font_size) for line in lines]
    required_width = sum(line_widths) + spacing_x * max(0, n - 1)
    return n, float(required_width), float(required_height)


# ---------------------------------------------------------------------------
# 公开入口
# ---------------------------------------------------------------------------

def solve_no_br_layout(
    text: str,
    horizontal: bool,
    seed_segments: int,
    seed_font_size: int,
    bubble_width: float,
    bubble_height: float,
    min_font_size: int,
    max_font_size: int,
    line_spacing_multiplier: float,
    target_lang: str = "en_US",
    config: Any = None,
    iterations: int = 3,
) -> NoBrLayoutResult:
    clean_text = _normalize_no_br_text(text)
    if not clean_text:
        return NoBrLayoutResult("", max(1, min_font_size), 1, 0.0, 0.0)
    current_region = getattr(config, "_current_region", None) if config is not None else None
    force_no_wrap_single_region = bool(
        current_region is not None
        and hasattr(current_region, "lines")
        and len(current_region.lines) == 1
    )

    text_len = len(clean_text)
    safe_min_font = max(1, int(min_font_size))
    safe_max_font = max(safe_min_font, int(max_font_size))
    current_font = max(safe_min_font, min(int(seed_font_size), safe_max_font))
    current_segments = max(1, min(int(seed_segments), text_len))
    line_spacing_multiplier = line_spacing_multiplier or 1.0

    bw = bubble_width if isinstance(bubble_width, (int, float)) and bubble_width > 0 else 1.0
    bh = bubble_height if isinstance(bubble_height, (int, float)) and bubble_height > 0 else 1.0

    for _ in range(max(1, int(iterations))):
        lines = _find_best_lines_for_target_segments(
            clean_text,
            current_font,
            horizontal,
            current_segments,
            target_lang,
            config,
        )
        if force_no_wrap_single_region:
            text_with_br = clean_text
        elif lines and len(lines) > 1:
            text_with_br = "[BR]".join(lines)
        elif current_segments > 1:
            text_with_br = _insert_br_by_pixel_budget(clean_text, current_segments, current_font, horizontal)
        else:
            text_with_br = clean_text

        n_actual, required_width, required_height = _measure_required_size(
            text_with_br,
            current_font,
            horizontal,
            line_spacing_multiplier,
            target_lang,
            config,
        )

        if required_width <= 0 or required_height <= 0:
            break

        fit_scale = min(bw / required_width, bh / required_height)
        if not math.isfinite(fit_scale) or fit_scale <= 0:
            fit_scale = 1.0
        next_font = max(safe_min_font, min(int(current_font * fit_scale), safe_max_font))
        next_segments = max(1, min(n_actual, text_len))

        if next_font == current_font and next_segments == current_segments:
            return NoBrLayoutResult(text_with_br, current_font, n_actual, required_width, required_height)

        current_font = next_font
        current_segments = next_segments

    final_lines = _find_best_lines_for_target_segments(
        clean_text,
        current_font,
        horizontal,
        current_segments,
        target_lang,
        config,
    )
    if force_no_wrap_single_region:
        final_text = clean_text
    elif final_lines and len(final_lines) > 1:
        final_text = "[BR]".join(final_lines)
    elif current_segments > 1:
        final_text = _insert_br_by_pixel_budget(clean_text, current_segments, current_font, horizontal)
    else:
        final_text = clean_text

    n_final, required_width, required_height = _measure_required_size(
        final_text,
        current_font,
        horizontal,
        line_spacing_multiplier,
        target_lang,
        config,
    )
    return NoBrLayoutResult(final_text, current_font, n_final, required_width, required_height)
