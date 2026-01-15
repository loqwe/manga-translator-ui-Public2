from __future__ import annotations

import re
from typing import Iterable, Optional, Tuple, Union

# 可选依赖：更快的语言识别（pycld2）
try:
    import pycld2 as cld2
except Exception:
    cld2 = None


_HANGUL_RE = re.compile(r"[\uAC00-\uD7A3]")  # 가-힘
_KANA_RE = re.compile(r"[\u3040-\u30FF]")   # ぁ-ヿ (平/片假名)

# 轻量提示：越南语独有字母（排除与法语/葡萄牙语共享的 âêô 等）
# 只保留越南语独有的：đ/Đ（横杠d）、ă/Ă（短音a）、ơ/Ơ（带钩o）、ư/Ư（带钩u）
_VI_HINT_RE = re.compile(r"[đĐăĂơƠưƯ]")
# 轻量提示：西语常见符号/字母
_ES_HINT_RE = re.compile(r"[¿¡ñÑ]")

_LATIN_TOKEN_RE = re.compile(r"[a-z\u00C0-\u024F]+", re.IGNORECASE)

# 方法名称映射（用于日志显示中文）
_METHOD_NAMES = {
    'script': '文字特征',
    'heuristic': '启发式',
    'langid': 'langid统计',
    'pycld2': 'pycld2',
    'vi_fallback_es': '越语回退西语',
    'none': '未检测',
}

# 少量高区分度词表（用于在 langid 失准时兜底）
_SPANISH_HINT_WORDS = {
    'qué', 'porque', 'por', 'estoy', 'eres', 'soy', 'para', 'pero', 'como', 'cómo',
    'no', 'sí', 'si', 'con', 'sin', 'muy', 'más', 'menos', 'dónde', 'donde', 'cuando', 'cuándo',
}

_ENGLISH_HINT_WORDS = {
    'the', 'and', 'you', 'your', 'i', "i'm", 'im', 'me', 'my', 'to', 'of', 'in', 'is', 'are',
    'what', 'why', 'how', 'dont', "don't", 'can', 'cant', "can't",
}


def detect_source_lang(texts: Union[str, Iterable[str]]) -> Tuple[Optional[str], Optional[str], Optional[float], str]:
    """检测源语言（返回内部语言码，如 KOR/JPN/ENG/IND/ESP/VIN）。

    返回值:
        (lang_code, iso_lang, confidence, method)

    说明:
        - lang_code: 内部语言码（None 表示无法判定）
        - iso_lang: 语言检测返回的 ISO 639-1（或近似）代码（None 表示无）
        - confidence: 置信度（脚本启发式返回 None）
        - method: 检测方法名称（中文）
    """
    if isinstance(texts, str):
        merged = texts
    else:
        merged = "\n".join(t for t in texts if t)

    merged = (merged or "").strip()
    if not merged:
        return None, None, None, 'none'

    # 1) 预统计：避免“少量外语字符”误判整段语言
    hangul_count = len(_HANGUL_RE.findall(merged))
    kana_count = len(_KANA_RE.findall(merged))
    latin_count = len(re.findall(r"[A-Za-z]", merged))
    letter_total = hangul_count + kana_count + latin_count

    # 2) 脚本强特征（仅在占比足够高时判定为日韩语）
    if letter_total > 0:
        hangul_ratio = hangul_count / letter_total
        kana_ratio = kana_count / letter_total

        # 例：英语/西语里夹了 1 个韩语词，不应整体判定为韩语
        if hangul_count >= 4 and hangul_ratio >= 0.25:
            return 'KOR', 'ko', None, _METHOD_NAMES['script']
        if kana_count >= 4 and kana_ratio >= 0.25:
            return 'JPN', 'ja', None, _METHOD_NAMES['script']

    # 3) 明显提示优先（py3langid 对西语有时会失准）
    # 注：越南语启发式检测已禁用，避免误判其他语言
    # if _VI_HINT_RE.search(merged):
    #     return 'VIN', 'vi', None, _METHOD_NAMES['heuristic']
    if _ES_HINT_RE.search(merged):
        return 'ESP', 'es', None, _METHOD_NAMES['heuristic']

    # 4) 拉丁字母为主的文本：优先用 pycld2（英语/西语/印尼语）
    if cld2 is not None and latin_count >= 4 and latin_count >= hangul_count + kana_count:
        try:
            _, _, details = cld2.detect(merged)
            if details and len(details) > 0:
                iso_lang = details[0][1].lower()  # 第一个检测结果的语言代码
                confidence = details[0][2] / 100.0 if details[0][2] else None  # percent -> 0~1
                # 导入映射表
                from ..translators.common import ISO_639_1_TO_VALID_LANGUAGES
                lang_code = ISO_639_1_TO_VALID_LANGUAGES.get(iso_lang)
                # 只对英语/西语/印尼语返回 pycld2 结果
                if lang_code in ('ENG', 'ESP', 'IND'):
                    return lang_code, iso_lang, confidence, _METHOD_NAMES['pycld2']
        except Exception:
            pass

    # 5) 词表兗底（仅对拉丁字母文本启用）
    tokens = set(_LATIN_TOKEN_RE.findall(merged.lower()))
    if tokens:
        es_score = len(tokens & _SPANISH_HINT_WORDS)
        en_score = len(tokens & _ENGLISH_HINT_WORDS)

        # 需要一定票数，避免误判
        if es_score >= 3 and es_score >= en_score + 1:
            return 'ESP', 'es', None, _METHOD_NAMES['heuristic']
        if en_score >= 3 and en_score >= es_score + 1:
            return 'ENG', 'en', None, _METHOD_NAMES['heuristic']

    # 6) 统计模型（py3langid）
    try:
        import py3langid as langid
        iso_lang, confidence = langid.classify(merged)
    except Exception:
        return None, None, None, _METHOD_NAMES['none']

    iso_norm = (iso_lang or '').lower()

    # 优先覆盖：py3langid 可能返回 zh-cn / zh-tw
    extra_iso_map = {
        'zh-cn': 'CHS',
        'zh-tw': 'CHT',
    }

    # 尽量不引入 translators/__init__.py（会导入所有翻译器），使用 common 的轻量映射
    from ..translators.common import ISO_639_1_TO_VALID_LANGUAGES

    lang_code = extra_iso_map.get(iso_norm) or ISO_639_1_TO_VALID_LANGUAGES.get(iso_norm)
    if not lang_code and '-' in iso_norm:
        # 兗底：如 es-419
        lang_code = ISO_639_1_TO_VALID_LANGUAGES.get(iso_norm.split('-', 1)[0])

    # 禁用越南语检测：langid常误判其他语言为越南语
    # 二次检测：如果langid识别为越南语，尝试检测是否为西班牙语
    if lang_code == 'VIN' or iso_norm == 'vi':
        # 检查是否有西语特征词
        es_score = len(tokens & _SPANISH_HINT_WORDS) if tokens else 0
        # 放宽西语检测条件：只要1个特征词即可
        if es_score >= 1:
            return 'ESP', 'es', None, _METHOD_NAMES['vi_fallback_es']
        lang_code = None

    return lang_code, (iso_norm or None), confidence, _METHOD_NAMES['langid']
