import re
import logging
from typing import List

from .common import CommonTranslator

logger = logging.getLogger('manga_translator')

# ── OpenCC engine (lazy init) ───────────────────────────────────
_cc = None

def _get_cc():
    global _cc
    if _cc is None:
        import opencc
        _cc = opencc.OpenCC('tw2sp')
    return _cc

# ── Post-processing fixes ───────────────────────────────────────
# Applied AFTER OpenCC tw2sp conversion to correct over-conversions.
FIXES = {
    '妳': '你',
    # ── 著(zhù) phrase restoration ──
    '着名': '著名', '着作': '著作', '显着': '显著', '卓着': '卓著',
    '着称': '著称', '名着': '名著', '原着': '原著', '专着': '专著',
    '巨着': '巨著', '编着': '编著', '遗着': '遗著', '拙着': '拙著',
    # ── misc corrections ──
    '阖家': '合家', '磁砖': '瓷砖', '麴霉': '曲霉', '干隆': '乾隆', '了望': '瞭望',
    '慰借': '慰藉', '狼借': '狼藉', '计画': '计划', '人蓡': '人参',
    '角徵羽': '角征羽',
    # ── OpenCC tw2sp missed chars ──
    '擡': '抬', '瞇': '眯',
}

# ── Context-aware 著/着 disambiguation ──────────────────────────
# OpenCC mmseg may group "著名" as one token (famous), but the real
# meaning is "V着"(particle) + "名…". Detect via positive match:
# CJK char (verb) + 著名 + trigger suffix.
_ZHU_PARTICLE_RE = re.compile(
    r'(?<=[\u4e00-\u9fff])著名(?=[为叫是牌画曲片单义额次份句字册流号角堂门姓望])'
)


def smart_convert(text: str) -> str:
    """Taiwan Traditional -> Simplified: OpenCC tw2sp + FIXES + context rules."""
    cc = _get_cc()
    step1 = cc.convert(text)
    result = step1
    for old, new in FIXES.items():
        result = result.replace(old, new)
    # Context-aware: fix mmseg "著名" mis-segmentation
    result = _ZHU_PARTICLE_RE.sub('着名', result)
    return result


class T2STranslator(CommonTranslator):
    """Traditional Chinese to Simplified Chinese translator.

    Pure local conversion using OpenCC tw2sp engine with post-processing
    fixes for common over-conversion errors. No API calls needed.
    """

    def supports_languages(self, from_lang: str, to_lang: str, fatal: bool = False) -> bool:
        return True

    async def translate(self, from_lang: str, to_lang: str, queries: List[str], use_mtpe: bool = False, ctx=None) -> List[str]:
        """Override base translate to bypass from_lang==to_lang short-circuit.

        langid cannot distinguish Traditional/Simplified Chinese — both map to
        'CHS', causing the base class to skip _translate entirely. T2S is a
        deterministic char-level converter that must always run.
        """
        self.logger.info(f'T2S: converting {len(queries)} texts (Traditional -> Simplified)')
        return [smart_convert(q) for q in queries]

    async def _translate(self, from_lang: str, to_lang: str, queries: List[str], ctx=None) -> List[str]:
        # Fallback if called directly; normally translate() is the entry point.
        return [smart_convert(q) for q in queries]
