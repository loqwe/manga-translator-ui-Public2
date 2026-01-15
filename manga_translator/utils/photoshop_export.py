"""
Photoshop PSD 导出模块
使用 ExtendScript (.jsx) 生成可编辑的 PSD 文件
"""

import os
import tempfile
import subprocess
import platform
import logging
# import math
from typing import Optional
from . import Context
from .generic import imwrite_unicode

logger = logging.getLogger(__name__)


# 对齐方式映射到 Photoshop 的 Justification 枚举
ALIGNMENT_TO_PS_JUSTIFICATION = {
    "left": "Justification.LEFT",
    "right": "Justification.RIGHT",
    "center": "Justification.CENTER",
}

# 文字方向映射
DIRECTION_TO_PS_DIRECTION = {
    "h": "Direction.HORIZONTAL",
    "v": "Direction.VERTICAL",
    "hr": "Direction.HORIZONTAL",  # 从右到左，需要额外处理
    "vr": "Direction.VERTICAL",
}


# JSX 脚本模板
JSX_TEMPLATE = """
#target photoshop

// 设置单位为像素
app.preferences.rulerUnits = Units.PIXELS;
app.preferences.typeUnits = TypeUnits.PIXELS;

// 定义错误日志文件路径
var ERROR_FILE_PATH = '{error_file}';

// 字体查找函数：根据字体名称查找 PostScript 名称
// Photoshop 的 textItem.font 需要使用 PostScript 名称
function findFontPostScriptName(fontName) {{
    if (!fontName) return null;
    
    var lowerName = fontName.toLowerCase();
    var fonts = app.fonts;
    
    // 第一轮：精确匹配 PostScript 名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].postScriptName.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第二轮：精确匹配字体名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].name.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第三轮：精确匹配字体家族名称（不区分大小写）
    for (var i = 0; i < fonts.length; i++) {{
        if (fonts[i].family.toLowerCase() === lowerName) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第四轮：部分匹配（字体名称包含搜索词）
    for (var i = 0; i < fonts.length; i++) {{
        var psName = fonts[i].postScriptName.toLowerCase();
        var name = fonts[i].name.toLowerCase();
        var family = fonts[i].family.toLowerCase();
        
        if (psName.indexOf(lowerName) !== -1 || 
            name.indexOf(lowerName) !== -1 || 
            family.indexOf(lowerName) !== -1) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    // 第五轮：搜索词包含在字体名称中（反向匹配）
    for (var i = 0; i < fonts.length; i++) {{
        var psName = fonts[i].postScriptName.toLowerCase();
        var name = fonts[i].name.toLowerCase();
        var family = fonts[i].family.toLowerCase();
        
        if (lowerName.indexOf(psName) !== -1 || 
            lowerName.indexOf(name) !== -1 || 
            lowerName.indexOf(family) !== -1) {{
            return fonts[i].postScriptName;
        }}
    }}
    
    return null;
}}

// 縦中横（Tate-chu-yoko）处理函数
// 通过设置 baselineDirection 为 "Crs " (Cross) 实现
function applyTateChuYoko(textLayer, charStart, charEnd, fontSize) {{
    try {{
        app.activeDocument.activeLayer = textLayer;
        
        var idsetd = charIDToTypeID("setd");
        var desc1 = new ActionDescriptor();
        var idnull = charIDToTypeID("null");
        var ref1 = new ActionReference();
        ref1.putEnumerated(charIDToTypeID("TxLr"), charIDToTypeID("Ordn"), charIDToTypeID("Trgt"));
        desc1.putReference(idnull, ref1);
        
        var idT = charIDToTypeID("T   ");
        var desc2 = new ActionDescriptor();
        var idTxtt = charIDToTypeID("Txtt");
        var list1 = new ActionList();
        var desc3 = new ActionDescriptor();
        
        desc3.putInteger(charIDToTypeID("From"), charStart);
        desc3.putInteger(charIDToTypeID("T   "), charEnd);
        
        var idTxtS = charIDToTypeID("TxtS");
        var desc4 = new ActionDescriptor();
        desc4.putBoolean(stringIDToTypeID("styleSheetHasParent"), true);
        
        // 保留字体大小
        var idPxl = charIDToTypeID("#Pxl");
        desc4.putUnitDouble(charIDToTypeID("Sz  "), idPxl, fontSize);
        desc4.putUnitDouble(stringIDToTypeID("impliedFontSize"), idPxl, fontSize);
        
        // 关键：设置 baselineDirection 为 "Crs " (Cross) 实现縦中横
        var idbaselineDirection = stringIDToTypeID("baselineDirection");
        desc4.putEnumerated(idbaselineDirection, idbaselineDirection, charIDToTypeID("Crs "));
        
        desc3.putObject(idTxtS, idTxtS, desc4);
        list1.putObject(idTxtt, desc3);
        desc2.putList(idTxtt, list1);
        desc1.putObject(idT, charIDToTypeID("TxLr"), desc2);
        
        executeAction(idsetd, desc1, DialogModes.NO);
        $.writeln('Applied TateChuYoko at position ' + charStart + '-' + charEnd);
        return true;
    }} catch (e) {{
        $.writeln('WARNING: Failed to apply TateChuYoko: ' + e.message);
        return false;
    }}
}}

// 查找文本中需要縦中横的字符位置
function findTateChuYokoPositions(text) {{
    var positions = [];
    var tcyChars = ['⁉', '⁈', '‼', '⁇'];
    
    for (var i = 0; i < text.length; i++) {{
        var ch = text.charAt(i);
        for (var j = 0; j < tcyChars.length; j++) {{
            if (ch === tcyChars[j]) {{
                positions.push([i, i + 1]);
                break;
            }}
        }}
    }}
    return positions;
}}

try {{
    // 打开原始图片
    var inputFile = new File('{input_file}');
    
    // 检查文件是否存在
    if (!inputFile.exists) {{
        $.writeln('ERROR: Input file does not exist: ' + inputFile.fsName);
        throw new Error('Input file does not exist: ' + inputFile.fsName);
    }}
    
    $.writeln('Opening input file: ' + inputFile.fsName);
    var doc = app.open(inputFile);
    $.writeln('Document opened successfully');
    
    // 设置文档颜色模式为 RGB（如果不是的话）
    try {{
        if (doc.mode != DocumentMode.RGB) {{
            $.writeln('Converting to RGB mode');
            doc.changeMode(ChangeMode.RGB);
        }}
    }} catch (modeError) {{
        $.writeln('WARNING: Could not convert to RGB mode: ' + modeError.message);
    }}
    
    // 重命名背景层或第一个图层为"原图"
    // Photoshop 2020+ 打开纯色PNG时可能不创建背景层，而是普通图层
    var originalLayer = null;
    
    try {{
        // 尝试获取背景层
        if (doc.backgroundLayer) {{
            originalLayer = doc.backgroundLayer;
            $.writeln('Found background layer');
        }}
    }} catch (e) {{
        $.writeln('No background layer found: ' + e.message);
    }}
    
    // 如果没有背景层，使用第一个图层（最底层）
    if (!originalLayer && doc.layers.length > 0) {{
        originalLayer = doc.layers[doc.layers.length - 1];
        $.writeln('Using bottom layer as original: ' + originalLayer.name);
    }}
    
    // 重命名并锁定原图层
    if (originalLayer) {{
        originalLayer.name = '原图 (original)';
        originalLayer.allLocked = true;
        $.writeln('Original layer renamed and locked');
    }} else {{
        $.writeln('WARNING: No original layer found');
    }}
    
    {inpainted_layer_code}
    
    {mask_layer_code}
    
    {text_layers_code}
    
    // 保存为 PSD
    var psdFile = new File('{output_file}');
    $.writeln('Saving PSD to: ' + psdFile.fsName);
    var psdOptions = new PhotoshopSaveOptions();
    psdOptions.embedColorProfile = true;
    psdOptions.alphaChannels = true;
    psdOptions.layers = true;
    psdOptions.spotColors = true;
    
    doc.saveAs(psdFile, psdOptions, true);
    $.writeln('PSD saved successfully');
    doc.close(SaveOptions.DONOTSAVECHANGES);
    $.writeln('Document closed');
    
}} catch (e) {{
    var errorMsg = 'ERROR: ' + e.message + '\\nLine: ' + e.line + '\\nFile: ' + (e.fileName || 'unknown');
    $.writeln(errorMsg);
    
    // 写入错误文件
    try {{
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(errorMsg);
        errFile.close();
    }} catch (writeErr) {{
        $.writeln('Failed to write error file: ' + writeErr.message);
    }}
}}
"""

# 单个文本层的 JSX 代码模板
# 按 BallonsTranslator 的顺序：类型 -> contents -> position -> width/height -> size
# 使用点文本（PointText），没有边界框限制
TEXT_LAYER_TEMPLATE = """
    // 文本层 {index}: {name}
    var textLayer{index} = doc.artLayers.add();
    textLayer{index}.kind = LayerKind.TEXT;
    textLayer{index}.name = '{name}';
    
    var textItem{index} = textLayer{index}.textItem;
    // 点文本：不设置 kind，默认就是 POINTTEXT
    
    {font_setup_code}

    // 根据文档分辨率调整尺寸值（Photoshop 内部使用 72 DPI 作为基准）
    // 设置属性顺序：方向 -> 对齐 -> 位置 -> 内容 -> 大小
    var dpiScale{index} = 72 / doc.resolution;
    
    textItem{index}.direction = {direction};
    textItem{index}.justification = {justification};
    textItem{index}.position = [{x}, {y}];
    textItem{index}.contents = '{text}';
    textItem{index}.size = new UnitValue({font_size} * dpiScale{index}, 'pt');
    
    // 设置文字颜色
    var textColor{index} = new SolidColor();
    textColor{index}.rgb.red = {color_r};
    textColor{index}.rgb.green = {color_g};
    textColor{index}.rgb.blue = {color_b};
    textItem{index}.color = textColor{index};
    
    {tracking_code}
    {leading_code}
    {rotation_code}
    {tcy_code}
"""

# 修复图层代码模板
INPAINTED_LAYER_TEMPLATE = """
    // 添加修复后的图层
    $.writeln('Adding inpainted layer from: {inpainted_file}');
    var inpaintedFile = new File('{inpainted_file}');
    if (!inpaintedFile.exists) {{
        $.writeln('WARNING: Inpainted file does not exist');
    }} else {{
        var inpaintedDoc = app.open(inpaintedFile);
        inpaintedDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        inpaintedDoc.close(SaveOptions.DONOTSAVECHANGES);
        doc.activeLayer.name = '修复图 (inpainted)';
        $.writeln('Inpainted layer added successfully');
    }}
"""

# 遮罩层代码模板
MASK_LAYER_TEMPLATE = """
    // 添加遮罩层
    $.writeln('Adding mask layer from: {mask_file}');
    var maskFile = new File('{mask_file}');
    if (!maskFile.exists) {{
        $.writeln('WARNING: Mask file does not exist');
    }} else {{
        var maskDoc = app.open(maskFile);
        maskDoc.activeLayer.duplicate(doc, ElementPlacement.PLACEATBEGINNING);
        maskDoc.close(SaveOptions.DONOTSAVECHANGES);
        doc.activeLayer.name = '遮罩 (mask)';
        $.writeln('Mask layer added successfully');
    }}
"""


def escape_jsx_string(text: str) -> str:
    """转义 JSX 字符串中的特殊字符（用于单引号包裹的字符串）"""
    if not text:
        return ""
    # 必须先处理反斜杠，再处理其他转义
    text = text.replace("\\", "\\\\")   # 反斜杠（如果文本中有）
    text = text.replace("'", "\\'")    # 单引号
    text = text.replace("\n", "\\r")   # 换行符（PS 使用 \r）
    text = text.replace("\r", "\\r")   # 回车符
    text = text.replace("\t", "    ")  # 制表符转为空格
    text = text.replace("\t", "    ")  # 制表符转为空格
    
    # 使用正则替换 [BR] 及其周围的空白，并不区分大小写
    # 支持半角 [BR] 和全角 【BR】
    import re
    text = re.sub(r'\s*(?:\[|【)BR(?:\]|】)\s*', '\\r', text, flags=re.IGNORECASE)

    # 终极处理：处理所有可能的垂直空白符
    # 包括 \n, \r, \u2028 (Line Separator), \u2029 (Paragraph Separator), \v (Vertical Tab), \f (Form Feed)
    # 这一步将所有的物理换行都转换为转义的 \r 字符
    text = re.sub(r'[\r\n\u2028\u2029\v\f]+', '\\r', text)
    
    return text


# 竖排文字中需要縦中横（横排显示）的符号映射
# 将多字符符号替换为单个全角字符，避免竖排时分开显示
VERTICAL_HORIZONTAL_MAP = {
    # 组合标点 -> 单个全角字符
    "!?": "⁉",      # 感叹问号组合
    "?!": "⁈",      # 问号感叹组合
    "!!": "‼",      # 双感叹号
    "??": "⁇",      # 双问号
    # 半角 -> 全角（竖排时显示更好）
    "!": "！",
    "?": "？",
}


def preprocess_vertical_text(text: str, is_vertical: bool) -> str:
    """
    预处理竖排文字，处理縦中横（横排内嵌）
    
    Args:
        text: 原始文本
        is_vertical: 是否为竖排文字
        
    Returns:
        处理后的文本
    """
    if not is_vertical or not text:
        return text
    
    result = text
    
    # 处理全角符号组合（中文翻译常用全角）
    result = result.replace("！？", "⁉")
    result = result.replace("？！", "⁈")
    result = result.replace("！！", "‼")
    result = result.replace("？？", "⁇")
    
    # 处理半角符号组合
    result = result.replace("!?", "⁉")
    result = result.replace("?!", "⁈")
    result = result.replace("!!", "‼")
    result = result.replace("??", "⁇")
    
    # 单个半角 -> 全角
    result = result.replace("!", "！")
    result = result.replace("?", "？")
    
    return result


def generate_text_layer_jsx(index: int, text_region, default_font: str, line_spacing: float = None) -> str:
    """生成单个文本层的 JSX 代码"""
    
    # 文字方向（提前判断，用于文本预处理）
    direction = text_region.direction
    is_vertical = direction.startswith('v') if direction else False
    
    # 文本内容（使用翻译后的文本）
    # 先进行竖排文字预处理（縦中横等）
    raw_text = text_region.translation
    processed_text = preprocess_vertical_text(raw_text, is_vertical)
    text = escape_jsx_string(processed_text)
    
    # 双重保险：强制移除所有可能的物理换行符，防止脚本语法错误
    if '\n' in text or '\r' in text:
        # 如果 escape_jsx_string 没有处理干净（理论上不应发生），这里强制处理
        logger.warning(f"检测到文本层 {index} 内容中仍有物理换行符，正在强制清理")
        text = text.replace('\n', '\\r').replace('\r', '\\r')
    
    if not text:
        logger.warning(f"文本层 {index} 的translation为空，跳过")
        return ""
    
    logger.debug(f"文本层 {index}: 原文='{' '.join(text_region.text)[:30]}', 译文='{text_region.translation[:30]}'")
    
    # 位置和尺寸 - 使用渲染阶段计算的 dst_points
    # dst_points 是 shape (1, 4, 2) 的数组，包含4个角点
    pts = text_region.dst_points.reshape(-1, 2)
    x_min = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min())
    x_max = float(pts[:, 0].max())
    y_max = float(pts[:, 1].max())
    w = x_max - x_min
    h = y_max - y_min
    
    # 文字方向（PS枚举值）
    direction_ps = DIRECTION_TO_PS_DIRECTION.get(direction, "Direction.HORIZONTAL")
    
    # 计算行数 (使用与 escape_jsx_string 相同的正则逻辑)
    import re
    # 支持半角 [BR] 和全角 【BR】
    num_lines = len(re.split(r'\s*(?:\[|【)BR(?:\]|】)\s*', text_region.translation, flags=re.IGNORECASE))
    
    # 字体大小
    font_size = text_region.font_size
    
    # 行间距系数 (leading factor)
    # 竖排基准间距 0.2, 横排基准间距 0.01
    base_spacing = 0.2 if is_vertical else 0.01
    multiplier = line_spacing if line_spacing is not None else 1.0
    leading_factor = 1.0 + base_spacing * multiplier
    
    # 点文本的 position
    # 居中对齐时，position 应该是第一行(横排)或第一列(竖排)的基线中心点
    # 需要根据行数进行修正，让整体文本块居中
    
    box_center_x = x_min + w / 2
    box_center_y = y_min + h / 2
    
    # 计算 leading 的像素近似值 (px)
    # font_size 本身是像素值(px)，因为 text_region.font_size 来自图像分析
    line_height_px = font_size * leading_factor
    
    if is_vertical:
        # 竖排：顶对齐 (Justification.LEFT 通常对应 Start/Top)
        # y 是顶部
        y = y_min
        
        # x 是第一列的中心
        # 列从右向左排，所以第一列在最右侧
        # 为了让Block整体水平居中：
        offset_x = (num_lines - 1) * line_height_px / 2
        x = box_center_x + offset_x
        
        justification = "Justification.LEFT"
        
    else:
        # 横排：居中对齐 (Justification.CENTER)
        # x 保持中心
        x = box_center_x
        
        # y 是第一行的基线
        baseline_offset = font_size * 0.35
        offset_y = (num_lines - 1) * line_height_px / 2
        y = box_center_y - offset_y + baseline_offset
        
        justification = "Justification.CENTER"
    
    # 颜色
    color_r, color_g, color_b = text_region.fg_colors[:3] if len(text_region.fg_colors) >= 3 else (0, 0, 0)
    
    # 字间距（tracking）- 不设置，使用 Photoshop 默认值
    tracking_code = ""
    
    # 行间距（leading）
    # 确保数值是 float
    leading_val_px = float(font_size) * float(leading_factor)
    # 使用 toFixed(2) 确保 JS 中是数字
    leading_code = f"""
    var leadingVal{index} = {leading_val_px} * dpiScale{index};
    textItem{index}.useAutoLeading = false;
    textItem{index}.leading = new UnitValue(leadingVal{index}, 'pt');
    """
    
    # 旋转
    rotation_code = ""
    if abs(text_region.angle) > 1:  # 只有角度大于1度才旋转
        # 直接使用原始角度值
        rotation_code = f"textLayer{index}.rotate({text_region.angle}, AnchorPosition.MIDDLECENTER);"
    
    # 縦中横处理（仅竖排文字）
    tcy_code = ""
    if is_vertical:
        tcy_chars = ['⁉', '⁈', '‼', '⁇']
        has_tcy = any(c in processed_text for c in tcy_chars)
        if has_tcy:
            tcy_code = f"""
    // 应用縦中横（竖排内横排）
    var tcyPositions{index} = findTateChuYokoPositions(textItem{index}.contents);
    var tcyFontSize{index} = {font_size};
    for (var ti{index} = 0; ti{index} < tcyPositions{index}.length; ti{index}++) {{
        applyTateChuYoko(textLayer{index}, tcyPositions{index}[ti{index}][0], tcyPositions{index}[ti{index}][1], tcyFontSize{index});
    }}
"""
    
    # 字体设置代码
    if default_font:
        font_setup_code = f"""// 查找字体的 PostScript 名称
    var requestedFont{index} = '{default_font}';
    var fontPS{index} = findFontPostScriptName(requestedFont{index});
    
    if (!fontPS{index}) {{
        var msg = 'Font not found: "' + requestedFont{index} + '". Please install this font or use a different font.';
        $.writeln('ERROR: ' + msg);
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(msg);
        errFile.close();
        throw new Error(msg);
    }}
    
    $.writeln('Font mapping: "' + requestedFont{index} + '" -> "' + fontPS{index} + '"');
    
    // 设置字体（使用 PostScript 名称）
    try {{
        textItem{index}.font = fontPS{index};
    }} catch (e) {{
        var msg = 'Failed to set font "' + fontPS{index} + '": ' + e.message;
        $.writeln('ERROR: ' + msg);
        var errFile = new File(ERROR_FILE_PATH);
        errFile.open('w');
        errFile.write(msg);
        errFile.close();
        throw new Error(msg);
    }}
"""
    else:
        font_setup_code = f"""// 使用 Photoshop 默认字体
    $.writeln('Using Photoshop default font for layer {index}');
"""
    
    # 文本层名称：使用译文，但必须移除所有换行符以免破坏 JSX 语法
    # 1. 删除 [BR] 及全角 【BR】 (替换为空字符串)
    safe_name = re.sub(r'\s*(?:\[|【)BR(?:\]|】)\s*', '', text_region.translation, flags=re.IGNORECASE)
    # 2. 删除所有物理换行符 (替换为空字符串)
    safe_name = re.sub(r'[\r\n\u2028\u2029\v\f]+', '', safe_name)
    # 3. 转义特殊字符并截断
    name = escape_jsx_string(safe_name[:50])

    return TEXT_LAYER_TEMPLATE.format(
        index=index,
        name=name,
        text=text,
        x=x,
        y=y,
        w=w,
        h=h,
        font_size=font_size,
        color_r=color_r,
        color_g=color_g,
        color_b=color_b,
        justification=justification,
        direction=direction_ps,
        is_vertical='true' if is_vertical else 'false',
        tracking_code=tracking_code,
        leading_code=leading_code,
        rotation_code=rotation_code,
        tcy_code=tcy_code,
        font_setup_code=font_setup_code,
    )


def get_psd_output_path(image_path: str) -> str:
    """
    获取PSD文件的输出路径
    
    在原图所在目录下创建 manga_translator_work/psd/ 文件夹
    
    Args:
        image_path: 原图路径
        
    Returns:
        PSD文件的完整路径
    """
    # 获取原图所在目录和文件名
    image_dir = os.path.dirname(os.path.abspath(image_path))
    image_name = os.path.basename(image_path)
    base_name, _ = os.path.splitext(image_name)
    
    # 创建 manga_translator_work/psd 目录
    psd_dir = os.path.join(image_dir, 'manga_translator_work', 'psd')
    os.makedirs(psd_dir, exist_ok=True)
    
    # 生成PSD文件路径
    psd_path = os.path.join(psd_dir, f"{base_name}.psd")
    
    return psd_path


def photoshop_export(output_file: str, ctx: Context, default_font: str = None, image_path: str = None, verbose: bool = False, result_path_fn=None, line_spacing: float = None, script_only: bool = False):
    """
    使用 Photoshop 导出 PSD 文件
    
    图层结构（从下到上）：
    1. 原图 (original) - 锁定
    2. 修复图 (inpainted) - 从工作目录读取
    3. 遮罩 (mask) - 如果有
    4. 文字图层 - 可编辑
    
    Args:
        output_file: 输出 PSD 文件路径
        ctx: 翻译上下文，包含图片和文本区域信息
        default_font: 默认字体名称，如果为 None 则使用 Photoshop 默认字体
        image_path: 原图路径（用于查找工作目录中的修复图）
        verbose: 是否启用调试模式（保存JSX脚本到result文件夹）
        result_path_fn: 结果路径生成函数（用于保存调试脚本）
        line_spacing: 行间距系数
        script_only: 如果为True，只生成JSX脚本而不执行Photoshop
    """
    
    # 如果 default_font 是文件路径，提取字体名称
    if default_font and (os.path.sep in default_font or '/' in default_font or default_font.endswith('.ttf') or default_font.endswith('.otf')):
        # 从路径中提取文件名（不含扩展名）作为字体名称
        font_basename = os.path.splitext(os.path.basename(default_font))[0]
        logger.warning(f"检测到 default_font 是文件路径: {default_font}")
        logger.warning(f"已提取字体名称: {font_basename}")
        logger.warning("提示: 请在配置中使用 'psd_font' 参数指定字体名称，而不是文件路径")
        default_font = font_basename
    
    # 创建临时文件（只用于修复图和遮罩）
    temp_dir = tempfile.gettempdir()
    inpainted_file = os.path.join(temp_dir, ".ps_inpainted.png")
    mask_file = os.path.join(temp_dir, ".ps_mask.png")
    jsx_file = os.path.join(temp_dir, ".ps_script.jsx")
    error_file = os.path.join(temp_dir, ".ps_error.txt")
    
    # 清理旧的错误文件
    if os.path.exists(error_file):
        try:
            os.unlink(error_file)
        except Exception:
            pass
    
    try:
        # 直接使用原图路径，不需要保存临时文件
        if not image_path or not os.path.exists(image_path):
            raise ValueError(f"原图路径无效或文件不存在: {image_path}")
        
        input_file = image_path
        logger.info(f"使用原图: {input_file}")
        
        # 从工作目录读取修复后的图（如果存在）
        inpainted_layer_code = ""
        if image_path:
            # 构建修复图路径：原图目录/manga_translator_work/inpainted/文件名.png
            from .path_manager import get_inpainted_path
            inpainted_path = get_inpainted_path(image_path, create_dir=False)
            
            if os.path.exists(inpainted_path):
                # 直接使用工作目录中的修复图
                inpainted_layer_code = INPAINTED_LAYER_TEMPLATE.format(
                    inpainted_file=inpainted_path.replace("\\", "/")  # 使用正斜杠
                )
                logger.info(f"找到修复图: {inpainted_path}")
            else:
                logger.debug(f"未找到修复图: {inpainted_path}")
        
        # 如果工作目录没有，尝试从ctx读取
        if not inpainted_layer_code and hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
            import cv2
            from PIL import Image
            # 将numpy数组转换为PIL图像
            inpainted_pil = Image.fromarray(cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_BGR2RGB))
            inpainted_pil.save(inpainted_file)
            inpainted_layer_code = INPAINTED_LAYER_TEMPLATE.format(
                inpainted_file=inpainted_file.replace("\\", "/")  # 使用正斜杠
            )
            logger.info("使用Context中的修复图")
        
        # 蒙版层 - 不添加
        mask_layer_code = ""
        
        # 生成文本层代码
        if default_font:
            logger.info(f"PSD导出使用字体: {default_font}")
        else:
            logger.info("PSD导出使用 Photoshop 默认字体")
        text_layers_code = ""
        if hasattr(ctx, 'text_regions') and ctx.text_regions:
            filtered_regions = [r for r in ctx.text_regions if r.translation]
            logger.info(f"准备添加 {len(filtered_regions)} 个文本层到 PSD")
            for i, region in enumerate(filtered_regions):
                text_layers_code += generate_text_layer_jsx(i, region, default_font, line_spacing)
        
        # 生成完整的 JSX 脚本
        # 路径转义：Windows路径的反斜杠需要转义为双反斜杠
        jsx_script = JSX_TEMPLATE.format(
            input_file=input_file.replace("\\", "/"),  # 使用正斜杠，JSX支持
            output_file=output_file.replace("\\", "/"),
            error_file=error_file.replace("\\", "/"),
            inpainted_layer_code=inpainted_layer_code,
            mask_layer_code=mask_layer_code,
            text_layers_code=text_layers_code,
        )
        
        # 保存 JSX 脚本（使用UTF-8 BOM编码，确保Photoshop能正确读取中文）
        with open(jsx_file, 'w', encoding='utf-8-sig') as f:
            f.write(jsx_script)
        
        logger.info(f"生成 JSX 脚本: {jsx_file}")
        
        # 如果启用verbose模式或script_only模式
        saved_script_path = None
        if verbose or script_only:
            try:
                image_name = os.path.basename(image_path) if image_path else "unknown"
                base_name, _ = os.path.splitext(image_name)

                if script_only and image_path:
                    # script_only 模式下，保存到 manga_translator_work/psd
                    image_dir = os.path.dirname(os.path.abspath(image_path))
                    psd_dir = os.path.join(image_dir, 'manga_translator_work', 'psd')
                    os.makedirs(psd_dir, exist_ok=True)
                    debug_jsx_path = os.path.join(psd_dir, f"{base_name}_photoshop_script.jsx")
                elif result_path_fn:
                    # verbose 模式或无 image_path，保存到 result 目录
                    debug_jsx_path = result_path_fn(f"{base_name}_photoshop_script.jsx")
                else:
                    debug_jsx_path = None

                if debug_jsx_path:
                    with open(debug_jsx_path, 'w', encoding='utf-8') as f:
                        f.write(jsx_script)
                    saved_script_path = debug_jsx_path
                    logger.info(f"📝 JSX脚本已保存: {debug_jsx_path}")
            except Exception as e:
                logger.warning(f"保存JSX脚本失败: {e}")
        
        # 如果只生成脚本，直接返回
        if script_only:
            logger.info("✅ 仅生成脚本模式：JSX脚本已保存，跳过Photoshop执行")
            if saved_script_path:
                logger.info(f"   脚本路径: {saved_script_path}")
            return
        
        # 执行 Photoshop
        ps_executable = find_photoshop_executable()
        if not ps_executable:
            raise FileNotFoundError(
                "未找到 Photoshop 可执行文件。请确保已安装 Photoshop，"
                "或设置环境变量 PHOTOSHOP_PATH 指向 Photoshop.exe"
            )
        
        logger.info(f"使用 Photoshop: {ps_executable}")
        logger.info(f"执行脚本: {jsx_file}")
        
        # 运行 Photoshop（不等待进程退出，只等待 PSD 文件生成）
        import time
        
        # 记录输出文件修改时间（如果已存在）
        old_mtime = os.path.getmtime(output_file) if os.path.exists(output_file) else 0
        
        # 启动 Photoshop（不等待）
        process = subprocess.Popen(
            [ps_executable, '-r', jsx_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        
        # 轮询等待 PSD 文件生成（最多等待 300 秒）
        timeout = 300
        poll_interval = 0.5
        elapsed = 0
        
        while elapsed < timeout:
            time.sleep(poll_interval)
            elapsed += poll_interval
            
            # 检查 PSD 文件是否生成/更新
            if os.path.exists(output_file):
                new_mtime = os.path.getmtime(output_file)
                if new_mtime > old_mtime and os.path.getsize(output_file) > 0:
                    # 文件已生成，等待一小段时间确保写入完成
                    time.sleep(0.5)
                    break
        
        # 读取输出（非阻塞）
        try:
            stdout, stderr = process.communicate(timeout=1)
            if stdout:
                stdout_text = stdout.decode('utf-8', errors='replace')
                logger.info(f"Photoshop 输出:\n{stdout_text}")
            if stderr:
                stderr_text = stderr.decode('utf-8', errors='replace')
                logger.warning(f"Photoshop 错误输出:\n{stderr_text}")
        except subprocess.TimeoutExpired:
            # Photoshop 还在运行，这是正常的
            pass
            
        # 检查是否有脚本错误报告
        if os.path.exists(error_file):
            with open(error_file, 'r') as f:
                error_msg = f.read()
            logger.error(f"Photoshop 脚本执行错误: {error_msg}")
            raise RuntimeError(f"Photoshop 脚本错误: {error_msg}")
        
        # 检查 PSD 文件是否成功生成
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            logger.info(f"PSD 文件已生成: {output_file}")
        else:
            if elapsed >= timeout:
                raise RuntimeError(f"Photoshop 执行超时 ({timeout}秒)，PSD 文件未生成")
            else:
                raise RuntimeError(f"PSD 文件生成失败: {output_file}")
        
    finally:
        # 清理临时文件
        for temp_file in [inpainted_file, mask_file, error_file]:
            if os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                except Exception as e:
                    logger.warning(f"无法删除临时文件 {temp_file}: {e}")
        
        # 如果不是verbose模式且不是script_only模式，删除JSX脚本
        if not verbose and not script_only and os.path.exists(jsx_file):
            try:
                os.unlink(jsx_file)
            except Exception as e:
                logger.warning(f"无法删除JSX脚本 {jsx_file}: {e}")


def find_photoshop_from_registry() -> Optional[str]:
    """
    从 Windows 注册表查找 Photoshop 安装路径
    
    Returns:
        Photoshop 可执行文件路径，如果未找到则返回 None
    """
    if platform.system() != "Windows":
        return None
    
    try:
        import winreg
    except ImportError:
        logger.warning("无法导入 winreg 模块，跳过注册表查询")
        return None
    
    # 可能的注册表路径
    registry_paths = [
        # Photoshop CC 及更新版本
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Adobe\Photoshop"),
        (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Adobe\Photoshop"),
        # 32位程序在64位系统上的路径
        (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Adobe\Photoshop"),
    ]
    
    for hkey, subkey_path in registry_paths:
        try:
            # 打开 Photoshop 主键
            with winreg.OpenKey(hkey, subkey_path) as key:
                # 枚举所有版本子键
                i = 0
                versions = []
                while True:
                    try:
                        version_key = winreg.EnumKey(key, i)
                        versions.append(version_key)
                        i += 1
                    except OSError:
                        break
                
                # 按版本号降序排序（优先使用最新版本）
                versions.sort(reverse=True)
                
                # 尝试每个版本
                for version in versions:
                    try:
                        version_path = f"{subkey_path}\\{version}"
                        with winreg.OpenKey(hkey, version_path) as version_key:
                            # 尝试读取 ApplicationPath 或 InstallPath
                            for value_name in ["ApplicationPath", "InstallPath", "Path"]:
                                try:
                                    install_path, _ = winreg.QueryValueEx(version_key, value_name)
                                    if install_path:
                                        # 构建可执行文件路径
                                        ps_exe = os.path.join(install_path, "Photoshop.exe")
                                        if os.path.exists(ps_exe):
                                            logger.info(f"从注册表找到 Photoshop: {ps_exe}")
                                            return ps_exe
                                except FileNotFoundError:
                                    continue
                    except Exception as e:
                        logger.debug(f"读取注册表版本 {version} 失败: {e}")
                        continue
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug(f"读取注册表路径 {subkey_path} 失败: {e}")
            continue
    
    return None


def find_photoshop_executable() -> Optional[str]:
    """
    查找 Photoshop 可执行文件路径
    
    查找顺序：
    1. 环境变量 PHOTOSHOP_PATH
    2. Windows 注册表（仅 Windows）
    3. 常见安装路径
    4. 遍历 Adobe 目录
    
    Returns:
        Photoshop 可执行文件的完整路径，如果未找到则返回 None
    """
    
    # 1. 优先使用环境变量
    ps_path = os.getenv("PHOTOSHOP_PATH")
    if ps_path and os.path.exists(ps_path):
        logger.info(f"从环境变量找到 Photoshop: {ps_path}")
        return ps_path
    
    system = platform.system()
    
    if system == "Windows":
        # 2. 从注册表查找（最可靠）
        ps_path = find_photoshop_from_registry()
        if ps_path:
            return ps_path
        
        # 3. Windows 常见安装路径
        possible_paths = [
            r"C:\Program Files\Adobe\Adobe Photoshop 2024\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2023\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2022\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop 2021\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop CC 2019\Photoshop.exe",
            r"C:\Program Files\Adobe\Adobe Photoshop CC 2018\Photoshop.exe",
        ]
        
        # 也检查 Program Files (x86)
        program_files_x86 = os.getenv("ProgramFiles(x86)")
        if program_files_x86:
            for path in list(possible_paths):
                x86_path = path.replace(r"C:\Program Files", program_files_x86)
                possible_paths.append(x86_path)
        
        # 搜索所有可能的路径
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"从常见路径找到 Photoshop: {path}")
                return path
        
        # 4. 遍历 Program Files 中的 Adobe 目录
        for program_files_var in ["ProgramFiles", "ProgramFiles(x86)"]:
            program_files = os.getenv(program_files_var)
            if not program_files:
                continue
            
            adobe_dir = os.path.join(program_files, "Adobe")
            if os.path.exists(adobe_dir):
                try:
                    folders = sorted(os.listdir(adobe_dir), reverse=True)  # 降序，优先新版本
                    for folder in folders:
                        if "Photoshop" in folder:
                            ps_exe = os.path.join(adobe_dir, folder, "Photoshop.exe")
                            if os.path.exists(ps_exe):
                                logger.info(f"从 Adobe 目录找到 Photoshop: {ps_exe}")
                                return ps_exe
                except Exception as e:
                    logger.debug(f"遍历 Adobe 目录失败: {e}")
    
    elif system == "Darwin":  # macOS
        possible_paths = [
            "/Applications/Adobe Photoshop 2024/Adobe Photoshop 2024.app/Contents/MacOS/Adobe Photoshop 2024",
            "/Applications/Adobe Photoshop 2023/Adobe Photoshop 2023.app/Contents/MacOS/Adobe Photoshop 2023",
            "/Applications/Adobe Photoshop 2022/Adobe Photoshop 2022.app/Contents/MacOS/Adobe Photoshop 2022",
            "/Applications/Adobe Photoshop 2021/Adobe Photoshop 2021.app/Contents/MacOS/Adobe Photoshop 2021",
            "/Applications/Adobe Photoshop CC 2019/Adobe Photoshop CC 2019.app/Contents/MacOS/Adobe Photoshop CC 2019",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"从常见路径找到 Photoshop: {path}")
                return path
    
    logger.warning("未找到 Photoshop 安装")
    return None


def test_photoshop_installation() -> bool:
    """
    测试 Photoshop 是否已正确安装并可用
    
    Returns:
        如果 Photoshop 可用返回 True，否则返回 False
    """
    ps_exe = find_photoshop_executable()
    if not ps_exe:
        logger.error("未找到 Photoshop 安装")
        return False
    
    logger.info(f"找到 Photoshop: {ps_exe}")
    return True
