
import re
import logging

import cv2
import numpy as np
from manga_translator.config import Config, RenderConfig
from manga_translator.rendering import text_render
from manga_translator.rendering.text_render import (
    auto_add_horizontal_tags,
    set_font,
)
from manga_translator.utils import TextBlock
from PyQt6.QtCore import QPointF
from PyQt6.QtGui import QImage, QPixmap, QPolygonF

logger = logging.getLogger('manga_translator')


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    import os
    import sys
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    return os.path.join(base_path, relative_path)

def update_font_config(font_filename: str):
    """更新字体配置"""
    if not font_filename:
        return
    import os
    font_path = resource_path(os.path.join('fonts', font_filename))
    if os.path.exists(font_path):
        try:
            set_font(font_path)
        except Exception as e:
            pass  # Silently ignore font update errors

def render_text_for_region(text_block: TextBlock, dst_points: np.ndarray, transform, render_params: dict, pure_zoom: float = 1.0, total_regions: int = 1):
    """
    为单个区域渲染文本的核心函数
    返回一个包含 (QPixmap, QPointF) 的元组用于绘制，或者在失败时返回 None
    """
    original_translation = text_block.translation
    try:
        # --- 1. 文本预处理 ---
        text_to_render = original_translation or text_block.text
        if not text_to_render:
            logger.warning(f"[EDITOR RENDER SKIPPED] Text is empty")
            return None

        processed_text = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', '\n', text_to_render.replace('↵', '\n'), flags=re.IGNORECASE)
        if not text_block.horizontal and render_params.get('auto_rotate_symbols'):
            processed_text = auto_add_horizontal_tags(processed_text)
        text_block.translation = processed_text

        # --- 2. 渲染 ---
        # 与后端逻辑一致：
        # 1. 当AI断句开启(disable_auto_wrap=True)且文本中有[BR]标记时，关闭自动换行
        # 2. 当AI断句开启但文本中没有[BR]标记时，启用自动换行（回退到自动换行模式）
        disable_auto_wrap_param = render_params.get('disable_auto_wrap', False)
        
        # 检测文本中是否有BR标记
        has_br = bool(re.search(r'(\[BR\]|【BR】|<br>|\n)', processed_text, flags=re.IGNORECASE))
        
        # 只有当AI断句开启且文本中有BR标记时才真正禁用自动换行
        # 否则回退到自动换行模式
        effective_disable_auto_wrap = disable_auto_wrap_param and has_br
        
        # 横排使用hyphenate参数控制
        hyphenate = not effective_disable_auto_wrap
        
        if disable_auto_wrap_param and not has_br:
            logger.debug(f"[EDITOR RENDER] AI断句开启但无BR标记，回退到自动换行模式")
        elif effective_disable_auto_wrap:
            logger.debug(f"[EDITOR RENDER] AI断句开启且有BR标记，禁用自动换行")

        render_params.get('line_spacing')
        disable_font_border = render_params.get('disable_font_border', False)
        
        middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)

        render_w = round(norm_h[0])
        render_h = round(norm_v[0])
        font_size = text_block.font_size

        fg_color, bg_color = text_block.get_font_colors()
        if disable_font_border:
            bg_color = None

        if render_w <= 0 or render_h <= 0:
            logger.warning(f"[EDITOR RENDER SKIPPED] Invalid render dimensions: width={render_w}, height={render_h}")
            return None

        config_data = render_params.copy()
        if config_data.get('direction') == 'v':
            config_data['direction'] = 'vertical'
        elif config_data.get('direction') == 'h':
            config_data['direction'] = 'horizontal'

        if 'font_color' in config_data and isinstance(config_data['font_color'], list):
            try:
                rgb = config_data['font_color']
                config_data['font_color'] = f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
            except (IndexError, TypeError):
                config_data.pop('font_color')
        
        # ✅ 修复：将后端参数名映射回RenderConfig期望的字段名
        if 'text_stroke_width' in config_data:
            config_data['stroke_width'] = config_data.pop('text_stroke_width')
        if 'text_stroke_color' in config_data:
            config_data['bg_color'] = config_data.pop('text_stroke_color')
        
        # ✅ 关键修复：根据是否有BR标记来设置effective的disable_auto_wrap
        # 这样竖排渲染时也能正确回退到自动换行模式
        config_data['disable_auto_wrap'] = effective_disable_auto_wrap

        config_obj = Config(render=RenderConfig(**config_data)) if config_data else Config()
        line_spacing_from_params = render_params.get('line_spacing')

        # 获取区域数（lines数组的长度），用于智能排版模式的换行判断
        region_count = 1
        if hasattr(text_block, 'lines') and text_block.lines is not None:
            try:
                region_count = len(text_block.lines)
            except:
                region_count = 1

        # 使用 freetype 渲染器（稳定可靠）
        if text_block.horizontal:
            rendered_surface = text_render.put_text_horizontal(
                font_size, 
                text_block.get_translation_for_rendering(), 
                render_w, 
                render_h, 
                text_block.alignment, 
                text_block.direction == 'hl', 
                fg_color, 
                bg_color, 
                text_block.target_lang, 
                hyphenate, 
                line_spacing_from_params, 
                config=config_obj, 
                region_count=region_count
            )
        else:
            rendered_surface = text_render.put_text_vertical(
                font_size, 
                text_block.get_translation_for_rendering(), 
                render_h, 
                text_block.alignment, 
                fg_color, 
                bg_color, 
                line_spacing_from_params, 
                config=config_obj, 
                region_count=region_count
            )

        if rendered_surface is None or rendered_surface.size == 0:
            logger.warning(f"[EDITOR RENDER SKIPPED] Rendered surface is None or empty. Text: '{text_block.translation[:50] if hasattr(text_block, 'translation') else 'N/A'}...'")
            return None
        
        # --- 3. 宽高比校正 (与后端渲染逻辑完全同步) ---
        h_temp, w_temp, _ = rendered_surface.shape
        if h_temp == 0 or w_temp == 0:
            logger.warning(f"[EDITOR RENDER SKIPPED] Rendered surface has zero dimensions: width={w_temp}, height={h_temp}")
            return None
        r_temp = w_temp / h_temp
        
        middle_pts = (dst_points[:, [1, 2, 3, 0]] + dst_points) / 2
        norm_h = np.linalg.norm(middle_pts[:, 1] - middle_pts[:, 3], axis=1)
        norm_v = np.linalg.norm(middle_pts[:, 2] - middle_pts[:, 0], axis=1)
        r_orig = np.mean(norm_h / norm_v)

        box = None
        if text_block.horizontal:
            if r_temp > r_orig:
                h_ext = int((w_temp / r_orig - h_temp) // 2) if r_orig > 0 else 0
                if h_ext >= 0:
                    box = np.zeros((h_temp + h_ext * 2, w_temp, 4), dtype=np.uint8)
                    # 垂直方向：根据center_text_in_bubble决定是否居中
                    if config_obj.render.center_text_in_bubble:
                        # 居中开启：垂直居中
                        box[h_ext:h_ext+h_temp, 0:w_temp] = rendered_surface
                    else:
                        # 默认：贴顶部
                        box[0:h_temp, 0:w_temp] = rendered_surface
                else:
                    box = rendered_surface.copy()
            else:
                w_ext = int((h_temp * r_orig - w_temp) // 2)
                if w_ext >= 0:
                    box = np.zeros((h_temp, w_temp + w_ext * 2, 4), dtype=np.uint8)
                    # 横排文本默认水平居中
                    box[0:h_temp, w_ext:w_ext+w_temp] = rendered_surface
                else:
                    box = rendered_surface.copy()
        else: # Vertical
            if r_temp > r_orig:
                h_ext = int(w_temp / (2 * r_orig) - h_temp / 2) if r_orig > 0 else 0
                if h_ext >= 0:
                    box = np.zeros((h_temp + h_ext * 2, w_temp, 4), dtype=np.uint8)
                    # 竖排文本垂直方向：根据center_text_in_bubble决定是否居中
                    if config_obj.render.center_text_in_bubble:
                        # 居中开启：垂直居中
                        box[h_ext:h_ext+h_temp, 0:w_temp] = rendered_surface
                    else:
                        # 默认：贴顶部
                        box[0:h_temp, 0:w_temp] = rendered_surface
                else:
                    box = rendered_surface.copy()
            else:
                w_ext = int((h_temp * r_orig - w_temp) / 2)
                if w_ext >= 0:
                    box = np.zeros((h_temp, w_temp + w_ext * 2, 4), dtype=np.uint8)
                    # 竖排文本水平居中
                    box[0:h_temp, w_ext:w_ext+w_temp] = rendered_surface
                else:
                    box = rendered_surface.copy()

        if box is None:
            box = rendered_surface.copy()

        # --- 4. 坐标变换与扭曲 (Warping) ---
        src_points = np.float32([[0, 0], [box.shape[1], 0], [box.shape[1], box.shape[0]], [0, box.shape[0]]])

        # 将图像坐标转换为视图(屏幕)坐标
        qpoly = transform.map(QPolygonF([QPointF(p[0], p[1]) for p in dst_points[0]]))
        dst_points_screen = np.float32([ [p.x(), p.y()] for p in qpoly ])

        # 计算屏幕上的最小边界框
        x_s, y_s, w_s, h_s = cv2.boundingRect(np.round(dst_points_screen).astype(np.int32))
        if w_s <= 0 or h_s <= 0:
            logger.warning(f"[EDITOR RENDER SKIPPED] Screen bounding box has invalid dimensions: x={x_s}, y={y_s}, width={w_s}, height={h_s}. Text may be outside visible area.")
            return None

        # 将目标点偏移到边界框的局部坐标
        dst_points_warp = dst_points_screen - [x_s, y_s]

        matrix, _ = cv2.findHomography(src_points, dst_points_warp, cv2.RANSAC, 5.0)
        if matrix is None:
            logger.warning(f"[EDITOR RENDER SKIPPED] Failed to compute homography matrix for text transformation")
            return None

        warped_image = cv2.warpPerspective(box, matrix, (w_s, h_s), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0,0))

        # --- 5. 转换为QPixmap并返回绘制信息 ---
        h, w, ch = warped_image.shape
        if ch == 4:
            # Convert RGBA to BGRA for QImage Format_ARGB32
            bgra_image = warped_image.copy()
            bgra_image[:, :, [0, 2]] = bgra_image[:, :, [2, 0]]  # Swap R and B channels
            # 【关键修复】使用.copy()确保QImage拥有自己的内存，防止numpy数组被回收后崩溃
            final_image = QImage(bgra_image.data, w, h, w * 4, QImage.Format.Format_ARGB32).copy()
            final_pixmap = QPixmap.fromImage(final_image)
            # 返回pixmap和它在屏幕(视图)上的绘制位置
            return (final_pixmap, QPointF(x_s, y_s))

    except Exception as e:
        print(f"Error during backend text rendering: {e}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        text_block.translation = original_translation
