#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
导出服务
负责将编辑器中的内容导出为后端渲染的图片
"""

import asyncio
import json
import logging
import os
import tempfile
import threading
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from utils.json_encoder import CustomJSONEncoder

# 全局输出目录存储
_global_output_directory = None

def set_global_output_directory(output_dir: str):
    """设置全局输出目录"""
    global _global_output_directory
    _global_output_directory = output_dir

def get_global_output_directory() -> Optional[str]:
    """获取全局输出目录"""
    return _global_output_directory


class ExportService:
    """导出服务类"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def get_output_directory(self) -> Optional[str]:
        """获取设置的输出目录"""
        # 首先检查全局存储的输出目录
        global_dir = get_global_output_directory()
        if global_dir and os.path.exists(global_dir):
            self.logger.info(f"使用全局输出目录: {global_dir}")
            return global_dir
        
        try:
            # 作为备选方案，尝试通过UI控件获取
            import tkinter as tk
            
            # 获取根窗口
            root = tk._default_root
            if root is None:
                return None
            
            # 查找应用控制器
            for child in root.winfo_children():
                if hasattr(child, 'controller'):
                    controller = child.controller
                    if hasattr(controller, 'main_view_widgets'):
                        output_entry = controller.main_view_widgets.get('output_folder_entry')
                        if output_entry:
                            output_dir = output_entry.get().strip()
                            if output_dir and os.path.exists(output_dir):
                                self.logger.info(f"找到输出目录: {output_dir}")
                                # 更新全局存储
                                set_global_output_directory(output_dir)
                                return output_dir
                            elif output_dir:
                                self.logger.warning(f"输出目录不存在: {output_dir}")
                            break
                    break
                    
        except Exception as e:
            self.logger.warning(f"无法获取输出目录: {e}")
        
        return None
    
    def get_output_format_from_config(self, config: Dict[str, Any]) -> str:
        """从配置中获取输出格式"""
        cli_config = config.get('cli', {})
        output_format = cli_config.get('format', '').strip()
        
        # 如果没有指定格式或格式为空，返回空字符串表示使用原格式
        if not output_format or output_format == "不指定":
            return ""
        
        return output_format.lower()
    
    def generate_output_filename(self, original_image_path: str, output_format: str = "", add_prefix: bool = False) -> str:
        """生成输出文件名，可选择是否添加前缀"""
        base_name = os.path.splitext(os.path.basename(original_image_path))[0]
        
        # 根据参数决定是否添加前缀
        if add_prefix:
            output_name = f"translated_{base_name}"
        else:
            # 使用原始文件名（编辑器导出时的默认行为）
            output_name = base_name
        
        # 确定文件扩展名
        if output_format:
            # 使用配置中指定的格式
            extension = f".{output_format}"
        else:
            # 使用原文件的格式
            original_ext = os.path.splitext(original_image_path)[1].lower()
            extension = original_ext if original_ext else ".png"
        
        return output_name + extension
    
    def export_rendered_image(self, image: Image.Image, regions_data: List[Dict[str, Any]], 
                            config: Dict[str, Any], output_path: str, 
                            mask: Optional[np.ndarray] = None,
                            progress_callback: Optional[callable] = None,
                            success_callback: Optional[callable] = None,
                            error_callback: Optional[callable] = None):
        """
        导出后端渲染的图片
        
        Args:
            image: 当前图片
            regions_data: 区域数据
            config: 配置字典
            output_path: 输出路径
            mask: (新增) 预计算的蒙版
            progress_callback: 进度回调
            success_callback: 成功回调
            error_callback: 错误回调
        """
        if not image:
            if error_callback:
                error_callback("没有图片可导出")
            return
        
        # regions_data 可以为空列表，此时导出原图（可能经过上色/超分处理）
        if regions_data is None:
            regions_data = []
        
        if progress_callback:
            progress_callback("开始导出渲染图片...")
        
        # 在后台线程中执行导出
        export_thread = threading.Thread(
            target=self._perform_backend_render_export,
            args=(image, regions_data, config, output_path, mask, progress_callback, success_callback, error_callback),
            daemon=True
        )
        export_thread.start()
    
    def _perform_backend_render_export(self, image: Image.Image, regions_data: List[Dict[str, Any]],
                                     config: Dict[str, Any], output_path: str,
                                     mask: Optional[np.ndarray] = None,
                                     progress_callback: Optional[callable] = None,
                                     success_callback: Optional[callable] = None,
                                     error_callback: Optional[callable] = None):
        """在后台线程中执行后端渲染导出"""
        import gc
        import os
        try:
            if progress_callback:
                progress_callback("准备导出环境...")

            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 保存当前图片到临时文件
                temp_image_path = os.path.join(temp_dir, "temp_image.png")
                image.save(temp_image_path)
                
                # 保存区域数据到JSON文件，使用load_text模式期望的文件名格式
                base_name = os.path.splitext(os.path.basename(temp_image_path))[0]
                regions_json_path = os.path.join(temp_dir, f"{base_name}_translations.json")
                self._save_regions_data(regions_data, regions_json_path, mask, config)
                
                if progress_callback:
                    progress_callback("初始化翻译引擎...")
                
                # 准备翻译器参数
                translator_params = self._prepare_translator_params(config)
                
                # 创建翻译器实例并执行渲染
                rendered_text_layer = self._execute_backend_render(
                    temp_image_path, regions_json_path, translator_params, config, progress_callback
                )
                
                if rendered_text_layer:
                    # 后端返回的 ctx.result 已经是完整的渲染结果（inpainted背景 + 文字）
                    # 直接使用，不需要再合成到原图上
                    final_image = rendered_text_layer
                    rendered_text_layer = None  # 避免后续释放

                    # --- Safer saving logic ---
                    temp_output_path = output_path + ".tmp"

                    try:
                        # Handle image saving, applying quality settings for supported formats
                        output_lower = output_path.lower()

                        if output_lower.endswith(('.jpg', '.jpeg')):
                            self.logger.info("Output is JPEG, converting from RGBA to RGB...")
                            temp_img = final_image.convert('RGB')
                            save_quality = config.get('cli', {}).get('save_quality', 95)
                            self.logger.info(f"Saving JPEG with quality: {save_quality}")
                            temp_img.save(temp_output_path, format='JPEG', quality=save_quality)
                            temp_img.close()  # 释放转换后的图像
                        elif output_lower.endswith('.webp'):
                            save_quality = config.get('cli', {}).get('save_quality', 95)
                            self.logger.info(f"Saving WEBP with quality: {save_quality}")
                            final_image.save(temp_output_path, format='WEBP', quality=save_quality)
                        else:
                            # For other formats like PNG, save directly
                            final_image.save(temp_output_path, format='PNG')

                        # 确保文件已写入磁盘
                        if not os.path.exists(temp_output_path):
                            raise Exception(f"临时文件未成功创建: {temp_output_path}")

                        # 释放 final_image（只在保存成功后）
                        try:
                            final_image.close()
                        except:
                            pass

                        # If save is successful, rename the temp file to the final output path
                        os.replace(temp_output_path, output_path)
                        self.logger.info(f"Successfully saved and replaced file at: {output_path}")

                    except Exception as e:
                        self.logger.error(f"Failed to save image to {output_path}: {e}")
                        # 确保释放图像
                        try:
                            if 'final_image' in locals() and final_image:
                                final_image.close()
                        except:
                            pass
                        # Re-raise the exception to be caught by the main try-except block
                        raise
                    finally:
                        # Clean up the temporary file if it still exists
                        if os.path.exists(temp_output_path):
                            os.remove(temp_output_path)
                    
                    if success_callback:
                        success_callback(f"图片已导出到: {output_path}")

                    self.logger.info(f"图片已成功导出到: {output_path}")
                else:
                    if error_callback:
                        error_callback("导出失败: 没有生成结果图片")

        except Exception as e:
            self.logger.error(f"后端渲染导出失败: {e}")
            import traceback
            traceback.print_exc()

            if error_callback:
                error_callback(f"后端渲染导出失败: {e}")
        finally:
            # 强制执行垃圾回收，释放内存
            gc.collect()
            
            # 清理GPU显存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    self.logger.info("GPU显存已清理")
            except Exception:
                pass
    
    def _save_regions_data_with_path(self, regions_data: List[Dict[str, Any]], json_path: str, image_path: str, mask: Optional[np.ndarray] = None, config: Optional[Dict[str, Any]] = None):
        """保存区域数据到JSON文件，使用正确的图片路径作为键（用于编辑器保存）"""
        # 使用图片的绝对路径作为键，与加载时保持一致
        image_key = os.path.abspath(image_path)
        self._save_regions_data_internal(regions_data, json_path, image_key, mask, config)
    
    def _save_regions_data(self, regions_data: List[Dict[str, Any]], json_path: str, mask: Optional[np.ndarray] = None, config: Optional[Dict[str, Any]] = None):
        """保存区域数据到JSON文件，确保格式与TextBlock兼容（用于导出）"""
        # 使用文件名作为键（向后兼容）
        image_key = os.path.splitext(os.path.basename(json_path.replace('_translations.json', '')))[0]
        self._save_regions_data_internal(regions_data, json_path, image_key, mask, config)
    
    def _save_regions_data_internal(self, regions_data: List[Dict[str, Any]], json_path: str, image_key: str, mask: Optional[np.ndarray] = None, config: Optional[Dict[str, Any]] = None):
        """保存区域数据到JSON文件的内部实现"""
        # 获取超分倍率，用于放大坐标
        upscale_ratio = 1
        if config:
            upscale_config = config.get('upscale', {})
            upscale_ratio = upscale_config.get('upscale_ratio', 0) or 1
        
        # 准备保存数据，确保数据格式正确
        save_data = []
        for region in regions_data:
            region_copy = region.copy()

            # 确保必要字段存在
            if 'translation' not in region_copy:
                region_copy['translation'] = region_copy.get('text', '')
            
            # 保留富文本信息（如果存在）
            # rich_text 字段包含 HTML 格式的文本，保存了字体大小、颜色、粗体、斜体等格式
            # 如果没有富文本格式，该字段为空字符串或不存在
            if 'rich_text' not in region_copy:
                region_copy['rich_text'] = ''
            
            # 确保lines字段存在且格式正确
            if 'lines' not in region_copy:
                self.logger.warning(f"Region missing 'lines' field: {region_copy}")
                continue
            
            # 验证和转换lines数据格式
            lines_data = region_copy['lines']
            if isinstance(lines_data, list):
                # 确保每个多边形都有足够的点
                valid_polygons = []
                for poly in lines_data:
                    if isinstance(poly, list) and len(poly) >= 4:
                        # 确保每个点都是[x, y]格式
                        valid_points = []
                        for point in poly:
                            if isinstance(point, (list, tuple)) and len(point) >= 2:
                                valid_points.append([float(point[0]), float(point[1])])
                            else:
                                self.logger.warning(f"Invalid point format in polygon: {point}")
                                break
                        else:
                            if len(valid_points) >= 4:
                                # 确保是矩形格式（4个点）
                                if len(valid_points) == 4:
                                    valid_polygons.append(valid_points)
                                else:
                                    # 如果超过4个点，取前4个点
                                    self.logger.warning(f"Polygon has {len(valid_points)} points, using first 4")
                                    valid_polygons.append(valid_points[:4])
                    else:
                        self.logger.warning(f"Invalid polygon format: {poly}")
                
                if valid_polygons:
                    # 恢复到正确的 (N, 4, 2) 形状
                    region_copy['lines'] = np.array(valid_polygons, dtype=np.float64)
                else:
                    self.logger.warning(f"No valid polygons found in region: {region_copy}")
                    continue
            elif isinstance(lines_data, np.ndarray):
                # 如果已经是numpy数组，验证并修正形状
                lines_arr = lines_data
                if lines_arr.ndim == 2 and lines_arr.shape == (4, 2):
                    # 单个多边形，需要添加一个维度变成 (1, 4, 2)
                    lines_arr = lines_arr.reshape(1, 4, 2)
                    self.logger.debug("Reshaped lines from (4, 2) to (1, 4, 2)")
                elif lines_arr.ndim != 3 or lines_arr.shape[1] != 4 or lines_arr.shape[2] != 2:
                    self.logger.warning(f"Invalid lines array shape: {lines_arr.shape}, expected (N, 4, 2)")
                    continue
                region_copy['lines'] = lines_arr.astype(np.float64)
            else:
                self.logger.warning(f"Lines data is not a list or numpy array: {type(lines_data)}")
                continue
            
            # --- Foreground Color ---
            # 优先使用 font_color (hex格式),如果没有才使用 fg_colors/fg_color (tuple格式)
            if 'font_color' not in region_copy or region_copy['font_color'] is None:
                fg_tuple = region_copy.pop('fg_colors', None)
                if fg_tuple is None:
                    fg_tuple = region_copy.pop('fg_color', None) # Fallback for singular

                if isinstance(fg_tuple, (list, tuple)) and len(fg_tuple) == 3:
                    try:
                        r, g, b = fg_tuple
                        region_copy['font_color'] = f"#{int(r):02x}{int(g):02x}{int(b):02x}"
                    except (ValueError, TypeError) as e:
                        self.logger.warning(f"Could not convert fg_color tuple to hex for saving: {e}")
            else:
                # font_color 已存在,移除 fg_colors/fg_color 避免冲突
                region_copy.pop('fg_colors', None)
                region_copy.pop('fg_color', None)

            # --- Background/Stroke Color ---
            bg_tuple = region_copy.pop('bg_colors', None)
            if bg_tuple is None:
                bg_tuple = region_copy.pop('bg_color', None) # Fallback
            
            # Ensure bg_color (singular) is present in the final dict if it exists
            if bg_tuple:
                region_copy['bg_color'] = bg_tuple

            # 确保其他必要字段存在
            if 'texts' not in region_copy:
                region_copy['texts'] = [region_copy.get('text', '')]
            
            # 确保其他必要字段存在
            if 'language' not in region_copy:
                region_copy['language'] = 'unknown'
            if 'font_size' not in region_copy:
                region_copy['font_size'] = 12
            if 'angle' not in region_copy:
                region_copy['angle'] = 0
            if 'target_lang' not in region_copy:
                region_copy['target_lang'] = 'CHS'  # 默认目标语言
            
            # 转换 direction 值：'v' -> 'vertical', 'h' -> 'horizontal'
            if 'direction' in region_copy:
                direction_value = region_copy['direction']
                if direction_value == 'v':
                    region_copy['direction'] = 'vertical'
                elif direction_value == 'h':
                    region_copy['direction'] = 'horizontal'
            
            save_data.append(region_copy)
        
        # load_text模式期望的格式：字典，键为图片路径，值为包含regions的字典
        # image_key 由调用方传入（可以是完整路径或文件名）
        formatted_data = {
            image_key: {
                'regions': save_data
            }
        }
        
        # 添加超分和上色配置信息
        if config:
            upscale_config = config.get('upscale', {})
            upscale_ratio = upscale_config.get('upscale_ratio', 0)
            if upscale_ratio:
                formatted_data[image_key]['upscale_ratio'] = upscale_ratio
                upscaler = upscale_config.get('upscaler', '')
                if upscaler:
                    formatted_data[image_key]['upscaler'] = upscaler
                self.logger.info(f"在JSON中记录超分信息: ratio={upscale_ratio}, upscaler={upscaler}")
            
            colorizer_config = config.get('colorizer', {})
            colorizer = colorizer_config.get('colorizer', '')
            if colorizer and colorizer != 'none':
                formatted_data[image_key]['colorizer'] = colorizer
                self.logger.info(f"在JSON中记录上色信息: colorizer={colorizer}")
        
        # 如果有蒙版数据，则添加到JSON中
        if mask is not None:
            self.logger.info("在导出JSON中加入预计算的蒙版（已编辑的refined mask）。")
            # 使用base64编码保存蒙版，避免JSON文件过大
            import base64
            import cv2
            _, encoded_mask = cv2.imencode('.png', mask)
            mask_base64 = base64.b64encode(encoded_mask).decode('utf-8')
            formatted_data[image_key]['mask_raw'] = mask_base64
            formatted_data[image_key]['mask_is_refined'] = True  # 标记为已精炼的蒙版，跳过后端的蒙版优化
            self.logger.info("蒙版已保存（base64编码），标记为已精炼，后端将跳过蒙版优化")

        # 添加调试信息
        self.logger.info(f"保存区域数据到: {json_path}")
        self.logger.info(f"区域数量: {len(save_data)}")
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(formatted_data, f, indent=2, ensure_ascii=False, cls=CustomJSONEncoder)
    
    def _prepare_translator_params(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """准备翻译器参数"""
        translator_params = {}
        
        # 字体路径处理：让后端自己处理，不在这里转换
        # 后端的 set_font 会在路径不存在时自动回退到 DEFAULT_FONT
        # 只有当配置中是绝对路径且存在时才传递
        render_config = config.get('render', {})
        font_path_value = render_config.get('font_path')
        if font_path_value and os.path.isabs(font_path_value) and os.path.exists(font_path_value):
            translator_params['font_path'] = font_path_value
            self.logger.info(f"设置字体路径: {font_path_value}")
        else:
            # 清除相对路径，让后端使用默认字体
            if 'render' in config and 'font_path' in config['render']:
                config['render']['font_path'] = None
            self.logger.info("使用后端默认字体")
        
        # 提取输出格式
        output_format = self.get_output_format_from_config(config)
        if output_format:
            translator_params['format'] = output_format
            self.logger.info(f"设置输出格式: {output_format}")
        
        # 提取并传递GPU配置
        cli_config = config.get('cli', {})
        if 'use_gpu' in cli_config:
            translator_params['use_gpu'] = cli_config['use_gpu']
            self.logger.info(f"设置GPU配置: use_gpu={cli_config['use_gpu']}")
        
        # 设置其他参数
        translator_params.update(config)
        translator_params['load_text'] = True  # 关键：启用加载文本模式
        translator_params['save_text'] = False  # 不保存文本
        
        # 添加调试日志
        self.logger.info(f"Config keys: {list(config.keys())}")
        if 'upscale' in config:
            self.logger.info(f"Upscale config: {config['upscale']}")
        else:
            self.logger.warning("No upscale config found in config")
        if 'colorizer' in config:
            self.logger.info(f"Colorizer config: {config['colorizer']}")
        else:
            self.logger.warning("No colorizer config found in config")
        
        # 关键：设置翻译器为none，跳过翻译步骤，直接渲染
        translator_params['translator'] = 'none'
        self.logger.info("设置翻译器为none，启用load_text模式，跳过翻译步骤，直接进行渲染")
        
        return translator_params
    
    def _execute_backend_render(self, image_path: str, regions_json_path: str,
                              translator_params: Dict[str, Any], config: Dict[str, Any],
                              progress_callback: Optional[callable] = None) -> Optional[Image.Image]:
        """执行后端渲染"""
        image = None
        try:
            from manga_translator.config import Config, RenderConfig
            from manga_translator.manga_translator import MangaTranslator

            if progress_callback:
                progress_callback("创建翻译器实例...")

            # 创建翻译器实例
            translator = MangaTranslator(params=translator_params)

            if progress_callback:
                progress_callback("加载图片和配置...")

            # 加载图片
            image = Image.open(image_path)
            image.name = image_path  # 确保图片名称正确，用于load_text模式查找翻译文件

            # 创建配置对象
            render_config = config.get('render', {}).copy()  # 使用copy避免修改原配置
            
            # 转换 direction 值：'v' -> 'vertical', 'h' -> 'horizontal'
            if 'direction' in render_config:
                direction_value = render_config['direction']
                if direction_value == 'v':
                    render_config['direction'] = 'vertical'
                elif direction_value == 'h':
                    render_config['direction'] = 'horizontal'
            
            render_config['font_color'] = None # Explicitly disable global font color
            render_cfg = RenderConfig(**render_config)

            # 创建翻译器配置，设置为none以跳过翻译
            from manga_translator.config import TranslatorConfig, UpscaleConfig, ColorizerConfig
            translator_cfg = TranslatorConfig(translator='none')
            
            # 从config中提取upscale和colorizer配置
            upscale_config = config.get('upscale', {})
            colorizer_config = config.get('colorizer', {})
            upscale_cfg = UpscaleConfig(**upscale_config) if upscale_config else UpscaleConfig()
            colorizer_cfg = ColorizerConfig(**colorizer_config) if colorizer_config else ColorizerConfig()
            
            self.logger.info(f"Creating Config with upscale_ratio={upscale_cfg.upscale_ratio}, colorizer={colorizer_cfg.colorizer}")

            cfg = Config(render=render_cfg, translator=translator_cfg, upscale=upscale_cfg, colorizer=colorizer_cfg)

            if progress_callback:
                progress_callback("执行后端渲染...")

            # 执行翻译（实际是渲染）
            import sys
            # 在Windows上的工作线程中，需要手动初始化Windows Socket
            if sys.platform == 'win32':
                # 使用ctypes直接调用WSAStartup
                import ctypes
                try:
                    WSADATA_SIZE = 400
                    wsa_data = ctypes.create_string_buffer(WSADATA_SIZE)
                    ws2_32 = ctypes.WinDLL('ws2_32')
                    ws2_32.WSAStartup(0x0202, wsa_data)
                except:
                    pass
                
                asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                ctx = loop.run_until_complete(translator.translate(image, cfg, image_name=image.name))

                if ctx.result is not None:
                    # ctx.result is already a PIL Image object, no conversion needed.
                    # 复制图像以避免 "Operation on closed image" 错误
                    result_image = ctx.result.copy()
                    # 关闭原始结果图像
                    ctx.result.close()
                    # 关闭输入图像以释放内存
                    if image:
                        image.close()
                        image = None
                    return result_image
                else:
                    # 详细记录失败原因
                    error_msg = "后端渲染没有生成结果"
                    if hasattr(ctx, 'translation_error') and ctx.translation_error:
                        error_msg += f", 错误: {ctx.translation_error}"
                    if hasattr(ctx, 'text_regions'):
                        error_msg += f", 区域数: {len(ctx.text_regions) if ctx.text_regions else 0}"
                    self.logger.error(error_msg)
                    return None

            finally:
                loop.close()

        except Exception as e:
            self.logger.error(f"执行后端渲染时出错: {e}")
            raise
        finally:
            # 确保输入图像被关闭
            if image:
                image.close()
    
    def export_regions_json(self, regions_data: List[Dict[str, Any]], output_path: str, config: Optional[Dict[str, Any]] = None) -> bool:
        """导出区域数据为JSON文件"""
        try:
            self._save_regions_data(regions_data, output_path, None, config)
            self.logger.info(f"区域数据已导出到: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"导出区域数据失败: {e}")
            return False


# 创建全局导出服务实例
_export_service = None

def get_export_service() -> ExportService:
    """获取导出服务实例"""
    global _export_service
    if _export_service is None:
        _export_service = ExportService()
    return _export_service
