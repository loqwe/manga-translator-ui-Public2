
import asyncio
import torch
import cv2
import json
import langcodes
import os
import regex as re
import sys
import time
import logging
import traceback
import numpy as np
from PIL import Image
from typing import Optional, Any, List
import py3langid as langid

from .config import Config, Colorizer, Translator, Renderer, Inpainter
from .utils import (
    BASE_PATH,
    LANGUAGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    load_image,
    dump_image,
    visualize_textblocks,
    is_valuable_text,
    is_mostly_noise_text,
    sort_regions,
    TextBlock,
    imwrite_unicode
)
from .utils.text_filter import match_filter, ensure_filter_list_exists
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端
from matplotlib import cm
from .utils.path_manager import (
    get_json_path,
    get_inpainted_path,
    find_json_path
)

from .detection import dispatch as dispatch_detection, prepare as prepare_detection, unload as unload_detection
from .upscaling import dispatch as dispatch_upscaling, prepare as prepare_upscaling, unload as unload_upscaling
from .ocr import dispatch as dispatch_ocr, prepare as prepare_ocr, unload as unload_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import dispatch as dispatch_inpainting, prepare as prepare_inpainting, unload as unload_inpainting
from .translators import (
    dispatch as dispatch_translation,
    prepare as prepare_translation,
    unload as unload_translation,
)
from .translators.common import ISO_639_1_TO_VALID_LANGUAGES
from .colorization import dispatch as dispatch_colorization, prepare as prepare_colorization, unload as unload_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render, dispatch_eng_render_pillow

# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

def set_main_logger(l):
    global logger
    logger = l

class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass

def load_dictionary(file_path):
    dictionary = []
    if file_path:
        path_to_check = file_path if os.path.isabs(file_path) else os.path.join(BASE_PATH, file_path)
        if os.path.exists(path_to_check):
            with open(path_to_check, 'r', encoding='utf-8') as file:
                for line_number, line in enumerate(file, start=1):
                    # Ignore empty lines and lines starting with '#' or '//'
                    if not line.strip() or line.strip().startswith('#') or line.strip().startswith('//'):
                        continue
                    # Remove comment parts
                    line = line.split('#')[0].strip()
                    line = line.split('//')[0].strip()
                    parts = line.split()
                    if len(parts) == 1:
                        # If there is only the left part, the right part defaults to an empty string, meaning delete the left part
                        pattern = re.compile(parts[0])
                        dictionary.append((pattern, '', line_number))
                    elif len(parts) == 2:
                        # If both left and right parts are present, perform the replacement
                        pattern = re.compile(parts[0])
                        dictionary.append((pattern, parts[1], line_number))
                    else:
                        logger.error(f'Invalid dictionary entry at line {line_number}: {line.strip()}')
    return dictionary

def apply_dictionary(text, dictionary):
    for pattern, value, line_number in dictionary:
        _original_text = text  
        text = pattern.sub(value, text)
    return text

def parse_upscale_ratio(upscale_ratio) -> float:
    """
    解析超分倍率，支持多种格式：
    - 数字: 2, 3, 4
    - 字符串数字: "2", "3", "4"
    - mangajanai格式: "x2", "x4", "DAT2 x4"
    - realcugan格式: "2x-conservative", "3x-denoise1x" 等
    
    返回浮点数倍率，如果无法解析则返回 0
    """
    if not upscale_ratio:
        return 0
    
    try:
        if isinstance(upscale_ratio, (int, float)):
            return float(upscale_ratio)
        
        if isinstance(upscale_ratio, str):
            # 移除空格并转小写
            upscale_ratio = upscale_ratio.strip().lower()
            # 优先匹配开头的数字（如 "2x-conservative" 中的 2）
            match = re.match(r'^(\d+)x', upscale_ratio)
            if not match:
                # 如果没匹配到，尝试匹配任意位置的数字（如 "x2" 或 "DAT2 x4"）
                match = re.search(r'(\d+)', upscale_ratio)
            
            if match:
                return float(match.group(1))
        
        return 0
    except (ValueError, TypeError):
        logger.warning(f"无法解析超分倍率: {upscale_ratio}, 将忽略")
        return 0

class MangaTranslator:
    verbose: bool
    ignore_errors: bool
    _gpu_limited_memory: bool
    device: Optional[str]
    kernel_size: Optional[int]
    models_ttl: int
    _progress_hooks: list[Any]
    result_sub_folder: str
    batch_size: int
    
    # 水印过滤规则缓存（类级别，避免重复加载）
    _watermark_filter_cache = None
    _watermark_filter_cache_time = 0

    def __init__(self, params: dict = {}):
        self.pre_dict = params.get('pre_dict', None)
        self.post_dict = params.get('post_dict', None)
        self.font_path = None
        self.kernel_size = None
        self.device = None
        self.text_output_file = params.get('save_text_file', None)
        self._gpu_limited_memory = False
        self.ignore_errors = False
        self.verbose = False
        self.models_ttl = 0
        self.batch_size = 1  # 默认不批量处理

        self._progress_hooks = []
        self._add_logger_hook()
        
        # 取消检查回调（用于Web服务器等场景）
        self._cancel_check_callback = None

        params = params or {}
        
        self._batch_contexts = []  # 存储批量处理的上下文
        self._batch_configs = []   # 存储批量处理的配置
        # batch_concurrent 四并发模式（默认关闭，可通过配置开启）
        self.batch_concurrent = params.get('batch_concurrent', False)
        
        # 添加模型加载状态标志
        self._models_loaded = False
        
        self.parse_init_params(params)
        self.result_sub_folder = ''

        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False
        # in PyTorch 1.12 and later.
        torch.backends.cuda.matmul.allow_tf32 = True

        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        torch.backends.cudnn.allow_tf32 = True

        self._model_usage_timestamps = {}
        self._detector_cleanup_task = None
        self.context_size = params.get('context_size', 0)
        self.all_page_translations = []
        self._original_page_texts = []  # 存储原文页面数据，用于并发模式下的上下文

        # 调试图片管理相关属性
        self._current_image_context = None  # 存储当前处理图片的上下文信息
        self._saved_image_contexts = {}     # 存储批量处理中每个图片的上下文信息
        
        # Task-level translator cache: {(task_id, translator_key): translator}
        self._task_translator_cache = {}
        
        # 日志文件现在由UI层管理，这里不再创建
        self._log_file_path = None
        
        # 过滤列表开关（默认启用）
        self.filter_text_enabled = params.get('filter_text_enabled', True)
        
        # 确保过滤列表文件存在
        try:
            ensure_filter_list_exists()
        except Exception:
            pass

    def parse_init_params(self, params: dict):
        self.verbose = params.get('verbose', False)
        # font_path 优先从配置文件读取，如果没有则使用命令行参数
        self.font_path = params.get('font_path', None)
        self.models_ttl = params.get('models_ttl', 0)
        self.batch_size = params.get('batch_size', 3)  # 批量大小（翻译批次）
        
        # batch_concurrent 四并发流水线处理（可选功能）
        # 开启后：检测、OCR、修复、翻译四并发，提升处理速度
            
        self.ignore_errors = params.get('ignore_errors', False)
        # check mps for apple silicon or cuda for nvidia
        device = 'mps' if torch.backends.mps.is_available() else 'cuda'
        self.device = device if params.get('use_gpu', False) else 'cpu'
        if self.using_gpu and ( not torch.cuda.is_available() and not torch.backends.mps.is_available()):
            # GPU不可用时，自动回退到CPU而不是抛出异常
            logger.warning(
                'CUDA or Metal compatible device could not be found in torch whilst --use-gpu was set. '
                'Automatically falling back to CPU mode.'
            )
            self.device = 'cpu'
        if params.get('model_dir'):
            ModelWrapper._MODEL_DIR = params.get('model_dir')
        #todo: fix why is kernel size loaded in the constructor
        self.kernel_size=int(params.get('kernel_size', 3))
        # Set input files
        self.input_files = params.get('input', [])
        # Set save_text
        self.save_text = params.get('save_text', False)
        # Set load_text
        self.load_text = params.get('load_text', False)
        self.save_mask = not params.get('no_save_mask', False)
        self.template = params.get('template', False)
        self.attempts = params.get('attempts', -1)
        self.save_quality = params.get('save_quality', 100)
        self.skip_no_text = params.get('skip_no_text', False)
        self.generate_and_export = params.get('generate_and_export', False)
        self.colorize_only = params.get('colorize_only', False)
        self.upscale_only = params.get('upscale_only', False)
        self.inpaint_only = params.get('inpaint_only', False)
        
        # 替换翻译模式（从已翻译图片复制翻译数据到生肉图片）
        self.replace_translation = params.get('replace_translation', False)
        
        # 翻译并发数（并发流水线模式下，同时发起的翻译请求数）
        self.translation_concurrency = params.get('translation_concurrency', 1)
        
        
        # batch_concurrent 已在初始化时设置并验证
        

        
    def _set_image_context(self, config: Config, image=None):
        """设置当前处理图片的上下文信息，用于生成调试图片子文件夹"""
        from .utils.generic import get_image_md5

        # 使用毫秒级时间戳确保唯一性
        timestamp = str(int(time.time() * 1000))
        detection_size = str(getattr(config.detector, 'detection_size', 1024))
        target_lang = getattr(config.translator, 'target_lang', 'unknown')
        translator = getattr(config.translator, 'translator', 'unknown')

        # 计算图片MD5哈希值
        if image is not None:
            file_md5 = get_image_md5(image)
        else:
            file_md5 = "unknown"

        # 生成子文件夹名：{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}
        subfolder_name = f"{timestamp}-{file_md5}-{detection_size}-{target_lang}-{translator}"

        self._current_image_context = {
            'subfolder': subfolder_name,
            'file_md5': file_md5,
            'config': config
        }
        
    def _get_image_subfolder(self) -> str:
        """获取当前图片的调试子文件夹名"""
        if self._current_image_context:
            return self._current_image_context['subfolder']
        return ''
    
    def _save_current_image_context(self, image_md5: str):
        """保存当前图片上下文，用于批量处理中保持一致性"""
        if self._current_image_context:
            self._saved_image_contexts[image_md5] = self._current_image_context.copy()

    def _restore_image_context(self, image_md5: str):
        """恢复保存的图片上下文"""
        if image_md5 in self._saved_image_contexts:
            self._current_image_context = self._saved_image_contexts[image_md5].copy()
            return True
        return False

    @property
    def using_gpu(self):
        return self.device.startswith('cuda') or self.device == 'mps'

    async def translate(self, image: Image.Image, config: Config, image_name: str = None, skip_context_save: bool = False, save_info: dict = None) -> Context:
        """
        Translates a single image by calling translate_batch with batch_size=1.
        
        This is a compatibility wrapper. All translation logic is now unified in translate_batch().

        :param image: Input image.
        :param config: Translation config.
        :param image_name: Image file name for saving results.
        :param save_info: Save configuration (output_folder, format, etc.)
        :return: Translation context.
        """
        # Attach image_name to image object for batch processing
        if image_name and not hasattr(image, 'name'):
            image.name = image_name
        
        # Call unified batch translation with single image
        results = await self.translate_batch(
            images_with_configs=[(image, config)],
            batch_size=1,
            save_info=save_info
        )
        
        # Return the single result
        return results[0] if results else Context()

    def _calculate_output_path(self, image_path: str, save_info: dict) -> str:
        """
        计算输出文件的完整路径
        
        Args:
            image_path: 输入图片的路径
            save_info: 包含输出配置的字典，包括：
                - output_folder: 输出文件夹
                - input_folders: 输入文件夹集合
                - format: 输出格式（可选）
                - save_to_source_dir: 是否输出到原图目录的 manga_translator_work/result 子目录
                
        Returns:
            str: 计算后的输出文件完整路径
        """
        output_folder = save_info.get('output_folder')
        input_folders = save_info.get('input_folders', set())
        output_format = save_info.get('format')
        save_to_source_dir = save_info.get('save_to_source_dir', False)
        
        file_path = image_path
        parent_dir = os.path.normpath(os.path.dirname(file_path))
        
        # 检查是否启用了"输出到原图目录"模式
        if save_to_source_dir:
            # 输出到原图所在目录的 manga_translator_work/result 子目录
            final_output_dir = os.path.join(parent_dir, 'manga_translator_work', 'result')
        else:
            # 原有逻辑：使用配置的输出目录
            final_output_dir = output_folder
            
            # 计算相对路径以保持文件夹结构
            for folder in input_folders:
                if parent_dir.startswith(folder):
                    relative_path = os.path.relpath(parent_dir, folder)
                    # Normalize path and avoid adding '.' as a directory component
                    if relative_path == '.':
                        final_output_dir = os.path.join(output_folder, os.path.basename(folder))
                    else:
                        final_output_dir = os.path.join(output_folder, os.path.basename(folder), relative_path)
                    # Normalize to use consistent separators
                    final_output_dir = os.path.normpath(final_output_dir)
                    break
        
        os.makedirs(final_output_dir, exist_ok=True)
        
        # 处理输出文件名和格式
        base_filename, _ = os.path.splitext(os.path.basename(file_path))
        if output_format and output_format.strip() and output_format.lower() != 'none':
            output_filename = f"{base_filename}.{output_format}"
        else:
            output_filename = os.path.basename(file_path)
        
        final_output_path = os.path.join(final_output_dir, output_filename)
        return final_output_path

    def _save_translated_image(self, image: Image.Image, output_path: str, image_path: str, overwrite: bool = True, mode_label: str = "BATCH") -> bool:
        """
        保存翻译后的图片到指定路径
        
        Args:
            image: 要保存的PIL图片对象
            output_path: 输出文件路径
            image_path: 原始图片路径（用于更新翻译映射表）
            overwrite: 是否覆盖已存在的文件
            mode_label: 模式标签（用于日志）
            
        Returns:
            bool: 是否成功保存
        """
        if not overwrite and os.path.exists(output_path):
            logger.info(f"  -> ⚠️ [{mode_label}] Skipping existing file: {os.path.basename(output_path)}")
            return False
        
        try:
            image_to_save = image
            # 处理RGBA到RGB转换（JPEG格式）
            if output_path.lower().endswith(('.jpg', '.jpeg')) and image_to_save.mode in ('RGBA', 'LA'):
                image_to_save = image_to_save.convert('RGB')
            
            # 保存图片并应用save_quality设置
            image_to_save.save(output_path, quality=self.save_quality)
            logger.info(f"  -> ✅ [{mode_label}] Saved successfully: {os.path.basename(output_path)}")
            
            # 创建"出生证明"文件：记录原始源文件夹路径
            self._create_source_path_file(image_path, output_path)
            
            # 更新翻译映射表
            self._update_translation_map(image_path, output_path)
            return True
        except Exception as e:
            logger.error(f"Error saving image to {output_path}: {e}")
            return False
    
    def _create_source_path_file(self, image_path: str, output_path: str):
        """
        创建"出生证明"文件：在输出文件夹中记录原始源文件夹路径
        
        Args:
            image_path: 原始图片路径
            output_path: 输出图片路径
        """
        try:
            output_dir = os.path.dirname(output_path)
            source_path_file = os.path.join(output_dir, "_source_path.txt")
            
            # 如果文件已存在，跳过（同一批次只需创建一次）
            if os.path.exists(source_path_file):
                return
            
            # 获取原始源文件夹的完整路径
            source_dir = os.path.dirname(os.path.abspath(image_path))
            
            # 写入源路径
            with open(source_path_file, 'w', encoding='utf-8') as f:
                f.write(source_dir)
            
            logger.debug(f"  -> 创建出生证明: _source_path.txt -> {source_dir}")
        except Exception as e:
            logger.warning(f"创建_source_path.txt失败: {e}")

    def _save_and_cleanup_context(self, ctx: Context, save_info: dict, config: Config = None, mode_label: str = "BATCH") -> bool:
        """
        统一的保存和清理方法：保存翻译结果、导出PSD并清理内存
        
        Args:
            ctx: Context对象
            save_info: 保存信息字典
            config: Config对象（用于PSD导出）
            mode_label: 模式标签（用于日志）
            
        Returns:
            bool: 是否成功保存
        """
        if not save_info or not ctx.result:
            return False
        
        try:
            overwrite = save_info.get('overwrite', True)
            final_output_path = self._calculate_output_path(ctx.image_name, save_info)
            success = self._save_translated_image(ctx.result, final_output_path, ctx.image_name, overwrite, mode_label)
            
            # 标记成功
            if success or not overwrite:  # 跳过已存在的文件也算成功
                ctx.success = True
            
            # 导出可编辑PSD（如果启用）
            if config and hasattr(config, 'cli') and hasattr(config.cli, 'export_editable_psd') and config.cli.export_editable_psd:
                try:
                    from .utils.photoshop_export import photoshop_export, get_psd_output_path
                    psd_path = get_psd_output_path(ctx.image_name)
                    cli_cfg = getattr(config, 'cli', None)
                    default_font = getattr(cli_cfg, 'psd_font', None)
                    line_spacing = getattr(config.render, 'line_spacing', None) if hasattr(config, 'render') else None
                    script_only = getattr(cli_cfg, 'psd_script_only', False)
                    photoshop_export(psd_path, ctx, default_font, ctx.image_name, self.verbose, self._result_path, line_spacing, script_only)
                    logger.info(f"  -> ✅ [PSD] Exported editable PSD: {os.path.basename(psd_path)}")
                except Exception as psd_err:
                    logger.error(f"Error exporting PSD for {os.path.basename(ctx.image_name)}: {psd_err}")
            
            # ✅ 保存后立即清理result以释放内存
            ctx.result = None
            
            return success
        except Exception as e:
            logger.error(f"Error in _save_and_cleanup_context: {e}")
            return False

    def _save_text_to_file(self, image_path: str, ctx: Context, config: Config = None):
        """保存翻译数据到JSON文件，使用新的目录结构"""
        text_output_file = self.text_output_file
        if not text_output_file:
            # 使用新的路径管理器生成JSON路径
            text_output_file = get_json_path(image_path, create_dir=True)

        data = {}
        image_key = os.path.abspath(image_path)

        # Prepare data for JSON serialization
        regions_data = [region.to_dict() for region in ctx.text_regions]

        # 强制使用Config中的排版方向和对齐方式覆盖（如果存在）
        # 这是为了确保即使 textline_merge 检测过程使用了 auto，
        # 最终保存时也会反映用户的强制设置（例如全书强制横排）
        if config and hasattr(config, 'render'):
            try:
                # 覆盖方向
                if hasattr(config.render, 'direction'):
                    dir_val = config.render.direction
                    if hasattr(dir_val, 'value'): dir_val = dir_val.value
                    
                    forced_direction = None
                    if dir_val == 'vertical': forced_direction = 'v'
                    elif dir_val == 'horizontal': forced_direction = 'h'
                    
                    if forced_direction:
                        for region in regions_data:
                            region['direction'] = forced_direction
                
                # 覆盖对齐方式
                if hasattr(config.render, 'alignment'):
                    align_val = config.render.alignment
                    if hasattr(align_val, 'value'): align_val = align_val.value
                    
                    if align_val in ('left', 'center', 'right'):
                        for region in regions_data:
                            region['alignment'] = align_val
                            
            except Exception as e:
                logger.warning(f"Failed to override region settings from config: {e}")

        # 获取图片尺寸（优先使用保存的尺寸，兼容并发模式）
        if hasattr(ctx, 'original_size') and ctx.original_size:
            original_width, original_height = ctx.original_size
        elif ctx.input and hasattr(ctx.input, 'size'):
            original_width, original_height = ctx.input.size
        else:
            # 如果都没有，使用默认值或从图片文件读取
            logger.warning("无法获取图片尺寸，使用默认值")
            original_width, original_height = 0, 0
        
        data_to_save = {
            'regions': regions_data,
            'original_width': original_width,
            'original_height': original_height
        }
        
        # 添加超分和上色配置信息
        if config:
            if config.upscale and config.upscale.upscale_ratio:
                data_to_save['upscale_ratio'] = config.upscale.upscale_ratio
                if config.upscale.upscaler:
                    data_to_save['upscaler'] = config.upscale.upscaler
                logger.info(f"在JSON中记录超分信息: ratio={config.upscale.upscale_ratio}, upscaler={config.upscale.upscaler}")
            
            if config.colorizer and config.colorizer.colorizer and config.colorizer.colorizer != 'none':
                data_to_save['colorizer'] = config.colorizer.colorizer
                logger.info(f"在JSON中记录上色信息: colorizer={config.colorizer.colorizer}")

        if self.save_mask and ctx.mask_raw is not None:
            try:
                import base64
                import cv2
                _, buffer = cv2.imencode('.png', ctx.mask_raw)
                mask_base64 = base64.b64encode(buffer).decode('utf-8')
                data_to_save['mask_raw'] = mask_base64
                # 保存蒙版是否已优化的标志
                data_to_save['mask_is_refined'] = getattr(ctx, 'mask_is_refined', False)
            except Exception as e:
                logger.error(f"Failed to encode mask to base64: {e}")

        data[image_key] = data_to_save

        try:
            # Use a custom encoder to handle numpy types
            class NumpyEncoder(json.JSONEncoder):
                def default(self, obj):
                    if isinstance(obj, np.integer):
                        return int(obj)
                    if isinstance(obj, np.floating):
                        return float(obj)
                    if isinstance(obj, np.ndarray):
                        return obj.tolist()
                    return super(NumpyEncoder, self).default(obj)

            json_string = json.dumps(data, ensure_ascii=False, indent=4, cls=NumpyEncoder)

            with open(text_output_file, 'wb') as f:
                f.write(json_string.encode('utf-8'))
            logger.info(f"JSON saved to: {text_output_file}")
        except Exception as e:
            logger.error(f"Failed to write translation file to {text_output_file}: {e}")

    def _save_inpainted_image(self, image_path: str, inpainted_img: np.ndarray):
        """保存修复后的图片到inpainted目录"""
        try:
            inpainted_path = get_inpainted_path(image_path, create_dir=True)
            imwrite_unicode(inpainted_path, cv2.cvtColor(inpainted_img, cv2.COLOR_RGB2BGR), logger)
            logger.info(f"Inpainted image saved to: {inpainted_path}")
        except Exception as e:
            logger.error(f"Failed to save inpainted image: {e}")

    def _preprocess_load_text_mode(self, images_with_configs: List[tuple]):
        """
        load_text模式预处理：自动从TXT文件导入翻译到JSON
        这个方法在翻译开始前统一执行，确保CLI和UI都能使用
        """
        try:
            from manga_translator.utils.path_manager import find_json_path, find_txt_files, get_json_path
            
            # 获取默认模板路径
            template_path = self._get_default_template_path()
            if not template_path or not os.path.exists(template_path):
                logger.warning("Template file not found, skipping TXT to JSON import")
                return
            
            # 收集需要处理的图片路径
            image_paths = []
            for image, config in images_with_configs:
                if hasattr(image, 'name') and image.name:
                    image_paths.append(image.name)
            
            if not image_paths:
                return
            
            # 批量处理TXT导入
            success_count = 0
            skip_count = 0
            
            for image_path in image_paths:
                try:
                    # 查找JSON和TXT文件
                    json_path = find_json_path(image_path)
                    original_txt_path, translated_txt_path = find_txt_files(image_path)
                    
                    # 如果没有JSON文件，跳过（稍后会报错）
                    if not json_path:
                        skip_count += 1
                        continue
                    
                    # 优先使用原文TXT，其次使用翻译TXT
                    txt_path = original_txt_path if original_txt_path else translated_txt_path
                    
                    if not txt_path:
                        skip_count += 1
                        continue
                    
                    # 执行TXT到JSON的导入
                    from desktop_qt_ui.services.workflow_service import safe_update_large_json_from_text
                    result = safe_update_large_json_from_text(txt_path, json_path, template_path)
                    
                    if not result.startswith("错误"):
                        success_count += 1
                        logger.debug(f"Imported TXT to JSON: {os.path.basename(image_path)}")
                    
                except Exception as e:
                    logger.debug(f"Failed to import TXT for {os.path.basename(image_path)}: {e}")
                    continue
            
            if success_count > 0:
                logger.info(f"TXT to JSON import completed: {success_count} successful, {skip_count} skipped")
            elif skip_count > 0:
                logger.debug(f"No TXT files found for import ({skip_count} images)")
                
        except ImportError as e:
            logger.warning(f"Cannot import workflow_service, skipping TXT to JSON import: {e}")
        except Exception as e:
            logger.warning(f"Error during TXT to JSON import: {e}")
    
    def _get_default_template_path(self) -> Optional[str]:
        """获取默认模板文件路径"""
        try:
            # 尝试多个可能的路径
            possible_paths = [
                os.path.join(os.path.dirname(__file__), '..', 'examples', 'translation_template.json'),
                os.path.join(os.getcwd(), 'examples', 'translation_template.json'),
            ]
            
            # 如果是打包环境
            if getattr(sys, 'frozen', False):
                if hasattr(sys, '_MEIPASS'):
                    possible_paths.insert(0, os.path.join(sys._MEIPASS, 'examples', 'translation_template.json'))
                else:
                    exe_dir = os.path.dirname(sys.executable)
                    possible_paths.insert(0, os.path.join(exe_dir, 'examples', 'translation_template.json'))
            
            for path in possible_paths:
                abs_path = os.path.abspath(path)
                if os.path.exists(abs_path):
                    return abs_path
            
            # 如果都不存在，尝试创建默认模板
            default_path = os.path.abspath(possible_paths[0])
            os.makedirs(os.path.dirname(default_path), exist_ok=True)
            
            default_content = '''翻译模板文件

原文: <original>
译文: <translated>

'''
            with open(default_path, 'w', encoding='utf-8') as f:
                f.write(default_content)
            
            logger.info(f"Created default template at: {default_path}")
            return default_path
            
        except Exception as e:
            logger.warning(f"Failed to get/create default template: {e}")
            return None
    
    def _load_text_and_regions_from_file(self, image_path: str, config: Config) -> (Optional[List[TextBlock]], Optional[np.ndarray], bool):
        """加载翻译数据，支持新的目录结构和向后兼容"""
        if not image_path:
            return None, None, False

        # 使用path_manager查找JSON文件（新位置优先）
        text_file_path = find_json_path(image_path)

        if not text_file_path:
            # 检查旧的TXT格式
            base_path, _ = os.path.splitext(image_path)
            text_file_path_txt = base_path + '_translations.txt'
            if os.path.exists(text_file_path_txt):
                # If the old format is found, load from it
                regions = self._load_text_and_regions_from_txt_file(image_path)
                # Since old format doesn't have mask, we return None for mask and refined status
                return regions, None, False
            else:
                logger.info(f"Translation file not found for: {image_path}")
                return None, None, False

        try:
            # Force UTF-8 encoding to handle potential file encoding issues
            with open(text_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to read or parse translation file {text_file_path}: {e}")
            return None, None, False

        # Don't check the image key. Assume the user knows what they are doing
        # and that the first entry in the JSON is the one they want to load.
        if not data or len(data.values()) == 0:
            logger.warning(f"JSON file {text_file_path} is empty or invalid.")
            return None, None, False

        # Get the first value from the dictionary, regardless of the key.
        image_data = next(iter(data.values()))
        mask_is_refined = False

        # Handle both old and new JSON formats
        if isinstance(image_data, list):
            # Old format: value is a list of regions
            regions_data = image_data
            mask_raw_data = None
        elif isinstance(image_data, dict):
            # New format: value is a dict with 'regions' and 'mask_raw'
            regions_data = image_data.get('regions', [])
            mask_raw_data = image_data.get('mask_raw', None)
            mask_is_refined = image_data.get('mask_is_refined', False)
        else:
            logger.warning(f"Invalid data format in JSON file {text_file_path}.")
            return None, None, False

        regions = []
        for region_data in regions_data:
            try:
                # Convert literal '\\n' to newline characters for the rendering engine
                if 'text' in region_data and isinstance(region_data['text'], str):
                    region_data['text'] = region_data['text'].replace('\\n', '\n')
                if 'translation' in region_data and isinstance(region_data['translation'], str):
                    region_data['translation'] = region_data['translation'].replace('\\n', '\n')

                # If target_lang is missing or empty, set it from the config.
                if not region_data.get('target_lang'):
                    if config and config.translator and config.translator.target_lang:
                        region_data['target_lang'] = config.translator.target_lang
                        logger.debug(f"Region target_lang missing in JSON, falling back to config's target_lang: {config.translator.target_lang}")

                # Convert hex font_color from editor to an 'fg_color' tuple for TextBlock
                if 'font_color' in region_data and isinstance(region_data['font_color'], str):
                    hex_color = region_data.pop('font_color') # Use pop to remove the old key
                    if hex_color.startswith('#') and len(hex_color) == 7:
                        try:
                            r = int(hex_color[1:3], 16)
                            g = int(hex_color[3:5], 16)
                            b = int(hex_color[5:7], 16)
                            region_data['fg_color'] = (r, g, b)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Could not parse font_color '{hex_color}': {e}")
                
                # Map 'fg_colors' (list) to 'fg_color' (tuple) if present
                if 'fg_colors' in region_data:
                    fg_val = region_data.pop('fg_colors')
                    if isinstance(fg_val, list):
                        region_data['fg_color'] = tuple(fg_val)
                
                # Map 'bg_colors' or 'text_stroke_color' to 'bg_color'
                if 'bg_colors' in region_data:
                    bg_val = region_data.pop('bg_colors')
                    if isinstance(bg_val, list):
                        region_data['bg_color'] = tuple(bg_val)
                elif 'text_stroke_color' in region_data: # Handle UI specific name
                    bg_val = region_data.pop('text_stroke_color')
                    if isinstance(bg_val, list): # List RGB
                         region_data['bg_color'] = tuple(bg_val)
                    elif isinstance(bg_val, str) and bg_val.startswith('#'): # Hex string
                        try:
                            r = int(bg_val[1:3], 16)
                            g = int(bg_val[3:5], 16)
                            b = int(bg_val[5:7], 16)
                            region_data['bg_color'] = (r, g, b)
                        except (ValueError, TypeError):
                             pass
                
                # 处理 stroke_color_type（新参数）
                # 如果有 stroke_color_type，根据它设置 bg_color 和 adjust_bg_color
                if 'stroke_color_type' in region_data:
                    stroke_color_type = region_data.get('stroke_color_type')
                    if stroke_color_type == "white":
                        region_data['bg_color'] = (255, 255, 255)
                        region_data['adjust_bg_color'] = False  # 禁用自动调整
                    elif stroke_color_type == "black":
                        region_data['bg_color'] = (0, 0, 0)
                        region_data['adjust_bg_color'] = False  # 禁用自动调整
                    # stroke_color_type 会被传递给 TextBlock 构造函数
                
                # 描边宽度 - stroke_width 优先级高于 default_stroke_width
                # 用户在编辑器中设置的 stroke_width 应该覆盖原始的 default_stroke_width
                if 'stroke_width' in region_data:
                    region_data['default_stroke_width'] = region_data.pop('stroke_width')
                
                # 确保 line_spacing 和 default_stroke_width 被正确传递
                # 这些参数已经在 region_data 中，会被 TextBlock 构造函数接收

                # Recreate the TextBlock object by unpacking the dictionary
                # This restores all saved attributes
                if 'lines' in region_data and isinstance(region_data['lines'], list):
                    # Fix: Use np.float64 to match TextBlock expectation, not np.int32
                    lines_arr = np.array(region_data['lines'], dtype=np.float64)
                    # 验证并修正形状，确保是 (N, 4, 2)
                    if lines_arr.ndim == 2 and lines_arr.shape == (4, 2):
                        lines_arr = lines_arr.reshape(1, 4, 2)
                    elif lines_arr.ndim != 3 or lines_arr.shape[1] != 4 or lines_arr.shape[2] != 2:
                        logger.warning(f"[加载JSON] 无效的lines形状: {lines_arr.shape}, 跳过此区域")
                        continue
                    region_data['lines'] = lines_arr
                
                region = TextBlock(**region_data)
                regions.append(region)
            except Exception as e:
                logger.error(f"Failed to parse a region in {text_file_path}: {e}")
                continue
        
        mask_raw = None
        if isinstance(mask_raw_data, str):
            try:
                import base64
                import cv2
                img_bytes = base64.b64decode(mask_raw_data)
                img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                mask_raw = cv2.imdecode(img_array, cv2.IMREAD_UNCHANGED)
            except Exception as e:
                logger.error(f"Failed to decode base64 mask: {e}")
        elif isinstance(mask_raw_data, list):
            mask_raw = np.array(mask_raw_data, dtype=np.uint8)
        
        logger.info(f"Loaded {len(regions)} regions from {text_file_path}")
        if mask_raw is not None:
            logger.info(f"Loaded mask_raw from {text_file_path}")

        return regions, mask_raw, mask_is_refined

    def _load_text_and_regions_from_txt_file(self, image_path: str) -> Optional[List[TextBlock]]:
        """
        旧的TXT格式加载方法（已废弃）
        现在只支持JSON格式，此方法保留用于向后兼容但不再实现
        """
        logger.warning("TXT format is deprecated and no longer supported. Please use JSON format instead.")
        return None

    async def _translate(self, config: Config, ctx: Context) -> Context:
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input  # Fallback to input image if colorization fails

        else:
            ctx.img_colorized = ctx.input

        # -- Upscaling
        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if config.upscale.upscale_ratio:
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized # Fallback to colorized (or input) image if upscaling fails
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
        
        # 验证加载的图片
        if ctx.img_rgb is None or ctx.img_rgb.size == 0:
            logger.error("加载图片失败: img_rgb为空或无效")
            if not self.ignore_errors:
                raise Exception("加载图片失败: img_rgb为空或无效")
            # 尝试从原始输入重新加载
            ctx.img_rgb, ctx.img_alpha = load_image(ctx.input)
        
        if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
            logger.error(f"加载的图片尺寸无效: {ctx.img_rgb.shape}")
            if not self.ignore_errors:
                raise Exception(f"加载的图片尺寸无效: {ctx.img_rgb.shape}")

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            # 生成带置信度颜色映射和颜色条的热力图
            logger.info(f"Generating confidence heatmap for mask_raw (shape: {ctx.mask_raw.shape}, dtype: {ctx.mask_raw.dtype})")
            heatmap = self._create_confidence_heatmap(ctx.mask_raw, equalize=False)
            logger.info(f"Heatmap generated (shape: {heatmap.shape}), saving to mask_raw.png")
            imwrite_unicode(self._result_path('mask_raw.png'), heatmap, logger)
            
            # 如果有raw_mask_mask，生成对比图
            if hasattr(ctx, 'raw_mask_mask') and ctx.raw_mask_mask is not None:
                try:
                    logger.info("Generating mask vs db comparison heatmap...")
                    logger.info(f"[DEBUG] raw_mask_mask shape: {ctx.raw_mask_mask.shape}, ctx.mask_raw shape: {ctx.mask_raw.shape}")
                    
                    # 确保两个mask尺寸一致
                    if ctx.raw_mask_mask.shape != ctx.mask_raw.shape:
                        logger.info(f"[DEBUG] Resizing raw_mask_mask from {ctx.raw_mask_mask.shape} to {ctx.mask_raw.shape}")
                        raw_mask_mask_resized = cv2.resize(ctx.raw_mask_mask, 
                                                          (ctx.mask_raw.shape[1], ctx.mask_raw.shape[0]), 
                                                          interpolation=cv2.INTER_LINEAR)
                    else:
                        raw_mask_mask_resized = ctx.raw_mask_mask
                    
                    heatmap_mask = self._create_confidence_heatmap(raw_mask_mask_resized, equalize=False)
                    heatmap_db = heatmap  # 复用刚生成的db热力图
                    comparison = np.hstack([heatmap_mask, heatmap_db])
                    comparison_path = self._result_path('mask_comparison.png')
                    imwrite_unicode(comparison_path, comparison, logger)
                    logger.info(f'Saved mask vs db comparison to {comparison_path}')
                except Exception as e:
                    logger.error(f'Failed to generate mask vs db comparison: {e}')

        # --- BEGIN: Save raw detection boxes image in verbose mode ---
        if self.verbose and ctx.textlines:
            try:
                logger.info("Verbose mode: Saving raw detection boxes image...")
                raw_detection_image = np.copy(ctx.img_rgb)
                for textline in ctx.textlines:
                    # Draw each polygon with a unique color to distinguish them
                    # Using a simple hash of the textline object to get a color
                    color_val = hash(str(textline.pts)) % (256 * 256 * 256)
                    color = (color_val & 0xFF, (color_val >> 8) & 0xFF, (color_val >> 16) & 0xFF)
                    cv2.polylines(raw_detection_image, [textline.pts.astype(np.int32)], isClosed=True, color=color, thickness=2)
                
                # Convert to BGR for saving with OpenCV
                raw_detection_image_bgr = cv2.cvtColor(raw_detection_image, cv2.COLOR_RGB2BGR)
                imwrite_unicode(self._result_path('detection_raw_boxes.png'), raw_detection_image_bgr, logger)
                logger.info("Saved raw detection boxes to detection_raw_boxes.png")
            except Exception as e:
                logger.error(f"Failed to save raw detection boxes image: {e}")
        # --- END: Save raw detection boxes image ---

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # 设置为空列表，以便保存空的JSON
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            imwrite_unicode(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR), logger)

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] # Fallback to empty textlines if OCR fails

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # 设置为空列表，以便保存空的JSON
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = [] # Fallback to empty text_regions if textline merge fails

        # -- 过滤列表：根据 OCR 识别的原文过滤
        if ctx.text_regions and self.filter_text_enabled:
            filtered_regions = []
            for region in ctx.text_regions:
                match_result = match_filter(region.text)
                if match_result:
                    matched_word, match_type = match_result
                    logger.info(f'过滤文本区域 ({match_type}匹配): "{region.text}" -> 匹配: "{matched_word}"')
                else:
                    filtered_regions.append(region)
            if len(filtered_regions) < len(ctx.text_regions):
                logger.info(f'过滤列表: 过滤了 {len(ctx.text_regions) - len(filtered_regions)} 个文本区域')
            ctx.text_regions = filtered_regions

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 当不使用简单排序时显示panel
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            imwrite_unicode(self._result_path('bboxes.png'), bboxes, logger)

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text  
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")
            
        # -- Translation
        # 判断是否需要跳过翻译步骤
        should_skip_translation = False
        
        # 1. 保存文本+模板配置：只执行检测器和OCR步骤，不进行翻译，然后为当前文件快速退出
        if self.save_text and self.template:
            logger.info("Save text + Template mode: Running up to OCR, then saving and stopping for this file.")
            should_skip_translation = True
            # 设置原文作为翻译结果, 并设置目标语言以防万一
            for region in ctx.text_regions:
                region.translation = region.text
                # Set target_lang to avoid downstream errors if the pipeline were to continue (belt and braces)
                if not hasattr(region, 'target_lang') or not region.target_lang:
                    region.target_lang = config.translator.target_lang

            # 保存JSON文件
            if hasattr(ctx, 'image_name') and ctx.image_name:
                self._save_text_to_file(ctx.image_name, ctx, config)
                logger.info(f"JSON template saved for {os.path.basename(ctx.image_name)}.")
                
                # 直接导出TXT文件（原文）
                try:
                    json_path = find_json_path(ctx.image_name)
                    if json_path and os.path.exists(json_path):
                        from desktop_qt_ui.services.workflow_service import generate_original_text, get_template_path_from_config
                        template_path = get_template_path_from_config()
                        if template_path and os.path.exists(template_path):
                            # 导出原文
                            original_result = generate_original_text(json_path, template_path)
                            logger.info(f"Original text export result: {original_result}")
                        else:
                            logger.warning(f"Template file not found: {template_path}")
                    else:
                        logger.warning(f"JSON file not found for TXT export: {ctx.image_name}")
                except Exception as e:
                    logger.error(f"Failed to export TXT: {e}")
            else:
                logger.warning("Could not save translation file, image_name not in context.")

            # ✅ 标记成功（模板模式成功生成原文）
            ctx.success = True
            
            # 设置占位符结果并为当前文件提前返回，以便主循环可以处理下一个文件
            ctx.result = None
            return ctx
        # 2. 模板配置+加载文本：TXT文件内容就是翻译，不进行翻译处理
        elif self.template and self.load_text:
            logger.info("Template + Load text mode: TXT content is translation, skipping translation.")
            should_skip_translation = True
            # TXT文件的内容本身就是翻译结果，无需额外处理
        # 3. 单独模板配置：没有任何作用，正常进行翻译
        elif self.template and not self.save_text and not self.load_text:
            logger.info("Template only mode: No effect, proceeding with normal translation.")
            should_skip_translation = False
        
        if not should_skip_translation:
            await self._report_progress('translating')
            try:
                ctx.text_regions = await self._run_text_translation(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during translating:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.text_regions = [] # Fallback to empty text_regions if translation fails

        if hasattr(ctx, 'pipeline_should_stop') and ctx.pipeline_should_stop:
            ctx.result = ctx.input
            return ctx

        await self._report_progress('after-translating')

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        # (Delayed to take advantage of the region filtering done after ocr and translation)
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0] # Fallback to raw mask or empty mask

        if self.verbose and ctx.mask is not None:
            inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                          self.device, self.verbose)
            imwrite_unicode(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR), logger)
            imwrite_unicode(self._result_path('mask_final.png'), ctx.mask, logger)

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)

        except Exception as _e:
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                imwrite_unicode(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), logger)
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # 保存inpainted图片到新目录结构（用于可编辑图片功能）
        # 与JSON保存逻辑保持一致：save_text或text_output_file任一满足即保存
        if (self.save_text or self.text_output_file) and hasattr(ctx, 'image_name') and ctx.image_name and ctx.img_inpainted is not None:
            self._save_inpainted_image(ctx.image_name, ctx.img_inpainted)
        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as _e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted # Fallback to inpainted (or original RGB) image if rendering fails

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)

        return await self._revert_upscale(config, ctx)
    
    # If `revert_upscaling` is True, revert to input size
    # Else leave `ctx` as-is
    async def _revert_upscale(self, config: Config, ctx: Context):
        if config.upscale.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        # 在verbose模式下保存final.png到调试文件夹
        if ctx.result and self.verbose:
            try:
                final_img = np.array(ctx.result)
                if len(final_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    final_img = cv2.cvtColor(final_img, cv2.COLOR_RGB2BGR)
                final_path = self._result_path('final.png')
                imwrite_unicode(final_path, final_img, logger)
            except Exception as e:
                logger.error(f"Error saving final.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        return ctx

    async def _run_colorizer(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("colorizer", config.colorizer.colorizer)] = current_time
        #todo: im pretty sure the ctx is never used. does it need to be passed in?
        return await dispatch_colorization(
            config.colorizer.colorizer,
            colorization_size=config.colorizer.colorization_size,
            denoise_sigma=config.colorizer.denoise_sigma,
            device=self.device,
            image=ctx.input,
            **ctx
        )

    async def _run_upscaling(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("upscaling", config.upscale.upscaler)] = current_time
        
        # Prepare kwargs for Real-CUGAN (NCNN version) and MangaJaNai
        upscaler_kwargs = {}
        if config.upscale.upscaler == 'realcugan':
            realcugan_model = getattr(config.upscale, 'realcugan_model', None)
            if realcugan_model:
                upscaler_kwargs['model_name'] = realcugan_model
            # tile_size: None=use upscaler default, 0=no tiling, >0=manual tile size
            tile_size = getattr(config.upscale, 'tile_size', None)
            if tile_size is not None:
                upscaler_kwargs['tile_size'] = tile_size
        elif config.upscale.upscaler == 'mangajanai':
            # mangajanai 的 upscale_ratio 可以是字符串 (x2, x4, DAT2 x4) 或数字
            ratio = config.upscale.upscale_ratio
            if isinstance(ratio, str):
                upscaler_kwargs['model_name'] = ratio
                # 从字符串解析实际倍率
                if 'x2' in ratio.lower():
                    actual_ratio = 2
                else:
                    actual_ratio = 4
            elif ratio == 2:
                upscaler_kwargs['model_name'] = 'x2'
                actual_ratio = 2
            else:
                upscaler_kwargs['model_name'] = 'x4'
                actual_ratio = 4

            tile_size = getattr(config.upscale, 'tile_size', None)
            if tile_size is not None:
                upscaler_kwargs['tile_size'] = tile_size
        
        # 获取实际的数字倍率
        if config.upscale.upscaler == 'mangajanai':
            upscale_ratio_num = actual_ratio
        else:
            upscale_ratio_num = config.upscale.upscale_ratio
        
        result = (await dispatch_upscaling(
            config.upscale.upscaler, 
            [ctx.img_colorized], 
            upscale_ratio_num, 
            self.device,
            **upscaler_kwargs
        ))[0]
        
        # 如果 models_ttl > 0，则由清理任务自动卸载；否则立即卸载以释放显存
        if self.models_ttl > 0:
            logger.info(f"Upscaling model {config.upscale.upscaler} will be unloaded after {self.models_ttl}s of inactivity")
        else:
            # models_ttl == 0 表示永久保留，但 upscaling 模型占用显存较大，仍然立即卸载
            logger.info(f"Unloading upscaling model {config.upscale.upscaler} immediately to free VRAM")
            await self._unload_model('upscaling', config.upscale.upscaler, **upscaler_kwargs)
            del self._model_usage_timestamps[("upscaling", config.upscale.upscaler)]
        
        return result

    async def _run_detection(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        current_time = time.time()
        self._model_usage_timestamps[("detection", config.detector.detector)] = current_time
        result = await dispatch_detection(config.detector.detector, ctx.img_rgb, config.detector.detection_size, config.detector.text_threshold,
                                        config.detector.box_threshold,
                                        config.detector.unclip_ratio, config.detector.det_invert, config.detector.det_gamma_correct, config.detector.det_rotate,
                                        config.detector.det_auto_rotate,
                                        self.device, self.verbose,
                                        config.detector.use_yolo_obb, config.detector.yolo_obb_conf, config.detector.yolo_obb_iou, config.detector.yolo_obb_overlap_threshold,
                                        config.detector.min_box_area_ratio, self._result_path)
        
        # 处理bbox调试图（如果检测器返回了）
        if self.verbose and result and len(result) == 3 and result[2] is not None:
            third_elem = result[2]
            # 检查是否是tuple（包含三张图）
            if isinstance(third_elem, tuple) and len(third_elem) == 3:
                try:
                    logger.info(f'[DEBUG] Processing 3-element tuple: {[type(x) for x in third_elem]}')
                    bbox_img, binary_mask_img, raw_mask_mask = third_elem
                    # 保存边框调试图
                    bbox_debug_path = self._result_path('bboxes_with_scores.png')
                    imwrite_unicode(bbox_debug_path, bbox_img, logger)
                    logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                    # 保存二值化mask
                    binary_mask_path = self._result_path('mask_binary.png')
                    imwrite_unicode(binary_mask_path, binary_mask_img, logger)
                    logger.info(f'Saved binary mask to {binary_mask_path}')
                    # 暂存raw_mask_mask到ctx以便后续生成对比图
                    ctx.raw_mask_mask = raw_mask_mask
                    logger.info(f'[DEBUG] Stored raw_mask_mask for later comparison (shape: {raw_mask_mask.shape})')
                    result = (result[0], result[1], None)
                except Exception as e:
                    logger.error(f'Failed to save bbox debug images: {e}')
            # 兼容2张图的情况
            elif isinstance(third_elem, tuple) and len(third_elem) == 2:
                try:
                    bbox_img, binary_mask_img = third_elem
                    bbox_debug_path = self._result_path('bboxes_with_scores.png')
                    imwrite_unicode(bbox_debug_path, bbox_img, logger)
                    logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                    binary_mask_path = self._result_path('mask_binary.png')
                    imwrite_unicode(binary_mask_path, binary_mask_img, logger)
                    logger.info(f'Saved binary mask to {binary_mask_path}')
                    result = (result[0], result[1], None)
                except Exception as e:
                    logger.error(f'Failed to save bbox debug images: {e}')
            # 兼容单张图的情况（包括混合检测调试图）
            elif isinstance(third_elem, np.ndarray) and len(third_elem.shape) == 3:
                try:
                    # 如果启用了YOLO辅助检测，优先保存为混合检测调试图
                    if config.detector.use_yolo_obb:
                        # 保存混合检测调试图
                        hybrid_debug_path = self._result_path('hybrid_detection_boxes.png')
                        imwrite_unicode(hybrid_debug_path, cv2.cvtColor(third_elem, cv2.COLOR_RGB2BGR), logger)
                        logger.info(f'✅ 已保存混合检测调试图: {hybrid_debug_path}')
                    else:
                        # 保存普通bbox调试图
                        bbox_debug_path = self._result_path('bboxes_with_scores.png')
                        imwrite_unicode(bbox_debug_path, third_elem, logger)
                        logger.info(f'Saved bbox debug image to {bbox_debug_path}')
                    result = (result[0], result[1], None)
                except Exception as e:
                    logger.error(f'Failed to save bbox debug image: {e}')
        
        # --- BEGIN NON-MAXIMUM SUPPRESSION (NMS) FOR DE-DUPLICATION ---
        if result and result[0]:
            try:
                from shapely.geometry import Polygon

                def calculate_iou(box_1, box_2):
                    poly_1 = Polygon(box_1.pts)
                    poly_2 = Polygon(box_2.pts)
                    if not poly_1.is_valid or not poly_2.is_valid:
                        return 0.0
                    intersection_area = poly_1.intersection(poly_2).area
                    union_area = poly_1.union(poly_2).area
                    if union_area == 0:
                        return 0.0
                    return intersection_area / union_area

                textlines = result[0][:] # Work on a copy
                textlines.sort(key=lambda x: x.prob, reverse=True)
                
                kept_textlines = []
                while textlines:
                    current_box = textlines.pop(0)
                    kept_textlines.append(current_box)
                    remaining_textlines = []
                    for box in textlines:
                        iou = calculate_iou(current_box, box)
                        if iou < 0.9: # IoU threshold, 0.9 means very high overlap
                            remaining_textlines.append(box)
                    textlines = remaining_textlines

                if len(result[0]) != len(kept_textlines):
                    logger.info(f"Removed {len(result[0]) - len(kept_textlines)} duplicate lines via NMS.")
                    result = (kept_textlines, result[1], result[2])

            except Exception as e:
                logger.error(f"An error occurred during Non-Maximum Suppression: {e}")
                pass
        # --- END NON-MAXIMUM SUPPRESSION (NMS) ---

        return result
    async def _unload_model(self, tool: str, model: str, **kwargs):
        logger.info(f"Unloading {tool} model: {model}")
        match tool:
            case 'colorizer':
                await unload_colorization(model)
            case 'detection':
                await unload_detection(model)
            case 'inpainting':
                await unload_inpainting(model)
            case 'ocr':
                await unload_ocr(model)
            case 'upscaling':
                await unload_upscaling(model, **kwargs)
            case 'translation':
                await unload_translation(model)
            case 'textline_merge':
                # textline_merge 不需要卸载（无模型）
                logger.debug("textline_merge does not require unloading")
            case 'rendering':
                # rendering 不需要卸载（无模型）
                logger.debug("rendering does not require unloading")
            case _:
                logger.warning(f"Unknown tool type for unloading: {tool}")
        
        # 清理 Python 内存（对 CPU 和 GPU 都有效）
        import gc
        gc.collect()
        
        # 清理 GPU 显存（如果有）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info(f"模型 {tool}/{model} 已卸载，内存已清理")

    def _cleanup_gpu_memory(self):
        """清理GPU显存的辅助方法"""
        import gc
        gc.collect()
        if hasattr(self, 'device') and (self.device == 'cuda' or self.device == 'mps'):
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception:
                pass
    
    def _cleanup_context_memory(self, ctx, keep_result=True):
        """
        清理单个上下文的中间数据（用于特殊模式）
        
        Args:
            ctx: Context对象
            keep_result: bool - 是否保留 ctx.result
        """
        # 清理输入图像
        if hasattr(ctx, 'input') and ctx.input is not None:
            if hasattr(ctx.input, 'close'):
                try:
                    ctx.input.close()
                except:
                    pass
            del ctx.input
            ctx.input = None
        
        # 清理中间处理图像
        if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
            del ctx.img_rgb
            ctx.img_rgb = None
        
        if hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
            del ctx.img_colorized
            ctx.img_colorized = None
        
        if hasattr(ctx, 'upscaled') and ctx.upscaled is not None:
            del ctx.upscaled
            ctx.upscaled = None
        
        if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
            del ctx.img_inpainted
            ctx.img_inpainted = None
        
        if hasattr(ctx, 'img_rendered') and ctx.img_rendered is not None:
            del ctx.img_rendered
            ctx.img_rendered = None
        
        if hasattr(ctx, 'img_alpha') and ctx.img_alpha is not None:
            del ctx.img_alpha
            ctx.img_alpha = None
        
        if hasattr(ctx, 'mask') and ctx.mask is not None:
            del ctx.mask
            ctx.mask = None
        
        if hasattr(ctx, 'mask_raw') and ctx.mask_raw is not None:
            del ctx.mask_raw
            ctx.mask_raw = None
        
        # 如果不保留结果，也清理 result
        if not keep_result and hasattr(ctx, 'result') and ctx.result is not None:
            del ctx.result
            ctx.result = None
        
        # 强制垃圾回收和GPU显存清理
        self._cleanup_gpu_memory()
        logger.debug('[MEMORY] Context cleanup completed')

    
    def _cleanup_batch_memory(self, current_batch_images=None, preprocessed_contexts=None, translated_contexts=None, keep_results=True):
        """
        统一的批次内存清理方法
        
        Args:
            current_batch_images: List[(image, config)] - 当前批次的原始图片
            preprocessed_contexts: List[(ctx, config)] - 预处理后的上下文
            translated_contexts: List[(ctx, config)] - 翻译后的上下文
            keep_results: bool - 是否保留 ctx.result（用于返回结果）
        """
#         import gc
        
        # 1. 清理原始图片
        if current_batch_images:
            for i, (image, _) in enumerate(current_batch_images):
                if hasattr(image, 'close'):
                    try:
                        image.close()
                    except:
                        pass
                # 显式删除引用
                del current_batch_images[i]
        
        # 2. 清理预处理上下文中的输入图像
        if preprocessed_contexts:
            for ctx, _ in preprocessed_contexts:
                if hasattr(ctx, 'input') and ctx.input is not None:
                    # 先关闭再删除
                    if hasattr(ctx.input, 'close'):
                        try:
                            ctx.input.close()
                        except:
                            pass
                    del ctx.input
                    ctx.input = None
            preprocessed_contexts.clear()
        
        # 3. 清理翻译上下文中的中间图像
        if translated_contexts:
            for ctx, _ in translated_contexts:
                # 清理中间处理图像（使用 del 显式删除）
                if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                    del ctx.img_rgb
                    ctx.img_rgb = None
                if hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
                    del ctx.img_colorized
                    ctx.img_colorized = None
                if hasattr(ctx, 'upscaled') and ctx.upscaled is not None:
                    del ctx.upscaled
                    ctx.upscaled = None
                if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
                    del ctx.img_inpainted
                    ctx.img_inpainted = None
                if hasattr(ctx, 'img_rendered') and ctx.img_rendered is not None:
                    del ctx.img_rendered
                    ctx.img_rendered = None
                if hasattr(ctx, 'img_alpha') and ctx.img_alpha is not None:
                    del ctx.img_alpha
                    ctx.img_alpha = None
                if hasattr(ctx, 'mask') and ctx.mask is not None:
                    del ctx.mask
                    ctx.mask = None
                if hasattr(ctx, 'mask_raw') and ctx.mask_raw is not None:
                    del ctx.mask_raw
                    ctx.mask_raw = None
                
                # 如果不保留结果，也清理 result
                if not keep_results and hasattr(ctx, 'result') and ctx.result is not None:
                    del ctx.result
                    ctx.result = None
            
            translated_contexts.clear()
        
        # 4. 强制垃圾回收和GPU显存清理
        self._cleanup_gpu_memory()
        
        # 5. Windows 特定：强制释放物理内存
        try:
            import ctypes
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
            logger.debug('[MEMORY] Windows working set trimmed')
        except:
            pass  # 非 Windows 系统时忽略
        
        logger.debug('[MEMORY] Batch cleanup completed')

    # Background models cleanup job.
    async def _detector_cleanup_job(self):
        logger.info(f"Model cleanup job started with models_ttl={self.models_ttl} seconds")
        while True:
            if self.models_ttl == 0:
                await asyncio.sleep(1)
                continue
            now = time.time()
            for (tool, model), last_used in list(self._model_usage_timestamps.items()):
                time_since_last_use = now - last_used
                if time_since_last_use > self.models_ttl:
                    logger.info(f"Model {tool}/{model} has been idle for {time_since_last_use:.1f}s (TTL: {self.models_ttl}s), unloading...")
                    await self._unload_model(tool, model)
                    del self._model_usage_timestamps[(tool, model)]
            await asyncio.sleep(1)

    async def _run_ocr(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        current_time = time.time()
        self._model_usage_timestamps[("ocr", config.ocr.ocr)] = current_time
        
        # 为OCR创建子文件夹（只在verbose模式下）
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, 'ocrs')
                else:
                    ocr_result_dir = os.path.join(BASE_PATH, 'result', image_subfolder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
            else:
                ocr_result_dir = os.path.join(BASE_PATH, 'result', self.result_sub_folder, 'ocrs')
                os.makedirs(ocr_result_dir, exist_ok=True)
        else:
            # 非verbose模式下使用临时目录或不创建OCR结果目录
            ocr_result_dir = None
        
        # 临时设置环境变量供OCR模块使用
        old_ocr_dir = os.environ.get('MANGA_OCR_RESULT_DIR', None)
        if ocr_result_dir:
            os.environ['MANGA_OCR_RESULT_DIR'] = ocr_result_dir
        
        try:
            # --- Primary OCR run ---
            primary_ocr_engine = config.ocr.ocr
            ocr_name = primary_ocr_engine.value if hasattr(primary_ocr_engine, 'value') else primary_ocr_engine
            logger.info(f"Running primary OCR with: {ocr_name}")
            textlines = await dispatch_ocr(primary_ocr_engine, ctx.img_rgb, ctx.textlines, config.ocr, self.device, self.verbose)

            # --- BEGIN: HYBRID OCR LOGIC ---
            if config.ocr.use_hybrid_ocr:
                # Identify textlines that failed recognition or have low confidence
                # 判断失败条件：文本为空 或 置信度低于阈值
                prob_threshold = config.ocr.prob if config.ocr.prob is not None else 0.1
                failed_indices = [
                    i for i, tl in enumerate(textlines) 
                    if not tl.text.strip() or tl.prob < prob_threshold
                ]
                
                if failed_indices:
                    # Use textlines[i] instead of ctx.textlines[i] because OCR may have changed the order
                    failed_textlines = [textlines[i] for i in failed_indices]
                    logger.info(f"{len(failed_textlines)} textlines failed or have low confidence (< {prob_threshold}) with primary OCR. Trying secondary OCR...")
                    
                    secondary_ocr_engine = config.ocr.secondary_ocr
                    # We can reuse the same config object, just switching the engine
                    secondary_config = config.ocr
                    
                    secondary_ocr_name = secondary_ocr_engine.value if hasattr(secondary_ocr_engine, 'value') else secondary_ocr_engine
                    logger.info(f"Running secondary OCR with: {secondary_ocr_name}")
                    secondary_results = await dispatch_ocr(secondary_ocr_engine, ctx.img_rgb, failed_textlines, secondary_config, self.device, self.verbose)
                    
                    # Merge the results back into the original list
                    for i, result_tl in zip(failed_indices, secondary_results):
                        textlines[i] = result_tl # Replace the failed textline with the new result
                    
                    logger.info("Secondary OCR processing finished.")
                    
                    # ✅ 混合OCR完成后清理,防止两次OCR调用累积
#                     import gc
                    if 'secondary_results' in locals():
                        del secondary_results
                    if 'failed_textlines' in locals():
                        del failed_textlines
                    if 'failed_indices' in locals():
                        del failed_indices
                    self._cleanup_gpu_memory()
            # --- END: HYBRID OCR LOGIC ---

        finally:
            # 恢复环境变量
            if old_ocr_dir is not None:
                os.environ['MANGA_OCR_RESULT_DIR'] = old_ocr_dir
            elif 'MANGA_OCR_RESULT_DIR' in os.environ:
                del os.environ['MANGA_OCR_RESULT_DIR']

        new_textlines = []
        for textline in textlines:
            if textline.text.strip():
                if config.render.font_color_fg:
                    textline.fg_r, textline.fg_g, textline.fg_b = config.render.font_color_fg
                if config.render.font_color_bg:
                    textline.bg_r, textline.bg_g, textline.bg_b = config.render.font_color_bg
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, config: Config, ctx: Context):
        current_time = time.time()
        self._model_usage_timestamps[("textline_merge", "textline_merge")] = current_time
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],
                                                     config, verbose=self.verbose)
        for region in text_regions:
            if not hasattr(region, "text_raw"):
                region.text_raw = region.text      # <- Save the initial OCR results to expand the render detection box. Also, prevent affecting the forbidden translation function.       
        # Filter out languages to skip  
        if config.translator.skip_lang is not None:  
            skip_langs = [lang.strip().upper() for lang in config.translator.skip_lang.split(',')]
            filtered_textlines = []  
            for txtln in ctx.textlines:  
                try:  
                    detected_lang, confidence = langid.classify(txtln.text)
                    source_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
                    if source_language != 'UNKNOWN':
                        source_language = source_language.upper()
                except Exception:  
                    source_language = 'UNKNOWN'  
    
                # Print detected source_language and whether it's in skip_langs  
                # logger.info(f'Detected source language: {source_language}, in skip_langs: {source_language in skip_langs}, text: "{txtln.text}"')  
    
                if source_language in skip_langs:  
                    logger.info(f'Filtered out: {txtln.text}')  
                    logger.info(f'Reason: Detected language {source_language} is in skip_langs')  
                    continue  # Skip this region  
                filtered_textlines.append(txtln)  
            ctx.textlines = filtered_textlines  
    
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0],  
                                                     config, verbose=self.verbose)  

        # 应用合并后的面积过滤（基于合并后的大框）
        # 只过滤单个检测框的小区域，保留包含多个检测框的合并区域
        if config.detector.min_box_area_ratio > 0:
            img_h, img_w = ctx.img_rgb.shape[:2]
            img_total_pixels = img_h * img_w
            
            # 模拟检测器的切割逻辑，判断是否需要切割
            h, w = img_h, img_w
            transpose = False
            if h < w:
                _transpose = True
                h, w = w, h
            
            asp_ratio = h / w
            tgt_size = config.detector.detection_size
            down_scale_ratio = h / tgt_size
            require_rearrange = down_scale_ratio > 2.5 and asp_ratio > 3
            
            # 如果需要切割，计算切割块的大小
            if require_rearrange:
                pw_num = max(int(np.floor(2 * tgt_size / w)), 2)
                patch_size = _ph = pw_num * w
                
                # 限制切割后块的最大长宽比（不超过 3:1）
                # 注意：ph = pw_num * w，所以 ph/w = pw_num
                max_patch_aspect_ratio = 3.0
                _original_pw_num = pw_num
                if pw_num > max_patch_aspect_ratio:
                    # pw_num 太大，说明切割块太高，需要减小
                    # 但是不能直接减小 pw_num，因为这是检测器的逻辑
                    # 我们只是用来计算面积过滤的参考值
                    # 所以这里使用限制后的 pw_num 来计算 tile_pixels
                    adjusted_pw_num = max_patch_aspect_ratio
                    adjusted_ph = adjusted_pw_num * w
                    tile_pixels = adjusted_ph * w
                    logger.info(f'检测到极端长宽比图片 (长宽比={asp_ratio:.2f}), 限制面积过滤参考块长宽比: 原始切割块={patch_size}x{w} (长宽比={pw_num:.2f}), 过滤参考块={adjusted_ph:.0f}x{w} (长宽比={adjusted_pw_num:.2f}), 面积={tile_pixels:.0f}像素')
                else:
                    tile_pixels = patch_size * w
                    logger.info(f'检测到极端长宽比图片 (长宽比={asp_ratio:.2f}), 使用切割块面积 ({patch_size}x{w}={tile_pixels}像素) 进行过滤')
            else:
                tile_pixels = img_total_pixels  # 不切割，使用整图
            
            before_filter_count = len(text_regions)
            filtered_out_regions = []
            filtered_in_regions = []
            
            for region in text_regions:
                # 计算合并的检测框数量
                num_textlines = len(region.lines)
                
                # 如果包含多个检测框，说明是真正的文本区域，保留
                if num_textlines > 1:
                    filtered_in_regions.append(region)
                    continue
                
                # 只对单个检测框的区域进行面积过滤
                region_area = region.real_area
                # 使用切割块面积（如果有切割）或整图面积
                area_ratio = region_area / tile_pixels
                
                if region_area <= 16 or area_ratio <= config.detector.min_box_area_ratio:
                    filtered_out_regions.append((region, area_ratio, num_textlines, require_rearrange))
                else:
                    filtered_in_regions.append(region)
            
            text_regions = filtered_in_regions
            after_filter_count = len(text_regions)
            
            if filtered_out_regions:
                reference_desc = f'切割块({patch_size}x{w})' if require_rearrange else f'整图({img_w}x{img_h})'
                filter_ratio = len(filtered_out_regions) / before_filter_count * 100 if before_filter_count > 0 else 0
                # Info级别：只显示摘要
                logger.info(f'合并后面积过滤: 参考={reference_desc}, 最小面积比例={config.detector.min_box_area_ratio:.4f} ({config.detector.min_box_area_ratio*100:.2f}%), '
                           f'过滤前={before_filter_count}, 过滤后={after_filter_count}, 移除={len(filtered_out_regions)} ({filter_ratio:.1f}%, 仅单框区域)')
                # Verbose模式：显示详细信息
                if self.verbose:
                    for idx, (region, ratio, num_lines, was_rearranged) in enumerate(filtered_out_regions):
                        # 获取框的宽高
                        x1, y1, x2, y2 = region.xyxy
                        width = x2 - x1
                        height = y2 - y1
                        logger.debug(f'  移除单框区域[{idx+1}]: 大小={width:.0f}x{height:.0f}, 面积={region.real_area:.1f}像素, 占比={ratio*100:.3f}%, 文本="{region.text[:20]}"')

        new_text_regions = []
        for region in text_regions:
            # 跳过text为None的区域
            if region.text is None:
                logger.warning('跳过text为None的区域')
                continue
                
            # Remove leading spaces after pre-translation dictionary replacement                
            original_text = region.text  
            stripped_text = original_text.strip()  
            
            # Record removed leading characters  
            removed_start_chars = original_text[:len(original_text) - len(stripped_text)]  
            if removed_start_chars:  
                logger.info(f'Removed leading characters: "{removed_start_chars}" from "{original_text}"')  
            
            # Modified filtering condition: handle incomplete parentheses  
            bracket_pairs = {  
                '(': ')', '（': '）', '[': ']', '【': '】', '{': '}', '〔': '〕', '〈': '〉', '「': '」',  
                '"': '"', '＂': '＂', "'": "'", "“": "”", '《': '》', '『': '』', '"': '"', '〝': '〞', '﹁': '﹂', '﹃': '﹄',  
                '⸂': '⸃', '⸄': '⸅', '⸉': '⸊', '⸌': '⸍', '⸜': '⸝', '⸠': '⸡', '‹': '›', '«': '»', '＜': '＞', '<': '>'  
            }   
            left_symbols = set(bracket_pairs.keys())  
            right_symbols = set(bracket_pairs.values())  
            
            has_brackets = any(s in stripped_text for s in left_symbols) or any(s in stripped_text for s in right_symbols)  
            
            if has_brackets:  
                result_chars = []  
                stack = []  
                to_skip = []    
                
                # 第一次遍历：标记匹配的括号  
                # First traversal: mark matching brackets
                for i, char in enumerate(stripped_text):  
                    if char in left_symbols:  
                        stack.append((i, char))  
                    elif char in right_symbols:  
                        if stack:  
                            # 有对应的左括号，出栈  
                            # There is a corresponding left bracket, pop the stack
                            stack.pop()  
                        else:  
                            # 没有对应的左括号，标记为删除  
                            # No corresponding left parenthesis, marked for deletion
                            to_skip.append(i)  
                
                # 标记未匹配的左括号为删除
                # Mark unmatched left brackets as delete  
                for pos, _ in stack:  
                    to_skip.append(pos)  
                
                has_removed_symbols = len(to_skip) > 0  
                
                # 第二次遍历：处理匹配但不对应的括号
                # Second pass: Process matching but mismatched brackets
                stack = []  
                for i, char in enumerate(stripped_text):  
                    if i in to_skip:  
                        # 跳过孤立的括号
                        # Skip isolated parentheses
                        continue  
                        
                    if char in left_symbols:  
                        stack.append(char)  
                        result_chars.append(char)  
                    elif char in right_symbols:  
                        if stack:  
                            left_bracket = stack.pop()  
                            expected_right = bracket_pairs.get(left_bracket)  
                            
                            if char != expected_right:  
                                # 替换不匹配的右括号为对应左括号的正确右括号
                                # Replace mismatched right brackets with the correct right brackets corresponding to the left brackets
                                result_chars.append(expected_right)  
                                logger.info(f'Fixed mismatched bracket: replaced "{char}" with "{expected_right}"')  
                            else:  
                                result_chars.append(char)  
                    else:  
                        result_chars.append(char)  
                
                new_stripped_text = ''.join(result_chars)  
                
                if has_removed_symbols:  
                    logger.info(f'Removed unpaired bracket from "{stripped_text}"')  
                
                if new_stripped_text != stripped_text and not has_removed_symbols:  
                    logger.info(f'Fixed brackets: "{stripped_text}" → "{new_stripped_text}"')  
                
                stripped_text = new_stripped_text  
              
            region.text = stripped_text.strip()     
            
            # 过滤空文本、过短文本、无价值文本
            if not region.text \
                    or len(region.text) < config.ocr.min_text_length \
                    or not is_valuable_text(region.text) \
                    or (not config.translator.no_text_lang_skip and langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0):
                if region.text and region.text.strip():
                    logger.info(f'Filtered out: {region.text}')
                    if len(region.text) < config.ocr.min_text_length:
                        logger.info('Reason: Text length is less than the minimum required length.')
                    elif not is_valuable_text(region.text):
                        logger.info('Reason: Text is not considered valuable.')
                    elif langcodes.tag_distance(region.source_lang, config.translator.target_lang) == 0:
                        logger.info('Reason: Text language matches the target language and no_text_lang_skip is False.')
                elif not region.text:
                    logger.info('Filtered out: Empty text region')
            else:
                if config.render.font_color_fg or config.render.font_color_bg:
                    if config.render.font_color_bg:
                        region.adjust_bg_color = False
                new_text_regions.append(region)
        text_regions = new_text_regions

        text_regions = sort_regions(
            text_regions,
            right_to_left=config.render.rtl,
            img=ctx.img_rgb,
            force_simple_sort=config.force_simple_sort
        )   
        
        
        
        return text_regions

    def _prune_context_history(self):
        """
        Prune translation history to prevent memory leaks in large batch tasks.
        Keeps only the most recent pages needed for context.
        """
        # Minimum history to keep (context_size + buffer)
        # If context_size is 0, keep a small buffer (e.g., 5) just in case
        keep_size = max(self.context_size, 1) + 5
        
        if len(self.all_page_translations) > keep_size:
            # Remove oldest entries
            trim_count = len(self.all_page_translations) - keep_size
            self.all_page_translations = self.all_page_translations[trim_count:]
            if len(self._original_page_texts) >= trim_count:
                self._original_page_texts = self._original_page_texts[trim_count:]
            # Also clean up saved image contexts if they are too old (simple heuristic)
            if len(self._saved_image_contexts) > keep_size * 2:
                # Keep only the last N keys
                keys = list(self._saved_image_contexts.keys())
                keys_to_remove = keys[:-keep_size*2]
                for k in keys_to_remove:
                    del self._saved_image_contexts[k]

    def _get_current_task_id(self) -> int:
        """Return the current asyncio task id, or 0 if not available."""
        try:
            task = asyncio.current_task()
        except Exception:
            task = None
        return id(task) if task else 0

    def _get_task_translator(self, translator_key: Translator, factory):
        """Get or create a translator instance scoped to the current asyncio Task."""
        task_id = self._get_current_task_id()
        cache_key = (task_id, translator_key)
        translator = self._task_translator_cache.get(cache_key)
        if translator is None:
            translator = factory()
            self._task_translator_cache[cache_key] = translator
        return translator

    def _release_task_translators(self, task_id: int):
        """Release cached translators for a finished asyncio Task to avoid memory leaks."""
        if not task_id:
            return
        keys_to_delete = [k for k in self._task_translator_cache.keys() if k[0] == task_id]
        for k in keys_to_delete:
            self._task_translator_cache.pop(k, None)

    def _build_prev_context(self, use_original_text=False, current_page_index=None, batch_index=None, batch_original_texts=None):
        """
        跳过句子数为0的页面，取最近 context_size 个非空页面，拼成：
        <|1|>句子
        <|2|>句子
        ...
        的格式；如果没有任何非空页面，返回空串。

        Args:
            use_original_text: 是否使用原文而不是译文作为上下文（当前未使用）
            current_page_index: 当前页面索引，用于确定上下文范围
            batch_index: 当前页面在批次中的索引（当前未使用）
            batch_original_texts: 当前批次的原文数据（当前未使用）
        """
        if self.context_size <= 0:
            return ""

        trans_pages = self.all_page_translations or []
        orig_pages = self._original_page_texts or []

        if current_page_index is not None:
            trans_pages = trans_pages[:current_page_index]
            orig_pages = orig_pages[:current_page_index]

        max_len = max(len(trans_pages), len(orig_pages))
        if max_len == 0:
            return ""

        if len(trans_pages) < max_len:
            trans_pages = trans_pages + ([{}] * (max_len - len(trans_pages)))
        if len(orig_pages) < max_len:
            orig_pages = orig_pages + ([{}] * (max_len - len(orig_pages)))

        # Prefer translations, and fall back to originals when translations are missing.
        selected_pages = []
        for i in range(max_len):
            t_page = trans_pages[i] or {}
            o_page = orig_pages[i] or {}

            has_translation = any(str(sent).strip() for sent in t_page.values() if sent is not None)
            has_original = any(str(sent).strip() for sent in o_page.values() if sent is not None)

            if has_translation:
                selected_pages.append(t_page)
            elif has_original:
                selected_pages.append(o_page)

        if not selected_pages:
            return ""

        pages_used = min(self.context_size, len(selected_pages))
        if pages_used <= 0:
            return ""

        tail = selected_pages[-pages_used:]

        lines = []
        for page in tail:
            for sent in page.values():
                if sent is None:
                    continue
                s = str(sent).strip()
                if s:
                    lines.append(s)

        if not lines:
            return ""

        numbered = [f"<|{i+1}|>{s}" for i, s in enumerate(lines)]
        return "Here are the previous translation results for reference:\n" + "\n".join(numbered)

    async def _dispatch_with_context(self, config: Config, texts: list[str], ctx: Context):
        # Attach config to context for translators that need it
        ctx.config = config

        # 计算实际要使用的上下文页数和跳过的空页数
        # Calculate the actual number of context pages to use and empty pages to skip
        done_trans_pages = self.all_page_translations
        done_orig_pages = self._original_page_texts
        max_done = max(len(done_trans_pages), len(done_orig_pages))

        if self.context_size > 0 and max_done > 0:
            pages_expected = min(self.context_size, max_done)

            non_empty_count = 0
            for i in range(max_done):
                trans_page = done_trans_pages[i] if i < len(done_trans_pages) else {}
                orig_page = done_orig_pages[i] if i < len(done_orig_pages) else {}

                has_translation = any(str(sent).strip() for sent in trans_page.values() if sent is not None)
                has_original = any(str(sent).strip() for sent in orig_page.values() if sent is not None)

                if has_translation or has_original:
                    non_empty_count += 1

            pages_used = min(self.context_size, non_empty_count)
            skipped = pages_expected - pages_used
        else:
            pages_used = skipped = 0

        if self.context_size > 0:
            logger.info(f"已启用上下文翻译：使用最近 {self.context_size} 页历史作为参考")

        # 构建上下文字符串
        # Build the context string
        prev_ctx = self._build_prev_context()

        # 如果是 OpenAI 翻译器，则专门处理上下文注入
        # Special handling for OpenAI translator: inject context
        if config.translator.translator == Translator.openai:
            from .translators.openai import OpenAITranslator
            translator = self._get_task_translator(Translator.openai, OpenAITranslator)

            translator.parse_args(config.translator)
            translator.set_prev_context(prev_ctx)

            if pages_used > 0:
                context_count = prev_ctx.count("<|")
                logger.info(f"[历史上下文] 携带 {pages_used} 页上下文, {context_count} 条句子作为翻译参考")
            if skipped > 0:
                logger.warning(f"[历史上下文] 跳过了 {skipped} 页无句子的空页")
                


            # OpenAI 需要传递 ctx 参数（用于AI断句）
            return await translator._translate(ctx.from_lang, config.translator.target_lang, texts, ctx)
        else:
            return await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config,
                False,  # use_mtpe removed
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
        
    async def _load_and_prepare_prompts(self, config: Config, ctx: Context):
        """Loads custom HQ and line break prompts into the context object."""
        # Load custom high-quality prompt from JSON file if specified
        ctx.custom_prompt_json = None
        if config.translator.high_quality_prompt_path:
            try:
                prompt_path = config.translator.high_quality_prompt_path
                if not os.path.isabs(prompt_path):
                    prompt_path = os.path.join(BASE_PATH, prompt_path)
                
                if os.path.exists(prompt_path):
                    with open(prompt_path, 'r', encoding='utf-8') as f:
                        ctx.custom_prompt_json = json.load(f)

                    # 无作品区术语仅存放不应用：
                    # - 若 glossary 未使用 works 结构：运行时从 prompt 中剥离 unassigned，避免被模型看到
                    # - 若 glossary 使用 works 结构：保留 unassigned，后续按作品/生肉名筛选后会自动瘦身
                    try:
                        glossary_obj = ctx.custom_prompt_json.get("glossary") if isinstance(ctx.custom_prompt_json, dict) else None
                        if isinstance(glossary_obj, dict) and "unassigned" in glossary_obj:
                            works_obj = glossary_obj.get("works")
                            # 仅当未采用 works 分作品结构时才剥离 unassigned
                            if not isinstance(works_obj, dict):
                                new_glossary_obj = dict(glossary_obj)
                                new_glossary_obj.pop("unassigned", None)
                                new_prompt_obj = dict(ctx.custom_prompt_json)
                                new_prompt_obj["glossary"] = new_glossary_obj
                                ctx.custom_prompt_json = new_prompt_obj
                    except Exception:
                        pass

                    # 运行时按“作品”筛选/合并 glossary（如果 prompt.json 使用 works 结构）
                    # glossary_write_work_name：用于“术语提取写回”的目标作品名；若无法明确识别则为 None（写入无作品区）
                    ctx.glossary_write_work_name = None
                    try:
                        from .utils.work_resolver import (
                            resolve_work_name,
                            resolve_raw_work_name,
                            filter_custom_prompt_json_by_work,
                            merge_custom_prompt_json_by_works,
                        )

                        # 1) 混作品批次：合并多个 works 的术语为一份扁平 glossary
                        if getattr(ctx, 'merge_work_glossary', False):
                            work_names = getattr(ctx, 'merge_work_names', None) or []
                            merged_prompt, selected_works = merge_custom_prompt_json_by_works(
                                ctx.custom_prompt_json,
                                work_names,
                                include_default=True,
                            )
                            ctx.custom_prompt_json = merged_prompt

                            # 记录（仅用于日志/调试；混作品默认禁用写回）
                            if selected_works:
                                ctx.work_name = " + ".join(selected_works)
                            ctx.glossary_write_work_name = None

                        # 2) 单作品：按作品筛选 glossary
                        else:
                            # 某些场景（例如：多作品混批次）会显式禁用按作品过滤
                            if not getattr(ctx, 'disable_work_glossary_filter', False):
                                # 若外部已提前设置 ctx.work_name，则优先使用
                                inferred_work = getattr(ctx, 'work_name', None)
                                image_name = getattr(ctx, 'image_name', None)

                                # 1) 熟肉名：优先用 ctx.work_name，否则尝试从路径解析
                                if not inferred_work and image_name:
                                    inferred_work = resolve_work_name(image_name)

                                # 2) 生肉名：无论是否已有熟肉名，都尝试解析（用于 unassigned 匹配/术语写回）
                                raw_work_name = resolve_raw_work_name(image_name) if image_name else None

                                # 传入 raw_work_name，支持从 unassigned 按生肉名匹配术语
                                filtered_prompt, selected_work = filter_custom_prompt_json_by_work(
                                    ctx.custom_prompt_json,
                                    inferred_work,
                                    raw_work_name=raw_work_name
                                )
                                ctx.custom_prompt_json = filtered_prompt

                                # 记录最终作品名（用于提示/日志）
                                if selected_work:
                                    ctx.work_name = selected_work
                                elif inferred_work:
                                    ctx.work_name = inferred_work

                                # 保存 raw_work_name 到 ctx，用于术语提取写回
                                ctx.raw_work_name = raw_work_name

                                # 术语提取写回：只有在“明确识别到作品名”时才写入 works
                                # - selected_work == inferred_work：表示作品名匹配成功
                                # - 否则：多半是回退到默认/第一个作品，此时写回会污染作品区 → 写入无作品区
                                if selected_work and inferred_work and selected_work == inferred_work:
                                    ctx.glossary_write_work_name = selected_work
                                else:
                                    ctx.glossary_write_work_name = None

                    except Exception as e:
                        logger.debug(f"[Glossary] Failed to select work-specific glossary: {e}")

                    # 批次入口：记录本批次使用的术语来源
                    try:
                        if getattr(ctx, 'merge_work_glossary', False):
                            merged = getattr(ctx, 'merge_work_names', None) or []
                            if merged:
                                logger.info(f"[术语] 本批次合并多作品术语: {' + '.join(merged)}")
                            else:
                                logger.info("[术语] 本批次合并多作品术语 (自动/默认)")
                        else:
                            work = getattr(ctx, 'work_name', None)
                            raw_work = getattr(ctx, 'raw_work_name', None)
                            if work:
                                logger.info(f"[术语] 本批次使用作品术语: {work} (生肉名={raw_work or '无'})")
                            elif raw_work:
                                logger.info(f"[术语] 本批次使用无作品区术语 (生肉名={raw_work})")
                            else:
                                logger.info("[术语] 本批次使用默认术语 (未匹配到作品)")
                    except Exception:
                        logger.debug("[术语] 日志记录失败")

                    logger.info(f"Successfully loaded custom HQ prompt from: {prompt_path}")
                    # Log the parsed content for user verification
                    from .translators.gemini_hq import _flatten_prompt_data
                    _parsed_content = _flatten_prompt_data(ctx.custom_prompt_json)
                    # logger.info(f"--- Parsed Custom Prompt Content ---\n{parsed_content}\n------------------------------------")
                else:
                    logger.warning(f"Custom HQ prompt file not found at: {prompt_path}")
            except Exception as e:
                logger.error(f"Error loading custom HQ prompt: {e}")

        # Load AI line break prompt if enabled
        ctx.line_break_prompt_json = None
        if config.render.disable_auto_wrap: # This is the "AI断句" switch
            try:
                line_break_prompt_path = os.path.join(BASE_PATH, 'dict', 'system_prompt_line_break.json')
                if os.path.exists(line_break_prompt_path):
                    with open(line_break_prompt_path, 'r', encoding='utf-8') as f:
                        ctx.line_break_prompt_json = json.load(f)
                    logger.info("AI断句已启用，已加载断句提示词。")
                else:
                    logger.warning("AI断句已启用，但未找到断句提示词文件。")
            except Exception as e:
                logger.error(f"加载断句提示词失败: {e}")
        return ctx

    async def _run_text_translation(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        # Centralized prompt loading logic
        ctx = await self._load_and_prepare_prompts(config, ctx)

        # 检查text_regions是否为None或空
        if not ctx.text_regions:
            return []
    
        current_time = time.time()
        self._model_usage_timestamps[("translation", config.translator.translator)] = current_time

        # --- Main translation logic ---
        if config.translator.translator == Translator.none:
            for region in ctx.text_regions:  
                region.translation = ""  # 空翻译将创建空白区域 / Empty translation will create blank areas
        else: # Actual network translation
            # --- BEGIN PRE-TRANSLATION DE-DUPLICATION ---
            # Per user request, clean up duplicate lines within regions before sending to translator.
            logger.debug("Starting pre-translation de-duplication of regions...")
            for region in ctx.text_regions:
                if hasattr(region, 'lines') and hasattr(region, 'texts') and isinstance(region.lines, np.ndarray) and isinstance(region.texts, list) and len(region.lines) > 1:
                    unique_lines = []
                    unique_texts = []
                    seen_coords = set()
                    
                    original_line_count = len(region.lines)

                    for i, line_coords in enumerate(region.lines):
                        # Flatten for hashing, ensuring consistent representation
                        coords_tuple = tuple(line_coords.reshape(-1))
                        if coords_tuple not in seen_coords:
                            seen_coords.add(coords_tuple)
                            unique_lines.append(line_coords)
                            if i < len(region.texts):
                                unique_texts.append(region.texts[i])

                    if len(unique_lines) < original_line_count:
                        logger.info(f"Pre-translation cleanup: Found and removed {original_line_count - len(unique_lines)} internal duplicate lines in a region.")
                        region.lines = np.array(unique_lines, dtype=np.float64)
                        region.texts = unique_texts
                        region.text = '\n'.join(unique_texts)
            logger.debug("Pre-translation de-duplication finished.")
            # --- END PRE-TRANSLATION DE-DUPLICATION ---

            texts = [region.text.replace('\ufffd', '') for region in ctx.text_regions]
            translated_sentences = await self._dispatch_with_context(config, texts, ctx)

            for region, translation in zip(ctx.text_regions, translated_sentences):
                if config.render.uppercase:
                    translation = translation.upper()
                elif config.render.lowercase:
                    translation = translation.lower()
                region.translation = translation

        # 统一设置所有region的通用属性（避免重复代码）
        for region in ctx.text_regions:
            region.target_lang = config.translator.target_lang
            region._alignment = config.render.alignment
            region._direction = config.render.direction

        # --- Save results (moved to after post-processing) ---
        # JSON保存移到后处理（标点符号替换等）之后，确保保存的是最终结果
        # (JSON saving is deferred until after post-processing to ensure final results are saved)

        # --- NEW: Generate and Export Workflow ---
        if self.generate_and_export:
            logger.info("'Generate and Export' mode: Halting pipeline after translation and exporting clean text.")
            # 即使没有文本也要导出（创建空文件）
            if hasattr(ctx, 'image_name') and ctx.image_name:
                try:
                    json_path = find_json_path(ctx.image_name)
                    if json_path and os.path.exists(json_path):
                        from desktop_qt_ui.services.workflow_service import generate_translated_text, get_template_path_from_config
                        template_path = get_template_path_from_config()
                        if template_path and os.path.exists(template_path):
                            # 导出翻译（即使没有文本也会创建空文件）
                            translated_result = generate_translated_text(json_path, template_path)
                            logger.info(f"Translated text export result: {translated_result}")
                        else:
                            logger.warning(f"Template file not found, cannot export clean text: {template_path}")
                    else:
                        logger.warning(f"JSON file not found, cannot export clean text: {ctx.image_name}")
                except Exception as e:
                    logger.error(f"Failed to export clean text in 'Generate and Export' mode: {e}")
            
            # ✅ 标记成功（导出翻译模式完成）
            ctx.success = True
            ctx.pipeline_should_stop = True # Set flag to stop before rendering
            # Do not return here, let the function complete to return all regions

        # === 模板+保存文本模式退出逻辑 ===
        if self.template and self.save_text:
            logger.info("Template + Save Text mode: Stopping pipeline for this file to generate text template only.")
            ctx.pipeline_should_stop = True
            # Return early to skip all post-processing
            return ctx.text_regions

        # --- Post-processing (当不是"仅生成文本模板"模式时执行) ---
        
        # Punctuation correction logic. for translators often incorrectly change quotation marks from the source language to those commonly used in the target language.
        check_items = [
            # 圆括号处理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括号处理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引号处理
            ["「", """, "'", "『", "【"],
            ["」", """, "'", "』", "】"],
            ["『", """, "'", "「", "【"],
            ["』", """, "'", "」", "】"],
            
            # 新增【】处理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", """],
            ["「", "'"],
            ["」", """],
            ["」", "'"],
            ["【", "["],  
            ["】", "]"],
        ]
        
        # 全角句点替换（必须先替换长的，再替换短的，避免部分替换）
        full_width_period_replace = [
            ["…", "．．．"],  # 把三个全角句点替换为省略号
            ["…", "．．"],    # 把两个全角句点替换为省略号
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # === 优化后的数量判断逻辑 ===
                # === Optimized quantity judgment logic ===
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 全角句点替换（必须先替换，避免被两个点的规则部分替换）
                # Full-width period replacement (must be done first to avoid partial replacement)
                for v in full_width_period_replace:
                    region.translation = region.translation.replace(v[1], v[0])
                
                # 强制替换规则
                # Forced replacement rules
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # Apply post dictionary after translating
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")

        # 译后检查和重试逻辑 - 第一阶段：单个region幻觉检测
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # 单个region级别的幻觉检测（在过滤前进行）
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只检查重复内容幻觉，不进行页面级目标语言检查
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 对失败的区域进行重试
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    await self._retry_translation_with_validation(region, config, ctx)
                logger.info("Repetition check retry finished.")

        # 译后检查和重试逻辑 - 第二阶段：页面级目标语言检查（使用过滤后的区域）
        if config.translator.enable_post_translation_check:
            
            # 页面级目标语言检查（使用过滤后的区域数量）
            page_lang_check_result = True
            if ctx.text_regions and len(ctx.text_regions) > 5:
                logger.info(f"Starting page-level target language check with {len(ctx.text_regions)} regions...")
                page_lang_check_result = await self._check_target_language_ratio(
                    ctx.text_regions,
                    config.translator.target_lang,
                    min_ratio=0.5
                )
                
                if not page_lang_check_result:
                    logger.warning("Page-level target language ratio check failed")
                    
                    # 第二阶段：整个批次重新翻译逻辑
                    max_batch_retry = config.translator.post_check_max_retry_attempts
                    batch_retry_count = 0
                    
                    while batch_retry_count < max_batch_retry and not page_lang_check_result:
                        batch_retry_count += 1
                        logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry} for page-level target language check...")
                        
                        # 重新翻译所有区域
                        original_texts = []
                        for region in ctx.text_regions:
                            if hasattr(region, 'text') and region.text:
                                original_texts.append(region.text)
                            else:
                                original_texts.append("")
                        
                        if original_texts:
                            try:
                                # 重新批量翻译
                                logger.info(f"Retrying translation for {len(original_texts)} regions...")
                                new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                
                                # 更新翻译结果到regions
                                for i, region in enumerate(ctx.text_regions):
                                    if i < len(new_translations) and new_translations[i]:
                                        old_translation = region.translation
                                        region.translation = new_translations[i]
                                        logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")
                                    
                                # 重新检查目标语言比例
                                logger.info(f"Re-checking page-level target language ratio after batch retry {batch_retry_count}...")
                                page_lang_check_result = await self._check_target_language_ratio(
                                    ctx.text_regions,
                                    config.translator.target_lang,
                                    min_ratio=0.5
                                )
                                
                                if page_lang_check_result:
                                    logger.info("Page-level target language check passed")
                                    break
                                else:
                                    logger.warning("Page-level target language check still failed")
                                    
                            except Exception as e:
                                logger.error(f"Error during batch retry {batch_retry_count}: {e}")
                                break
                        else:
                            logger.warning("No text found for batch retry")
                            break
                    
                    if not page_lang_check_result:
                        logger.error(f"Page-level target language check failed after all {max_batch_retry} batch retries")
                else:
                    logger.info(f"Skipping page-level target language check: only {len(ctx.text_regions)} regions (threshold: 5)")
            
            # 统一的成功信息
            if page_lang_check_result:
                logger.info("All translation regions passed post-translation check.")
            else:
                logger.warning("Some translation regions failed post-translation check.")

        # 清理翻译文本
        for region in ctx.text_regions:
            if region.translation:
                # 1. 去掉BR标记周围的空白
                region.translation = re.sub(r'\s*(\[BR\]|<br>|【BR】)\s*', r'\1', region.translation, flags=re.IGNORECASE)
                
                # 2. 如果没有开启AI断句，去掉所有BR标记（避免显示在结果中）
                if not (config.render and config.render.disable_auto_wrap):
                    region.translation = re.sub(r'(\[BR\]|<br>|【BR】)', ' ', region.translation, flags=re.IGNORECASE)

        # 过滤逻辑（简化版本，保留主要过滤条件）
        new_text_regions = []
        for region in ctx.text_regions:
            should_filter = False
            filter_reason = ""

            if not region.translation.strip():
                should_filter = True
                filter_reason = "Translation contain blank areas"
            elif config.translator.translator != Translator.none:
                if region.translation.isnumeric():
                    should_filter = True
                    filter_reason = "Numeric translation"
                elif not config.translator.translator == Translator.original:
                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                    if text_equal:
                        should_filter = True
                        filter_reason = "Translation identical to original"

            if should_filter:
                if region.translation.strip():
                    logger.info(f'Filtered out: {region.translation}')
                    logger.info(f'Reason: {filter_reason}')
            else:
                new_text_regions.append(region)

        # 更新context中的文本区域列表（使用过滤后的）
        ctx.text_regions = new_text_regions

        # --- Save JSON after all post-processing (including punctuation replacement and filtering) ---
        if self.save_text or self.text_output_file:
            if hasattr(ctx, 'image_name') and ctx.image_name:
                # 使用ctx中保存的config，如果没有则使用当前config参数
                config_to_use = getattr(ctx, 'config', config) if hasattr(ctx, 'config') else config
                self._save_text_to_file(ctx.image_name, ctx, config_to_use)
                logger.info(f"Translations saved to JSON for {ctx.image_name} (after post-processing).")
            else:
                logger.warning("Could not save translation file, image_name not in context.")

        return new_text_regions

    async def _run_mask_refinement(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text',
                                              config.mask_dilation_offset, config.ocr.ignore_bubble, self.verbose,self.kernel_size)

    async def _run_inpainting(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        current_time = time.time()
        self._model_usage_timestamps[("inpainting", config.inpainter.inpainter)] = current_time
        return await dispatch_inpainting(config.inpainter.inpainter, ctx.img_rgb, ctx.mask, config.inpainter, config.inpainter.inpainting_size, self.device,
                                         self.verbose)

    async def _run_text_rendering(self, config: Config, ctx: Context):
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        current_time = time.time()
        self._model_usage_timestamps[("rendering", config.render.renderer)] = current_time
        
        # 优先使用配置文件中的 font_path，如果没有则使用命令行参数
        font_path = config.render.font_path or self.font_path
        
        if config.render.renderer == Renderer.none:
            output = ctx.img_inpainted
        # manga2eng currently only supports horizontal left to right rendering
        elif (config.render.renderer == Renderer.manga2Eng or config.render.renderer == Renderer.manga2EngPillow) and ctx.text_regions and LANGUAGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang) == 'h':
            if config.render.renderer == Renderer.manga2EngPillow:
                output = await dispatch_eng_render_pillow(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, font_path, config.render.line_spacing)
            else:
                output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, font_path, config.render.line_spacing)
        else:
            # Request debug image for balloon_fill mode when verbose
            need_debug_img = self.verbose and config.render.layout_mode == 'balloon_fill'
            result = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, font_path, config, ctx.img_rgb, return_debug_img=need_debug_img)
            
            # Handle debug image if returned
            if need_debug_img and isinstance(result, tuple):
                output, debug_img = result
                # Save balloon_fill debug image
                if debug_img is not None:
                    try:
                        debug_path = self._result_path('balloon_fill_boxes.png')
                        imwrite_unicode(debug_path, cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR), logger)
                        logger.info(f"📸 Balloon fill debug image saved: {debug_path}")
                    except Exception as e:
                        logger.error(f"Failed to save balloon_fill debug image: {e}")
            else:
                output = result
        
        # ✅ 渲染完成后立即清理不再需要的图像数据
        if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
            del ctx.img_rgb
            ctx.img_rgb = None
        if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
            del ctx.img_inpainted
            ctx.img_inpainted = None
        
        # 强制垃圾回收，释放内存
        import gc
        gc.collect()
        
        return output

    def _create_confidence_heatmap(self, mask: np.ndarray, vmin: float = 0.0, vmax: float = 1.0, equalize: bool = True) -> np.ndarray:
        """
        将灰度mask转换为带颜色条的置信度热力图
        
        Args:
            mask: 灰度mask数组 (0-255)
            vmin: 颜色映射的最小值 (0-1)，低于此值显示为最低颜色
            vmax: 颜色映射的最大值 (0-1)，高于此值显示为最高颜色
            equalize: 是否应用直方图均衡化增强对比度
        
        Returns:
            带颜色条的BGR图像
        """
        # 如果是多通道图像，转换为单通道
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # 直方图均衡化增强对比度
        if equalize:
            mask = cv2.equalizeHist(mask)
        
        # 归一化mask到0-1范围
        if mask.max() > 0:
            mask_normalized = mask.astype(np.float32) / 255.0
        else:
            mask_normalized = mask.astype(np.float32)
        
        # 调整映射范围：将[vmin, vmax]映射到[0, 1]
        if vmin != 0.0 or vmax != 1.0:
            mask_normalized = np.clip((mask_normalized - vmin) / (vmax - vmin), 0, 1)
        
        # 应用颜色映射（使用jet colormap）
        colormap = cm.get_cmap('jet')
        colored_mask = colormap(mask_normalized)
        
        # 转换为BGR格式 (matplotlib返回RGBA)
        colored_mask_bgr = (colored_mask[:, :, :3] * 255).astype(np.uint8)
        colored_mask_bgr = cv2.cvtColor(colored_mask_bgr, cv2.COLOR_RGB2BGR)
        
        # 创建带颜色条的图像
        h, w = mask.shape
        # 创建颜色条 (宽度为图像宽度的10%，最小50像素)
        colorbar_width = max(50, int(w * 0.1))
        colorbar_height = h
        
        # 生成颜色条
        colorbar = np.linspace(1, 0, colorbar_height).reshape(-1, 1)
        colorbar = np.tile(colorbar, (1, colorbar_width))
        colored_colorbar = colormap(colorbar)
        colored_colorbar_bgr = (colored_colorbar[:, :, :3] * 255).astype(np.uint8)
        colored_colorbar_bgr = cv2.cvtColor(colored_colorbar_bgr, cv2.COLOR_RGB2BGR)
        
        # 创建带文字标注的颜色条
        # 添加白色边框和文字背景
        colorbar_with_labels = np.ones((colorbar_height, colorbar_width + 100, 3), dtype=np.uint8) * 255
        colorbar_with_labels[:, :colorbar_width] = colored_colorbar_bgr
        
        # 添加刻度和文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        num_ticks = 11  # 显示11个刻度
        
        for i in range(num_ticks):
            # 映射到实际值范围 [vmax, vmin]
            normalized_value = 1.0 - i / (num_ticks - 1)  # 从1.0到0.0
            actual_value = vmin + normalized_value * (vmax - vmin)  # 映射到[vmin, vmax]
            y_pos = int(i * (colorbar_height - 1) / (num_ticks - 1))
            
            # 绘制刻度线
            cv2.line(colorbar_with_labels, 
                    (colorbar_width, y_pos), 
                    (colorbar_width + 10, y_pos), 
                    (0, 0, 0), 1)
            
            # 绘制文字
            text = f'{actual_value:.2f}'
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_y = y_pos + text_size[1] // 2
            cv2.putText(colorbar_with_labels, text, 
                       (colorbar_width + 15, text_y), 
                       font, font_scale, (0, 0, 0), font_thickness)
        
        # 添加标题
        title = 'Confidence'
        title_size = cv2.getTextSize(title, font, font_scale, font_thickness)[0]
        title_x = colorbar_width + (100 - title_size[0]) // 2
        cv2.putText(colorbar_with_labels, title, 
                   (title_x, 20), 
                   font, font_scale, (0, 0, 0), font_thickness)
        
        # 拼接原图和颜色条
        result_image = np.hstack([colored_mask_bgr, colorbar_with_labels])
        
        return result_image

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        # 只有在verbose模式下才使用图片级子文件夹
        if self.verbose:
            image_subfolder = self._get_image_subfolder()
            if image_subfolder:
                if self.result_sub_folder:
                    result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, image_subfolder, path)
                else:
                    result_path = os.path.join(BASE_PATH, 'result', image_subfolder, path)
                # 确保目录存在
                os.makedirs(os.path.dirname(result_path), exist_ok=True)
                return result_path
        
        # 在server/web模式下（result_sub_folder为空）且为非verbose模式时
        # 需要创建一个子文件夹来保存final.png
        if not self.result_sub_folder:
            # When no subfolder is specified (like in desktop-ui mode),
            # the 'path' parameter is expected to be an absolute path to the output file.
            # Therefore, we don't join it with BASE_PATH.
            base_dir = os.path.join(BASE_PATH, 'result')
            result_path = os.path.join(base_dir, path)
        else:
            result_path = os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)
        
        # 确保目录存在
        dir_to_create = os.path.dirname(result_path)
        if dir_to_create:
            os.makedirs(dir_to_create, exist_ok=True)
        return result_path

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)
    
    def set_cancel_check_callback(self, callback):
        """设置取消检查回调函数"""
        self._cancel_check_callback = callback
    
    def _check_cancelled(self):
        """检查任务是否被取消"""
        if self._cancel_check_callback and self._cancel_check_callback():
            logger.warning('[阶段] 任务被取消')
            raise asyncio.CancelledError("Task cancelled")

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling': 'Running upscaling',
            'detection': 'Running text detection',
            'ocr': 'Running ocr',
            'mask-generation': 'Running mask refinement',
            'translating': 'Running text translation',
            'rendering': 'Running rendering',
            'colorizing': 'Running colorization',
            'downscaling': 'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions': 'No text regions! - Skipping',
            'skip-no-text': 'No text regions with text! - Skipping',
            'error-translating': 'Text translator returned empty queries',
            'cancelled': 'Image translation cancelled',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warning(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    async def translate_batch(self, images_with_configs: List[tuple], batch_size: int = None, image_names: List[str] = None, save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
        """
        批量翻译多张图片，在翻译阶段进行批量处理以提高效率
        
        Args:
            images_with_configs: List of (image, config) tuples
            batch_size: 批量大小，如果为None则使用实例的batch_size
            image_names: 已弃用的参数，保留用于兼容性
            save_info: 保存配置，包含output_folder、input_folders、format等
            global_offset: 全局偏移量，用于显示正确的图片编号（前端分批加载时使用）
            global_total: 全局总图片数，用于显示正确的总批次数（前端分批加载时使用）
            
        Returns:
            List of Context objects with translation results
        """
        # 每次翻译任务开始时重新加载过滤列表（仅在启用时）
        if self.filter_text_enabled:
            from .utils.text_filter import load_filter_list
            load_filter_list(force_reload=True)
        
        batch_size = batch_size or self.batch_size
        
        # 如果提供了全局总数，使用它来计算总批次数；否则使用当前批次的图片数
        display_total = global_total if global_total is not None else len(images_with_configs)
        
        # === 步骤0: load_text模式预处理 - 自动从TXT导入到JSON ===
        if self.load_text and images_with_configs:
            logger.info("Load text mode detected: Auto-importing translations from TXT to JSON...")
            self._preprocess_load_text_mode(images_with_configs)
        
        # === 步骤1: 检查是否需要使用高质量翻译模式 ===
        is_hq_translator = False
        if images_with_configs:
            first_config = images_with_configs[0][1]
            if first_config and hasattr(first_config.translator, 'translator'):
                from manga_translator.config import Translator
                translator_type = first_config.translator.translator
                is_hq_translator = translator_type in [Translator.openai_hq, Translator.gemini_hq]
                is_import_export_mode = self.load_text or self.template

                # 如果是高质量翻译且未启用并发模式，使用专用的高质量翻译流程
                if is_hq_translator and not is_import_export_mode and not self.batch_concurrent:
                    logger.info(f"检测到高质量翻译器 {translator_type}，自动启用高质量翻译模式")
                    return await self._translate_batch_high_quality(images_with_configs, save_info, global_offset, global_total)
                
                if is_hq_translator and is_import_export_mode:
                    logger.warning("检测到导入/导出翻译模式，高质量翻译流程将被跳过，将使用标准流程进行渲染。")
        
        # === 步骤1.5: 替换翻译模式 ===
        if self.replace_translation and images_with_configs:
            logger.info("Replace translation mode detected: Will extract translations from translated images")
            return await self._translate_batch_replace_translation(images_with_configs, save_info, global_offset, global_total)
        
        # === 步骤2: 检查是否需要使用顺序处理模式 ===
        # 注意：不要在这里调用 translate()，因为 translate() 会调用 translate_batch()，造成无限循环
        # 相反，我们直接使用批量处理逻辑，但 batch_size 设置为 1
        is_template_save_mode = self.template and self.save_text
        if is_template_save_mode:
            logger.info("Template+SaveText mode detected. Forcing sequential processing to save files one by one.")
            batch_size = 1  # 强制使用 batch_size=1
        elif batch_size <= 1 and not self.batch_concurrent:
            logger.debug('Batch size <= 1, using sequential processing')
            batch_size = 1
        
        # === 步骤3: 检查是否使用并发流水线模式 ===
        # 并发流水线支持：普通翻译、高质量翻译、单文件翻译
        # 不支持的特殊模式：
        # - load_text: 从JSON加载翻译
        # - template + save_text: 导出原文
        # - generate_and_export: 导出翻译
        # - colorize_only: 仅上色
        # - upscale_only: 仅超分
        # - inpaint_only: 仅修复
        
        # 检查是否有不兼容的特殊模式
        has_incompatible_mode = (
            self.load_text or 
            is_template_save_mode or 
            self.generate_and_export or 
            self.colorize_only or 
            self.upscale_only or 
            self.inpaint_only or
            self.replace_translation  # 替换翻译模式也不支持并发
        )
        
        # 如果启用了并发但有不兼容模式，给出提示
        if self.batch_concurrent and has_incompatible_mode:
            incompatible_modes = []
            if self.load_text:
                incompatible_modes.append("加载翻译")
            if is_template_save_mode:
                incompatible_modes.append("导出原文")
            if self.generate_and_export:
                incompatible_modes.append("导出翻译")
            if self.colorize_only:
                incompatible_modes.append("仅上色")
            if self.upscale_only:
                incompatible_modes.append("仅超分")
            if self.inpaint_only:
                incompatible_modes.append("仅修复")
            if self.replace_translation:
                incompatible_modes.append("替换翻译")
            
            logger.info(f'⚠️  并发流水线已禁用：当前模式 [{", ".join(incompatible_modes)}] 不支持并发处理')
        
        if self.batch_concurrent and not has_incompatible_mode:
            mode_desc = "高质量翻译" if is_hq_translator else "标准翻译"
            logger.info(f'🚀 启用并发流水线模式 ({mode_desc}): {len(images_with_configs)} 张图片, 翻译批量大小: {batch_size}')
            from .concurrent_pipeline import ConcurrentPipeline
            
            # 保存save_info供并发流水线使用
            self._current_save_info = save_info
            
            # 从配置中获取翻译并发参数
            translation_concurrency = getattr(self, 'translation_concurrency', 1)
            
            # 从配置与环境变量获取RPM限制（每分钟最大请求数）
            preset_rpm = 0
            if images_with_configs:
                first_config = images_with_configs[0][1] if isinstance(images_with_configs[0], tuple) else None
                if first_config and hasattr(first_config, 'translator'):
                    preset_rpm = int(getattr(first_config.translator, 'max_requests_per_minute', 0) or 0)
            # 如果配置未设置或为0，则回退到环境变量 MAX_REQUESTS_PER_MINUTE
            if preset_rpm <= 0:
                try:
                    env_rpm = os.getenv('MAX_REQUESTS_PER_MINUTE', '0')
                    preset_rpm = int(env_rpm.strip().strip('"')) if env_rpm else 0
                except Exception:
                    preset_rpm = 0
            # 兆底：直接从项目根目录的 .env 文件解析
            if preset_rpm <= 0:
                try:
                    from dotenv import dotenv_values
                    env_path = os.path.join(BASE_PATH, '.env')
                    if os.path.exists(env_path):
                        values = dotenv_values(env_path)
                        raw = values.get('MAX_REQUESTS_PER_MINUTE') if values else None
                        if raw:
                            preset_rpm = int(str(raw).strip().strip('"'))
                except Exception:
                    pass
            
            pipeline = ConcurrentPipeline(
                self, 
                batch_size,
                translation_concurrency=translation_concurrency,
                preset_rpm=preset_rpm
            )
            
            # 提取文件路径和配置
            file_paths = []
            configs = []
            for item in images_with_configs:
                # item 可能是 (image, config) 或 image
                if isinstance(item, tuple):
                    image, config = item
                    # 如果 image 是 PIL.Image 对象且有 name 属性（文件路径）
                    if hasattr(image, 'name'):
                        file_paths.append(image.name)
                    else:
                        # 如果是字符串，直接作为路径
                        file_paths.append(str(image))
                    configs.append(config)
                else:
                    # 单个图片对象
                    if hasattr(item, 'name'):
                        file_paths.append(item.name)
                    else:
                        file_paths.append(str(item))
                    # 使用默认配置
                    configs.append(images_with_configs[0][1] if isinstance(images_with_configs[0], tuple) else None)
            
            # 使用并发流水线处理（分批加载图片）
            contexts = await pipeline.process_batch(file_paths, configs)
            
            return contexts
        
        # === 步骤4: 批量处理模式（顺序处理） ===
        logger.info(f'Starting batch translation: {len(images_with_configs)} images, batch size: {batch_size}')
        logger.info(f'[阶段] 批量翻译任务启动')
        
        # ✅ 修复：如果 images_with_configs 中包含文件路径字符串（并发模式格式），需要先加载图片
        # 这种情况发生在：用户启用并发模式，但因为特殊模式（如仅上色）导致并发被禁用
        needs_image_loading = False
        if images_with_configs:
            first_item = images_with_configs[0]
            if isinstance(first_item, tuple):
                first_image = first_item[0]
                # 检查是否是字符串路径
                if isinstance(first_image, str):
                    needs_image_loading = True
                    logger.info("检测到文件路径格式，将先加载图片...")
        
        if needs_image_loading:
            from PIL import Image as PILImage
            loaded_images_with_configs = []
            for item in images_with_configs:
                if isinstance(item, tuple):
                    file_path, config = item
                    try:
                        # 加载图片
                        with open(file_path, 'rb') as f:
                            image = PILImage.open(f)
                            image.load()  # 立即加载图片数据
                        image.name = file_path  # 保存文件路径
                        loaded_images_with_configs.append((image, config))
                    except Exception as e:
                        logger.error(f"加载图片失败 {file_path}: {e}")
                        continue
                else:
                    loaded_images_with_configs.append(item)
            images_with_configs = loaded_images_with_configs
            logger.info(f"成功加载 {len(images_with_configs)} 张图片")
        
        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())
        
        results = []
        total_images = len(images_with_configs)

        # 分批处理所有图片
        for batch_start in range(0, total_images, batch_size):
            current_batch_images = []
            preprocessed_contexts = []
            translated_contexts = []
            
            try:
                await asyncio.sleep(0)  # 检查是否被取消
                self._check_cancelled()  # 检查取消标志

                batch_end = min(batch_start + batch_size, total_images)
                current_batch_images = images_with_configs[batch_start:batch_end]

                # 计算全局图片编号（考虑前端分批加载的偏移量）
                global_batch_start = global_offset + batch_start + 1
                global_batch_end = global_offset + batch_end
                global_batch_num = (global_offset + batch_start) // batch_size + 1
                global_total_batches = (display_total + batch_size - 1) // batch_size
                
                logger.info(f"Processing rolling batch {global_batch_num}/{global_total_batches} (images {global_batch_start}-{global_batch_end})")
                logger.info(f'[阶段] 开始处理批次 {global_batch_num}/{global_total_batches}')
                
                # 报告批次进度给前端
                await self._report_progress(f"batch:{global_batch_start}:{global_batch_end}:{display_total}")

                # --- 阶段1: 预处理（检测、OCR、文本行合并） ---
                
                # 特殊情况：load_text模式（从JSON加载翻译）
                if self.load_text:
                    logger.info("Load text mode: Loading translations from JSON and skipping detection/OCR/translation")
                    for i, (image, config) in enumerate(current_batch_images):
                        await asyncio.sleep(0)
                        self._check_cancelled()  # 检查取消标志
                        try:
                            self._set_image_context(config, image)
                            image_name = image.name if hasattr(image, 'name') else None
                            
                            # 直接处理 load_text 模式，不调用 translate() 避免无限循环
                            ctx = Context()
                            ctx.input = image
                            ctx.image_name = image_name
                            ctx.verbose = self.verbose
                            ctx.save_quality = self.save_quality
                            ctx.config = config
                            
                            # 加载翻译数据
                            loaded_regions, loaded_mask, mask_is_refined = self._load_text_and_regions_from_file(image_name, config)
                            if loaded_regions is None:
                                json_path = os.path.splitext(image_name)[0] + '_translations.json' if image_name else 'unknown'
                                raise FileNotFoundError(f"Translation file not found or invalid: {json_path}")
                            
                            # 如果regions是空列表，记录日志但继续处理（渲染原图）
                            if not loaded_regions:
                                logger.info(f"No text regions found in JSON for {os.path.basename(image_name)}, will render original image")
                            
                            # 设置字体大小和默认translation
                            for region in loaded_regions:
                                if not hasattr(region, 'font_size') or not region.font_size:
                                    try:
                                        # 确保lines形状正确 (N, 4, 2)
                                        if region.lines.ndim == 3 and region.lines.shape[1] >= 4 and region.lines.shape[2] >= 2:
                                            box_height = np.max(region.lines[:,:,1]) - np.min(region.lines[:,:,1])
                                            region.font_size = min(int(box_height * 0.8), 128)
                                        else:
                                            logger.warning(f"Invalid lines shape {region.lines.shape}, using default font_size=24")
                                            region.font_size = 24
                                    except Exception as e:
                                        logger.warning(f"Error calculating font_size from lines: {e}, using default font_size=24")
                                        region.font_size = 24
                                
                                # 如果translation为空或None，使用原文text作为默认值
                                if not region.translation:
                                    region.translation = region.text
                                    logger.debug(f"Region translation is empty, using original text: {region.text[:50]}...")
                            
                            ctx.text_regions = loaded_regions
                            
                            # 执行上色和超分
                            if config.colorizer.colorizer != Colorizer.none:
                                await self._report_progress('colorizing')
                                ctx.img_colorized = await self._run_colorizer(config, ctx)
                            else:
                                ctx.img_colorized = ctx.input
                            
                            if config.upscale.upscale_ratio:
                                await self._report_progress('upscaling')
                                ctx.upscaled = await self._run_upscaling(config, ctx)
                            else:
                                ctx.upscaled = ctx.img_colorized
                            
                            ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
                            
                            # 验证加载的图片
                            if ctx.img_rgb is None or ctx.img_rgb.size == 0:
                                logger.error("[批量] 加载图片失败: img_rgb为空或无效")
                                continue
                            
                            if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
                                logger.error(f"[批量] 加载的图片尺寸无效: {ctx.img_rgb.shape}")
                                continue
                            
                            # 处理 mask
                            if loaded_mask is not None:
                                if mask_is_refined:
                                    ctx.mask = loaded_mask
                                else:
                                    ctx.mask_raw = loaded_mask
                            else:
                                if ctx.mask_raw is None:
                                    mask = np.zeros_like(ctx.img_rgb[:, :, 0])
                                    polygons = [p.reshape((-1, 1, 2)) for r in ctx.text_regions for p in r.lines]
                                    cv2.fillPoly(mask, polygons, 255)
                                    ctx.mask_raw = mask
                            
                            # 如果执行了超分，需要将mask和坐标也超分
                            if config.upscale.upscale_ratio:
                                upscale_ratio = parse_upscale_ratio(config.upscale.upscale_ratio)
                                if upscale_ratio > 0:
                                    if ctx.mask_raw is not None:
                                        ctx.mask_raw = cv2.resize(ctx.mask_raw, (ctx.img_rgb.shape[1], ctx.img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    if ctx.mask is not None:
                                        ctx.mask = cv2.resize(ctx.mask, (ctx.img_rgb.shape[1], ctx.img_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
                                    
                                    for region in ctx.text_regions:
                                        region.lines = region.lines * upscale_ratio
                                        if hasattr(region, 'font_size') and region.font_size:
                                            region.font_size = int(region.font_size * upscale_ratio)
                            
                            # 如果没有文本区域，跳过mask refinement、inpainting和rendering，直接返回原图
                            if not ctx.text_regions:
                                logger.info(f"No text regions to render for {os.path.basename(image_name)}, returning original image")
                                await self._report_progress('finished', True)
                                ctx.result = ctx.upscaled  # 返回上采样后的原图
                                ctx = await self._revert_upscale(config, ctx)
                            else:
                                # Mask refinement
                                if ctx.mask is None:
                                    await self._report_progress('mask-generation')
                                    ctx.mask = await self._run_mask_refinement(config, ctx)
                                
                                # Inpainting
                                await self._report_progress('inpainting')
                                ctx.img_inpainted = await self._run_inpainting(config, ctx)
                                
                                # Rendering
                                await self._report_progress('rendering')
                                ctx.img_rendered = await self._run_text_rendering(config, ctx)
                                
                                await self._report_progress('finished', True)
                                ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
                                ctx = await self._revert_upscale(config, ctx)
                            
                            preprocessed_contexts.append((ctx, config))
                            
                            # ✅ 每处理完一张图片后立即清理内存（保留result）
                            self._cleanup_context_memory(ctx, keep_result=True)
                            
                        except Exception as e:
                            logger.error(f"Error loading text for image {i+1} in batch: {e}")
                            ctx = Context()
                            ctx.input = image
                            ctx.text_regions = []
                            if hasattr(image, 'name'):
                                ctx.image_name = image.name
                            ctx.translation_error = str(e)
                            ctx.result = image
                            preprocessed_contexts.append((ctx, config))
                    
                    # load_text模式下已经完成了所有处理（包括渲染），直接保存并返回
                    for ctx, config in preprocessed_contexts:
                        if save_info and ctx.result:
                            try:
                                # 使用统一的保存和清理方法（包含PSD导出）
                                self._save_and_cleanup_context(ctx, save_info, config, "LOAD_TEXT")
                            except Exception as save_err:
                                logger.error(f"Error saving load_text result for {os.path.basename(ctx.image_name)}: {save_err}")
                        
                        results.append(ctx)
                    
                    # ✅ load_text模式：批次完成后清理批次数据（图片已在循环内清理）
                    if current_batch_images:
                        for i, (image, _) in enumerate(current_batch_images):
                            if hasattr(image, 'close'):
                                try:
                                    image.close()
                                except:
                                    pass
                    
                    # load_text模式处理完成，继续下一批
                    continue

                # 标准模式：执行检测、OCR等预处理
                logger.info(f'[阶段] 开始预处理阶段（检测、OCR）')
                for i, (image, config) in enumerate(current_batch_images):
                    # 检查是否被取消
                    await asyncio.sleep(0)
                    self._check_cancelled()  # 检查取消标志
                    try:
                        self._set_image_context(config, image)
                        # ✅ 保存context以便渲染阶段复用，避免生成两个文件夹
                        from .utils.generic import get_image_md5
                        image_md5 = get_image_md5(image)
                        self._save_current_image_context(image_md5)
                        ctx = await self._translate_until_translation(image, config)
                        if hasattr(image, 'name'):
                            ctx.image_name = image.name
                        preprocessed_contexts.append((ctx, config))
                    except Exception as e:
                        logger.error(f"Error pre-processing image {i+1} in batch: {e}")
                        ctx = Context()
                        ctx.input = image
                        ctx.text_regions = []
                        if hasattr(image, 'name'):
                            ctx.image_name = image.name
                        preprocessed_contexts.append((ctx, config))

                # --- 阶段2: 翻译 ---
                logger.info(f'[阶段] 预处理完成，开始翻译阶段')
                if self.colorize_only or self.upscale_only or self.inpaint_only:
                    # 特殊情况：仅上色/仅超分/仅修复模式，跳过翻译
                    mode_name = "Colorize Only" if self.colorize_only else ("Upscale Only" if self.upscale_only else "Inpaint Only")
                    logger.info(f"{mode_name} mode: Skipping translation and rendering stages.")
                    translated_contexts = preprocessed_contexts
                elif is_template_save_mode:
                    # 特殊情况：导出原文模式，跳过翻译
                    logger.info("Template+SaveText mode: Skipping translation, will export original text only.")
                    translated_contexts = preprocessed_contexts
                else:
                    # 标准翻译流程
                    try:
                        translated_contexts = await self._batch_translate_contexts(preprocessed_contexts, batch_size)
                    except Exception as e:
                        logger.error(f"Error during batch translation stage: {e}")
                        raise

                # --- 阶段3: 渲染和保存 ---
                # 特殊情况：导出原文模式（跳过渲染，只保存JSON和导出原文）
                if is_template_save_mode:
                    logger.info("Template+SaveText mode: Skipping rendering, exporting original text only.")
                    for ctx, config in translated_contexts:
                        if hasattr(ctx, 'text_regions') and ctx.text_regions is not None and hasattr(ctx, 'image_name') and ctx.image_name:
                            # 保存JSON文件
                            self._save_text_to_file(ctx.image_name, ctx, config)
                            try:
                                json_path = find_json_path(ctx.image_name)
                                if json_path and os.path.exists(json_path):
                                    from desktop_qt_ui.services.workflow_service import generate_original_text, get_template_path_from_config
                                    template_path = get_template_path_from_config()
                                    if template_path and os.path.exists(template_path):
                                        # 导出原文
                                        original_result = generate_original_text(json_path, template_path)
                                        logger.info(f"Original text export for {os.path.basename(ctx.image_name)}: {original_result}")
                                    else:
                                        logger.warning(f"Template file not found for {os.path.basename(ctx.image_name)}: {template_path}")
                                else:
                                    logger.warning(f"JSON file not found for {os.path.basename(ctx.image_name)}")
                            except Exception as e:
                                logger.error(f"Failed to export original text for {os.path.basename(ctx.image_name)}: {e}")
                        # ✅ 标记成功（导出原文完成）
                        ctx.success = True
                        results.append(ctx)
                        
                        # ✅ 每处理完一张图片后立即清理内存（保留result）
                        self._cleanup_context_memory(ctx, keep_result=True)
                    
                    # ✅ 批次完成后清理批次数据（图片已在循环内清理）
                    if current_batch_images:
                        for i, (image, _) in enumerate(current_batch_images):
                            if hasattr(image, 'close'):
                                try:
                                    image.close()
                                except:
                                    pass
                    
                    continue  # 跳过渲染，继续下一批次
                
                # 特殊情况：生成并导出模式（跳过渲染）
                if self.generate_and_export:
                    logger.info("'Generate and Export' mode enabled for standard batch. Skipping rendering.")
                    for ctx, config in translated_contexts:
                        if ctx.text_regions and hasattr(ctx, 'image_name') and ctx.image_name:
                            self._save_text_to_file(ctx.image_name, ctx, config)
                            try:
                                json_path = find_json_path(ctx.image_name)
                                if json_path and os.path.exists(json_path):
                                    from desktop_qt_ui.services.workflow_service import generate_translated_text, get_template_path_from_config
                                    template_path = get_template_path_from_config()
                                    if template_path and os.path.exists(template_path):
                                        # 导出翻译
                                        translated_result = generate_translated_text(json_path, template_path)
                                        logger.info(f"Translated text export for {os.path.basename(ctx.image_name)}: {translated_result}")
                                    else:
                                        logger.warning(f"Template file not found for {os.path.basename(ctx.image_name)}: {template_path}")
                                else:
                                    logger.warning(f"JSON file not found for {os.path.basename(ctx.image_name)}")
                            except Exception as e:
                                logger.error(f"Failed to export clean text for {os.path.basename(ctx.image_name)}: {e}")
                        # ✅ 标记成功（导出翻译完成）
                        ctx.success = True
                        results.append(ctx)
                        
                        # ✅ 每处理完一张图片后立即清理内存（保留result）
                        self._cleanup_context_memory(ctx, keep_result=True)
                    
                    # ✅ 批次完成后清理批次数据（图片已在循环内清理）
                    if current_batch_images:
                        for i, (image, _) in enumerate(current_batch_images):
                            if hasattr(image, 'close'):
                                try:
                                    image.close()
                                except:
                                    pass
                    
                    continue  # 跳过渲染，继续下一批次

                # 标准流程：渲染并保存
                logger.info(f'[阶段] 翻译完成，开始渲染阶段')
                for idx, (ctx, config) in enumerate(translated_contexts):
                    await asyncio.sleep(0)  # 检查是否被取消
                    self._check_cancelled()  # 检查取消标志
                    try:
                        if hasattr(ctx, 'input'):
                            from .utils.generic import get_image_md5
                            image_md5 = get_image_md5(ctx.input)
                            if not self._restore_image_context(image_md5):
                                self._set_image_context(config, ctx.input)
                        
                        # Colorize/Upscale/Inpaint Only Mode: Skip rendering pipeline
                        if not self.colorize_only and not self.upscale_only and not self.inpaint_only:
                            ctx = await self._complete_translation_pipeline(ctx, config)
                        if save_info and ctx.result:
                            try:
                                # 使用统一的保存和清理方法（包含PSD导出）
                                self._save_and_cleanup_context(ctx, save_info, config, "BATCH")
                            except Exception as save_err:
                                logger.error(f"Error saving standard batch result for {os.path.basename(ctx.image_name)}: {save_err}")

                        # 只在save_text或text_output_file启用时保存JSON（包括空的text_regions）
                        if (self.save_text or self.text_output_file) and hasattr(ctx, 'text_regions') and ctx.text_regions is not None and hasattr(ctx, 'image_name') and ctx.image_name:
                            # 使用循环变量中的config，而不是从ctx中获取
                            self._save_text_to_file(ctx.image_name, ctx, config)

                        results.append(ctx)

                        # ✅ 渲染完一张立即清理这张图片的中间数据（不等整个批次完成）
                        if hasattr(ctx, 'input') and ctx.input is not None:
                            if hasattr(ctx.input, 'close'):
                                try:
                                    ctx.input.close()
                                except:
                                    pass
                            ctx.input = None
                        if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                            del ctx.img_rgb
                            ctx.img_rgb = None
                        if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
                            del ctx.img_inpainted
                            ctx.img_inpainted = None
                        if hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
                            del ctx.img_colorized
                            ctx.img_colorized = None
                        
                        # 每渲染3张图片就强制垃圾回收一次
                        if (idx + 1) % 3 == 0:
                            import gc
                            gc.collect()
                            
                    except Exception as e:
                        logger.error(f"Error rendering image in batch: {e}")
                        results.append(ctx)
            
            finally:
                # ✅ 批次完成后（无论成功还是失败）立即清理内存
                logger.info(f'[阶段] 批次 {batch_start//batch_size + 1} 处理完成，开始清理内存')
                self._cleanup_batch_memory(
                    current_batch_images=current_batch_images,
                    preprocessed_contexts=preprocessed_contexts,
                    translated_contexts=translated_contexts,
                    keep_results=True
                )

                # Release task-scoped translators at the batch boundary.
                self._release_task_translators(self._get_current_task_id())

                logger.debug(f'[MEMORY] Batch {batch_start//batch_size + 1} cleanup completed')

        logger.info(f"Batch translation completed: processed {len(results)} images")
        return results

    async def _translate_until_translation(self, image: Image.Image, config: Config) -> Context:
        """
        执行翻译之前的所有步骤（彩色化、上采样、检测、OCR、文本行合并）
        """
        
        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()
        
        ctx = Context()
        ctx.input = image
        ctx.result = None
        
        # 保存原始输入图片用于调试
        if self.verbose:
            try:
                input_img = np.array(image)
                if len(input_img.shape) == 3:  # 彩色图片，转换BGR顺序
                    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)
                result_path = self._result_path('input.png')
                imwrite_unicode(result_path, input_img, logger)
            except Exception as e:
                logger.error(f"Error saving input.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # preload and download models (not strictly necessary, remove to lazy load)
        if self.models_ttl == 0 and not self._models_loaded:
            logger.info('Loading models')
            
            # ✅ 检查停止标志
            await asyncio.sleep(0)
            self._check_cancelled()
            
            if config.upscale.upscale_ratio:
                # 传递超分配置参数
                upscaler_kwargs = {}
                if config.upscale.upscaler == 'realcugan':
                    if config.upscale.realcugan_model:
                        upscaler_kwargs['model_name'] = config.upscale.realcugan_model
                    if config.upscale.tile_size is not None:
                        upscaler_kwargs['tile_size'] = config.upscale.tile_size
                elif config.upscale.upscaler == 'mangajanai':
                    # mangajanai 的 upscale_ratio 可以是字符串 (x2, x4, DAT2 x4) 或数字
                    ratio = config.upscale.upscale_ratio
                    if isinstance(ratio, str):
                        upscaler_kwargs['model_name'] = ratio
                    elif ratio == 2:
                        upscaler_kwargs['model_name'] = 'x2'
                    else:
                        upscaler_kwargs['model_name'] = 'x4'
                    if config.upscale.tile_size is not None:
                        upscaler_kwargs['tile_size'] = config.upscale.tile_size
                await prepare_upscaling(config.upscale.upscaler, **upscaler_kwargs)
            
            await prepare_detection(config.detector.detector)
            
            await prepare_ocr(config.ocr.ocr, self.device)
            
            await prepare_inpainting(config.inpainter.inpainter, self.device)
            
            await prepare_translation(config.translator.translator_gen)
            
            if config.colorizer.colorizer != Colorizer.none:
                await prepare_colorization(config.colorizer.colorizer)
            
            self._models_loaded = True  # 标记模型已加载

        # Start the background cleanup job once if not already started.
        if self._detector_cleanup_task is None:
            self._detector_cleanup_task = asyncio.create_task(self._detector_cleanup_job())

        # ✅ 检查停止标志
        await asyncio.sleep(0)
        self._check_cancelled()

        # -- Colorization
        if config.colorizer.colorizer != Colorizer.none:
            await self._report_progress('colorizing')
            try:
                ctx.img_colorized = await self._run_colorizer(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during colorizing:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.img_colorized = ctx.input
        else:
            ctx.img_colorized = ctx.input

        # --- Colorize Only Mode Check (for batch processing) ---
        if self.colorize_only:
            logger.info("Colorize Only mode (batch): Running colorization only, skipping detection, OCR, translation and rendering.")
            ctx.result = ctx.img_colorized
            ctx.text_regions = []  # Empty text regions
            await self._report_progress('colorize-only-complete', True)
            # 不在这里清理，让调用方在保存JSON后统一清理
            return ctx

        # -- Upscaling
        if config.upscale.upscale_ratio:
            # ✅ 检查停止标志
            await asyncio.sleep(0)
            self._check_cancelled()
            
            await self._report_progress('upscaling')
            try:
                ctx.upscaled = await self._run_upscaling(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during upscaling:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise  
                ctx.upscaled = ctx.img_colorized
        else:
            ctx.upscaled = ctx.img_colorized

        # --- Upscale Only Mode Check (for batch processing) ---
        if self.upscale_only:
            logger.info("Upscale Only mode (batch): Running upscaling only, skipping detection, OCR, translation and rendering.")
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # Empty text regions
            await self._report_progress('upscale-only-complete', True)
            # 不在这里清理，让调用方在保存JSON后统一清理
            return ctx

        # --- Inpaint Only Mode Check (for batch processing) ---
        if self.inpaint_only:
            logger.info("=== Inpaint Only Mode ===")
            logger.info("Pipeline: Detection → Fill Text → Textline Merge → Mask Refinement → Inpainting")
            
            ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
            
            # 验证加载的图片
            if ctx.img_rgb is None or ctx.img_rgb.size == 0:
                logger.error("[批量流程] 加载图片失败: img_rgb为空或无效")
                raise Exception("加载图片失败: img_rgb为空或无效")
            
            if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
                logger.error(f"[批量流程] 加载的图片尺寸无效: {ctx.img_rgb.shape}")
                raise Exception(f"加载的图片尺寸无效: {ctx.img_rgb.shape}")
            
            # Step 1: 检测 - 获取textlines（检测框）和mask_raw（原始蒙版）
            await self._report_progress('detection')
            try:
                ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
                logger.info(f"✓ Step 1 - Detection: Found {len(ctx.textlines) if ctx.textlines else 0} textlines")
                logger.info(f"  - mask_raw: {ctx.mask_raw.shape if ctx.mask_raw is not None else 'None'}")
                if ctx.mask_raw is not None:
                    logger.info(f"  - mask_raw non-zero pixels: {np.count_nonzero(ctx.mask_raw)}")
            except Exception as _e:
                logger.error(f"Error during detection:\n{traceback.format_exc()}")
                if not self.ignore_errors:
                    raise
                ctx.textlines = []
                ctx.mask_raw = None
                ctx.mask = None
            
            if not ctx.textlines or ctx.mask_raw is None:
                logger.warning("No textlines or mask_raw detected, skipping inpainting.")
                ctx.img_inpainted = ctx.img_rgb
                ctx.result = ctx.img_inpainted
                ctx.text_regions = []
                await self._report_progress('inpaint-only-complete', True)
                # 不在这里清理，让调用方在保存JSON后统一清理
                return ctx
            
            # Step 2: 填充文本 - 跳过OCR，为每个textline填充占位文本
            for textline in ctx.textlines:
                textline.text = "TEXT"
            logger.info(f"✓ Step 2 - Fill Text: Filled {len(ctx.textlines)} textlines with placeholder 'TEXT'")
            
            # Step 3: Textline Merge - 将textlines合并成text_regions（大框）
            try:
                ctx.text_regions = await dispatch_textline_merge(
                    ctx.textlines, 
                    ctx.img_rgb.shape[1], 
                    ctx.img_rgb.shape[0],
                    config, 
                    verbose=self.verbose
                )
                logger.info(f"✓ Step 3 - Textline Merge: Merged {len(ctx.textlines)} textlines into {len(ctx.text_regions)} text_regions")
            except Exception as _e:
                logger.error(f"Error during textline merge:\n{traceback.format_exc()}")
                # 降级：为每个textline创建一个简单的TextBlock
                logger.warning("Falling back to simple text_regions (1 textline = 1 region)")
                ctx.text_regions = []
                for textline in ctx.textlines:
                    region = TextBlock(
                        lines=[textline.pts],
                        texts=["TEXT"],
                        font_size=int(textline.font_size) if hasattr(textline, 'font_size') else 20,
                        angle=0,
                        prob=textline.prob if hasattr(textline, 'prob') else 1.0,
                        fg_color=(0, 0, 0),
                        bg_color=(255, 255, 255)
                    )
                    ctx.text_regions.append(region)
                logger.info(f"Created {len(ctx.text_regions)} simple text_regions")
            
            if not ctx.text_regions:
                logger.warning("No text_regions created, skipping mask refinement and inpainting.")
                ctx.img_inpainted = ctx.img_rgb
                ctx.result = ctx.img_inpainted
                await self._report_progress('inpaint-only-complete', True)
                # 不在这里清理，让调用方在保存JSON后统一清理
                return ctx
            
            # Step 4: Mask Refinement - 使用text_regions和mask_raw优化蒙版
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
                mask_pixels = np.count_nonzero(ctx.mask) if ctx.mask is not None else 0
                logger.info(f"✓ Step 4 - Mask Refinement: Generated mask with {mask_pixels} non-zero pixels")
            except Exception as _e:
                logger.error(f"Error during mask refinement:\n{traceback.format_exc()}")
                # 降级到简单膨胀
                logger.warning("Falling back to simple mask dilation")
                kernel = np.ones((config.kernel_size, config.kernel_size), np.uint8)
                ctx.mask = cv2.dilate(ctx.mask_raw, kernel, iterations=config.mask_dilation_offset // config.kernel_size)
                mask_pixels = np.count_nonzero(ctx.mask) if ctx.mask is not None else 0
                logger.info(f"Simple dilated mask has {mask_pixels} non-zero pixels")
            
            # Step 5: Inpainting - 使用优化后的mask进行修复
            if ctx.mask is None or np.count_nonzero(ctx.mask) == 0:
                logger.warning("Mask is empty! Skipping inpainting.")
                ctx.img_inpainted = ctx.img_rgb
            else:
                await self._report_progress('inpainting')
                try:
                    ctx.img_inpainted = await self._run_inpainting(config, ctx)
                    logger.info("✓ Step 5 - Inpainting: Completed successfully")
                except Exception as _e:
                    logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
                    if not self.ignore_errors:
                        raise
                    ctx.img_inpainted = ctx.img_rgb
            
            # 设置结果 - 转换为PIL Image（保存函数需要PIL格式）
            from PIL import Image
            if isinstance(ctx.img_inpainted, np.ndarray):
                ctx.result = Image.fromarray(ctx.img_inpainted)
            else:
                ctx.result = ctx.img_inpainted
            
            ctx.text_regions = []

            # 设置标志，告诉_complete_translation_pipeline跳过处理
            ctx.inpaint_only_complete = True

            logger.info("=== Inpaint Only Mode Complete ===")
            await self._report_progress('inpaint-only-complete', True)
            # 不在这里清理，让调用方在保存JSON后统一清理
            return ctx

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)
        
        # 验证加载的图片
        if ctx.img_rgb is None or ctx.img_rgb.size == 0:
            logger.error("加载图片失败: img_rgb为空或无效")
            if not self.ignore_errors:
                raise Exception("加载图片失败: img_rgb为空或无效")
            # 尝试从原始输入重新加载
            ctx.img_rgb, ctx.img_alpha = load_image(ctx.input)
        
        if len(ctx.img_rgb.shape) < 2 or ctx.img_rgb.shape[0] == 0 or ctx.img_rgb.shape[1] == 0:
            logger.error(f"加载的图片尺寸无效: {ctx.img_rgb.shape}")
            if not self.ignore_errors:
                raise Exception(f"加载的图片尺寸无效: {ctx.img_rgb.shape}")

        # -- Detection
        await self._report_progress('detection')
        try:
            ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during detection:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = [] 
            ctx.mask_raw = None
            ctx.mask = None

        if self.verbose and ctx.mask_raw is not None:
            # 生成带置信度颜色映射和颜色条的热力图
            logger.info(f"Generating confidence heatmap for mask_raw (shape: {ctx.mask_raw.shape}, dtype: {ctx.mask_raw.dtype})")
            heatmap = self._create_confidence_heatmap(ctx.mask_raw, equalize=False)
            logger.info(f"Heatmap generated (shape: {heatmap.shape}), saving to mask_raw.png")
            imwrite_unicode(self._result_path('mask_raw.png'), heatmap, logger)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # 设置为空列表，以便保存空的JSON
            return await self._revert_upscale(config, ctx)

        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            imwrite_unicode(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR), logger)

        # -- OCR
        await self._report_progress('ocr')
        try:
            ctx.textlines = await self._run_ocr(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during ocr:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.textlines = []

        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            ctx.result = ctx.upscaled
            ctx.text_regions = []  # 设置为空列表，以便保存空的JSON
            return await self._revert_upscale(config, ctx)

        # -- Textline merge
        await self._report_progress('textline_merge')
        try:
            ctx.text_regions = await self._run_textline_merge(config, ctx)
        except Exception as _e:  
            logger.error(f"Error during textline_merge:\n{traceback.format_exc()}")  
            if not self.ignore_errors:  
                raise 
            ctx.text_regions = []

        # -- 过滤列表：根据 OCR 识别的原文过滤
        if ctx.text_regions and self.filter_text_enabled:
            filtered_regions = []
            filtered_count = 0
            for region in ctx.text_regions:
                # 1. 先检查文本过滤列表（examples/filter_list.txt）
                match_result = match_filter(region.text)
                if match_result:
                    matched_word, match_type = match_result
                    logger.info(f'过滤文本区域 ({match_type}匹配): "{region.text}" -> 匹配: "{matched_word}"')
                    filtered_count += 1
                    continue
                
                # 2. 检测网站水印（支持本地自定义规则）
                is_watermark = False
                if config.enable_watermark_filter:
                    watermark_patterns, exact_match_patterns = self._load_watermark_filter_rules()
                    norm_text = re.sub(r'[^\w\s]', ' ', region.text.lower()).strip()
                    norm_text = ' '.join(norm_text.split())
                    original_text = region.text
                    
                    # 部分匹配检查
                    for pattern in watermark_patterns:
                        try:
                            if pattern.get('type') == 'regex':
                                if pattern.get('compiled') and pattern['compiled'].search(original_text):
                                    is_watermark = True
                                    logger.debug(f"[水印过滤] regex命中: {pattern.get('pattern')}")
                                    break
                            else:
                                norm_pattern = pattern.get('norm')
                                if norm_pattern and norm_pattern in norm_text:
                                    is_watermark = True
                                    logger.debug(f"[水印过滤] substr命中: {pattern.get('pattern')}")
                                    break
                        except Exception:
                            continue
                    
                    # 完整匹配检查
                    if not is_watermark and exact_match_patterns:
                        cleaned_text = norm_text
                        for pattern in exact_match_patterns:
                            try:
                                if pattern.get('type') == 'regex':
                                    if pattern.get('compiled') and pattern['compiled'].search(original_text):
                                        is_watermark = True
                                        logger.debug(f"[水印过滤] regex命中(exact): {pattern.get('pattern')}")
                                        break
                                else:
                                    norm_pattern = pattern.get('norm')
                                    if not norm_pattern:
                                        continue
                                    if cleaned_text == norm_pattern or cleaned_text.startswith(norm_pattern) or cleaned_text.endswith(norm_pattern):
                                        is_watermark = True
                                        logger.debug(f"[水印过滤] substr命中(exact): {pattern.get('pattern')}")
                                        break
                            except Exception:
                                continue
                
                if is_watermark:
                    logger.info(f'过滤水印文本: "{region.text}"')
                    filtered_count += 1
                    continue
                
                if is_mostly_noise_text(region.text):
                    logger.info(f'过滤噪声文本: "{region.text}"')
                    filtered_count += 1
                    continue
                
                filtered_regions.append(region)
            
            if filtered_count > 0:
                logger.info(f'过滤列表: 过滤了 {filtered_count} 个文本区域')
            ctx.text_regions = filtered_regions

        if self.verbose and ctx.text_regions:
            show_panels = not config.force_simple_sort  # 当不使用简单排序时显示panel
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions, 
                                        show_panels=show_panels, img_rgb=ctx.img_rgb, right_to_left=config.render.rtl)
            imwrite_unicode(self._result_path('bboxes.png'), bboxes, logger)

        # Apply pre-dictionary after textline merge
        pre_dict = load_dictionary(self.pre_dict)
        pre_replacements = []
        for region in ctx.text_regions:
            original = region.text  
            region.text = apply_dictionary(region.text, pre_dict)
            if original != region.text:
                pre_replacements.append(f"{original} => {region.text}")

        if pre_replacements:
            logger.info("Pre-translation replacements:")
            for replacement in pre_replacements:
                logger.info(replacement)
        else:
            logger.info("No pre-translation replacements made.")

        # Source language detection for per-language prompt loading
        if ctx.text_regions:
            try:
                from .utils.lang_detect import detect_source_lang
                texts_for_detect = [r.text for r in ctx.text_regions if getattr(r, 'text', None)]
                lang_code, iso_lang, confidence, method = detect_source_lang(texts_for_detect)
                ctx.from_lang = lang_code or 'auto'
                if lang_code:
                    logger.info(f"[Source Language] Detected: {lang_code} (ISO: {iso_lang}, method: {method})")
                else:
                    logger.info(f"[Source Language] Detection failed, using 'auto'")
            except Exception as e:
                ctx.from_lang = 'auto'
                logger.warning(f"[Source Language] Detection error: {e}, using 'auto'")
        else:
            ctx.from_lang = 'auto'

        # 保存当前图片上下文到ctx中，用于并发翻译时的路径管理
        if self._current_image_context:
            ctx.image_context = self._current_image_context.copy()

        return ctx

    async def _batch_translate_contexts(self, contexts_with_configs: List[tuple], batch_size: int) -> List[tuple]:
        """
        批量处理翻译步骤，防止内存溢出
        """
        results = []
        total_contexts = len(contexts_with_configs)
        
        # 按批次处理，防止内存溢出
        for i in range(0, total_contexts, batch_size):
            await asyncio.sleep(0)  # 检查是否被取消
            self._check_cancelled()  # 检查取消标志
            batch = contexts_with_configs[i:i + batch_size]
            logger.info(f'Processing translation batch {i//batch_size + 1}/{(total_contexts + batch_size - 1)//batch_size}')
            
            # 收集当前批次的所有文本
            all_texts = []
            batch_text_mapping = []  # 记录每个文本属于哪个context和region
            
            for ctx_idx, (ctx, config) in enumerate(batch):
                if not ctx.text_regions:
                    continue
                    
                region_start_idx = len(all_texts)
                for region_idx, region in enumerate(ctx.text_regions):
                    # 跳过 None 值，避免后续处理时出错
                    if region.text is not None:
                        all_texts.append(region.text)
                        batch_text_mapping.append((ctx_idx, region_idx))
                
            if not all_texts:
                # 当前批次没有需要翻译的文本
                results.extend(batch)
                continue
                
            # 批量翻译
            try:
                await self._report_progress('translating')
                # 使用第一个配置进行翻译（假设批次内配置相同）
                sample_config = batch[0][1] if batch else None
                if sample_config:
                    # ✅ 合并当前批次所有图片的text_regions（用于AI断句）
                    # 创建临时ctx，避免影响原始ctx
                    merged_ctx = Context()
                    merged_ctx.config = sample_config  # 复制配置
                    
                    all_regions = []
                    for ctx, _ in batch:
                        if ctx.text_regions:
                            all_regions.extend(ctx.text_regions)
                    merged_ctx.text_regions = all_regions
                    
                    # 复制第一个ctx的其他必要属性
                    first_ctx = batch[0][0]
                    if hasattr(first_ctx, 'from_lang'):
                        merged_ctx.from_lang = first_ctx.from_lang
                    # ✅ 复制 image_name，用于术语表作品匹配
                    if hasattr(first_ctx, 'image_name'):
                        merged_ctx.image_name = first_ctx.image_name
                    
                    # ✅ 复制同批次局部上下文（用于延迟重试单图时参考上一张）
                    if hasattr(first_ctx, 'local_prev_context'):
                        merged_ctx.local_prev_context = getattr(first_ctx, 'local_prev_context', None)

                    # ✅ 加载AI断句prompt和自定义HQ prompt
                    merged_ctx = await self._load_and_prepare_prompts(sample_config, merged_ctx)
                    
                    logger.debug(f"[Batch] Merged {len(all_regions)} text regions from {len(batch)} images for AI line breaking")
                    
                    # 支持批量翻译 - 传递合并后的上下文（仅用于AI断句）
                    batch_contexts = [ctx for ctx, config in batch]
                    
                    # 计算当前批次在所有页面中的索引（用于上下文）
                    # i 是当前批次的起始索引（相对于本次translate_batch调用的所有图片）
                    # self.all_page_translations 包含之前所有已翻译的页面
                    page_index = len(self.all_page_translations) + i
                    
                    # 准备batch_original_texts（用于并发模式的上下文）
                    batch_original_texts = []
                    for ctx, _ in batch:
                        if ctx.text_regions:
                            image_data = {
                                'original_texts': [region.text for region in ctx.text_regions if region.text is not None]
                            }
                            batch_original_texts.append(image_data)
                    
                    # ✅ 为HQ翻译器准备high_quality_batch_data（包含图片和text_regions）
                    # 这是HQ翻译器进入高质量批量模式的必要条件，也是AI断句检查能正常工作的前提
                    if sample_config.translator.translator in [Translator.openai_hq, Translator.gemini_hq]:
                        hq_batch_data = []
                        global_text_index = 1  # 全局文本编号从1开始（与提示词中的编号一致）
                        for ctx, _ in batch:
                            if ctx.text_regions:
                                num_regions = len(ctx.text_regions)
                                # 为当前图片生成全局连续的文本编号
                                text_order = list(range(global_text_index, global_text_index + num_regions))
                                global_text_index += num_regions
                                
                                upscaled_size = None
                                # 使用超分后的图片尺寸（如果有超分），否则使用上色后的图片尺寸
                                # 注意：需要处理 PIL Image 和 numpy array 两种情况
                                from PIL import Image as PILImage
                                if hasattr(ctx, 'upscaled') and ctx.upscaled is not None:
                                    if isinstance(ctx.upscaled, PILImage.Image):
                                        w, h = ctx.upscaled.size
                                        upscaled_size = (h, w)  # 转换为 (height, width)
                                    else:
                                        upscaled_size = ctx.upscaled.shape[:2]  # numpy: (height, width)
                                elif hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
                                    if isinstance(ctx.img_colorized, PILImage.Image):
                                        w, h = ctx.img_colorized.size
                                        upscaled_size = (h, w)
                                    else:
                                        upscaled_size = ctx.img_colorized.shape[:2]
                                elif hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                                    if isinstance(ctx.img_rgb, PILImage.Image):
                                        w, h = ctx.img_rgb.size
                                        upscaled_size = (h, w)
                                    else:
                                        upscaled_size = ctx.img_rgb.shape[:2]
                                
                                img_data = {
                                    'image': ctx.input if hasattr(ctx, 'input') else None,
                                    'text_regions': ctx.text_regions,
                                    'original_texts': [region.text for region in ctx.text_regions if region.text is not None],
                                    'text_order': text_order,
                                    'upscaled_size': upscaled_size
                                }
                                hq_batch_data.append(img_data)
                        
                        if hq_batch_data:
                            merged_ctx.high_quality_batch_data = hq_batch_data
                            logger.debug(f"[Batch] Prepared high_quality_batch_data for {len(hq_batch_data)} images")
                    
                    translated_texts = await self._batch_translate_texts(
                        all_texts, 
                        sample_config, 
                        merged_ctx, 
                        batch_contexts,
                        page_index=page_index,
                        batch_index=0,  # 批量处理时第一张图的批次索引为0
                        batch_original_texts=batch_original_texts
                    )
                else:
                    translated_texts = all_texts  # 无法翻译时保持原文
                    
                # 将翻译结果分配回各个context
                text_idx = 0
                for ctx_idx, (ctx, config) in enumerate(batch):
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region_idx, region in enumerate(ctx.text_regions):
                        if text_idx < len(translated_texts):
                            region.translation = translated_texts[text_idx]
                            region.target_lang = config.translator.target_lang
                            region._alignment = config.render.alignment
                            region._direction = config.render.direction
                            text_idx += 1
                        
                # 应用后处理逻辑（括号修正、过滤等）
                for ctx, config in batch:
                    if ctx.text_regions:
                        ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                
                # ✅ 立即保存当前批次的翻译结果到all_page_translations，供下一个批次使用上下文
                for ctx, config in batch:
                    if ctx.text_regions:
                        page_trans = {}
                        for region in ctx.text_regions:
                            if region.translation:
                                page_trans[region.text] = region.translation
                        self.all_page_translations.append(page_trans)
                        logger.debug(f"[Batch Context] Saved {len(page_trans)} translations for next batch context")

                        page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                              for i, r in enumerate(ctx.text_regions)}
                        self._original_page_texts.append(page_original_texts)
                        logger.debug(f"[Batch Context] Saved {len(page_original_texts)} original texts for next batch context")
                        
                # Prune history to prevent memory leak
                self._prune_context_history()
                        
                # 批次级别的目标语言检查
                if batch and batch[0][1].translator.enable_post_translation_check:
                    # 收集批次内所有页面的filtered regions
                    all_batch_regions = []
                    for ctx, config in batch:
                        if ctx.text_regions:
                            all_batch_regions.extend(ctx.text_regions)
                    
                    # 进行批次级别的目标语言检查
                    batch_lang_check_result = True
                    if all_batch_regions and len(all_batch_regions) > 10:
                        sample_config = batch[0][1]
                        logger.info(f"Starting batch-level target language check with {len(all_batch_regions)} regions...")
                        batch_lang_check_result = await self._check_target_language_ratio(
                            all_batch_regions,
                            sample_config.translator.target_lang,
                            min_ratio=0.5
                        )
                        
                        if not batch_lang_check_result:
                            logger.warning("Batch-level target language ratio check failed")
                            
                            # 批次重新翻译逻辑
                            max_batch_retry = sample_config.translator.post_check_max_retry_attempts
                            batch_retry_count = 0
                            
                            while batch_retry_count < max_batch_retry and not batch_lang_check_result:
                                batch_retry_count += 1
                                logger.warning(f"Starting batch retry {batch_retry_count}/{max_batch_retry}")
                                
                                # 重新翻译批次内所有区域
                                all_original_texts = []
                                region_mapping = []  # 记录每个text属于哪个ctx
                                
                                for ctx_idx, (ctx, config) in enumerate(batch):
                                    if ctx.text_regions:
                                        for region in ctx.text_regions:
                                            if hasattr(region, 'text') and region.text:
                                                all_original_texts.append(region.text)
                                                region_mapping.append((ctx_idx, region))
                                
                                if all_original_texts:
                                    try:
                                        # 重新批量翻译
                                        logger.info(f"Retrying translation for {len(all_original_texts)} regions...")
                                        new_translations = await self._batch_translate_texts(all_original_texts, sample_config, batch[0][0])
                                        
                                        # 更新翻译结果到各个region
                                        for i, (ctx_idx, region) in enumerate(region_mapping):
                                            if i < len(new_translations) and new_translations[i]:
                                                old_translation = region.translation
                                                region.translation = new_translations[i]
                                                logger.debug(f"Region {i+1} translation updated: '{old_translation}' -> '{new_translations[i]}'")
                                        
                                        # 重新收集所有regions并检查目标语言比例
                                        all_batch_regions = []
                                        for ctx, config in batch:
                                            if ctx.text_regions:
                                                all_batch_regions.extend(ctx.text_regions)
                                        
                                        logger.info(f"Re-checking batch-level target language ratio after batch retry {batch_retry_count}...")
                                        batch_lang_check_result = await self._check_target_language_ratio(
                                            all_batch_regions,
                                            sample_config.translator.target_lang,
                                            min_ratio=0.5
                                        )
                                        
                                        if batch_lang_check_result:
                                            logger.info("Batch-level target language check passed")
                                            break
                                        else:
                                            logger.warning("Batch-level target language check still failed")
                                            
                                    except Exception as e:
                                        logger.error(f"Error during batch retry {batch_retry_count}: {e}")
                                        break
                                else:
                                    logger.warning("No text found for batch retry")
                                    break
                            
                            if not batch_lang_check_result:
                                logger.error(f"Batch-level target language check failed after all {max_batch_retry} batch retries")
                    else:
                        logger.info(f"Skipping batch-level target language check: only {len(all_batch_regions)} regions (threshold: 10)")
                    
                    # 统一的成功信息
                    if batch_lang_check_result:
                        logger.info("All translation regions passed post-translation check.")
                    else:
                        logger.warning("Some translation regions failed post-translation check.")
                        
                # 过滤逻辑（简化版本，保留主要过滤条件）
                for ctx, config in batch:
                    if ctx.text_regions:
                        new_text_regions = []
                        for region in ctx.text_regions:
                            should_filter = False
                            filter_reason = ""

                            if not region.translation.strip():
                                should_filter = True
                                filter_reason = "Translation contain blank areas"
                            elif config.translator.translator != Translator.none:
                                if region.translation.isnumeric():
                                    should_filter = True
                                    filter_reason = "Numeric translation"
                                elif not config.translator.translator == Translator.original:
                                    text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                    if text_equal:
                                        should_filter = True
                                        filter_reason = "Translation identical to original"

                            if should_filter:
                                if region.translation.strip():
                                    logger.info(f'Filtered out: {region.translation}')
                                    logger.info(f'Reason: {filter_reason}')
                            else:
                                new_text_regions.append(region)
                        ctx.text_regions = new_text_regions
                        
                results.extend(batch)
                
            except Exception as e:
                # 安全地获取异常信息
                try:
                    error_msg = str(e)
                except Exception:
                    error_msg = f"无法获取异常信息 (异常类型: {type(e).__name__})"
                
                logger.error(f"Error in batch translation: {error_msg}")
                logger.error(traceback.format_exc())
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                for ctx, config in batch:
                    if not ctx.text_regions:  # 检查text_regions是否为None或空
                        continue
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                results.extend(batch)
            
            # ✅ 翻译批次完成后清理内存
            # 清理merged_ctx和batch中的临时数据
            if 'merged_ctx' in locals() and merged_ctx:
                merged_ctx.text_regions = None
                merged_ctx = None
            batch = None
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return results

    async def _concurrent_translate_contexts(self, contexts_with_configs: List[tuple]) -> List[tuple]:
        """
        并发处理翻译步骤，为每个图片单独发送翻译请求，避免合并大批次
        """

        # 在并发模式下，先保存所有页面的原文用于上下文
        batch_original_texts = []  # 存储当前批次的原文
        if self.context_size > 0:
            for i, (ctx, config) in enumerate(contexts_with_configs):
                if ctx.text_regions:
                    # 保存当前页面的原文
                    page_texts = {}
                    for j, region in enumerate(ctx.text_regions):
                        page_texts[j] = region.text
                    batch_original_texts.append(page_texts)

                    # 确保 _original_page_texts 有足够的长度
                    while len(self._original_page_texts) <= len(self.all_page_translations) + i:
                        self._original_page_texts.append({})

                    self._original_page_texts[len(self.all_page_translations) + i] = page_texts
                else:
                    batch_original_texts.append({})

        async def translate_single_context(ctx_config_pair_with_index):
            """翻译单个context的异步函数"""
            ctx, config, page_index, batch_index = ctx_config_pair_with_index
            try:
                if not ctx.text_regions:
                    return ctx, config

                # 收集该context的所有文本
                texts = [region.text for region in ctx.text_regions if region.text is not None]

                if not texts:
                    return ctx, config

                logger.debug(f'Translating {len(texts)} regions for single image in concurrent mode (page {page_index}, batch {batch_index})')

                # 单独翻译这一张图片的文本，传递页面索引和批次索引用于正确的上下文
                translated_texts = await self._batch_translate_texts(
                    texts, config, ctx,
                    page_index=page_index,
                    batch_index=batch_index,
                    batch_original_texts=batch_original_texts
                )

                # 将翻译结果分配回各个region
                for i, region in enumerate(ctx.text_regions):
                    if i < len(translated_texts):
                        region.translation = translated_texts[i]
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                
                # 应用后处理逻辑（括号修正、过滤等）
                if ctx.text_regions:
                    ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                
                # 单页目标语言检查（如果启用）
                if config.translator.enable_post_translation_check and ctx.text_regions:
                    page_lang_check_result = await self._check_target_language_ratio(
                        ctx.text_regions,
                        config.translator.target_lang,
                        min_ratio=0.3  # 对单页使用更宽松的阈值
                    )
                    
                    if not page_lang_check_result:
                        logger.warning("Page-level target language check failed for single image")
                        
                        # 单页重试逻辑
                        max_retry = config.translator.post_check_max_retry_attempts
                        retry_count = 0
                        
                        while retry_count < max_retry and not page_lang_check_result:
                            retry_count += 1
                            logger.info(f"Retrying single image translation {retry_count}/{max_retry}")
                            
                            # 重新翻译
                            original_texts = [region.text for region in ctx.text_regions if hasattr(region, 'text') and region.text]
                            if original_texts:
                                try:
                                    new_translations = await self._batch_translate_texts(original_texts, config, ctx)
                                    
                                    # 更新翻译结果
                                    text_idx = 0
                                    for region in ctx.text_regions:
                                        if hasattr(region, 'text') and region.text and text_idx < len(new_translations):
                                            old_translation = region.translation
                                            region.translation = new_translations[text_idx]
                                            logger.debug(f"Region translation updated: '{old_translation}' -> '{new_translations[text_idx]}'")
                                            text_idx += 1
                                    
                                    # 重新检查
                                    page_lang_check_result = await self._check_target_language_ratio(
                                        ctx.text_regions,
                                        config.translator.target_lang,
                                        min_ratio=0.3
                                    )
                                    
                                    if page_lang_check_result:
                                        logger.info("Single image target language check passed")
                                        break
                                        
                                except Exception as e:
                                    logger.error(f"Error during single image retry {retry_count}: {e}")
                                    break
                            else:
                                break
                        
                        if not page_lang_check_result:
                            logger.warning(f"Single image target language check failed after all {max_retry} retries")
                
                # 过滤逻辑
                if ctx.text_regions:
                    new_text_regions = []
                    for region in ctx.text_regions:
                        should_filter = False
                        filter_reason = ""

                        if not region.translation.strip():
                            should_filter = True
                            filter_reason = "Translation contain blank areas"
                        elif config.translator.translator != Translator.none:
                            if region.translation.isnumeric():
                                should_filter = True
                                filter_reason = "Numeric translation"
                            elif not config.translator.translator == Translator.original:
                                text_equal = region.text.lower().strip() == region.translation.lower().strip()
                                if text_equal:
                                    should_filter = True
                                    filter_reason = "Translation identical to original"

                        if should_filter:
                            if region.translation.strip():
                                logger.info(f'Filtered out: {region.translation}')
                                logger.info(f'Reason: {filter_reason}')
                        else:
                            new_text_regions.append(region)
                    ctx.text_regions = new_text_regions
                
                return ctx, config
                
            except Exception as e:
                logger.error(f"Error in concurrent translation for single image: {e}")
                if not self.ignore_errors:
                    raise
                # 错误时保持原文
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                return ctx, config
        
        # 创建并发任务，为每个任务添加页面索引和批次索引
        tasks = []
        for i, ctx_config_pair in enumerate(contexts_with_configs):
            # 计算当前页面在整个翻译序列中的索引
            page_index = len(self.all_page_translations) + i
            batch_index = i  # 在当前批次中的索引
            ctx_config_pair_with_index = (*ctx_config_pair, page_index, batch_index)
            task = asyncio.create_task(translate_single_context(ctx_config_pair_with_index))
            tasks.append(task)
        
        logger.info(f'Starting concurrent translation of {len(tasks)} images...')
        
        # 等待所有任务完成
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as e:
            logger.error(f"Error in concurrent translation gather: {e}")

            # Best-effort cleanup of task-scoped translators.
            for t in tasks:
                self._release_task_translators(id(t))

            raise
        
        # 处理结果，检查是否有异常
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Image {i+1} concurrent translation failed: {result}")
                if not self.ignore_errors:
                    raise result
                # 创建失败的占位符
                ctx, config = contexts_with_configs[i]
                if ctx.text_regions:
                    for region in ctx.text_regions:
                        region.translation = region.text
                        region.target_lang = config.translator.target_lang
                        region._alignment = config.render.alignment
                        region._direction = config.render.direction
                final_results.append((ctx, config))
            else:
                final_results.append(result)
        
        logger.info(f'Concurrent translation completed: {len(final_results)} images processed')

        # Release task-scoped translators to avoid cache growth in long batch runs.
        for t in tasks:
            self._release_task_translators(id(t))

        return final_results

    async def _batch_translate_texts(self, texts: List[str], config: Config, ctx: Context, batch_contexts: List[Context] = None, page_index: int = None, batch_index: int = None, batch_original_texts: List[dict] = None) -> List[str]:
        """
        批量翻译文本列表，使用现有的翻译器接口

        Args:
            texts: 要翻译的文本列表
            config: 配置对象
            ctx: 上下文对象
            batch_contexts: 批处理上下文列表
            page_index: 当前页面索引，用于并发模式下的上下文计算
            batch_index: 当前页面在批次中的索引
            batch_original_texts: 当前批次的原文数据
        """
        if config.translator.translator == Translator.none:
            return ["" for _ in texts]



        # If this is an OpenAI/Gemini translator, we need to inject context.
        if config.translator.translator in [Translator.openai, Translator.gemini, Translator.openai_hq, Translator.gemini_hq]:
            translator_key = config.translator.translator

            if translator_key == Translator.openai:
                from .translators.openai import OpenAITranslator
                translator = self._get_task_translator(Translator.openai, OpenAITranslator)
            elif translator_key == Translator.gemini:
                from .translators.gemini import GeminiTranslator
                translator = self._get_task_translator(Translator.gemini, GeminiTranslator)
            elif translator_key == Translator.openai_hq:
                from .translators.openai_hq import OpenAIHighQualityTranslator
                translator = self._get_task_translator(Translator.openai_hq, OpenAIHighQualityTranslator)
            elif translator_key == Translator.gemini_hq:
                from .translators.gemini_hq import GeminiHighQualityTranslator
                translator = self._get_task_translator(Translator.gemini_hq, GeminiHighQualityTranslator)

            translator.parse_args(config.translator)
            # 注意：-1 表示无限重试，也是有效值
            # 这里不再覆盖 translator.attempts，让配置中的值生效

            # Reset per-call counters that should not leak across batches.
            if translator_key == Translator.openai_hq and hasattr(translator, '_rate_limit_attempt'):
                translator._rate_limit_attempt = 0
            
            # 传递取消检查回调给翻译器
            if self._cancel_check_callback:
                translator.set_cancel_check_callback(self._cancel_check_callback)

            # 为所有翻译器构建和设置文本上下文（包括HQ翻译器）
            done_trans_pages = self.all_page_translations
            done_orig_pages = self._original_page_texts
            max_done = max(len(done_trans_pages), len(done_orig_pages))

            if self.context_size > 0 and max_done > 0:
                pages_expected = min(self.context_size, max_done)

                non_empty_count = 0
                for i in range(max_done):
                    trans_page = done_trans_pages[i] if i < len(done_trans_pages) else {}
                    orig_page = done_orig_pages[i] if i < len(done_orig_pages) else {}

                    has_translation = any(str(sent).strip() for sent in trans_page.values() if sent is not None)
                    has_original = any(str(sent).strip() for sent in orig_page.values() if sent is not None)

                    if has_translation or has_original:
                        non_empty_count += 1

                pages_used = min(self.context_size, non_empty_count)
                skipped = pages_expected - pages_used
            else:
                pages_used = skipped = 0

            if self.context_size > 0:
                logger.info(f"已启用上下文翻译：使用最近 {self.context_size} 页历史作为参考")

            # 构建上下文（简化版，不使用批次内上下文）
            prev_ctx = self._build_prev_context(
                use_original_text=False,  # 优先使用译文，无译文回退原文
                current_page_index=page_index,
                batch_index=None,  # 不使用批次内上下文
                batch_original_texts=None
            )

            # ✅ 延迟重试单图：把同批次上一张的原文上下文附加到历史上下文后
            local_prev = getattr(ctx, 'local_prev_context', None)
            local_ref_count = 0
            if local_prev:
                prev_ctx = (prev_ctx + "\n\n---\n\n" + local_prev) if prev_ctx else local_prev
                # Count how many same-batch images are referenced
                local_ref_count = local_prev.count("同批次前序原文参考#")

            translator.set_prev_context(prev_ctx)

            # 输出合并后的上下文日志
            if local_ref_count > 0 and pages_used > 0:
                # 延迟重试单图：同时有历史上下文和局部上下文
                context_count = prev_ctx.count("<|")
                logger.info(f"[延迟重试] 携带 {pages_used} 页历史上下文 + 同批次前 {local_ref_count} 张图片原文, 共 {context_count} 条句子")
            elif local_ref_count > 0:
                # 延迟重试单图：只有局部上下文（无历史上下文）
                context_count = prev_ctx.count("<|") if prev_ctx else 0
                logger.info(f"[延迟重试] 携带同批次前 {local_ref_count} 张图片原文, {context_count} 条句子")
            elif pages_used > 0:
                # 正常翻译：只有历史上下文
                context_count = prev_ctx.count("<|")
                logger.info(f"[历史上下文] 携带 {pages_used} 页上下文, {context_count} 条句子作为翻译参考")
            if skipped > 0:
                logger.warning(f"[历史上下文] 跳过了 {skipped} 页无句子的空页")


            # 将config附加到ctx，供翻译器使用（例如AI断句功能）
            ctx.config = config
            
            # ✅ 确保 from_lang 有效（用于按语言加载提示词）
            source_lang = getattr(ctx, 'from_lang', None) or 'auto'
            
            # openai_hq, gemini_hq 等需要传递ctx参数
            if config.translator.translator in [Translator.openai_hq, Translator.gemini_hq]:
                # 所有需要上下文的翻译器都在这里传递ctx
                return await translator._translate(
                    source_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )
            else:
                # 普通OpenAI和Gemini需要ctx参数（用于AI断句）
                return await translator._translate(
                    source_lang,
                    config.translator.target_lang,
                    texts,
                    ctx
                )

        else:
            # 使用通用翻译调度器
            return await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config,
                False,  # use_mtpe removed
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
    
    async def _fallback_batch_text_translate(self, texts: List[str], config: Config, ctx: Context) -> List[str]:
        """
        降级到普通文本翻译（批量处理）
        
        当HQ翻译器遇到安全限制时，降级到对应的普通文本翻译器批量处理
        openai_hq → openai, gemini_hq → gemini
        """
        if not texts:
            return []
        
        # 确定降级的翻译器
        if config.translator.translator == Translator.openai_hq:
            from .translators.openai import OpenAITranslator
            fallback_translator = self._get_task_translator(Translator.openai, OpenAITranslator)
            logger.info(f"[降级翻译] openai_hq → openai，批量处理 {len(texts)} 条文本")
        elif config.translator.translator == Translator.gemini_hq:
            from .translators.gemini import GeminiTranslator
            fallback_translator = self._get_task_translator(Translator.gemini, GeminiTranslator)
            logger.info(f"[降级翻译] gemini_hq → gemini，批量处理 {len(texts)} 条文本")
        else:
            # 非HQ翻译器，使用通用调度
            from .translators import dispatch_translation
            result = await dispatch_translation(
                config.translator.translator_gen,
                texts,
                config,
                False,
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
            return result if result else texts
        
        # 配置降级翻译器
        fallback_translator.parse_args(config.translator)
        fallback_translator.attempts = 2  # 降级翻译允许2次尝试
        fallback_translator._max_total_attempts = 2  # 重置全局尝试次数
        fallback_translator._global_attempt_count = 0  # 重置计数器
        
        if self._cancel_check_callback:
            fallback_translator.set_cancel_check_callback(self._cancel_check_callback)
        
        # 执行批量翻译
        result = await fallback_translator._translate(
            ctx.from_lang,
            config.translator.target_lang,
            texts,
            ctx
        )
        return result if result else texts
    
    async def _fallback_text_translate(self, text: str, config: Config, ctx: Context) -> str:
        """
        降级到普通文本翻译（不使用图片，仅文本）
        
        当高质量翻译（openai_hq/gemini_hq）失败后，降级到对应的普通文本翻译器
        openai_hq → openai, gemini_hq → gemini
        """
        # 确定降级的翻译器
        if config.translator.translator == Translator.openai_hq:
            from .translators.openai import OpenAITranslator
            fallback_translator = self._get_task_translator(Translator.openai, OpenAITranslator)
            logger.info("[降级翻译] openai_hq → openai")
        elif config.translator.translator == Translator.gemini_hq:
            from .translators.gemini import GeminiTranslator
            fallback_translator = self._get_task_translator(Translator.gemini, GeminiTranslator)
            logger.info("[降级翻译] gemini_hq → gemini")
        else:
            # 非HQ翻译器，使用通用调度
            from .translators import dispatch_translation
            result = await dispatch_translation(
                config.translator.translator_gen,
                [text],
                config,
                False,
                ctx,
                'cpu' if self._gpu_limited_memory else self.device
            )
            return result[0] if result else text
        
        # 配置降级翻译器
        fallback_translator.parse_args(config.translator)
        fallback_translator.attempts = 2  # 降级翻译允许2次尝试
        fallback_translator._max_total_attempts = 2  # 重置全局尝试次数
        fallback_translator._global_attempt_count = 0  # 重置计数器
        
        if self._cancel_check_callback:
            fallback_translator.set_cancel_check_callback(self._cancel_check_callback)
        
        # 执行翻译
        result = await fallback_translator._translate(
            ctx.from_lang,
            config.translator.target_lang,
            [text],
            ctx
        )
        return result[0] if result else text
    
    def _translate_error_message(self, error_msg: str) -> str:
        """将英文错误消息转换为中文提示"""
        error_lower = error_msg.lower()
        
        # OpenAI API 错误
        if '404' in error_msg or 'not found' in error_lower:
            return "❌ 翻译失败：API端点未找到(404错误)\n💡 解决方法：\n1. 检查API地址是否正确配置\n2. 如使用第三方API，确认模型名称是否正确\n3. 确认API密钥是否有效"
        
        if '401' in error_msg or 'unauthorized' in error_lower or 'authentication' in error_lower:
            return "❌ 翻译失败：API认证失败(401错误)\n💡 解决方法：\n1. 检查API密钥是否正确\n2. 确认API密钥是否已激活\n3. 检查账户是否有足够余额"
        
        if '429' in error_msg or 'rate limit' in error_lower:
            return "❌ 翻译失败：API请求频率超限(429错误)\n💡 解决方法：\n1. 等待一段时间后重试\n2. 升级API套餐以提高请求限制\n3. 减小批处理大小"
        
        if '500' in error_msg or '502' in error_msg or '503' in error_msg or 'server error' in error_lower:
            return "❌ 翻译失败：API服务器错误(5xx错误)\n💡 解决方法：\n1. 稍后重试\n2. 检查API服务状态页面\n3. 尝试使用其他翻译服务"
        
        if 'timeout' in error_lower or 'timed out' in error_lower:
            return "❌ 翻译失败：请求超时\n💡 解决方法：\n1. 检查网络连接\n2. 增加超时时间设置\n3. 减小批处理大小或图片数量"
        
        if 'connection' in error_lower:
            return "❌ 翻译失败：网络连接错误\n💡 解决方法：\n1. 检查网络连接\n2. 检查防火墙设置\n3. 如使用代理，确认代理配置正确"
        
        if 'quota' in error_lower or 'balance' in error_lower or 'insufficient' in error_lower:
            return "❌ 翻译失败：API配额不足或余额不足\n💡 解决方法：\n1. 充值API账户\n2. 检查账户配额使用情况\n3. 升级API套餐"
        
        if 'budget' in error_lower or 'exceededbudget' in error_lower:
            return "❌ 翻译失败：API预算已用完\n💡 解决方法：\n1. 在API提供商后台增加预算限制\n2. 充值账户余额\n3. 等待下个计费周期\n4. 暂时使用其他翻译服务"
        
        # 通用错误
        return f"❌ 翻译失败：{error_msg}\n💡 建议：\n1. 检查API配置是否正确\n2. 查看完整日志以获取详细错误信息\n3. 尝试更换翻译服务"
            
    async def _apply_post_translation_processing(self, ctx: Context, config: Config) -> List:
        """
        应用翻译后处理逻辑（括号修正、过滤等）
        """
        # 检查text_regions是否为None或空
        if not ctx.text_regions:
            return []
            
        check_items = [
            # 圆括号处理
            ["(", "（", "「", "【"],
            ["（", "(", "「", "【"],
            [")", "）", "」", "】"],
            ["）", ")", "」", "】"],
            
            # 方括号处理
            ["[", "［", "【", "「"],
            ["［", "[", "【", "「"],
            ["]", "］", "】", "」"],
            ["］", "]", "】", "」"],
            
            # 引号处理
            ["「", "“", "‘", "『", "【"],
            ["」", "”", "’", "』", "】"],
            ["『", "“", "‘", "「", "【"],
            ["』", "”", "’", "」", "】"],
            
            # 新增【】处理
            ["【", "(", "（", "「", "『", "["],
            ["】", ")", "）", "」", "』", "]"],
        ]

        replace_items = [
            ["「", "“"],
            ["「", "‘"],
            ["」", "”"],
            ["」", "”"],
            ["【", "["],  
            ["】", "]"],  
        ]

        for region in ctx.text_regions:
            if region.text and region.translation:
                # 引号处理逻辑
                if '『' in region.text and '』' in region.text:
                    quote_type = '『』'
                elif '「' in region.text and '」' in region.text:
                    quote_type = '「」'
                elif '【' in region.text and '】' in region.text: 
                    quote_type = '【】'
                else:
                    quote_type = None
                
                if quote_type:
                    src_quote_count = region.text.count(quote_type[0])
                    dst_dquote_count = region.translation.count('"')
                    dst_fwquote_count = region.translation.count('＂')
                    
                    if (src_quote_count > 0 and
                        (src_quote_count == dst_dquote_count or src_quote_count == dst_fwquote_count) and
                        not region.translation.isascii()):
                        
                        if quote_type == '「」':
                            region.translation = re.sub(r'"([^"]*)"', r'「\1」', region.translation)
                        elif quote_type == '『』':
                            region.translation = re.sub(r'"([^"]*)"', r'『\1』', region.translation)
                        elif quote_type == '【】':  
                            region.translation = re.sub(r'"([^"]*)"', r'【\1】', region.translation)

                # 括号修正逻辑
                for v in check_items:
                    num_src_std = region.text.count(v[0])
                    num_src_var = sum(region.text.count(t) for t in v[1:])
                    num_dst_std = region.translation.count(v[0])
                    num_dst_var = sum(region.translation.count(t) for t in v[1:])
                    
                    if (num_src_std > 0 and
                        num_src_std != num_src_var and
                        num_src_std == num_dst_std + num_dst_var):
                        for t in v[1:]:
                            region.translation = region.translation.replace(t, v[0])

                # 强制替换规则
                # Forced replacement rules
                for v in replace_items:
                    region.translation = region.translation.replace(v[1], v[0])

        # 注意：翻译结果的保存移动到了translate方法的最后，确保保存的是最终结果

        # 应用后字典
        post_dict = load_dictionary(self.post_dict)
        post_replacements = []  
        for region in ctx.text_regions:  
            original = region.translation  
            region.translation = apply_dictionary(region.translation, post_dict)
            if original != region.translation:  
                post_replacements.append(f"{original} => {region.translation}")  

        if post_replacements:  
            logger.info("Post-translation replacements:")  
            for replacement in post_replacements:  
                logger.info(replacement)  
        else:  
            logger.info("No post-translation replacements made.")

        # 单个region幻觉检测
        failed_regions = []
        if config.translator.enable_post_translation_check:
            logger.info("Starting post-translation check...")
            
            # 单个region级别的幻觉检测
            for region in ctx.text_regions:
                if region.translation and region.translation.strip():
                    # 只检查重复内容幻觉
                    if await self._check_repetition_hallucination(
                        region.translation, 
                        config.translator.post_check_repetition_threshold,
                        silent=False
                    ):
                        failed_regions.append(region)
            
            # 对失败的区域进行重试
            if failed_regions:
                logger.warning(f"Found {len(failed_regions)} regions that failed repetition check, starting retry...")
                for region in failed_regions:
                    try:
                        logger.info(f"Retrying translation for region with text: '{region.text}'")
                        new_translation = await self._retry_translation_with_validation(region, config, ctx)
                        if new_translation:
                            old_translation = region.translation
                            region.translation = new_translation
                            logger.info(f"Region retry successful: '{old_translation}' -> '{new_translation}'")
                        else:
                            logger.warning(f"Region retry failed, keeping original translation: '{region.translation}'")
                            break
                    except Exception as e:
                        logger.error(f"Error during region retry: {e}")
                        break
        
        return ctx.text_regions

    async def _complete_translation_pipeline(self, ctx: Context, config: Config) -> Context:
        """
        完成翻译后的处理步骤（掩码细化、修复、渲染）
        """
        await self._report_progress('after-translating')

        # Inpaint Only Mode: Skip pipeline, ctx.result already set
        if hasattr(ctx, 'inpaint_only_complete') and ctx.inpaint_only_complete:
            logger.info("Skipping _complete_translation_pipeline (inpaint only mode already complete)")
            return ctx

        # Colorize Only Mode: Skip validation, ctx.result should already be set
        if self.colorize_only:
            return ctx

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)
        elif ctx.text_regions == 'cancel':
            await self._report_progress('cancelled', True)
            ctx.result = ctx.upscaled
            return await self._revert_upscale(config, ctx)

        # -- Mask refinement
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            try:
                ctx.mask = await self._run_mask_refinement(config, ctx)
            except Exception as _e:  
                logger.error(f"Error during mask-generation:\n{traceback.format_exc()}")  
                if not self.ignore_errors:  
                    raise 
                ctx.mask = ctx.mask_raw if ctx.mask_raw is not None else np.zeros_like(ctx.img_rgb, dtype=np.uint8)[:,:,0]

        if self.verbose and ctx.mask is not None:
            try:
                inpaint_input_img = await dispatch_inpainting(Inpainter.none, ctx.img_rgb, ctx.mask, config.inpainter,config.inpainter.inpainting_size,
                                                              self.device, self.verbose)
                
                # 保存inpaint_input.png
                inpaint_input_path = self._result_path('inpaint_input.png')
                imwrite_unicode(inpaint_input_path, cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR), logger)
                
                # 保存mask_final.png
                mask_final_path = self._result_path('mask_final.png')
                imwrite_unicode(mask_final_path, ctx.mask, logger)
            except Exception as e:
                logger.error(f"Error saving debug images (inpaint_input.png, mask_final.png): {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # -- Inpainting
        await self._report_progress('inpainting')
        try:
            ctx.img_inpainted = await self._run_inpainting(config, ctx)
            
            # ✅ Inpainting完成后强制GC和GPU清理
            self._cleanup_gpu_memory()

        except Exception as _e:
            logger.error(f"Error during inpainting:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            else:
                ctx.img_inpainted = ctx.img_rgb

        if self.verbose:
            try:
                inpainted_path = self._result_path('inpainted.png')
                imwrite_unicode(inpainted_path, cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), logger)
            except Exception as e:
                logger.error(f"Error saving inpainted.png debug image: {e}")
                logger.debug(f"Exception details: {traceback.format_exc()}")

        # 保存inpainted图片到新目录结构（用于可编辑图片功能）
        # 与JSON保存逻辑保持一致：save_text或text_output_file任一满足即保存
        if (self.save_text or self.text_output_file) and hasattr(ctx, 'image_name') and ctx.image_name and ctx.img_inpainted is not None:
            self._save_inpainted_image(ctx.image_name, ctx.img_inpainted)

        # -- Rendering
        await self._report_progress('rendering')

        # 在rendering状态后立即发送文件夹信息，用于前端精确检查final.png
        if hasattr(self, '_progress_hooks') and self._current_image_context:
            folder_name = self._current_image_context['subfolder']
            # 发送特殊格式的消息，前端可以解析
            await self._report_progress(f'rendering_folder:{folder_name}')

        try:
            ctx.img_rendered = await self._run_text_rendering(config, ctx)
        except Exception as _e:
            logger.error(f"Error during rendering:\n{traceback.format_exc()}")
            if not self.ignore_errors:
                raise
            ctx.img_rendered = ctx.img_inpainted

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
        
        # 保存debug文件夹信息到Context中（用于Web模式的缓存访问）
        if self.verbose:
            ctx.debug_folder = self._get_image_subfolder()

        return await self._revert_upscale(config, ctx)
    
    async def _check_repetition_hallucination(self, text: str, threshold: int = 5, silent: bool = False) -> bool:
        """
        检查文本是否包含重复内容（模型幻觉）
        Check if the text contains repetitive content (model hallucination)
        """
        if not text or len(text.strip()) < threshold:
            return False
            
        # 检查字符级重复
        consecutive_count = 1
        prev_char = None
        
        for char in text:
            if char == prev_char:
                consecutive_count += 1
                if consecutive_count >= threshold:
                    if not silent:
                        logger.warning(f'Detected character repetition hallucination: "{text}" - repeated character: "{char}", consecutive count: {consecutive_count}')
                    return True
            else:
                consecutive_count = 1
            prev_char = char
        
        # 检查词语级重复（按字符分割中文，按空格分割其他语言）
        segments = re.findall(r'[\u4e00-\u9fff]|\S+', text)
        
        if len(segments) >= threshold:
            consecutive_segments = 1
            prev_segment = None
            
            for segment in segments:
                if segment == prev_segment:
                    consecutive_segments += 1
                    if consecutive_segments >= threshold:
                        if not silent:
                            logger.warning(f'Detected word repetition hallucination: "{text}" - repeated segment: "{segment}", consecutive count: {consecutive_segments}')
                        return True
                else:
                    consecutive_segments = 1
                prev_segment = segment
        
        # 检查短语级重复
        words = text.split()
        if len(words) >= threshold * 2:
            for i in range(len(words) - threshold + 1):
                phrase = ' '.join(words[i:i + threshold//2])
                remaining_text = ' '.join(words[i + threshold//2:])
                if phrase in remaining_text:
                    phrase_count = text.count(phrase)
                    if phrase_count >= 3:  # 降低短语重复检测阈值
                        if not silent:
                            logger.warning(f'Detected phrase repetition hallucination: "{text}" - repeated phrase: "{phrase}", occurrence count: {phrase_count}')
                        return True
                        
        return False

    async def _check_target_language_ratio(self, text_regions: List, target_lang: str, min_ratio: float = 0.5) -> bool:
        """
        检查翻译结果中目标语言的占比是否达到要求
        使用py3langid进行语言检测
        Check if the target language ratio meets the requirement by detecting the merged translation text
        
        Args:
            text_regions: 文本区域列表
            target_lang: 目标语言代码
            min_ratio: 最小目标语言占比（此参数在新逻辑中不使用，保留为兼容性）
            
        Returns:
            bool: True表示通过检查，False表示未通过
        """
        if not text_regions or len(text_regions) <= 10:
            # 如果区域数量不超过10个，跳过此检查
            return True
            
        # 合并所有翻译文本
        all_translations = []
        for region in text_regions:
            translation = getattr(region, 'translation', '')
            if translation and translation.strip():
                all_translations.append(translation.strip())
        
        if not all_translations:
            logger.debug('No valid translation texts for language ratio check')
            return True
            
        # 将所有翻译合并为一个文本进行检测
        merged_text = ''.join(all_translations)
        
        # logger.info(f'Target language check - Merged text preview (first 200 chars): "{merged_text[:200]}"')
        # logger.info(f'Target language check - Total merged text length: {len(merged_text)} characters')
        # logger.info(f'Target language check - Number of regions: {len(all_translations)}')
        
        # 使用py3langid进行语言检测
        try:
            detected_lang, confidence = langid.classify(merged_text)
            detected_language = ISO_639_1_TO_VALID_LANGUAGES.get(detected_lang, 'UNKNOWN')
            if detected_language != 'UNKNOWN':
                detected_language = detected_language.upper()
            
            # logger.info(f'Target language check - py3langid result: "{detected_lang}" -> "{detected_language}" (confidence: {confidence:.3f})')
        except Exception as e:
            logger.debug(f'py3langid failed for merged text: {e}')
            detected_language = 'UNKNOWN'
            _confidence = -9999
        
        # 检查检测出的语言是否为目标语言
        is_target_lang = (detected_language == target_lang.upper())
        
        # logger.info(f'Target language check: Detected language "{detected_language}" using py3langid (confidence: {confidence:.3f})')
        # logger.info(f'Target language check: Target is "{target_lang.upper()}"')
        # logger.info(f'Target language check result: {"PASSED" if is_target_lang else "FAILED"}')
        
        return is_target_lang

    async def _validate_translation(self, original_text: str, translation: str, target_lang: str, config, ctx: Context = None, silent: bool = False, page_lang_check_result: bool = None) -> bool:
        """
        验证翻译质量（包含目标语言比例检查和幻觉检测）
        Validate translation quality (includes target language ratio check and hallucination detection)
        
        Args:
            page_lang_check_result: 页面级目标语言检查结果，如果为None则进行检查，如果已有结果则直接使用
        """
        if not config.translator.enable_post_translation_check:
            return True
            
        if not translation or not translation.strip():
            return True
        
        # 1. 目标语言比例检查（页面级别）
        if page_lang_check_result is None and ctx and ctx.text_regions and len(ctx.text_regions) > 10:
            # 进行页面级目标语言检查
            page_lang_check_result = await self._check_target_language_ratio(
                ctx.text_regions,
                target_lang,
                min_ratio=0.5
            )
            
        # 如果页面级检查失败，直接返回失败
        if page_lang_check_result is False:
            if not silent:
                logger.debug("Target language ratio check failed for this region")
            return False
        
        # 2. 检查重复内容幻觉（region级别）
        if await self._check_repetition_hallucination(
            translation, 
            config.translator.post_check_repetition_threshold,
            silent
        ):
            return False
                
        return True

    async def _retry_translation_with_validation(self, region, config: Config, ctx: Context) -> str:
        """
        带验证的重试翻译
        Retry translation with validation
        """
        original_translation = region.translation
        max_attempts = config.translator.post_check_max_retry_attempts
        
        for attempt in range(max_attempts):
            # 验证当前翻译 - 在重试过程中只检查单个region（幻觉检测），不进行页面级检查
            is_valid = await self._validate_translation(
                region.text, 
                region.translation, 
                config.translator.target_lang,
                config,
                ctx=None,  # 不传ctx避免页面级检查
                silent=True,  # 重试过程中禁用日志输出
                page_lang_check_result=True  # 传入True跳过页面级检查，只做region级检查
            )
            
            if is_valid:
                if attempt > 0:
                    logger.info(f'Post-translation check passed (Attempt {attempt + 1}/{max_attempts}): "{region.translation}"')
                return region.translation
            
            # 如果不是最后一次尝试，进行重新翻译
            if attempt < max_attempts - 1:
                logger.warning(f'Post-translation check failed (Attempt {attempt + 1}/{max_attempts}), re-translating: "{region.text}"')
                
                try:
                    # 单独重新翻译这个文本区域
                    if config.translator.translator != Translator.none:
                        from .translators import dispatch
                        retranslated = await dispatch(
                            config.translator.translator_gen,
                            [region.text],
                            config.translator,
                            False,  # use_mtpe removed
                            ctx,
                            'cpu' if self._gpu_limited_memory else self.device
                        )
                        if retranslated:
                            region.translation = retranslated[0]
                            
                            # 应用格式化处理
                            if config.render.uppercase:
                                region.translation = region.translation.upper()
                            elif config.render.lowercase:
                                region.translation = region.translation.lower()
                                
                            logger.info(f'Re-translation finished: "{region.text}" -> "{region.translation}"')
                        else:
                            logger.warning(f'Re-translation failed, keeping original translation: "{original_translation}"')
                            region.translation = original_translation
                            break
                    else:
                        logger.warning('Translator is none, cannot re-translate.')
                        break
                        
                except Exception as e:
                    logger.error(f'Error during re-translation: {e}')
                    region.translation = original_translation
                    break
            else:
                logger.warning(f'Post-translation check failed, maximum retry attempts ({max_attempts}) reached, keeping original translation: "{original_translation}"')
                region.translation = original_translation
        
        return region.translation

    def _update_translation_map(self, source_path: str, translated_path: str):
        """在输出目录创建或更新 translation_map.json"""
        try:
            output_dir = os.path.dirname(translated_path)
            map_path = os.path.join(output_dir, 'translation_map.json')
            
            # 规范化路径以确保一致性
            source_path_norm = os.path.normpath(source_path)
            translated_path_norm = os.path.normpath(translated_path)

            translation_map = {}
            if os.path.exists(map_path):
                with open(map_path, 'r', encoding='utf-8') as f:
                    try:
                        translation_map = json.load(f)
                    except json.JSONDecodeError:
                        logger.warning(f"Could not decode {map_path}, creating a new one.")
            
            # 使用翻译后的路径作为键，确保唯一性
            translation_map[translated_path_norm] = source_path_norm
            
            with open(map_path, 'w', encoding='utf-8') as f:
                json.dump(translation_map, f, ensure_ascii=False, indent=4)

        except Exception as e:
            logger.error(f"Failed to update translation map: {e}")

    def _load_watermark_filter_rules(self):
        """
        加载水印过滤规则（支持本地自定义配置文件）
        优先从 examples/config/watermark_filter.json 读取，如果不存在则使用默认规则
        """
        import time
        import re
        
        # 缓存5分钟，避免频繁读取文件
        cache_ttl = 300
        current_time = time.time()
        if (MangaTranslator._watermark_filter_cache is not None and 
            current_time - MangaTranslator._watermark_filter_cache_time < cache_ttl):
            return MangaTranslator._watermark_filter_cache
        
        # 默认规则
        default_partial = [
            '웹툰왕국뉴토끼', 'newtoki', 'manga18', '뉴토끼', 'webtoon', 'toonkor',
            'doujindesu', 'javdesu', 'igodesu', 'discord.gg', 'bit.ly', 'bit.lv',
            'sawerdesu', 'baca', 'download', 'reupload', 'mirror', 'copyright',
            'scan me', 'qr code', 'web lain', 'website pal', 'kunjungi', 'link website',
        ]
        default_exact = ['pfftt situ waras', 'situ waras']
        
        # 尝试从配置文件加载
        config_path = os.path.join(BASE_PATH, 'examples', 'config', 'watermark_filter.json')
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                # 新格式：groups（各组规则合并，保持运行时行为不变）
                if isinstance(config, dict) and isinstance(config.get('groups'), list):
                    merged_partial = []
                    merged_exact = []
                    for g in config.get('groups'):
                        if not isinstance(g, dict):
                            continue
                        gp = g.get('partial_match_patterns', []) or []
                        ge = g.get('exact_match_patterns', []) or []
                        if isinstance(gp, list):
                            merged_partial.extend(gp)
                        if isinstance(ge, list):
                            merged_exact.extend(ge)
                    partial_patterns = merged_partial or default_partial
                    exact_patterns = merged_exact or default_exact
                else:
                    partial_patterns = config.get('partial_match_patterns', default_partial)
                    exact_patterns = config.get('exact_match_patterns', default_exact)

                logger.debug(f"[水印过滤] 已加载自定义规则: {len(partial_patterns)}个部分匹配, {len(exact_patterns)}个完整匹配")
            else:
                partial_patterns = default_partial
                exact_patterns = default_exact
        except Exception as e:
            logger.warning(f"[水印过滤] 加载配置文件失败，使用默认规则: {e}")
            partial_patterns = default_partial
            exact_patterns = default_exact

        def _normalize_text(s: str) -> str:
            s = re.sub(r'[^\w\s]', ' ', s.lower()).strip()
            return ' '.join(s.split())

        def _compile_rule(rule, mode: str):
            """将规则编译为统一结构。

            返回字典：
            - type: 'substr' | 'regex'
            - pattern: str
            - case_sensitive: bool
            - compiled: Pattern | None
            - norm: str | None
            """
            if isinstance(rule, str):
                pat = _normalize_text(rule)
                return {
                    'type': 'substr',
                    'pattern': rule,
                    'case_sensitive': False,
                    'compiled': None,
                    'norm': pat,
                    'mode': mode,
                }

            if isinstance(rule, dict):
                pat = str(rule.get('pattern', '')).strip()
                if not pat:
                    return None
                is_regex = bool(rule.get('regex', False))
                case_sensitive = bool(rule.get('case_sensitive', False))

                if is_regex:
                    flags = 0
                    if not case_sensitive:
                        flags |= re.IGNORECASE
                    try:
                        compiled = re.compile(pat, flags)
                    except re.error as e:
                        logger.warning(f"[水印过滤] 正则编译失败，已忽略: {pat} ({e})")
                        return None
                    return {
                        'type': 'regex',
                        'pattern': pat,
                        'case_sensitive': case_sensitive,
                        'compiled': compiled,
                        'norm': None,
                        'mode': mode,
                    }

                # 非regex对象：按子串规则处理，默认不区分大小写
                norm = _normalize_text(pat if not case_sensitive else pat)
                return {
                    'type': 'substr',
                    'pattern': pat,
                    'case_sensitive': case_sensitive,
                    'compiled': None,
                    'norm': _normalize_text(pat) if not case_sensitive else pat,
                    'mode': mode,
                }

            # 未知类型
            return None

        compiled_partial = []
        for r in partial_patterns or []:
            cr = _compile_rule(r, mode='partial')
            if cr:
                compiled_partial.append(cr)

        compiled_exact = []
        for r in exact_patterns or []:
            cr = _compile_rule(r, mode='exact')
            if cr:
                compiled_exact.append(cr)

        MangaTranslator._watermark_filter_cache = (compiled_partial, compiled_exact)
        MangaTranslator._watermark_filter_cache_time = current_time
        return MangaTranslator._watermark_filter_cache

    async def _translate_batch_replace_translation(self, images_with_configs: List[tuple], save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
        """
        替换翻译模式：从翻译图提取OCR结果并应用到生肉图
        
        流程：
        1. 对生肉图执行检测+OCR，过滤低置信度区域
        2. 查找对应的翻译图，执行检测+OCR
        3. 区域匹配（考虑尺寸缩放）
        4. 使用匹配的区域执行修复和渲染
        
        Args:
            images_with_configs: List of (image, config) tuples
            save_info: 保存配置
            global_offset: 全局偏移量
            global_total: 全局总图片数
        """
        from .utils.replace_translation import translate_batch_replace_translation
        return await translate_batch_replace_translation(self, images_with_configs, save_info, global_offset, global_total)

    async def _translate_batch_high_quality(self, images_with_configs: List[tuple], save_info: dict = None, global_offset: int = 0, global_total: int = None) -> List[Context]:
        """
        高质量翻译模式：按批次滚动处理，每批独立完成预处理、翻译、渲染全流程。
        如果提供了save_info，则在每批处理后直接保存。
        
        Args:
            images_with_configs: List of (image, config) tuples
            save_info: 保存配置
            global_offset: 全局偏移量，用于显示正确的图片编号
            global_total: 全局总图片数，用于显示正确的总批次数
        """
        batch_size = self.batch_size if self.batch_size > 1 else 3  # 统一使用batch_size参数
        logger.info(f"Starting high quality translation in rolling batch mode with batch size: {batch_size}")
        results = []
        
        # 如果提供了全局总数，使用它来计算总批次数；否则使用当前批次的图片数
        display_total = global_total if global_total is not None else len(images_with_configs)
        
        total_images = len(images_with_configs)
        for batch_start in range(0, total_images, batch_size):
            # 检查是否被取消
            await asyncio.sleep(0)
            self._check_cancelled()  # 检查取消标志

            batch_end = min(batch_start + batch_size, total_images)
            current_batch_images = images_with_configs[batch_start:batch_end]

            # 计算全局图片编号（考虑前端分批加载的偏移量）
            global_batch_start = global_offset + batch_start + 1
            global_batch_end = global_offset + batch_end
            global_batch_num = (global_offset + batch_start) // batch_size + 1
            global_total_batches = (display_total + batch_size - 1) // batch_size
            
            logger.info(f"Processing rolling batch {global_batch_num}/{global_total_batches} (images {global_batch_start}-{global_batch_end})")
            
            # 报告批次进度给前端
            await self._report_progress(f"batch:{global_batch_start}:{global_batch_end}:{display_total}")

            # 阶段一：预处理当前批次
            preprocessed_contexts = []
            for i, (image, config) in enumerate(current_batch_images):
                # 检查是否被取消
                await asyncio.sleep(0)
                self._check_cancelled()  # 检查取消标志
                try:
                    self._set_image_context(config, image)
                    # ✅ 保存context以便渲染阶段复用，避免生成两个文件夹
                    from .utils.generic import get_image_md5
                    image_md5 = get_image_md5(image)
                    self._save_current_image_context(image_md5)
                    ctx = await self._translate_until_translation(image, config)
                    if hasattr(image, 'name'):
                        ctx.image_name = image.name
                    preprocessed_contexts.append((ctx, config))
                except Exception as e:
                    logger.error(f"Error pre-processing image {i+1} in batch: {e}")
                    ctx = Context()
                    ctx.input = image
                    ctx.text_regions = []
                    if hasattr(image, 'name'):
                        ctx.image_name = image.name
                    preprocessed_contexts.append((ctx, config))

            # 阶段二：翻译当前批次
            batch_data = []
            global_text_index = 1  # 全局文本编号从1开始（与提示词中的编号一致）
            for ctx, config in preprocessed_contexts:
                num_regions = len(ctx.text_regions) if ctx.text_regions else 0
                # 为当前图片生成全局连续的文本编号
                text_order = list(range(global_text_index, global_text_index + num_regions))
                global_text_index += num_regions
                
                # 获取超分后的尺寸（用于坐标转换）
                upscaled_size = None
                if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                    upscaled_size = ctx.img_rgb.shape[:2]  # (height, width)
                
                image_data = {
                    'image': ctx.input,
                    'text_regions': ctx.text_regions if ctx.text_regions else [],
                    'original_texts': [region.text for region in ctx.text_regions if region.text is not None] if ctx.text_regions else [],
                    'text_order': text_order,
                    'upscaled_size': upscaled_size
                }
                batch_data.append(image_data)

            if any(data['original_texts'] for data in batch_data):
                try:
                    sample_config = preprocessed_contexts[0][1] if preprocessed_contexts else None
                    if sample_config:
                        # ✅ 创建新的Context用于enhanced_ctx，避免污染第一张图片的context
                        enhanced_ctx = Context()
                        # 复制第一张图片的必要属性
                        if preprocessed_contexts:
                            first_ctx = preprocessed_contexts[0][0]
                            if hasattr(first_ctx, 'input'):
                                enhanced_ctx.input = first_ctx.input
                            if hasattr(first_ctx, 'img_rgb'):
                                enhanced_ctx.img_rgb = first_ctx.img_rgb
                            # ✅ 复制源语言（用于按语言加载提示词）
                            if hasattr(first_ctx, 'from_lang'):
                                enhanced_ctx.from_lang = first_ctx.from_lang
                            else:
                                enhanced_ctx.from_lang = 'auto'
                        
                        enhanced_ctx.high_quality_batch_data = batch_data

                        # ✅ 合并所有页面的text_regions到enhanced_ctx（用于AI断句）
                        all_regions = []
                        for ctx, _ in preprocessed_contexts:
                            if ctx.text_regions:
                                all_regions.extend(ctx.text_regions)
                        enhanced_ctx.text_regions = all_regions
                        logger.debug(f"[HQ Batch] Merged {len(all_regions)} text regions from {len(preprocessed_contexts)} pages")

                        # Centralized prompt loading logic
                        enhanced_ctx = await self._load_and_prepare_prompts(sample_config, enhanced_ctx)
                        
                        all_texts = [text for data in batch_data for text in data['original_texts']]
                        text_mapping = [(img_idx, region_idx) for img_idx, data in enumerate(batch_data) for region_idx, _ in enumerate(data['original_texts'])]
                        
                        logger.info(f"Sending batch data with {len(preprocessed_contexts)} images, {len(all_texts)} text regions to high quality translator")
                        
                        # 计算当前批次在所有页面中的索引（用于上下文）
                        # batch_start 是当前批次的起始索引（相对于本次translate_batch调用的所有图片）
                        # self.all_page_translations 包含之前所有已翻译的页面
                        page_index = len(self.all_page_translations) + batch_start
                        
                        # 高质量翻译模式：批量翻译
                        translated_texts = await self._batch_translate_texts(
                            all_texts, 
                            sample_config, 
                            enhanced_ctx,
                            page_index=page_index,
                            batch_index=None,  # 不使用批次内上下文
                            batch_original_texts=None
                        )
                        
                        for text_idx, (img_idx, region_idx) in enumerate(text_mapping):
                            if text_idx < len(translated_texts):
                                ctx, config = preprocessed_contexts[img_idx]
                                if ctx.text_regions and region_idx < len(ctx.text_regions):
                                    region = ctx.text_regions[region_idx]
                                    region.translation = translated_texts[text_idx]
                                    region.target_lang = config.translator.target_lang
                                    region._alignment = config.render.alignment
                                    region._direction = config.render.direction
                        
                        for ctx, config in preprocessed_contexts:
                            if ctx.text_regions:
                                ctx.text_regions = await self._apply_post_translation_processing(ctx, config)
                        
                        # ✅ 立即保存当前批次的翻译结果到all_page_translations，供下一个批次使用上下文
                        for ctx, config in preprocessed_contexts:
                            if ctx.text_regions:
                                # 保存译文
                                page_trans = {}
                                for region in ctx.text_regions:
                                    if region.translation:
                                        page_trans[region.text] = region.translation
                                self.all_page_translations.append(page_trans)
                                logger.debug(f"[HQ Batch Context] Saved {len(page_trans)} translations for next batch context")
                                
                                # 保存原文（用于并发模式的上下文）
                                page_original_texts = {i: (r.text_raw if hasattr(r, "text_raw") else r.text)
                                                      for i, r in enumerate(ctx.text_regions)}
                                self._original_page_texts.append(page_original_texts)
                                logger.debug(f"[HQ Batch Context] Saved {len(page_original_texts)} original texts for next batch context")
                                
                                # Prune history to prevent memory leak
                                self._prune_context_history()
                                
                except Exception as e:
                    logger.error(f"Error in high quality batch translation: {e}")
                    # 重新抛出异常，终止翻译流程
                    raise
            # --- NEW: Handle Generate and Export for High-Quality Mode ---
            if self.generate_and_export:
                logger.info("'Generate and Export' mode enabled for high-quality translation. Skipping rendering.")
                for ctx, config in preprocessed_contexts:
                    # Ensure JSON is saved first
                    if ctx.text_regions and hasattr(ctx, 'image_name') and ctx.image_name:
                        self._save_text_to_file(ctx.image_name, ctx, config)

                        # Export the clean text using the template
                        try:
                            json_path = find_json_path(ctx.image_name)
                            if json_path and os.path.exists(json_path):
                                from desktop_qt_ui.services.workflow_service import generate_translated_text, get_template_path_from_config
                                template_path = get_template_path_from_config()
                                if template_path and os.path.exists(template_path):
                                    # 导出翻译
                                    translated_result = generate_translated_text(json_path, template_path)
                                    logger.info(f"Translated text export for {os.path.basename(ctx.image_name)}: {translated_result}")
                                else:
                                    logger.warning(f"Template file not found, cannot export clean text for {os.path.basename(ctx.image_name)}: {template_path}")
                            else:
                                logger.warning(f"JSON file not found, cannot export clean text for {os.path.basename(ctx.image_name)}")
                        except Exception as e:
                            logger.error(f"Failed to export clean text for {os.path.basename(ctx.image_name)} in HQ mode: {e}")

                    # ✅ 标记成功（导出翻译完成）
                    ctx.success = True
                    results.append(ctx)
                
                # ✅ 批次完成后立即清理内存
                self._cleanup_batch_memory(
                    current_batch_images=current_batch_images,
                    preprocessed_contexts=preprocessed_contexts,
                    keep_results=True
                )

                # Release task-scoped translators at the batch boundary.
                self._release_task_translators(self._get_current_task_id())
                
                continue # BUG FIX: Continue to the next batch instead of returning

            # 阶段三：渲染并保存当前批次
            for ctx, config in preprocessed_contexts:
                # 检查是否被取消
                await asyncio.sleep(0)
                self._check_cancelled()  # 检查取消标志
                try:
                    if hasattr(ctx, 'input'):
                        from .utils.generic import get_image_md5
                        image_md5 = get_image_md5(ctx.input)
                        if not self._restore_image_context(image_md5):
                            self._set_image_context(config, ctx.input)
                    
                    # Colorize/Upscale/Inpaint Only Mode: Skip rendering pipeline
                    if not self.colorize_only and not self.upscale_only and not self.inpaint_only:
                        ctx = await self._complete_translation_pipeline(ctx, config)
                    
                    # --- BEGIN SAVE LOGIC ---
                    if save_info and ctx.result:
                        try:
                            self._save_and_cleanup_context(ctx, save_info, config, "HQ")
                        except Exception as save_err:
                            logger.error(f"Error saving high-quality result for {os.path.basename(ctx.image_name)}: {save_err}")
                            import traceback
                            logger.error(traceback.format_exc())
                    # --- END SAVE LOGIC ---

                    # 只在save_text或text_output_file启用时保存JSON（包括空的text_regions）
                    if (self.save_text or self.text_output_file) and hasattr(ctx, 'text_regions') and ctx.text_regions is not None and hasattr(ctx, 'image_name') and ctx.image_name:
                        # 使用循环变量中的config，而不是从ctx中获取
                        self._save_text_to_file(ctx.image_name, ctx, config)

                    # ✅ 标记成功
                    ctx.success = True
                    
                    # ✅ 清理中间处理图像（保留text_regions等元数据）
                    if hasattr(ctx, 'input') and ctx.input is not None:
                        if hasattr(ctx.input, 'close'):
                            try:
                                ctx.input.close()
                            except:
                                pass
                        ctx.input = None
                    if hasattr(ctx, 'img_rgb'):
                        ctx.img_rgb = None
                    if hasattr(ctx, 'img_inpainted'):
                        ctx.img_inpainted = None
                    if hasattr(ctx, 'img_rendered'):
                        ctx.img_rendered = None
                    if hasattr(ctx, 'img_colorized'):
                        ctx.img_colorized = None
                    if hasattr(ctx, 'img_alpha'):
                        ctx.img_alpha = None
                    if hasattr(ctx, 'mask'):
                        ctx.mask = None
                    if hasattr(ctx, 'mask_raw'):
                        ctx.mask_raw = None
                    # 注意：high_quality_batch_data 由翻译器的统一清理方法处理
                    
                    results.append(ctx)
                except Exception as e:
                    logger.error(f"Error rendering image: {e}")
                    # 渲染失败时抛出异常，而不是继续处理
                    raise RuntimeError(f"Rendering failed for {os.path.basename(ctx.image_name) if hasattr(ctx, 'image_name') else 'Unknown'}: {e}") from e
            
            # ✅ 批次完成后立即清理内存（但保留翻译历史供下一批次使用）
#             import gc
            # 1. 清理batch_data中的图像引用
            for data in batch_data:
                if 'image' in data:
                    data['image'] = None
            batch_data.clear()
            
            # 2. 使用统一清理方法
            self._cleanup_batch_memory(
                preprocessed_contexts=preprocessed_contexts,
                keep_results=True
            )

            # Release task-scoped translators at the batch boundary.
            self._release_task_translators(self._get_current_task_id())
            
            logger.debug(f'[MEMORY] Batch {batch_start//batch_size + 1} cleanup completed (kept translation history for context)')

        logger.info(f"High quality translation completed: processed {len(results)} images")
        return results
