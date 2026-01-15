#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
命令行翻译工具 - 直接使用 UI 层的翻译逻辑
支持子进程模式进行内存管理和断点续传
"""
import os
import sys
import argparse
import asyncio
import multiprocessing
from pathlib import Path

# 添加项目根目录到 Python 路径
ROOT_DIR = Path(__file__).parent.parent.parent  # 上两级目录
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(ROOT_DIR / 'desktop_qt_ui'))

# 内存管理默认值
DEFAULT_MEMORY_THRESHOLD_MB = 8000  # 默认8GB
DEFAULT_BATCH_SIZE_PER_RESTART = 50  # 每处理N张图片后检查


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='漫画翻译命令行工具 - 使用与 UI 相同的翻译逻辑',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 翻译单个图片
  python -m manga_translator local -i manga.jpg
  
  # 翻译文件夹
  python -m manga_translator local -i ./manga_folder/ -o ./output/
  
  # 使用自定义配置
  python -m manga_translator local -i manga.jpg --config my_config.json
  
  # 启用子进程模式（支持内存管理，每50张图片重启子进程释放内存）
  python -m manga_translator local -i ./manga_folder/ --subprocess
  
  # 自定义内存管理参数（每20张图片重启）
  python -m manga_translator local -i ./manga_folder/ --subprocess --batch-per-restart 20
  
  # 从断点继续（需要配合 --subprocess）
  python -m manga_translator local -i ./manga_folder/ --subprocess --resume
  
  # 详细日志
  python -m manga_translator local -i manga.jpg -v
        """
    )
    
    parser.add_argument('-i', '--input', required=True, nargs='+',
                        help='输入图片或文件夹路径')
    parser.add_argument('-o', '--output', default=None,
                        help='输出目录（默认：同目录加 -translated 后缀）')
    parser.add_argument('--config', default=None,
                        help='配置文件路径（默认：examples/config.json）')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='显示详细日志')
    parser.add_argument('--overwrite', action='store_true',
                        help='覆盖已存在的文件')
    
    # 内存管理参数
    parser.add_argument('--subprocess', action='store_true',
                        help='启用子进程模式（支持内存管理和断点续传）')
    parser.add_argument('--memory-limit', type=int, default=DEFAULT_MEMORY_THRESHOLD_MB,
                        help=f'绝对内存限制（MB），超过后自动重启子进程（默认：{DEFAULT_MEMORY_THRESHOLD_MB}，0表示不限制）')
    parser.add_argument('--memory-percent', type=int, default=80,
                        help='内存百分比限制，超过系统总内存的这个百分比时重启（默认：80）')
    parser.add_argument('--batch-per-restart', type=int, default=DEFAULT_BATCH_SIZE_PER_RESTART,
                        help=f'每处理N张图片后重启子进程释放内存（默认：{DEFAULT_BATCH_SIZE_PER_RESTART}）')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断的位置继续（需要配合 --subprocess 使用）')
    
    # 并发模式参数
    parser.add_argument('--concurrent', action='store_true',
                        help='启用并发流水线模式（检测、OCR、翻译、渲染并行处理）')
    
    return parser.parse_args()


async def translate_files(input_paths, output_dir, config_service, verbose=False, overwrite=False, args=None):
    """翻译文件（使用 UI 层的逻辑）"""
    
    # 延迟导入，避免 --help 时加载所有模块
    from desktop_qt_ui.services.file_service import FileService
    from manga_translator import MangaTranslator, Config
    from manga_translator.utils import init_logging, set_log_level, get_logger
    from PIL import Image
    import logging
    import logging.handlers
    
    init_logging()
    if verbose:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.INFO)
    
    # 确保 manga_translator 的日志也输出到控制台
    manga_logger = logging.getLogger('manga_translator')
    manga_logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    
    # 添加控制台 handler（如果还没有）
    if not any(isinstance(h, logging.StreamHandler) for h in manga_logger.handlers):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        formatter = logging.Formatter('[%(name)s] %(message)s')
        console_handler.setFormatter(formatter)
        manga_logger.addHandler(console_handler)
    
    # 添加文件日志（与 Qt UI 相同位置和格式）
    from datetime import datetime
    
    log_dir = ROOT_DIR / 'result'
    log_dir.mkdir(exist_ok=True)
    
    # 生成带时间戳的日志文件名（与 Qt UI 格式一致）
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    log_file = log_dir / f'log_{timestamp}.txt'
    
    # 检查是否已添加文件 handler
    has_file_handler = any(
        isinstance(h, logging.FileHandler)
        for h in logging.root.handlers
    )
    
    if not has_file_handler:
        file_handler = logging.FileHandler(
            str(log_file),
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logging.root.addHandler(file_handler)
        print(f"📝 日志文件: {log_file}")
    
    logger = get_logger('local')
    
    # 获取配置
    config = config_service.get_config()
    config_dict = config.model_dump()
    
    # 从配置文件读取 CLI 设置，命令行参数可以覆盖
    cli_config = config_dict.get('cli', {})
    
    # 应用命令行参数（如果提供了命令行参数，则覆盖配置文件）
    if verbose:
        cli_config['verbose'] = True
    else:
        verbose = cli_config.get('verbose', False)
    
    # overwrite: 命令行参数优先，否则使用配置文件
    if overwrite:
        cli_config['overwrite'] = True
    else:
        overwrite = cli_config.get('overwrite', False)
    
    # use_gpu: 命令行参数优先
    if hasattr(args, 'use_gpu') and args.use_gpu is not None:
        cli_config['use_gpu'] = args.use_gpu
    
    # format: 命令行参数优先
    if hasattr(args, 'format') and args.format is not None:
        cli_config['format'] = args.format
    
    # batch_size: 命令行参数优先
    if hasattr(args, 'batch_size') and args.batch_size is not None:
        cli_config['batch_size'] = args.batch_size
    
    # attempts: 命令行参数优先
    if hasattr(args, 'attempts') and args.attempts is not None:
        cli_config['attempts'] = args.attempts
    
    # concurrent: 命令行参数优先，否则使用配置文件中的值
    if hasattr(args, 'concurrent') and args.concurrent:
        cli_config['batch_concurrent'] = True
    # 如果命令行没有指定，保留配置文件中的 batch_concurrent 值（已在 cli_config 中）
    
    config_dict['cli'] = cli_config
    
    # 检查是否有不兼容并行的特殊模式
    load_text = cli_config.get('load_text', False)
    template = cli_config.get('template', False)
    save_text = cli_config.get('save_text', False)
    generate_and_export = cli_config.get('generate_and_export', False)
    colorize_only = cli_config.get('colorize_only', False)
    upscale_only = cli_config.get('upscale_only', False)
    inpaint_only = cli_config.get('inpaint_only', False)
    replace_translation = cli_config.get('replace_translation', False)
    
    is_template_save_mode = template and save_text
    has_incompatible_mode = (
        load_text or 
        is_template_save_mode or 
        generate_and_export or 
        colorize_only or 
        upscale_only or 
        inpaint_only or
        replace_translation
    )
    
    # 如果有不兼容模式，强制禁用并行
    if cli_config.get('batch_concurrent', False) and has_incompatible_mode:
        incompatible_modes = []
        if load_text:
            incompatible_modes.append("导入翻译")
        if is_template_save_mode:
            incompatible_modes.append("导出原文")
        if generate_and_export:
            incompatible_modes.append("导出翻译")
        if colorize_only:
            incompatible_modes.append("仅上色")
        if upscale_only:
            incompatible_modes.append("仅超分")
        if inpaint_only:
            incompatible_modes.append("仅修复")
        if replace_translation:
            incompatible_modes.append("替换翻译")
        
        print(f"⚠️  并发流水线已禁用：当前模式 [{', '.join(incompatible_modes)}] 不支持并发处理")
        cli_config['batch_concurrent'] = False
        config_dict['cli'] = cli_config
    
    print(f"\n{'='*60}")
    print(f"翻译器: {config_dict['translator']['translator']}")
    print(f"目标语言: {config_dict['translator']['target_lang']}")
    print(f"使用 GPU: {cli_config.get('use_gpu', True)}")
    print(f"批量大小: {cli_config.get('batch_size', 1)}")
    print(f"并发模式: {'启用' if cli_config.get('batch_concurrent', False) else '禁用'}")
    print(f"覆盖已存在文件: {overwrite}")
    print(f"输出格式: {cli_config.get('format') or '保持原格式'}")
    print(f"保存质量: {cli_config.get('save_quality', 95)}")
    print(f"{'='*60}\n")
    
    # 收集所有图片文件
    file_service = FileService()
    all_files = []
    
    # 分离文件和文件夹
    folders = []
    individual_files = []
    
    for input_path in input_paths:
        input_path = os.path.abspath(input_path)
        if os.path.isfile(input_path):
            individual_files.append(input_path)
        elif os.path.isdir(input_path):
            folders.append(input_path)
    
    # 对文件夹进行自然排序（与UI模式保持一致）
    folders.sort(key=file_service._natural_sort_key)
    
    # 按文件夹分组处理
    for folder in folders:
        # 递归获取文件夹中的所有图片（已经使用自然排序）
        folder_files = file_service.get_image_files_from_folder(folder, recursive=True)
        all_files.extend(folder_files)
    
    # 处理单独添加的文件（使用自然排序）
    individual_files.sort(key=file_service._natural_sort_key)
    all_files.extend(individual_files)
    
    if not all_files:
        print("❌ 未找到图片文件")
        return
    
    print(f"📁 找到 {len(all_files)} 个图片文件\n")
    
    # 确定输出目录
    if output_dir:
        final_output_dir = os.path.abspath(output_dir)
    else:
        # 使用配置文件中的输出目录，或默认规则
        if config_dict.get('app', {}).get('last_output_path'):
            final_output_dir = config_dict['app']['last_output_path']
        else:
            # 默认：在第一个输入路径旁边创建 -translated 文件夹
            first_input = input_paths[0]
            if os.path.isdir(first_input):
                final_output_dir = first_input.rstrip('/\\') + '-translated'
            else:
                final_output_dir = os.path.dirname(first_input)
    
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"📤 输出目录: {final_output_dir}\n")
    
    # 准备翻译参数（像 UI 一样）
    translator_params = config_dict.get('cli', {}).copy()
    # 保存 cli 中的关键参数，避免被覆盖
    cli_attempts = translator_params.get('attempts', -1)
    translator_params.update(config_dict)
    # 恢复 cli 参数（如果 config_dict 中没有 attempts）
    if 'attempts' not in config_dict:
        translator_params['attempts'] = cli_attempts
    
    # 处理 font_path
    font_filename = config_dict.get('render', {}).get('font_path')
    if font_filename and not os.path.isabs(font_filename):
        font_full_path = os.path.join(ROOT_DIR, 'fonts', font_filename)
        if os.path.exists(font_full_path):
            translator_params['font_path'] = font_full_path
            # 同时更新 config_dict 中的 font_path
            config_dict['render']['font_path'] = font_full_path
    
    # 创建翻译器
    print("🔧 初始化翻译器...")
    translator = MangaTranslator(params=translator_params)
    print("✅ 翻译器初始化完成")
    
    # 创建 Config 对象
    explicit_keys = {'render', 'upscale', 'translator', 'detector', 'colorizer', 'inpainter', 'ocr'}
    config_for_translate = {k: v for k, v in config_dict.items() if k in explicit_keys}
    for key in ['kernel_size', 'mask_dilation_offset', 'force_simple_sort']:
        if key in config_dict:
            config_for_translate[key] = config_dict[key]
    
    # 将 CLI 配置中的 attempts 复制到 translator 配置中（像 UI 一样）
    if 'translator' in config_for_translate:
        translator_config = config_for_translate['translator'].copy()
        cli_attempts = cli_config.get('attempts', -1)
        translator_config['attempts'] = cli_attempts
        config_for_translate['translator'] = translator_config
        logger.info(f"Setting translator attempts to: {cli_attempts} (from CLI config)")
    
    manga_config = Config(**config_for_translate)
    
    # 准备批量数据（像 UI 一样）
    images_with_configs = []
    
    # 收集输入文件夹（用于保持目录结构）
    input_folders = set()
    for input_path in input_paths:
        if os.path.isdir(input_path):
            input_folders.add(os.path.normpath(os.path.abspath(input_path)))
    
    print("\n📁 准备图片列表...")
    # ✅ 只保存文件路径，不加载图片数据
    file_paths_with_configs = []
    for file_path in all_files:
        # 只验证文件可读性，不加载图片数据
        try:
            if os.path.exists(file_path) and os.path.isfile(file_path):
                file_paths_with_configs.append((file_path, manga_config))
            else:
                print(f"❌ 文件不存在: {os.path.basename(file_path)}")
        except Exception as e:
            print(f"❌ 无法访问: {os.path.basename(file_path)} - {e}")
    
    if not file_paths_with_configs:
        print("没有需要翻译的图片")
        return
    
    # 准备 save_info（像 UI 一样）
    output_format = cli_config.get('format')
    if not output_format or output_format == "不指定":
        output_format = None
    
    save_info = {
        'output_folder': final_output_dir,
        'format': output_format,
        'overwrite': overwrite,
        'input_folders': input_folders  # 保持为 set，翻译器内部会处理
    }
    
    # 调试：检查输出目录是否存在
    if not os.path.exists(final_output_dir):
        os.makedirs(final_output_dir, exist_ok=True)
        print(f"✅ 创建输出目录: {final_output_dir}")
    
    # 过滤掉已存在的文件（如果 overwrite=False）
    skipped_count = 0
    if not overwrite:
        print("\n🔍 检查已存在的文件（覆盖检测已禁用）...")
        filtered_file_paths = []
        for file_path, config in file_paths_with_configs:
            try:
                should_skip = False
                skip_reason = ""
                
                # 检查导出原文/翻译的TXT文件（如果启用）
                if cli_config.get('template', False) and cli_config.get('save_text', False):
                    # 导出原文模式 - 检查TXT文件
                    from manga_translator.utils.path_manager import get_original_txt_path
                    txt_path = get_original_txt_path(file_path, create_dir=False)
                    if os.path.exists(txt_path):
                        should_skip = True
                        skip_reason = f"原文文件已存在: {os.path.basename(txt_path)}"
                elif cli_config.get('generate_and_export', False):
                    # 导出翻译模式 - 检查TXT文件
                    from manga_translator.utils.path_manager import get_translated_txt_path
                    txt_path = get_translated_txt_path(file_path, create_dir=False)
                    if os.path.exists(txt_path):
                        should_skip = True
                        skip_reason = f"翻译文件已存在: {os.path.basename(txt_path)}"
                else:
                    # 普通翻译模式 - 检查图片文件
                    output_path = translator._calculate_output_path(file_path, save_info)
                    if os.path.exists(output_path):
                        should_skip = True
                        skip_reason = f"输出文件已存在: {os.path.basename(file_path)}"
                
                if should_skip:
                    skipped_count += 1
                    if verbose:
                        print(f"⏭️  跳过 - {skip_reason}")
                else:
                    filtered_file_paths.append((file_path, config))
            except Exception as e:
                # 如果检查失败，默认保留
                if verbose:
                    logger.debug(f"检查文件时出错 {file_path}: {e}")
                filtered_file_paths.append((file_path, config))
        
        if skipped_count > 0:
            print(f"⏭️  已跳过 {skipped_count} 个已存在的文件（覆盖检测已禁用）")
            print("ℹ️  提示：如需重新翻译这些文件，请使用 --overwrite 参数")
            file_paths_with_configs = filtered_file_paths
        else:
            print("✅ 未发现已存在的文件，将处理所有文件")
            
    if not file_paths_with_configs:
        print("✅ 所有文件都已跳过，无需处理")
        print(f"\n{'='*60}")
        print(f"✅ 成功（跳过）: {skipped_count}")
        print("❌ 失败: 0")
        print(f"📊 总计: {len(all_files)}")
        print(f"{'='*60}")
        return

    batch_size = cli_config.get('batch_size', 3)
    total_images = len(file_paths_with_configs)
    total_batches = (total_images + batch_size - 1) // batch_size if batch_size > 0 else 1
    
    print(f"\n📊 批量处理模式：共 {total_images} 张图片，分 {total_batches} 个批次处理")
    print("📋 保存配置:")
    print(f"   输出目录: {final_output_dir}")
    print(f"   输出格式: {output_format or '保持原格式'}")
    print(f"   覆盖模式: {overwrite}")
    print(f"   保存质量: {cli_config.get('save_quality', 95)}")
    print(f"   批量大小: {batch_size} 张/批")
    if verbose and input_folders:
        print("   输入文件夹:")
        for folder in input_folders:
            print(f"      - {folder}")
    print()
    
    # 根据是否启用并发模式选择不同的处理方式
    use_concurrent = cli_config.get('batch_concurrent', False)
    
    try:
        print("🚀 开始翻译...")
        print("📋 传递给翻译器的 save_info:")
        print(f"   output_folder: {save_info['output_folder']}")
        print(f"   format: {save_info['format']}")
        print(f"   overwrite: {save_info['overwrite']}")
        print(f"   input_folders: {save_info['input_folders']}")
        print()
        
        import sys
        sys.stdout.flush()  # 强制刷新输出
        
        all_contexts = []
        
        if use_concurrent:
            # ✅ 并发模式：一次性传递所有文件路径，让 ConcurrentPipeline 内部管理加载
            print(f"⏳ 并发流水线模式：一次性处理 {total_images} 张图片...")
            logger.info(f"开始并发批量翻译，save_info={save_info}")
            
            # 创建带文件路径的 Image 对象（不加载数据，只设置 name 属性）
            images_with_configs = []
            for file_path, config in file_paths_with_configs:
                # 创建一个轻量级的 Image 占位符，只包含路径信息
                # ConcurrentPipeline 会根据 image.name 自己加载图片
                try:
                    image = Image.open(file_path)
                    # 不调用 load()，让 ConcurrentPipeline 按需加载
                    image.name = file_path
                    images_with_configs.append((image, config))
                except Exception as e:
                    logger.error(f"Failed to open image {file_path}: {e}")
                    print(f"❌ 无法打开: {os.path.basename(file_path)} - {e}")
                    from manga_translator.utils import Context
                    error_ctx = Context()
                    error_ctx.image_name = file_path
                    error_ctx.translation_error = str(e)
                    all_contexts.append(error_ctx)
            
            if images_with_configs:
                batch_contexts = await translator.translate_batch(
                    images_with_configs,
                    save_info=save_info,
                    global_offset=0,
                    global_total=total_images
                )
                all_contexts.extend(batch_contexts)
                
                # 清理 Image 对象
                for image, _ in images_with_configs:
                    if hasattr(image, 'close'):
                        try:
                            image.close()
                        except:
                            pass
                images_with_configs.clear()
                
                import gc
                gc.collect()
        else:
            # ✅ 非并发模式：按批次加载和处理图片，避免一次性加载所有图片到内存
            print("⏳ 开始批量翻译（按批次加载图片以节省内存）...")
            logger.info(f"开始批量翻译，save_info={save_info}")
            
            # 前端分批加载的批次大小（用于内存管理）
            frontend_batch_size = 10  # 每次最多加载10张图片到内存
            total_frontend_batches = (total_images + frontend_batch_size - 1) // frontend_batch_size
            
            processed_images_count = 0  # 已处理的图片总数
            
            for frontend_batch_num in range(total_frontend_batches):
                batch_start = frontend_batch_num * frontend_batch_size
                batch_end = min(batch_start + frontend_batch_size, total_images)
                current_batch_paths = file_paths_with_configs[batch_start:batch_end]
                
                # 加载当前批次的图片（静默加载，不显示前端批次信息）
                images_with_configs = []
                for file_path, config in current_batch_paths:
                    try:
                        with open(file_path, 'rb') as f:
                            image = Image.open(f)
                            image.load()  # 加载图片数据
                        image.name = file_path
                        images_with_configs.append((image, config))
                    except Exception as e:
                        logger.error(f"Failed to load image {file_path}: {e}")
                        print(f"❌ 无法加载: {os.path.basename(file_path)} - {e}")
                        # 创建一个错误上下文
                        from manga_translator.utils import Context
                        error_ctx = Context()
                        error_ctx.image_name = file_path
                        error_ctx.translation_error = str(e)
                        all_contexts.append(error_ctx)
                
                if images_with_configs:
                    # 传递全局偏移量给后端，让后端显示正确的全局图片编号
                    batch_contexts = await translator.translate_batch(
                        images_with_configs, 
                        save_info=save_info,
                        global_offset=processed_images_count,  # 传递已处理的图片数
                        global_total=total_images  # 传递总图片数
                    )
                    all_contexts.extend(batch_contexts)
                    processed_images_count += len(images_with_configs)
                    
                    # ✅ 批次处理完成后，立即清理图片对象
                    for image, _ in images_with_configs:
                        if hasattr(image, 'close'):
                            try:
                                image.close()
                            except:
                                pass
                    images_with_configs.clear()
                    
                    # 强制垃圾回收
                    import gc
                    gc.collect()
        
        contexts = all_contexts
        
        # 统计结果（像 UI 一样）
        success_count = 0
        failed_count = 0
        
        print("\n📊 翻译完成，检查结果...\n")
        logger.info(f"收到 {len(contexts)} 个翻译结果")
        
        for i, ctx in enumerate(contexts, 1):
            if ctx:
                has_result = hasattr(ctx, 'result') and ctx.result is not None
                has_success = hasattr(ctx, 'success') and ctx.success
                has_error = hasattr(ctx, 'translation_error') and ctx.translation_error
                logger.info(f"Context {i}: result={has_result}, success={has_success}, error={has_error}")
        
        for ctx in contexts:
            if ctx:
                # 检查是否有翻译错误
                if hasattr(ctx, 'translation_error') and ctx.translation_error:
                    failed_count += 1
                    print(f"❌ 翻译失败: {os.path.basename(ctx.image_name)}")
                    if verbose:
                        print(f"   错误: {ctx.translation_error}")
                elif hasattr(ctx, 'success') and ctx.success:
                    # 优先检查 success 标志（因为 result 可能被清理了）
                    success_count += 1
                    print(f"✅ 完成: {os.path.basename(ctx.image_name)}")
                elif ctx.result:
                    success_count += 1
                    print(f"✅ 完成: {os.path.basename(ctx.image_name)}")
                else:
                    failed_count += 1
                    print(f"❌ 翻译失败: {os.path.basename(ctx.image_name)} - 翻译结果为空")
            else:
                failed_count += 1
                print("❌ 翻译失败: 未知图片")
        
        if failed_count > 0:
            print(f"\n⚠️ 批量翻译完成：成功 {success_count}/{total_images} 张，失败 {failed_count}/{total_images} 张")
        else:
            print(f"\n✅ 批量翻译完成：成功 {success_count}/{total_images} 张")
        print(f"💾 文件已保存到：{final_output_dir}")
                
    except Exception as e:
        print(f"\n❌ 批量翻译错误: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        success_count = 0
        failed_count = len(images_with_configs)
    
    # 总结
    print(f"\n{'='*60}")
    if skipped_count > 0:
        print(f"✅ 成功: {success_count} (另有 {skipped_count} 个已跳过)")
    else:
        print(f"✅ 成功: {success_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"📊 总计: {len(all_files)}")
    print(f"{'='*60}")
    
    # 检查输出目录
    if os.path.exists(final_output_dir):
        output_files = [f for f in os.listdir(final_output_dir) if os.path.isfile(os.path.join(final_output_dir, f))]
        print(f"\n📁 输出目录: {final_output_dir}")
        print(f"   包含 {len(output_files)} 个文件")
        if verbose and output_files:
            for f in output_files[:10]:  # 只显示前10个
                file_path = os.path.join(final_output_dir, f)
                file_size = os.path.getsize(file_path) / 1024
                print(f"   - {f} ({file_size:.1f} KB)")
            if len(output_files) > 10:
                print(f"   ... 还有 {len(output_files) - 10} 个文件")
    else:
        print(f"\n⚠️  输出目录不存在: {final_output_dir}")
    print()


async def run_local_mode(args):
    """运行 local 模式的入口函数"""
    # 延迟导入配置服务
    from desktop_qt_ui.services.config_service import ConfigService
    from desktop_qt_ui.services.file_service import FileService
    
    # 初始化配置服务
    config_service = ConfigService(str(ROOT_DIR))
    
    # 如果指定了配置文件，加载它
    config_path = getattr(args, 'config', None)
    if config_path:
        if not config_service.load_config_file(config_path):
            print(f"❌ 无法加载配置文件: {config_path}")
            sys.exit(1)
    
    # 检查是否使用子进程模式
    use_subprocess = getattr(args, 'subprocess', False)
    verbose = getattr(args, 'verbose', False)
    overwrite = getattr(args, 'overwrite', False)
    
    if use_subprocess:
        # 子进程模式
        print("\n🔧 启用子进程模式（支持内存管理）")
        
        # 收集文件
        file_service = FileService()
        all_files = []
        input_paths = args.input
        
        folders = []
        individual_files = []
        
        for input_path in input_paths:
            input_path = os.path.abspath(input_path)
            if os.path.isfile(input_path):
                individual_files.append(input_path)
            elif os.path.isdir(input_path):
                folders.append(input_path)
        
        folders.sort(key=file_service._natural_sort_key)
        for folder in folders:
            folder_files = file_service.get_image_files_from_folder(folder, recursive=True)
            all_files.extend(folder_files)
        
        individual_files.sort(key=file_service._natural_sort_key)
        all_files.extend(individual_files)
        
        if not all_files:
            print("❌ 未找到图片文件")
            sys.exit(1)
        
        print(f"📁 找到 {len(all_files)} 个图片文件")
        
        # 确定输出目录
        output_dir = getattr(args, 'output', None)
        if not output_dir:
            config = config_service.get_config()
            if config.app.last_output_path:
                output_dir = config.app.last_output_path
            else:
                first_input = input_paths[0]
                if os.path.isdir(first_input):
                    output_dir = first_input.rstrip('/\\') + '-translated'
                else:
                    output_dir = os.path.dirname(first_input)
        
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        print(f"📤 输出目录: {output_dir}")
        
        # 预过滤已存在的文件
        skipped_count = 0
        if not overwrite:
            print("\n🔍 预检查已存在的文件（覆盖检测已禁用）...")
            try:
                from manga_translator import MangaTranslator
                config_dict = config_service.get_config().model_dump()
                cli_config = config_dict.get('cli', {})
                output_format = cli_config.get('format')
                if not output_format or output_format == "不指定":
                    output_format = None

                save_info = {
                    'output_folder': output_dir,
                    'format': output_format,
                    'overwrite': overwrite,
                    'input_folders': set()
                }
                
                temp_translator = MangaTranslator(params=cli_config)
                
                filtered_files = []
                for file_path in all_files:
                    try:
                        output_path = temp_translator._calculate_output_path(file_path, save_info)
                        if os.path.exists(output_path):
                            skipped_count += 1
                            if verbose:
                                print(f"⏭️  跳过已存在: {os.path.basename(file_path)}")
                        else:
                            filtered_files.append(file_path)
                    except:
                        filtered_files.append(file_path)
                
                if skipped_count > 0:
                    print(f"⏭️  已跳过 {skipped_count} 个已存在的文件（覆盖检测已禁用）")
                    print("ℹ️  提示：如需重新翻译这些文件，请使用 --overwrite 参数")
                    all_files = filtered_files
                else:
                    print("✅ 未发现已存在的文件，将处理所有文件")
            except Exception as e:
                print(f"⚠️ 预检查失败，将全部处理: {e}")
        
        if not all_files:
            print("✅ 所有文件都已跳过，无需处理")
            print(f"\n{'='*60}")
            print(f"✅ 成功（跳过）: {skipped_count}")
            print("❌ 失败: 0")
            print(f"📊 总计: {skipped_count}")
            print(f"{'='*60}")
            sys.exit(0)

        # 导入子进程管理器
        from .subprocess_manager import translate_with_subprocess
        
        try:
            config_dict = config_service.get_config().model_dump()
            
            success_count, failed_count = await translate_with_subprocess(
                all_files=all_files,
                output_dir=output_dir,
                config_dict=config_dict,
                config_path=config_path,
                verbose=verbose,
                overwrite=overwrite,
                memory_limit_mb=getattr(args, 'memory_limit', DEFAULT_MEMORY_THRESHOLD_MB),
                memory_limit_percent=getattr(args, 'memory_percent', 80),
                batch_per_restart=getattr(args, 'batch_per_restart', DEFAULT_BATCH_SIZE_PER_RESTART)
            )
            
            print(f"\n{'='*60}")
            if skipped_count > 0:
                print(f"✅ 成功: {success_count} (另有 {skipped_count} 个已跳过)")
                print(f"❌ 失败: {failed_count}")
                print(f"📊 总计: {len(all_files) + skipped_count}")
            else:
                print(f"✅ 成功: {success_count}")
                print(f"❌ 失败: {failed_count}")
                print(f"📊 总计: {len(all_files)}")
            print(f"💾 输出目录: {output_dir}")
            print(f"{'='*60}")
            
        except KeyboardInterrupt:
            print("\n\n⚠️  用户取消")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)
    else:
        # 原有的直接模式
        try:
            await translate_files(
                args.input,
                args.output if hasattr(args, 'output') else None,
                config_service,
                verbose=verbose,
                overwrite=overwrite,
                args=args
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  用户取消")
            sys.exit(0)
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            if verbose:
                import traceback
                traceback.print_exc()
            sys.exit(1)


def main():
    """主函数（用于直接运行）"""
    # Windows 下需要这个来支持子进程
    multiprocessing.freeze_support()
    
    args = parse_args()
    asyncio.run(run_local_mode(args))


if __name__ == '__main__':
    main()
