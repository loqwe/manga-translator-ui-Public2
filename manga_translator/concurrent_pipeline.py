"""
并发流水线处理模块
实现流水线并发：检测+OCR（顺序）→ 翻译线程（批量）+ 修复线程 + 渲染线程
使用线程池执行CPU/GPU密集型操作，避免阻塞事件循环
"""
import asyncio
import logging
import traceback
import os
import re
from typing import List, Dict
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import functools

from .utils import Context, load_image
from .utils.generic import is_mostly_noise_text
from .utils.failure_record import save_failure_record, remove_failure_record
from .config import Config
from .rate_limiter import RateLimiter

# 使用 manga_translator 的主 logger，确保日志能被UI捕获
logger = logging.getLogger('manga_translator')

# Queue sentinels for worker control
_RETRY_SWITCH = object()
_EXIT_SIGNAL = object()


class ConcurrentPipeline:
    """
    流水线并发处理器
    
    4个独立工作线程：
    1. 检测+OCR线程（顺序）→ 完成后放入翻译队列和修复队列
    2. 翻译线程（独立）→ 批量处理翻译队列
    3. 修复线程（独立）→ 处理修复队列
    4. 渲染线程（独立）→ 翻译+修复完成后渲染出图
    
    batch_size 控制翻译批量大小（一次翻译多少个文本块）
    """
    
    def __init__(self, translator_instance, batch_size: int = 3, max_workers: int = 4,
                 translation_concurrency: int = 1, preset_rpm: int = 0):
        """
        初始化并发流水线
        
        Args:
            translator_instance: MangaTranslator实例
            batch_size: 批量大小（一次翻译多少个文本块）
            max_workers: 线程池最大工作线程数（用于CPU/GPU密集型操作）
                        默认4个：检测+OCR、修复、渲染可以同时执行
            translation_concurrency: 翻译并发数（同时发起的翻译请求数）
            preset_rpm: 预设的每分钟最大请求数（0表示不限制）
        """
        self.translator = translator_instance
        self.batch_size = batch_size
        self.translation_concurrency = translation_concurrency
        self.preset_rpm = preset_rpm
        
        # ✅ 术语缓存初始化（启动时并发=1，直接写入模式）
        from .translators.glossary_cache import set_translation_concurrency, reset_delayed_flush_state
        set_translation_concurrency(1)
        reset_delayed_flush_state()  # 重置延迟刷新状态
        
        # 修复线程停止标记（用于切换术语缓存策略）
        self.inpaint_worker_stopped = False
        self.glossary_mode_switched = False  # 术语模式是否已切换
        self.rpm_restored = False  # RPM是否已恢复
        
        # 修复线程停止事件（用于异步通知翻译线程恢复并发数）
        self.inpaint_stopped_event = asyncio.Event()
        
        # 速率限制器（启动时线程数=1，修复线程停止后恢复用户配置）
        # 传入 RPM 配置启用令牌桶算法
        self.rate_limiter = RateLimiter(concurrency=1, rpm=preset_rpm)  # 启动时串行，但已启用 RPM 限制
        
        # 线程池：用于执行CPU/GPU密集型操作
        # 4个工作线程：允许检测、OCR、修复、渲染同时执行
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="pipeline_worker")
        
        # 队列
        self.translation_queue = asyncio.Queue()  # 翻译队列
        self.inpaint_queue = asyncio.Queue()      # 修复队列
        self.render_queue = asyncio.Queue()       # 渲染队列
        
        # Main-queue backpressure: pause OCR/detection when translation_queue is piled up, then focus on translation.
        self._ocr_pause_threshold = 100
        self._ocr_resume_threshold = 0
        self._ocr_can_run_event = asyncio.Event()
        self._ocr_can_run_event.set()
        self._main_queue_concurrent_enabled = False
        
        # Translation worker "active concurrency" limit:
        # - Main stage: default 1
        # - Backpressure triggered: temporarily raise to translation_concurrency
        # - After inpaint stops: raise to translation_concurrency (sticky)
        self._active_worker_limit = 1
        self._worker_scale_lock = asyncio.Lock()

        # Serialize GPU-heavy ops across stages to reduce HIP OOM spikes on AMD/ROCm.
        self._gpu_op_lock = asyncio.Lock()
        
        # 结果存储 {image_name: ctx}
        # ✅ 存储完整的ctx对象，而不是True/False标记
        self.translation_done = {}  # 翻译完成的ctx（包含翻译后的text_regions）
        self.inpaint_done = {}      # 修复完成的ctx（包含img_inpainted）
        
        # ✅ 存储基础ctx（检测+OCR的结果），供翻译和修复使用
        self.base_contexts = {}     # {image_name: ctx}
        
        # 控制标志
        self.stop_workers = False
        self.detection_ocr_done = False  # 检测+OCR是否全部完成
        self.has_critical_error = False  # 是否发生严重错误
        self.critical_error_msg = None   # 严重错误信息
        self.critical_error_exception = None  # 原始异常对象
        
        # 统计信息
        self.start_time = None
        self.total_images = 0
        self.stats = {
            'detection_ocr': 0,
            'translation': 0,
            'inpaint': 0,
            'rendering': 0
        }
        
        # 失败图片列表
        self.failed_images = []  # [(image_name, error_msg), ...]

        # Translation queue metrics (for logging/diagnostics)
        self.translation_enqueued_total = 0  # how many images have been enqueued to translation_queue
        
        # Worker busy flags (for serial mode when only one worker is used)
        self._worker_busy = {}  # {worker_id: bool}
        
        # 延迟重试队列（安全限制等错误时，先放入这里，等主队列完成后由Worker并行处理）
        self.retry_queue = asyncio.Queue()  # items: (ctx, config, error_type)
        self._main_workers_target = 0
        self._main_workers_finished = 0
        self._main_workers_lock = asyncio.Lock()

    def _should_serialize_gpu_ops(self, config) -> bool:
        """Return True if GPU-heavy ops should be serialized."""
        try:
            if config is None:
                return True
            cli = getattr(config, 'cli', None)
            if cli is None:
                return True
            return bool(getattr(cli, 'use_gpu', False))
        except Exception:
            return True
    
    async def _detection_ocr_worker(self, file_paths: List[str], configs: List):
        """
        检测+OCR工作线程（顺序处理，分批加载图片）
        完成后将上下文放入翻译队列和修复队列
        """
        # 让出控制权，确保其他线程有机会启动
        await asyncio.sleep(0)
        
        logger.info(f"[检测+OCR线程] 开始处理 {len(file_paths)} 张图片（分批加载）")
        
        from PIL import Image
        
        # 追踪当前作品名，便于在切换作品时显示日志
        current_work_name = None
        
        for idx, (file_path, config) in enumerate(zip(file_paths, configs)):
            # Main-queue backpressure: pause OCR/detection (only takes effect between images; won't interrupt the current image).
            while (not self.stop_workers) and (not self.has_critical_error) and (not self._ocr_can_run_event.is_set()):
                await asyncio.sleep(0.2)

            # 检查是否需要停止（其他线程出错）
            if self.stop_workers:
                logger.warning(f"[检测+OCR] 收到停止信号，已处理 {idx}/{len(file_paths)} 张图片")
                break
            
            try:
                # 提取作品名（使用 work_resolver 模块）
                try:
                    from .utils.work_resolver import resolve_work_name, resolve_raw_work_name
                    work_name = resolve_work_name(file_path)
                    raw_work_name = resolve_raw_work_name(file_path)
                    if work_name and work_name != current_work_name:
                        current_work_name = work_name
                        if raw_work_name and raw_work_name != work_name:
                            logger.info(f"[当前作品] {work_name} (生肉名: {raw_work_name})")
                        else:
                            logger.info(f"[当前作品] {work_name}")
                except Exception:
                    pass
                
                # 分批加载：只在需要时加载图片
                logger.debug(f"[检测+OCR] 加载图片: {file_path}")
                with open(file_path, 'rb') as f:
                    image = Image.open(f)
                    image.load()  # 立即加载图片数据
                image.name = file_path
                
                # 创建上下文
                ctx = Context()
                ctx.input = image
                ctx.image_name = file_path
                ctx.image_index = idx  # 保存图片索引（用于术语延迟刷新）
                ctx.verbose = self.translator.verbose
                ctx.save_quality = self.translator.save_quality
                ctx.config = config
                ctx.rate_limiter = self.rate_limiter  # 传递限速器引用，便于翻译器通知 429 错误
                
                logger.info(f"[检测+OCR] 处理 {idx+1}/{self.total_images}: {ctx.image_name}")

                serialize_gpu = self._should_serialize_gpu_ops(config)
                
                # 预处理：加载图片、上色、超分
                ctx.img_rgb, ctx.img_alpha = load_image(image)
                
                if config.colorizer.colorizer.value != 'none':
                    if serialize_gpu:
                        async with self._gpu_op_lock:
                            colorized_result = await self.translator._run_colorizer(config, ctx)
                    else:
                        colorized_result = await self.translator._run_colorizer(config, ctx)
                    # colorizer 返回 PIL Image，需要转换为 numpy
                    if hasattr(colorized_result, 'mode'):  # PIL Image
                        ctx.img_colorized, _ = load_image(colorized_result)
                    else:
                        ctx.img_colorized = colorized_result
                else:
                    ctx.img_colorized = ctx.img_rgb
                
                if config.upscale.upscale_ratio:
                    if serialize_gpu:
                        async with self._gpu_op_lock:
                            upscaled_result = await self.translator._run_upscaling(config, ctx)
                    else:
                        upscaled_result = await self.translator._run_upscaling(config, ctx)
                    # upscaler 返回 PIL Image，需要转换为 numpy
                    if hasattr(upscaled_result, 'mode'):  # PIL Image
                        ctx.upscaled, _ = load_image(upscaled_result)
                    else:
                        ctx.upscaled = upscaled_result
                else:
                    ctx.upscaled = ctx.img_colorized
                
                # 更新 img_rgb 为 upscaled 结果（现在都是 numpy.ndarray）
                ctx.img_rgb = ctx.upscaled
                
                # 检测（在线程池中执行，避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                detection_func = functools.partial(
                    asyncio.run,
                    self.translator._run_detection(config, ctx)
                )
                if serialize_gpu:
                    async with self._gpu_op_lock:
                        detection_result = await loop.run_in_executor(
                            self.executor, detection_func
                        )
                else:
                    detection_result = await loop.run_in_executor(
                        self.executor, detection_func
                    )
                # 检测结果现在返回4个值：(textlines, mask_raw, mask/debug_img, yolo_boxes)
                if len(detection_result) >= 4:
                    ctx.textlines, ctx.mask_raw, ctx.mask = detection_result[0], detection_result[1], detection_result[2]
                    # yolo_boxes 已经在 _run_detection 中保存到 ctx.yolo_boxes
                else:
                    ctx.textlines, ctx.mask_raw, ctx.mask = detection_result
                
                # OCR（在线程池中执行）
                ocr_func = functools.partial(
                    asyncio.run,
                    self.translator._run_ocr(config, ctx)
                )
                if serialize_gpu:
                    async with self._gpu_op_lock:
                        ctx.textlines = await loop.run_in_executor(
                            self.executor, ocr_func
                        )
                else:
                    ctx.textlines = await loop.run_in_executor(
                        self.executor, ocr_func
                    )
                
                # 文本行合并
                if ctx.textlines:
                    ctx.text_regions = await self.translator._run_textline_merge(config, ctx)
                
                # 文本过滤（根据 OCR 识别的原文过滤）
                if ctx.text_regions and self.translator.filter_text_enabled:
                    from .utils.text_filter import match_filter
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
                            watermark_patterns, exact_match_patterns = self.translator._load_watermark_filter_rules()
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
                        
                        # 3. 噪声文本过滤
                        if is_mostly_noise_text(region.text):
                            logger.info(f'过滤噪声文本: "{region.text}"')
                            filtered_count += 1
                            continue
                        
                        filtered_regions.append(region)
                    
                    if filtered_count > 0:
                        logger.info(f'过滤列表: 过滤了 {filtered_count} 个文本区域')
                    ctx.text_regions = filtered_regions

                # 源语言检测（用于按语言加载提示词），统一保证不为 None
                if ctx.text_regions:
                    logger.info(f"[检测+OCR] 准备源语言检测，文本块数: {len(ctx.text_regions)}")
                    try:
                        logger.info(f"[检测+OCR] 开始源语言检测，文本块数: {len(ctx.text_regions)}")
                        from .utils.lang_detect import detect_source_lang
                        texts_for_detect = [r.text for r in ctx.text_regions if getattr(r, 'text', None)]
                        lang_code, iso_lang, confidence, method = detect_source_lang(texts_for_detect)
                        ctx.from_lang = lang_code or 'auto'
                        if lang_code:
                            if confidence is None:
                                logger.info(f"[检测+OCR] 检测到源语言: {lang_code} (method: {method})")
                            else:
                                logger.info(f"[检测+OCR] 检测到源语言: {lang_code} (ISO: {iso_lang}, 置信度: {confidence:.2f}, method: {method})")
                        else:
                            logger.info(f"[检测+OCR] 源语言检测失败，使用 auto (ISO: {iso_lang}, method: {method})")
                    except Exception as e:
                        ctx.from_lang = getattr(ctx, 'from_lang', None) or 'auto'
                        logger.info(f"[检测+OCR] 源语言检测失败: {e}")
                else:
                    logger.info("[检测+OCR] 跳过源语言检测：无文本块")
                    ctx.from_lang = getattr(ctx, 'from_lang', None) or 'auto'

                # 无论检测结果如何，记录最终用于翻译的源语言
                logger.info(f"[检测+OCR] 源语言用于翻译: {ctx.from_lang}")

                self.stats['detection_ocr'] += 1
                logger.info(f"[检测+OCR] 完成 {idx+1}/{self.total_images}: {ctx.image_name} "
                           f"({len(ctx.text_regions) if ctx.text_regions else 0} 个文本块)")
                
                # 保存图片尺寸（用于保存JSON）
                if hasattr(image, 'size'):
                    ctx.original_size = image.size
                
                # ✅ 保留原始图片数据用于渲染（resize_regions_to_font_size需要original_img）
                # 注意：不关闭image对象，因为dump_image和渲染都需要使用它
                ctx.input = image  # 保留原始输入供dump_image使用
                # ✅ 保留img_rgb用于渲染时的original_img参数（balloon_fill等布局模式需要）
                # ctx.img_rgb 会在渲染完成后由渲染函数自动清理
                
                # ✅ 保存基础ctx，供后续合并使用
                self.base_contexts[ctx.image_name] = ctx
                
                # 放入翻译队列和修复队列（只传image_name和config，不传ctx）
                if ctx.text_regions:
                    await self.translation_queue.put((ctx.image_name, config))
                    await self.inpaint_queue.put((ctx.image_name, config))
                    self.translation_enqueued_total += 1
                    logger.info(
                        f"[检测+OCR] {ctx.image_name} 已加入翻译队列和修复队列 "
                        f"(翻译队列大小: {self.translation_queue.qsize()}，"
                        f"延迟队列大小: {self.retry_queue.qsize()}，"
                        f"累计入队: {self.translation_enqueued_total})"
                    )
                    # 让出控制权，让其他线程有机会运行
                    await asyncio.sleep(0)
                else:
                    # 无文本，直接标记完成并放入渲染队列
                    self.translation_done[ctx.image_name] = []  # 空列表表示无文本
                    self.inpaint_done[ctx.image_name] = True
                    ctx.text_regions = []  # 确保text_regions是空列表
                    # 增加修复统计计数（无文本图片也算“完成”）
                    self.stats['inpaint'] += 1
                    ctx.result = ctx.upscaled  # 设置result为upscaled图片
                    await self.render_queue.put((ctx, config))
                    logger.info(f"[检测+OCR] {ctx.image_name} 无文本（全部过滤），直接进入渲染队列")
                
            except Exception as e:
                # 安全地获取异常信息
                try:
                    error_msg = str(e)
                except Exception:
                    error_msg = f"无法获取异常信息 (异常类型: {type(e).__name__})"
                
                logger.error(f"[检测+OCR] 失败: {error_msg}")
                logger.error(traceback.format_exc())
                # 标记严重错误，停止所有线程
                self.has_critical_error = True
                self.critical_error_msg = f"检测+OCR失败: {error_msg}"
                self.critical_error_exception = e  # 保存原始异常
                self.stop_workers = True
                break
        
        # 标记检测+OCR全部完成
        self.detection_ocr_done = True
        logger.info("[检测+OCR线程] 处理完成")
    
    async def _translation_worker(self):
        """
        翻译工作线程（任务池模式）
        
        架构：
        1. 收集者协程：从 translation_queue 收集图片，凑够批次后放入 batch_queue
        2. N 个 worker 协程：各自从 batch_queue 拉取并处理，互不干扰
        3. 每个 worker 独立运行，一个慢了不影响其他 worker
        """
        await asyncio.sleep(0)
        
        logger.info(f"【翻译线程】 启动（任务池模式），批量大小: {self.batch_size}，初始worker数: 1，目标worker数: {self.translation_concurrency}，预设RPM: {self.preset_rpm}")
        
        # 内部批次队列（收集者 -> workers）
        batch_queue: asyncio.Queue = asyncio.Queue()
        
        # Worker control:
        # - _current_worker_count: spawned worker count (monotonic increase; used for worker_id allocation)
        # - _active_worker_limit: active worker limit (can switch between 1 and target)
        self._current_worker_count = 1  # 修复线程运行时只启动1个worker
        self._target_worker_count = max(1, int(self.translation_concurrency or 1))
        self._active_worker_limit = 1
        self._workers: List[asyncio.Task] = []
        self._worker_stop_event = asyncio.Event()  # 通知所有worker停止
        
        # 启动收集者协程
        collector_task = asyncio.create_task(
            self._batch_collector(batch_queue)
        )
        
        # 启动初始worker（修复线程运行时只启动1个）
        for i in range(self._current_worker_count):
            worker = asyncio.create_task(
                self._translation_pool_worker(i, batch_queue)
            )
            self._workers.append(worker)
        logger.info(f"[翻译] 已启动 {self._current_worker_count} 个worker")
        
        # 后台任务：等待修复线程停止后扩展worker数量
        expand_workers_task = asyncio.create_task(
            self._expand_workers_on_inpaint_stop(batch_queue)
        )

        # Main-queue backpressure controller: pause OCR when translation_queue is piled up, then temporarily enable concurrent translation.
        backpressure_task = asyncio.create_task(
            self._main_queue_backpressure_controller(batch_queue)
        )
        
        # Wait for the collector to finish (all images have been enqueued into batch_queue).
        await collector_task

        # Stop backpressure controller (avoid affecting the shutdown stage).
        if not backpressure_task.done():
            backpressure_task.cancel()
            try:
                await backpressure_task
            except asyncio.CancelledError:
                pass
        # Ensure OCR/detection won't be stuck.
        try:
            self._ocr_can_run_event.set()
        except Exception:
            pass

        # Shutdown stage: allow all spawned workers to participate (avoid idle workers hanging forever).
        self._active_worker_limit = max(1, int(self._current_worker_count or 1))

        # Send switch signals to all workers.
        # IMPORTANT: send based on the *target* worker count to avoid deadlocks when
        # additional workers are spawned later (e.g. after the inpaint worker stops).
        # ✅ 始终发送切换信号，Worker 先处理延迟队列，队列清空后再退出
        target_workers = max(int(self._target_worker_count or 1), len(self._workers))
        self._main_workers_target = target_workers
        self._main_workers_finished = 0
        logger.info(f"[翻译] 收集完成，发送 {target_workers} 个切换信号（Worker将处理延迟队列后退出）...")
        for _ in range(target_workers):
            await batch_queue.put(_RETRY_SWITCH)

        # Wait for all workers to finish.
        # NOTE: expand_workers_task may append new workers after the inpaint worker stops,
        # so we must wait dynamically instead of awaiting a one-time snapshot list.
        logger.info(f"[翻译] 等待所有worker完成...")
        while True:
            pending_workers = [w for w in self._workers if not w.done()]
            if not pending_workers:
                # No workers running. Stop late expansion to freeze the worker list,
                # then re-check once to avoid race conditions.
                if not expand_workers_task.done():
                    expand_workers_task.cancel()
                    try:
                        await expand_workers_task
                    except asyncio.CancelledError:
                        pass
                    continue
                break

            await asyncio.wait(pending_workers, return_when=asyncio.FIRST_COMPLETED)

        # Collect worker exceptions (avoid background Task exceptions being unobserved).
        if self._workers:
            await asyncio.gather(*self._workers, return_exceptions=True)
        
        # ✅ 主队列完成后，延迟重试由Worker在收到毒丸后并行处理
        
        # 翻译完成后，批量写入缓存的术语到文件
        try:
            from manga_translator.translators.glossary_cache import get_glossary_cache
            cache = get_glossary_cache()
            flushed_count = cache.flush_all()
            if flushed_count > 0:
                logger.info(f"[翻译] 术语表已保存: {flushed_count} 个术语")
        except Exception as e:
            logger.warning(f"[翻译] 术语表保存失败: {e}")
        
        # 检查是否所有图片都已翻译
        if self.stats['translation'] >= self.total_images:
            logger.info(f"[翻译线程] 所有图片已翻译 ({self.stats['translation']}/{self.total_images})")
        
        # 清理翻译器客户端
        try:
            translator_obj = getattr(self.translator, 'translator', None)
            if translator_obj and hasattr(translator_obj, '_cleanup'):
                await translator_obj._cleanup()
                logger.debug("[翻译] 翻译器客户端已清理")
        except Exception as e:
            logger.debug(f"[翻译] 翻译器客户端清理失败: {e}")
        
        logger.info("【翻译线程】 停止")
    
    async def _batch_collector(self, batch_queue: asyncio.Queue):
        """
        批次收集者：从 translation_queue 收集图片，凑够批次后放入 batch_queue
        """
        batch_wait_timeout = 90.0
        batch_start_time = None
        pending_batch = []

        collected_total = 0
        
        while not self.stop_workers:
            try:
                if self.has_critical_error:
                    logger.warning(f"[收集者] 检测到严重错误，停止收集")
                    break
                
                # Multi-worker mode: if all active workers are busy, pause collecting.
                if self._active_worker_limit > 1 and self._worker_busy:
                    active_ids = [i for i in range(int(self._active_worker_limit or 1))]
                    all_busy = all(self._worker_busy.get(i, False) for i in active_ids)
                    if all_busy:
                        await asyncio.sleep(0.5)
                        continue
                
                # 尝试从队列获取图片
                if len(pending_batch) < self.batch_size:
                    try:
                        image_name, config = await asyncio.wait_for(
                            self.translation_queue.get(), timeout=1.0
                        )
                        ctx = self.base_contexts.get(image_name)
                        if ctx:
                            pending_batch.append((ctx, config))
                            collected_total += 1
                            if batch_start_time is None:
                                batch_start_time = asyncio.get_event_loop().time()
                            enqueued_total = getattr(self, 'translation_enqueued_total', 0)
                            if self._active_worker_limit == 1:
                                logger.info(
                                    f"[收集者] 收集到图片 ({len(pending_batch)}/{self.batch_size})，"
                                    f"累计收集: {collected_total}，累计入队: {enqueued_total}，"
                                    f"翻译队列剩余: {self.translation_queue.qsize()}"
                                )
                            else:
                                logger.debug(
                                    f"[收集者] 收集到图片 ({len(pending_batch)}/{self.batch_size})，"
                                    f"累计收集: {collected_total}，累计入队: {enqueued_total}，"
                                    f"翻译队列剩余: {self.translation_queue.qsize()}"
                                )
                        else:
                            logger.error(f"[收集者] 找不到 {image_name} 的基础上下文")
                    except asyncio.TimeoutError:
                        pass
                
                # 检查是否完成
                if self.detection_ocr_done and self.translation_queue.empty() and not pending_batch:
                    break
                
                # 判断是否应该发送当前批次
                should_send = False
                reason = ""
                
                if len(pending_batch) >= self.batch_size:
                    should_send = True
                    reason = f"批次已满 ({len(pending_batch)}/{self.batch_size})"
                elif pending_batch and self.detection_ocr_done and self.translation_queue.empty():
                    should_send = True
                    reason = f"OCR完成，发送剩余 {len(pending_batch)} 张图片"
                elif pending_batch and batch_start_time:
                    elapsed = asyncio.get_event_loop().time() - batch_start_time
                    if elapsed >= batch_wait_timeout:
                        should_send = True
                        reason = f"等待超时({elapsed:.0f}秒)"
                
                if should_send:
                    enqueued_total = getattr(self, 'translation_enqueued_total', 0)
                    logger.info(
                        f"[收集者] 将 {len(pending_batch)}/{self.batch_size} 张图片放入批次队列"
                        f"（活跃worker={self._active_worker_limit}，已启动={self._current_worker_count}，目标={self._target_worker_count}，累计收集: {collected_total}，"
                        f"累计入队: {enqueued_total}，翻译队列剩余: {self.translation_queue.qsize()}，"
                        f"原因: {reason}）"
                    )
                    await batch_queue.put(pending_batch.copy())
                    pending_batch = []
                    batch_start_time = None
                    
                    # ✅ 单 Worker 模式：等待当前批次翻译完成后再收集下一批
                    # 多 Worker 模式：继续并行收集（流水线）
                    if self._active_worker_limit == 1:
                        # 等待单Worker空闲，确保上一批次翻译完成
                        while not self.stop_workers and not self.has_critical_error:
                            busy = any(self._worker_busy.values())
                            if not busy:
                                logger.debug("[收集者] 串行模式：上一批次已完成，继续收集")
                                break
                            await asyncio.sleep(0.5)
                    
            except Exception as e:
                logger.error(f"[收集者] 错误: {e}")
                logger.error(traceback.format_exc())
        
        enqueued_total = getattr(self, 'translation_enqueued_total', 0)
        logger.info(f"[收集者] 完成 (累计收集: {collected_total}，累计入队: {enqueued_total})")
    
    async def _translation_pool_worker(self, worker_id: int, batch_queue: asyncio.Queue):
        """
        翻译池worker：独立从 batch_queue 拉取批次并处理
        
        每个worker完全独立，一个慢了不影响其他worker
        """
        logger.info(f"[Worker-{worker_id}] 启动")
        processed_count = 0
        
        # 默认空闲
        self._worker_busy[worker_id] = False
        
        while not self.stop_workers:
            try:
                if self.has_critical_error:
                    logger.warning(f"[Worker-{worker_id}] 检测到严重错误，停止")
                    break

                # Main-queue backpressure: non-active workers should not pull tasks in serial mode.
                # worker_id=0 is always active; others are gated by _active_worker_limit.
                if worker_id > 0 and worker_id >= int(self._active_worker_limit or 1):
                    await asyncio.sleep(0.2)
                    continue
                
                # 从批次队列获取任务（阻塞等待）
                try:
                    batch = await asyncio.wait_for(batch_queue.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                
                # 控制信号：切换到延迟队列处理，完成后退出
                if batch is _RETRY_SWITCH:
                    # Guard: do not switch into retry stage until the main OCR/translation pipeline is fully done.
                    # This prevents retry tasks from stealing capacity when OCR/detection is expected to resume.
                    if (not self.detection_ocr_done) or (not self.translation_queue.empty()):
                        batch_queue.task_done()
                        await batch_queue.put(_RETRY_SWITCH)
                        await asyncio.sleep(0.2)
                        continue

                    batch_queue.task_done()
                    logger.info(f"[Worker-{worker_id}] 收到切换信号，已处理 {processed_count} 个批次，切换到延迟队列")
                    async with self._main_workers_lock:
                        self._main_workers_finished += 1
                    # ✅ 处理延迟队列，队列清空后自动退出
                    await self._process_retry_queue_worker(worker_id)
                    break
                if batch is _EXIT_SIGNAL:
                    # ✅ 兼容旧信号，直接退出（正常情况不会触发）
                    batch_queue.task_done()
                    logger.info(f"[Worker-{worker_id}] 收到退出信号，已处理 {processed_count} 个批次")
                    async with self._main_workers_lock:
                        self._main_workers_finished += 1
                    break
                
                # 处理批次（错误重试流程见 md/错误重试流程.md）
                self._worker_busy[worker_id] = True
                logger.info(f"[Worker-{worker_id}] 开始处理批次 ({len(batch)} 张图片)")

                # ✅ 将 worker_id 传递给 ctx，让翻译器使用独立连接池
                for ctx, _config in batch:
                    ctx.worker_id = worker_id

                try:
                    # ✅ 统一走批次级重试逻辑：
                    # - timeout: 直接进入 retry_queue（拆分单张），不做整批重试
                    # - network/auth: 整批重试 1 次，仍失败则进入 retry_queue（拆分单张）
                    # - safety/count/br/quality: 进入 retry_queue（拆分单张）
                    await self._process_translation_batch(batch)
                except Exception as e:
                    # Fallback: should be rare because _process_translation_batch already handles most errors.
                    await self._handle_translation_error(batch, e, worker_id)
                finally:
                    processed_count += 1
                    batch_queue.task_done()
                    self._worker_busy[worker_id] = False

                logger.info(f"[Worker-{worker_id}] 批次处理结束 (总进度: {self.stats['translation']}/{self.total_images})")
                
            except Exception as e:
                logger.error(f"[Worker-{worker_id}] 未捕获错误: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"[Worker-{worker_id}] 停止，共处理 {processed_count} 个批次")
    
    async def _handle_translation_error(self, batch: List[tuple], error: Exception, worker_id: int):
        """
        处理翻译错误（从 _process_translation_batch 提取的错误处理逻辑）
        """
        error_msg = str(error).lower()
        error_type = type(error).__name__
        
        # 检测安全限制错误
        is_safety_error = any(kw in error_msg for kw in [
            'safety', 'content_filter', 'blocked', 'inappropriate',
            'violated', 'policy', 'harmful'
        ])
        
        if is_safety_error:
            # 安全限制：放入延迟重试队列
            logger.warning(f"[Worker-{worker_id}] 安全限制错误，放入延迟重试队列")
            for ctx, config in batch:
                await self.retry_queue.put((ctx, config, 'safety_limit'))
        else:
            # 其他错误：标记失败
            logger.error(f"[Worker-{worker_id}] 翻译失败: {error}")
            for ctx, config in batch:
                ctx.translation_error = str(error)
                ctx.success = False
                self.translation_done[ctx.image_name] = 'FAILED'
                self.failed_images.append((ctx.image_name, str(error)))
            self.stats['translation'] += len(batch)
    
    async def _expand_workers_on_inpaint_stop(self, batch_queue: asyncio.Queue):
        """Wait for the inpaint worker to stop, then expand translation workers."""
        try:
            # Wait for the inpaint worker to stop.
            await self.inpaint_stopped_event.wait()
            
            logger.info("[扩展] 修复线程已停止，开始扩展worker")

            async with self._worker_scale_lock:
                # Switch glossary write strategy to delayed flush (concurrent-safe).
                if not self.glossary_mode_switched:
                    from .translators.glossary_cache import set_translation_concurrency
                    set_translation_concurrency(self.translation_concurrency)
                    self.glossary_mode_switched = True
                    logger.info("[扩展] 术语切换为延迟刷新模式")

                # Expand rate limiter concurrency.
                if not self.rpm_restored:
                    old_concurrency = self.rate_limiter.concurrency
                    if self._target_worker_count > old_concurrency:
                        self.rate_limiter.expand_concurrency(self._target_worker_count)
                    self.rpm_restored = True
                    logger.info(f"[扩展] 速率限制器扩展: {old_concurrency} -> {self._target_worker_count}")

                # Spawn extra workers (monotonic increase).
                workers_to_add = self._target_worker_count - self._current_worker_count
                if workers_to_add > 0:
                    logger.info(f"[扩展] 启动 {workers_to_add} 个额外worker")
                    for i in range(workers_to_add):
                        new_worker_id = self._current_worker_count + i
                        worker = asyncio.create_task(
                            self._translation_pool_worker(new_worker_id, batch_queue)
                        )
                        self._workers.append(worker)
                    self._current_worker_count = self._target_worker_count
                    logger.info(f"[扩展] worker数量: {self._current_worker_count}")

                # After inpaint stops, keep concurrent translation enabled.
                self._active_worker_limit = int(self._target_worker_count or 1)
                logger.info(f"[扩展] 活跃worker上限: {self._active_worker_limit}")
                
        except asyncio.CancelledError:
            logger.debug("[扩展] 任务被取消")
        except Exception as e:
            logger.error(f"[扩展] 错误: {e}")

    async def _main_queue_backpressure_controller(self, batch_queue: asyncio.Queue):
        """Backpressure controller (main stage only).

        Uses `translation_queue.qsize()` (OCR-finished but not yet collected) as the metric:
        - If queue size >= threshold: pause OCR/detection and temporarily enable concurrent translation.
        - If queue size == 0: resume OCR/detection and disable concurrent translation.

        Note:
        - This controller is only meaningful before the inpaint worker stops.
        - Once inpaint stops, the pipeline switches to full concurrency permanently.
        """
        target = int(self._target_worker_count or 1)
        if target <= 1:
            return

        try:
            while not self.stop_workers and not self.has_critical_error:
                # Only apply to the "main queue" stage.
                if self.inpaint_stopped_event.is_set() or self.detection_ocr_done:
                    try:
                        self._ocr_can_run_event.set()
                    except Exception:
                        pass
                    break

                q = int(self.translation_queue.qsize() or 0)

                if (not self._main_queue_concurrent_enabled) and q >= int(self._ocr_pause_threshold or 0):
                    await self._enable_main_queue_concurrency(batch_queue=batch_queue, qsize=q)
                elif self._main_queue_concurrent_enabled and q <= int(self._ocr_resume_threshold or 0):
                    await self._disable_main_queue_concurrency(qsize=q)

                await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            try:
                self._ocr_can_run_event.set()
            except Exception:
                pass
            raise

    async def _enable_main_queue_concurrency(self, *, batch_queue: asyncio.Queue, qsize: int):
        """Enable concurrent translation temporarily (main queue stage)."""
        async with self._worker_scale_lock:
            if self._main_queue_concurrent_enabled:
                return
            if self.inpaint_stopped_event.is_set() or self.detection_ocr_done:
                return

            # Pause OCR/detection.
            try:
                self._ocr_can_run_event.clear()
            except Exception:
                pass

            target = int(self._target_worker_count or 1)

            # Expand rate limiter concurrency once (monotonic increase).
            try:
                if target > int(self.rate_limiter.concurrency or 1):
                    self.rate_limiter.expand_concurrency(target)
            except Exception:
                pass

            # Spawn workers up to target.
            workers_to_add = target - int(self._current_worker_count or 0)
            if workers_to_add > 0:
                logger.info(f"[背压] 启动 {workers_to_add} 个额外worker（阈值触发）")
                for i in range(workers_to_add):
                    new_worker_id = int(self._current_worker_count or 0) + i
                    worker = asyncio.create_task(
                        self._translation_pool_worker(new_worker_id, batch_queue)
                    )
                    self._workers.append(worker)
                self._current_worker_count = target

            # Switch glossary cache strategy to delayed flush to avoid concurrent file writes.
            try:
                from .translators.glossary_cache import set_translation_concurrency
                set_translation_concurrency(target)
            except Exception:
                pass

            # Enable concurrent translation workers.
            prev_limit = int(self._active_worker_limit or 1)
            self._active_worker_limit = target
            self._main_queue_concurrent_enabled = True

            logger.warning(
                f"[背压] 翻译队列剩余={qsize} 达到阈值{self._ocr_pause_threshold}，暂停检测+OCR，"
                f"启用并发翻译（活跃worker: {prev_limit} -> {self._active_worker_limit}）"
            )

    async def _disable_main_queue_concurrency(self, *, qsize: int):
        """Disable concurrent translation temporarily (main queue stage)."""
        async with self._worker_scale_lock:
            if not self._main_queue_concurrent_enabled:
                return
            if self.inpaint_stopped_event.is_set() or self.detection_ocr_done:
                return

            prev_limit = int(self._active_worker_limit or 1)
            self._active_worker_limit = 1
            self._main_queue_concurrent_enabled = False

            # Switch glossary cache strategy back to write-through.
            try:
                from .translators.glossary_cache import set_translation_concurrency
                set_translation_concurrency(1)
            except Exception:
                pass

            # Resume OCR/detection.
            try:
                self._ocr_can_run_event.set()
            except Exception:
                pass

            logger.info(
                f"[背压] 翻译队列已清空(q={qsize})，停止并发翻译（活跃worker: {prev_limit} -> {self._active_worker_limit}），"
                f"恢复检测+OCR"
            )
    
    async def _process_translation_batch(self, batch: List[tuple], retry_count: int = 0, skip_rate_limiter: bool = False):
        """
        处理一个翻译批次
        
        直接复用 MangaTranslator._batch_translate_contexts 的翻译逻辑，
        确保与标准批量处理完全一致，便于维护。
        
重试策略：
        - 网络错误（连接超时/断开）：
          1. 等待3秒后整批重试（共2次，含初始）
          2. 若整批重试仍失败，等待2秒后拆分重试（不占用槽位，部分成功/失败均可）
        - 401认证错误：
          1. 等待2秒后整批重试（共2次，含初始）
          2. 若整批重试仍失败，等待2秒后拆分重试（不占用槽位）
        - 安全限制错误：拆分成单张重试
        - 其他错误：直接标记失败
        
        Args:
            batch: 翻译批次
            retry_count: 当前重试次数（0=初始，1=整批重试后）
            skip_rate_limiter: 是否跳过速率限制器（重试时为True）
        """
        if not batch:
            return
        
        # 网络错误和401错误：2次重试（含初始），即最多整批重试1次
        max_network_retries = 1
        max_auth_retries = 1  # 401认证错误重试次数
        logger.info(f"[翻译] 批量翻译 {len(batch)} 张图片")
        
        try:
            if skip_rate_limiter:
                # ✅ 重试：跳过速率限制器，直接翻译（占用原槽位，不等待新令牌）
                logger.debug(f"[翻译] 重试跳过速率限制器")
                translated_batch = await self.translator._batch_translate_contexts(batch, len(batch))
            else:
                # ✅ 正常流程：使用速率限制器控制请求频率
                async with self.rate_limiter:
                    # ✅ 直接调用标准的批量翻译方法，复用所有翻译逻辑
                    # 包括：翻译、后处理、过滤、译后检查等
                    translated_batch = await self.translator._batch_translate_contexts(batch, len(batch))
            
            self.stats['translation'] += len(batch)
            logger.info(f"[翻译] 批次完成 ({self.stats['translation']}/{self.total_images})")
            
            # ✅ 报告成功，触发预热模式下的下一个请求
            self.rate_limiter.report_success()
            
            # 更新翻译结果到batch（_batch_translate_contexts可能修改了text_regions）
            # 标记翻译完成，并立即逐张检查是否可以渲染
            await self._mark_translation_done(translated_batch)
            
        except Exception as e:
            error_msg = str(e).lower()
            error_type = type(e).__name__
            
            # 检测网络错误（连接超时、断开、重置、5xx服务器错误等）
            is_network_error = any(kw in error_msg for kw in [
                'connect', 'timeout', 'reset', 'closed', 'eof', 'broken pipe',
                'connection', 'network', 'socket', 'httpcore', 'httpx',
                '503', '504', '520', '522', '523', '524',
                'cloudflare', 'cf-error', '<!doctype html', '<!DOCTYPE html'
            ]) or any(kw in error_type.lower() for kw in [
                'connect', 'timeout', 'network', 'socket', 'http',
                '503', '504', '520', '522', '523', '524', 'cloudflare', 'cf-error'
            ])

            translator_obj = getattr(self.translator, 'translator', None)
            use_stream = bool(getattr(translator_obj, 'use_stream', False))

            is_stream_connection_reset_error = False
            if use_stream:
                err_type_l = error_type.lower()
                is_stream_connection_reset_error = (
                    'remoteprotocolerror' in err_type_l or
                    any(kw in error_msg for kw in [
                        'remoteprotocolerror',
                        'peer closed connection',
                        'incomplete message body',
                        'connection reset',
                        'reset by peer',
                        'server disconnected',
                    ])
                )

            # 更细分：超时错误（不做阻塞式整批重试，直接进入延迟队列）
            is_timeout_error = (
                'timeout' in error_msg or 'timed out' in error_msg or
                'readtimeout' in error_type.lower() or 'timeout' in error_type.lower() or
                is_stream_connection_reset_error
            )
            
            # 检测401认证错误
            is_auth_error = '401' in error_msg or 'unauthorized' in error_msg or 'authentication' in error_msg
            
            # 检测安全限制相关错误（API内容过滤、安全策略等）
            is_safety_error = any(kw in error_msg for kw in [
                'safety', 'content', 'filter', 'block', 'policy', 'violation',
                'inappropriate', 'harmful', 'moderation', 'flagged'
            ])
            
            # 检测数量不匹配错误
            is_count_mismatch = any(kw in error_msg for kw in [
                'count mismatch', '数量不匹配', 'expected', 'got'
            ]) and ('translation' in error_msg or '翻译' in error_msg)
            
            # 检测BR标记缺失错误
            is_br_error = 'brmarkersvalidation' in error_type.lower() or any(kw in error_msg for kw in [
                'br标记', 'br markers', '断句检查', '[br]'
            ])
            
            # 检测质量检查失败错误
            is_quality_error = 'quality check failed' in error_msg or '质量检查失败' in error_msg
            
            # 超时错误：直接进入延迟队列（避免在主翻译槽位里卡住），并拆分为单张重试
            if is_timeout_error:
                rl_stats = self.rate_limiter.get_stats() if self.rate_limiter else {}
                inpaint_running = not self.inpaint_stopped_event.is_set()
                if is_stream_connection_reset_error:
                    logger.warning(
                        f"[翻译] 流式连接中断({error_type})，将 {len(batch)} 张图片放入延迟队列（拆分为单张重试，不做整批重试）"
                    )
                else:
                    logger.warning(
                        f"[翻译] 请求超时，将 {len(batch)} 张图片放入延迟队列（拆分为单张重试，不做整批重试）"
                    )
                logger.debug(
                    f"[翻译] 断联诊断: stream={use_stream} inpaint_running={inpaint_running} rl={rl_stats} "
                    f"batch={len(batch)} error_type={error_type}"
                )
                prev_ctxs = []
                for ctx, config in batch:
                    # Attach up to 2 previous images' original texts as same-batch context for single-image retry.
                    if prev_ctxs:
                        ctx.local_prev_context = self._format_same_batch_prev_original_context(prev_ctxs[-2:])
                    await self.retry_queue.put((ctx, config, 'timeout_error'))
                    prev_ctxs.append(ctx)
                return

            # 网络错误处理
            if is_network_error:
                if retry_count < max_network_retries:
                    # 第一次失败：等待3秒后整批重试
                    retry_count += 1
                    logger.warning(f"[翻译] 网络错误，3秒后整批重试 ({retry_count}/{max_network_retries}): {error_type}")
                    await asyncio.sleep(3)
                    # 递归重试：跳过速率限制器（不等待令牌），但仍占用原semaphore槽位
                    await self._process_translation_batch(batch, retry_count, skip_rate_limiter=True)
                    return
                else:
                    # 整批重试失败：放入延迟队列
                    logger.warning(f"[翻译] 网络错误整批重试失败，将 {len(batch)} 张图片放入延迟队列")
                    prev_ctxs = []
                    for ctx, config in batch:
                        if prev_ctxs:
                            ctx.local_prev_context = self._format_same_batch_prev_original_context(prev_ctxs[-2:])
                        await self.retry_queue.put((ctx, config, 'network_error'))
                        prev_ctxs.append(ctx)
                    return
            
            # 401认证错误处理
            if is_auth_error:
                if retry_count < max_auth_retries:
                    # 第一次失败：等待2秒后整批重试
                    retry_count += 1
                    logger.warning(f"[翻译] 401认证错误，2秒后整批重试 ({retry_count}/{max_auth_retries}): {error_type}")
                    await asyncio.sleep(2)
                    # 递归重试：跳过速率限制器（不等待令牌），但仍占用原semaphore槽位
                    await self._process_translation_batch(batch, retry_count, skip_rate_limiter=True)
                    return
                else:
                    # 整批重试失败：放入延迟队列
                    logger.warning(f"[翻译] 401认证错误整批重试失败，将 {len(batch)} 张图片放入延迟队列")
                    prev_ctxs = []
                    for ctx, config in batch:
                        if prev_ctxs:
                            ctx.local_prev_context = self._format_same_batch_prev_original_context(prev_ctxs[-2:])
                        await self.retry_queue.put((ctx, config, 'auth_error'))
                        prev_ctxs.append(ctx)
                    return
            
            # 安全限制/数量不匹配/BR缺失/质量检查错误：放入延迟重试队列，等主队列完成后统一处理
            if is_safety_error or is_count_mismatch or is_br_error or is_quality_error:
                # 确定错误类型
                if is_safety_error:
                    error_type_str = 'safety_limit'
                    error_desc = '安全限制'
                elif is_count_mismatch:
                    error_type_str = 'count_mismatch'
                    error_desc = '数量不匹配'
                elif is_br_error:
                    error_type_str = 'br_missing'
                    error_desc = 'BR标记缺失'
                else:
                    error_type_str = 'quality_failed'
                    error_desc = '质量检查失败'
                
                if len(batch) > 1:
                    # 批次错误：拆分放入延迟队列
                    logger.warning(f"[翻译] 批量翻译触发{error_desc}，将 {len(batch)} 张图片放入延迟队列，稍后并行重试")
                    prev_ctxs = []
                    for ctx, config in batch:
                        if prev_ctxs:
                            ctx.local_prev_context = self._format_same_batch_prev_original_context(prev_ctxs[-2:])
                        await self.retry_queue.put((ctx, config, error_type_str))
                        prev_ctxs.append(ctx)
                else:
                    # 单张错误：放入延迟队列
                    ctx, config = batch[0]
                    logger.warning(f"[翻译] 单张翻译触发{error_desc}，放入延迟队列: {ctx.image_name}")
                    await self.retry_queue.put((ctx, config, error_type_str))
                # 注意：不增加 stats['translation']，等延迟队列处理时再计数
                return
            
            # 非可重试错误或单张图片失败，直接标记失败
            if not is_network_error and not is_auth_error:
                logger.error(f"[翻译] 批次失败: {e}")
            logger.error(traceback.format_exc())
            for ctx, config in batch:
                ctx.translation_error = str(e)
                ctx.success = False
                self.translation_done[ctx.image_name] = 'FAILED'
                self.failed_images.append((ctx.image_name, str(e)))
                
                # ✅ 保存失败记录
                try:
                    historical_ctx = {
                        'all_page_translations': self.translator.all_page_translations.copy() if hasattr(self.translator, 'all_page_translations') else [],
                        '_original_page_texts': self.translator._original_page_texts.copy() if hasattr(self.translator, '_original_page_texts') else [],
                        'context_size': getattr(self.translator, 'context_size', 3),
                    }
                    save_failure_record(
                        image_path=ctx.image_name,
                        error_type=error_type,
                        error_message=str(e),
                        ctx=ctx,
                        config=config,
                        historical_context=historical_ctx
                    )
                except Exception as save_err:
                    logger.debug(f"[失败记录] 保存异常: {save_err}")
            self.stats['translation'] += len(batch)
    
    async def _mark_translation_done(self, translated_batch: List[tuple]):
        """标记翻译完成并检查是否可以渲染"""
        ready_to_render = 0
        for ctx, config in translated_batch:
            # ✅ 保存翻译后的text_regions 到 translation_done
            self.translation_done[ctx.image_name] = ctx.text_regions
            
            # ✅ 同步更新 base_contexts（因为 _apply_post_translation_processing 可能替换了 text_regions 列表）
            if ctx.image_name in self.base_contexts:
                self.base_contexts[ctx.image_name].text_regions = ctx.text_regions
            
            # 立即检查：如果修复也完成了，立即放入渲染队列
            if ctx.image_name in self.inpaint_done:
                await self.render_queue.put((ctx, config))
                ready_to_render += 1
                logger.info(f"[翻译] {ctx.image_name} 翻译+修复都完成，立即加入渲染队列")
        
        # ✅ 小批量完成后，检查是否需要刷新术语（每偶数个小批量刷新一次）
        from .translators.glossary_cache import on_batch_translation_complete
        on_batch_translation_complete()
        
        if ready_to_render > 0:
            logger.info(f"[翻译] 批次中 {ready_to_render}/{len(translated_batch)} 张图片立即加入渲染队列")
        else:
            logger.debug(f"【翻译】 批次中 0/{len(translated_batch)} 张图片完成修复，等待修复完成后加入渲染队列")
    
    async def _process_retry_queue(self):
        """
        处理延迟重试队列（兼容旧逻辑，当前由Worker并行处理）
        """
        if self.retry_queue.empty():
            return
        
        # 将队列内容一次性取出并并行处理
        retry_items = []
        while True:
            try:
                retry_items.append(self.retry_queue.get_nowait())
                self.retry_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        retry_count = len(retry_items)
        if retry_count == 0:
            return
        
        logger.info(f"[翻译] ✅ 开始处理延迟重试队列: {retry_count} 个任务")
        
        running_tasks = set()
        success_count = 0
        fail_count = 0
        
        for idx, (ctx, config, error_type) in enumerate(retry_items):
            if self.stop_workers or self.has_critical_error:
                logger.warning(f"[延迟重试] 检测到停止信号，中断处理")
                break
            
            while len(running_tasks) >= self.rate_limiter.concurrency:
                done, running_tasks = await asyncio.wait(running_tasks, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                        if result:
                            success_count += 1
                        else:
                            fail_count += 1
                    except Exception as e:
                        logger.error(f"[延迟重试] 任务异常: {e}")
                        fail_count += 1
            
            logger.info(f"[延迟重试] 处理 {idx+1}/{retry_count}: {ctx.image_name} (错误类型: {error_type})")
            task = asyncio.create_task(self._retry_single_image(ctx, config, error_type))
            running_tasks.add(task)
        
        if running_tasks:
            results = await asyncio.gather(*running_tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    fail_count += 1
                elif result:
                    success_count += 1
                else:
                    fail_count += 1

    def _format_same_batch_prev_original_context(self, prev_ctxs) -> str:
        """Build a compact same-batch context string from up to 2 previous images' original texts."""
        try:
            import os

            if not prev_ctxs:
                return ""

            blocks = []
            # Keep it compact to avoid prompt bloat.
            max_items_per_image = 8
            max_chars_total = 1200
            total = 0

            for idx, pctx in enumerate(prev_ctxs, start=1):
                name = getattr(pctx, 'image_name', '') or ''
                base = os.path.basename(name) if name else ''

                texts = []
                for region in getattr(pctx, 'text_regions', []) or []:
                    raw = getattr(region, 'text_raw', None)
                    t = raw if raw is not None else getattr(region, 'text', None)
                    if t is None:
                        continue
                    s = str(t).strip().replace('\n', ' ')
                    if not s:
                        continue
                    texts.append(s)

                if not texts:
                    continue

                header = f"同批次前序原文参考#{idx}（{base}）:" if base else f"同批次前序原文参考#{idx}:"
                lines = []
                for j, s in enumerate(texts[:max_items_per_image], start=1):
                    if total + len(s) > max_chars_total:
                        break
                    lines.append(f"<|{j}|>{s}")
                    total += len(s)

                if lines:
                    blocks.append(header + "\n" + "\n".join(lines))

                if total >= max_chars_total:
                    break

            return "\n\n---\n\n".join(blocks) if blocks else ""
        except Exception:
            return ""

    async def _process_retry_queue_worker(self, worker_id: int):
        """
        Worker 在收到切换信号后处理延迟重试队列（多 Worker 并行）
        
        流程：
        1. 等待所有 Worker 都进入延迟队列处理模式
        2. 并行处理延迟队列中的任务
        3. 队列清空后退出
        """
        # ✅ 等待所有 Worker 都进入延迟队列模式（确保没有 Worker 还在处理主队列并可能往延迟队列添加任务）
        wait_start = asyncio.get_event_loop().time()
        while self._main_workers_finished < self._main_workers_target:
            if self.stop_workers or self.has_critical_error:
                break
            await asyncio.sleep(0.2)
            # 超时保护（60秒）
            if asyncio.get_event_loop().time() - wait_start > 60:
                logger.warning(f"[Worker-{worker_id}] 等待其他Worker超时，继续处理延迟队列")
                break
        
        retry_size = self.retry_queue.qsize()
        if retry_size > 0:
            logger.info(f"[Worker-{worker_id}] 开始处理延迟重试队列（{retry_size} 个任务）")
        else:
            logger.info(f"[Worker-{worker_id}] 延迟队列为空，直接退出")
            return

        success_count = 0
        fail_count = 0

        while True:
            if self.stop_workers or self.has_critical_error:
                logger.warning(f"[Worker-{worker_id}] 检测到停止信号，中断延迟队列处理")
                break

            try:
                ctx, config, error_type = await asyncio.wait_for(self.retry_queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # ✅ 所有 Worker 都已进入延迟模式且队列为空，则退出
                if self._main_workers_finished >= self._main_workers_target and self.retry_queue.empty():
                    break
                continue

            try:
                logger.info(f"[延迟重试] Worker-{worker_id} 处理: {ctx.image_name} (错误类型: {error_type})")
                ok = await self._retry_single_image(ctx, config, error_type)
                if ok:
                    success_count += 1
                else:
                    fail_count += 1
            except Exception as e:
                logger.error(f"[延迟重试] Worker-{worker_id} 任务异常: {e}")
                fail_count += 1
            finally:
                self.retry_queue.task_done()

        if success_count > 0 or fail_count > 0:
            logger.info(f"[Worker-{worker_id}] 延迟队列处理完成: 成功 {success_count} 张，失败 {fail_count} 张")
    
    async def _retry_single_image(self, ctx, config, error_type: str) -> bool:
        """
        重试单张图片的翻译
        
        Args:
            ctx: 图片上下文
            config: 配置
            error_type: 错误类型 ('safety_limit' 等)
            
        Returns:
            bool: 是否成功
        """
        single_batch = [(ctx, config)]
        
        try:
            # 使用速率限制器（确保不超过并发限制）
            async with self.rate_limiter:
                translated_single = await self.translator._batch_translate_contexts(single_batch, 1)
            
            # 标记翻译完成
            await self._mark_translation_done(translated_single)
            
            # ✅ 增加翻译计数
            self.stats['translation'] += 1
            
            # ✅ 重试成功，移除失败记录
            try:
                remove_failure_record(ctx.image_name)
            except Exception:
                pass
            
            self.rate_limiter.report_success()
            logger.info(f"[延迟重试] 成功: {ctx.image_name}")
            return True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[延迟重试] 失败: {ctx.image_name} - {error_msg}")
            
            # 标记失败
            ctx.translation_error = error_msg
            ctx.success = False
            self.translation_done[ctx.image_name] = 'FAILED'
            self.failed_images.append((ctx.image_name, error_msg))
            
            # ✅ 保存失败记录（包含上下文，供下次重试）
            try:
                historical_ctx = {
                    'all_page_translations': self.translator.all_page_translations.copy() if hasattr(self.translator, 'all_page_translations') else [],
                    '_original_page_texts': self.translator._original_page_texts.copy() if hasattr(self.translator, '_original_page_texts') else [],
                    'context_size': getattr(self.translator, 'context_size', 3),
                }
                save_failure_record(
                    image_path=ctx.image_name,
                    error_type=error_type,
                    error_message=error_msg,
                    ctx=ctx,
                    config=config,
                    historical_context=historical_ctx
                )
            except Exception as save_err:
                logger.debug(f"[失败记录] 保存异常: {save_err}")
            
            # ✅ 失败也增加翻译计数（表示已处理）
            self.stats['translation'] += 1
            return False
    
    async def _wait_for_inpaint_stop(self):
        """
        后台任务：等待修复线程停止后恢复翻译并发数
        
        流程：
        1. await self.inpaint_stopped_event.wait() 阻塞等待
        2. 修复线程停止时调用 event.set() 唤醒此任务
        3. 执行恢复翻译并发数的逻辑
        """
        try:
            logger.debug("【后台任务】 等待修复线程停止...")
            
            # 阻塞等待修复线程停止事件
            await self.inpaint_stopped_event.wait()
            
            logger.info("【后台任务】 修复线程已停止，开始恢复翻译并发数")
            
            # 切换术语缓存策略（从直接写入切换为延迟刷新）
            if not self.glossary_mode_switched:
                from .translators.glossary_cache import set_translation_concurrency
                set_translation_concurrency(self.translation_concurrency)
                self.glossary_mode_switched = True
                logger.info(f"【后台任务】 术语切换为延迟刷新模式（并发数: {self.translation_concurrency}）")
            
            # 恢复用户配置的并发数
            # 通过 expand_concurrency() 扩展 Semaphore 槽位，不替换对象
            # 正在 async with 中的任务不受影响
            if not self.rpm_restored:
                old_concurrency = self.rate_limiter.concurrency
                
                # 扩展并发数（通过多次 release() 增加槽位）
                if self.translation_concurrency > old_concurrency:
                    self.rate_limiter.expand_concurrency(self.translation_concurrency)
                
                self.rpm_restored = True
                logger.info(f"【后台任务】 恢复翻译线程数: {old_concurrency} -> {self.translation_concurrency}")
            
        except asyncio.CancelledError:
            logger.debug("【后台任务】 被取消")
        except Exception as e:
            logger.error(f"【后台任务】 错误: {e}")
    
    async def _inpaint_worker(self):
        """
        修复工作线程
        从修复队列中取出上下文，进行修复
        """
        # 让出控制权，确保线程能被调度
        await asyncio.sleep(0)
        
        logger.info("[修复线程] 启动")
        
        while not self.stop_workers:
            try:
                # 如果发生严重错误，立即退出
                if self.has_critical_error:
                    logger.warning(f"[修复] 检测到严重错误，停止修复 (已完成 {self.stats['inpaint']}/{self.total_images})")
                    break
                
                # 检查是否完成所有任务：检测+OCR完成 且 队列为空
                if self.detection_ocr_done and self.inpaint_queue.empty():
                    # 再等待一小段时间，确保没有新任务
                    await asyncio.sleep(0.5)
                    if self.inpaint_queue.empty():
                        logger.info(f"[修复线程] 所有任务已完成 ({self.stats['inpaint']}/{self.total_images})")
                        break
                
                # 尝试获取任务（超时1秒）
                try:
                    image_name, config = await asyncio.wait_for(self.inpaint_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 超时，检查是否所有工作都完成了
                    # 使用 self.stats['inpaint'] 因为它包含了无文本图片的计数
                    total_processed = self.stats['inpaint'] + len(self.failed_images)
                    # 检查退出条件：OCR完成 + 修复队列为空 + 已处理数 >= 总数
                    logger.debug(f"[修复] 超时检查: detection_ocr_done={self.detection_ocr_done}, "
                                f"inpaint_queue_empty={self.inpaint_queue.empty()}, "
                                f"inpaint_stats={self.stats['inpaint']}, failed={len(self.failed_images)}, "
                                f"total_processed={total_processed}/{self.total_images}")
                    if self.detection_ocr_done and self.inpaint_queue.empty():
                        if total_processed >= self.total_images:
                            logger.info(f"[修复线程] 所有图片处理完成（成功: {self.stats['inpaint']}, 失败: {len(self.failed_images)}），结束")
                            break
                    continue
                
                # ✅ 从base_contexts获取ctx
                ctx = self.base_contexts.get(image_name)
                if not ctx:
                    logger.error(f"[修复] 找不到 {image_name} 的基础上下文")
                    continue
                
                logger.info(f"[修复] 处理: {ctx.image_name}")

                serialize_gpu = self._should_serialize_gpu_ops(config)
                
                # Mask refinement（在线程池中执行）
                if ctx.mask is None and ctx.text_regions:
                    loop = asyncio.get_event_loop()
                    mask_func = functools.partial(
                        asyncio.run,
                        self.translator._run_mask_refinement(config, ctx)
                    )
                    if serialize_gpu:
                        async with self._gpu_op_lock:
                            ctx.mask = await loop.run_in_executor(self.executor, mask_func)
                    else:
                        ctx.mask = await loop.run_in_executor(self.executor, mask_func)
                
                # Inpainting（在线程池中执行）
                if ctx.text_regions:
                    loop = asyncio.get_event_loop()
                    inpaint_func = functools.partial(
                        asyncio.run,
                        self.translator._run_inpainting(config, ctx)
                    )
                    if serialize_gpu:
                        async with self._gpu_op_lock:
                            ctx.img_inpainted = await loop.run_in_executor(self.executor, inpaint_func)
                    else:
                        ctx.img_inpainted = await loop.run_in_executor(self.executor, inpaint_func)
                
                self.stats['inpaint'] += 1
                logger.info(f"[修复] 完成: {ctx.image_name} ({self.stats['inpaint']}/{self.total_images})")
                
                # ✅ 标记修复完成（img_inpainted已经设置到base_contexts中的ctx了）
                self.inpaint_done[ctx.image_name] = True
                
                # 如果翻译也完成了，放入渲染队列
                if ctx.image_name in self.translation_done:
                    # 检查是否翻译失败（translation_done为'FAILED'表示失败）
                    translated_regions = self.translation_done.get(ctx.image_name)
                    if translated_regions == 'FAILED':
                        # 翻译失败，跳过修复和渲染，不输出任何文件
                        logger.warning(f"[修复] {ctx.image_name} 翻译失败，跳过所有后续处理")
                        continue
                    # ✅ 从 base_contexts获取完整的ctx，合并翻译和修复结果
                    render_ctx = self.base_contexts.get(ctx.image_name)
                    if render_ctx:
                        # 使用翻译后的text_regions
                        # 确保translated_regions是列表类型
                        if isinstance(translated_regions, (list, tuple)):
                            render_ctx.text_regions = translated_regions
                        elif translated_regions:
                            logger.warning(f"[修复] {ctx.image_name} 的翻译结果类型异常: {type(translated_regions)}, 使用空列表")
                            render_ctx.text_regions = []
                        else:
                            render_ctx.text_regions = []
                        # img_inpainted已经在上面设置好了
                        await self.render_queue.put((render_ctx, config))
                        logger.info(f"[修复] {ctx.image_name} 翻译+修复都完成，加入渲染队列")
                    else:
                        logger.error(f"[修复] 找不到 {ctx.image_name} 的基础上下文")
                
            except Exception as e:
                logger.error(f"[修复线程] 错误: {e}")
                # 标记严重错误，停止所有线程
                self.has_critical_error = True
                self.critical_error_msg = f"修复线程错误: {e}"
                self.critical_error_exception = e
                self.stop_workers = True
                break
        
        self.inpaint_worker_stopped = True  # 标记修复线程已停止
        self.inpaint_stopped_event.set()  # 触发事件，唤醒等待的后台任务
        logger.info("【修复线程】 停止，已触发 inpaint_stopped_event")
    
    async def _render_worker(self, results: List[Context]):
        """
        渲染工作线程
        从渲染队列中取出上下文，进行渲染
        渲染完成后立即清理内存
        """
        # 让出控制权，确保线程能被调度
        await asyncio.sleep(0)
        
        logger.info("[渲染线程] 启动")
        
        rendered_count = 0
        
        while not self.stop_workers:
            try:
                # 如果发生严重错误，立即退出
                if self.has_critical_error:
                    logger.warning(f"[渲染] 检测到严重错误，停止渲染 (已完成 {rendered_count}/{self.total_images})")
                    break
                
                # 尝试获取任务（超时1秒）
                try:
                    ctx, config = await asyncio.wait_for(self.render_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    # 超时，检查是否所有工作都完成了
                    # 成功数 + 失败数 = 总数，则结束
                    total_processed = rendered_count + len(self.failed_images)
                    logger.debug(f"[渲染] 超时检查: detection_ocr_done={self.detection_ocr_done}, "
                                f"translation_done={len(self.translation_done)}/{self.total_images}, "
                                f"rendered={rendered_count}, failed={len(self.failed_images)}, "
                                f"total_processed={total_processed}/{self.total_images}")
                    if self.detection_ocr_done and len(self.translation_done) >= self.total_images:
                        if total_processed >= self.total_images:
                            logger.info(f"[渲染线程] 所有图片处理完成（成功: {rendered_count}, 失败: {len(self.failed_images)}），结束")
                            break
                    continue
                
                logger.info(f"[渲染] 从队列获取任务: {ctx.image_name} (队列剩余: {self.render_queue.qsize()})")
                # ✅ 验证：确保ctx是正确的（通过image_name匹配）
                # 从 base_contexts重新获取，确保使用最新的数据
                verified_ctx = self.base_contexts.get(ctx.image_name)
                if not verified_ctx:
                    logger.error(f"[渲染] 找不到 {ctx.image_name} 的基础上下文，跳过")
                    # 增加失败计数，避免卡住
                    self.failed_images.append((ctx.image_name, "找不到基础上下文"))
                    continue
                
                # 使用验证后的ctx
                ctx = verified_ctx
                logger.info(f"[渲染] 开始处理: {ctx.image_name}")
                
                # ✅ 检查渲染所需的数据是否完整
                if not hasattr(ctx, 'img_rgb') or ctx.img_rgb is None:
                    logger.error(f"[渲染] ctx.img_rgb 为 None，无法渲染！跳过此图片")
                    ctx.translation_error = "渲染失败：缺少原始图片数据"
                    # 增加失败计数
                    self.failed_images.append((ctx.image_name, "缺少原始图片数据"))
                    continue
                
                # 调试：检查关键数据
                logger.debug(f"[渲染调试] img_rgb shape: {ctx.img_rgb.shape if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None else 'None'}")
                logger.debug(f"[渲染调试] img_inpainted shape: {ctx.img_inpainted.shape if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None else 'None'}")
                logger.debug(f"[渲染调试] text_regions count: {len(ctx.text_regions) if isinstance(ctx.text_regions, (list, tuple)) else 0}")
                if isinstance(ctx.text_regions, (list, tuple)) and ctx.text_regions:
                    for i, region in enumerate(ctx.text_regions[:3]):  # 只显示前3个
                        logger.debug(f"[渲染调试] Region {i}: translation='{region.translation[:30]}...', font_size={region.font_size}, xywh={region.xywh}")
                
                # ✅ 在渲染之前先保存修复后图片的副本（用于后续保存到inpainted目录）
                img_inpainted_copy = None
                if (self.translator.save_text or self.translator.text_output_file) and hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
                    import numpy as np
                    img_inpainted_copy = np.copy(ctx.img_inpainted)
                    logger.debug(f"[渲染] 已备份修复后图片用于保存")
                
                # 检查是否跳过渲染（翻译失败的图片直接输出原图）
                if hasattr(ctx, 'skip_render') and ctx.skip_render:
                    # 直接使用原图
                    from .utils.generic import dump_image
                    ctx.result = dump_image(ctx.input, ctx.img_rgb, ctx.img_alpha)
                    logger.info(f"[渲染] 跳过渲染，直接输出原图: {ctx.image_name}")
                elif not ctx.text_regions:
                    # 无文本，直接使用upscaled
                    from .utils.generic import dump_image
                    ctx.result = dump_image(ctx.input, ctx.upscaled, ctx.img_alpha)
                else:
                    # 渲染（在线程池中执行）
                    # img_rgb和img_inpainted已经在修复阶段更新为upscaled版本
                    loop = asyncio.get_event_loop()
                    render_func = functools.partial(
                        asyncio.run,
                        self.translator._run_text_rendering(config, ctx)
                    )
                    ctx.img_rendered = await loop.run_in_executor(self.executor, render_func)
                    
                    # 使用dump_image合并alpha通道（与标准流程一致）
                    from .utils.generic import dump_image
                    ctx.result = dump_image(ctx.input, ctx.img_rendered, ctx.img_alpha)
                
                self.stats['rendering'] += 1
                rendered_count += 1
                logger.info(f"[渲染] 完成: {ctx.image_name} ({self.stats['rendering']}/{self.total_images})")
                
                # ✅ 每张图片渲染完成后发送进度更新（UI进度条适配）
                try:
                    await self.translator._report_progress(f"batch:1:{rendered_count}:{self.total_images}")
                except Exception as e:
                    logger.debug(f"[渲染] 进度报告失败（可忽略）: {e}")
                
                # 检查result是否存在
                if ctx.result is not None:
                    logger.info(f"[渲染] ctx.result 已设置，类型: {type(ctx.result)}")
                    
                    # 立即保存图片和JSON
                    try:
                        # 获取save_info（从translator获取）
                        if hasattr(self.translator, '_current_save_info') and self.translator._current_save_info:
                            save_info = self.translator._current_save_info
                            
                            # 保存图片（与主流程一致：不管是否保存成功都继续）
                            overwrite = save_info.get('overwrite', True)
                            final_output_path = self.translator._calculate_output_path(ctx.image_name, save_info)
                            
                            # ✅ 保存修复后的图片到inpainted目录（使用渲染前备份的副本）
                            if img_inpainted_copy is not None:
                                try:
                                    from .utils.path_manager import get_inpainted_path
                                    from .utils.generic import imwrite_unicode
                                    import cv2
                                    
                                    inpainted_path = get_inpainted_path(ctx.image_name, create_dir=True)
                                    imwrite_unicode(inpainted_path, cv2.cvtColor(img_inpainted_copy, cv2.COLOR_RGB2BGR), logger)
                                    logger.info(f"[渲染] 修复后图片已保存: {inpainted_path}")
                                except Exception as e:
                                    logger.error(f"[渲染] 保存修复后图片失败: {e}")
                                finally:
                                    # 释放副本内存
                                    del img_inpainted_copy
                                    img_inpainted_copy = None
                            
                            # 如果是失败的图片，在文件名前缀加failed_
                            if hasattr(ctx, 'is_failed') and ctx.is_failed:
                                # 在文件名前加failed_
                                dir_path = os.path.dirname(final_output_path)
                                file_name = os.path.basename(final_output_path)
                                final_output_path = os.path.join(dir_path, f"failed_{file_name}")
                                logger.info(f"[渲染] 失败图片保存为: {os.path.basename(final_output_path)}")
                            
                            self.translator._save_translated_image(ctx.result, final_output_path, ctx.image_name, overwrite, "CONCURRENT")
                            
                            # 保存JSON（如果需要）
                            if (self.translator.save_text or self.translator.text_output_file) and ctx.text_regions is not None:
                                self.translator._save_text_to_file(ctx.image_name, ctx, config)
                        else:
                            logger.warning(f"[渲染] 无save_info，跳过保存")
                        
                        # 标记成功/失败
                        if hasattr(ctx, 'is_failed') and ctx.is_failed:
                            ctx.success = False  # 翻译失败的图片标记为失败
                        else:
                            ctx.success = True
                                
                    except Exception as save_err:
                        logger.error(f"[渲染] 保存失败 {os.path.basename(ctx.image_name)}: {save_err}")
                        logger.error(traceback.format_exc())
                        ctx.translation_error = str(save_err)
                else:
                    logger.error(f"[渲染] ctx.result 为 None！")
                
                # 添加到结果列表
                results.append(ctx)
                
                # 渲染完成后立即清理内存（注意：_run_text_rendering 已经清理了 ctx.img_rgb）
                logger.debug(f"[渲染] 清理内存: {ctx.image_name}")
                # ctx.img_rgb 已在 _run_text_rendering 中清理
                if hasattr(ctx, 'img_rgb') and ctx.img_rgb is not None:
                    del ctx.img_rgb
                    ctx.img_rgb = None
                # 保留 ctx.img_alpha 用于dump_image
                if hasattr(ctx, 'img_colorized') and ctx.img_colorized is not None:
                    del ctx.img_colorized
                    ctx.img_colorized = None
                if hasattr(ctx, 'upscaled') and ctx.upscaled is not None:
                    del ctx.upscaled
                    ctx.upscaled = None
                if hasattr(ctx, 'mask') and ctx.mask is not None:
                    del ctx.mask
                    ctx.mask = None
                if hasattr(ctx, 'img_inpainted') and ctx.img_inpainted is not None:
                    del ctx.img_inpainted
                    ctx.img_inpainted = None
                if hasattr(ctx, 'img_rendered') and ctx.img_rendered is not None:
                    del ctx.img_rendered
                    ctx.img_rendered = None
                # 保留 ctx.result 和 ctx.img_alpha 用于保存
                
                # 强制垃圾回收
                import gc
                gc.collect()
                
                # ✅ 清理base_contexts中的ctx，释放内存
                if ctx.image_name in self.base_contexts:
                    del self.base_contexts[ctx.image_name]
                    logger.debug(f"[渲染] 已清理 {ctx.image_name} 的基础上下文")
                
            except Exception as e:
                logger.error(f"[渲染线程] 错误: {e}")
                # 标记严重错误，停止所有线程
                self.has_critical_error = True
                self.critical_error_msg = f"渲染线程错误: {e}"
                self.critical_error_exception = e
                self.stop_workers = True
                break
        
        logger.info("[渲染线程] 停止")
    
    async def process_batch(self, file_paths: List[str], configs: List) -> List[Context]:
        """
        并发处理一批图片（流水线模式，分批加载）
        
        Args:
            file_paths: 图片文件路径列表
            configs: 配置列表
            
        Returns:
            处理完成的Context列表
        """
        self.total_images = len(file_paths)
        self.start_time = datetime.now(timezone.utc)
        self.all_file_paths = file_paths  # ✅ 保存原始文件路径列表，用于取消时统计
        
        logger.info(f"[并发流水线] 开始处理 {self.total_images} 张图片")
        logger.info(f"[并发流水线] 流水线模式: 检测+OCR（顺序，分批加载）→ 翻译线程（批量={self.batch_size}）+ 修复线程 + 渲染线程")
        
        # 重置统计
        for key in self.stats:
            self.stats[key] = 0
        self.translation_done.clear()
        self.inpaint_done.clear()
        self.base_contexts.clear()  # ✅ 清理基础上下文
        self.detection_ocr_done = False  # 重置标志
        self.failed_images.clear()  # 重置失败图片列表
        self.translation_enqueued_total = 0
        
        # 结果列表
        results = []
        
        # 启动4个工作线程
        tasks = [
            asyncio.create_task(self._detection_ocr_worker(file_paths, configs)),
            asyncio.create_task(self._translation_worker()),
            asyncio.create_task(self._inpaint_worker()),
            asyncio.create_task(self._render_worker(results))
        ]
        
        # 是否被取消
        was_cancelled = False
        
        try:
            # 等待所有任务完成
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning("[并发流水线] 任务被取消")
            was_cancelled = True
            self.stop_workers = True
        except Exception as e:
            logger.error(f"[并发流水线] 错误: {e}")
            logger.error(traceback.format_exc())
            self.stop_workers = True
            raise
        finally:
            self.stop_workers = True
            # 关闭线程池
            self.executor.shutdown(wait=True)
        
        # 检查是否有严重错误（但取消不算严重错误）
        if self.has_critical_error and not was_cancelled:
            error_msg = self.critical_error_msg or "未知错误"
            logger.error(f"[并发流水线] 处理失败: {error_msg}")
            # 重新抛出原始异常（保留完整的异常链）
            if self.critical_error_exception:
                raise self.critical_error_exception
            else:
                raise RuntimeError(f"并发流水线处理失败: {error_msg}")
        
        # 统计
        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        success_count = self.stats['rendering']
        fail_count = len(self.failed_images)
        
        # 如果被取消，将未处理的图片标记为失败
        if was_cancelled:
            # 收集已处理的图片名（成功渲染的 + 已失败的）
            processed_images = set()
            for ctx in results:
                if hasattr(ctx, 'image_name') and ctx.image_name:
                    processed_images.add(ctx.image_name)
            for img_name, _ in self.failed_images:
                processed_images.add(img_name)
            
            # 从原始文件路径列表中找出未处理的图片
            for img_path in self.all_file_paths:
                if img_path not in processed_images:
                    self.failed_images.append((img_path, "用户取消"))
            
            fail_count = len(self.failed_images)
            logger.warning(f"[并发流水线] 任务被取消，已处理 {success_count} 张，未处理 {fail_count} 张")
        else:
            logger.info(f"[并发流水线] 完成！")
        
        logger.info(f"  总耗时: {elapsed:.2f}秒")
        logger.info(f"  成功: {success_count} 张，失败: {fail_count} 张")
        if self.total_images > 0:
            logger.info(f"  平均速度: {elapsed/self.total_images:.2f}秒/张")
        logger.info(f"  处理统计: 检测+OCR={self.stats['detection_ocr']}, "
                   f"翻译={self.stats['translation']}, 修复={self.stats['inpaint']}, "
                   f"渲染={self.stats['rendering']}")
        
        # 输出失败图片列表
        if self.failed_images:
            if was_cancelled:
                logger.warning(f"[并发流水线] ⚠️ 以下 {fail_count} 张图片未完成:")
            else:
                logger.warning(f"[并发流水线] ⚠️ 以下 {fail_count} 张图片翻译失败:")
            for img_name, error_msg in self.failed_images:
                # 只显示文件名，不显示完整路径
                import os
                short_name = os.path.basename(img_name)
                # 截断过长的错误信息
                short_error = error_msg[:100] + '...' if len(error_msg) > 100 else error_msg
                logger.warning(f"  ❌ {short_name}: {short_error}")
            
            # ✅ 将失败的图片也添加到结果列表中，以便前端记录
            from .utils import Context
            for img_name, error_msg in self.failed_images:
                failed_ctx = Context()
                failed_ctx.image_name = img_name
                failed_ctx.translation_error = error_msg
                failed_ctx.success = False
                results.append(failed_ctx)
        
        return results
