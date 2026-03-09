import asyncio
import gc
import logging
import os
import traceback
from datetime import datetime, timezone
from typing import List

from .concurrent_pipeline import ConcurrentPipeline
from .utils import Context

logger = logging.getLogger('manga_translator')


class T2SPipeline(ConcurrentPipeline):
    """Dedicated concurrent pipeline for T2S batch jobs."""

    def _finalize_completed_context(self, ctx: Context, results: List[Context]):
        logger.debug(f"[T2S][render] cleanup: {ctx.image_name}")

        image_name = getattr(ctx, 'image_name', None)
        self.translation_done.pop(image_name, None)
        self.inpaint_done.pop(image_name, None)

        light_ctx = Context()
        for attr in ('image_name', 'success', 'translation_error', 'skipped'):
            if hasattr(ctx, attr):
                setattr(light_ctx, attr, getattr(ctx, attr))

        closed_ids = set()
        for attr in ('input', 'result'):
            value = getattr(ctx, attr, None)
            if value is None:
                continue
            close = getattr(value, 'close', None)
            if callable(close) and id(value) not in closed_ids:
                try:
                    close()
                except Exception:
                    pass
                closed_ids.add(id(value))
            setattr(ctx, attr, None)

        for attr in (
            'img_rgb',
            'img_colorized',
            'upscaled',
            'mask',
            'mask_raw',
            'img_inpainted',
            'img_rendered',
            'img_alpha',
            'textlines',
            'text_regions',
            'config',
            'rate_limiter',
        ):
            if getattr(ctx, attr, None) is not None:
                setattr(ctx, attr, None)

        results.append(light_ctx)
        gc.collect()

        if image_name in self.base_contexts:
            del self.base_contexts[image_name]
            logger.debug(f"[T2S][render] base context cleared: {image_name}")

    def _should_stop_render_worker(self, rendered_count: int) -> bool:
        total_processed = rendered_count + len(self.failed_images)
        if not self.detection_ocr_done or total_processed < self.total_images:
            return False
        if not self.translation_queue.empty():
            return False
        if not self.render_queue.empty():
            return False
        return self.inpaint_stopped_event.is_set()


    async def process_batch(self, file_paths: List[str], configs: List) -> List[Context]:
        self.total_images = len(file_paths)
        self.start_time = datetime.now(timezone.utc)
        self.all_file_paths = file_paths

        logger.info(f"[T2S专属流水线] 开始处理 {self.total_images} 张图片")
        logger.info(
            "[T2S专属流水线] 流程: 检测+OCR -> T2S转换 -> 修复线程 + 渲染线程"
        )

        self.stop_workers = False
        self.has_critical_error = False
        self.critical_error_msg = None
        self.critical_error_exception = None
        self._ocr_can_run_event.set()

        self.translation_queue = asyncio.Queue()
        self.inpaint_queue = asyncio.Queue()
        self.render_queue = asyncio.Queue(maxsize=20)
        self.retry_queue = asyncio.Queue()

        for key in self.stats:
            self.stats[key] = 0
        self.translation_done.clear()
        self.inpaint_done.clear()
        self.base_contexts.clear()
        self.detection_ocr_done = False
        self.failed_images.clear()
        self.translation_enqueued_total = 0
        self.inpaint_worker_stopped = False
        self.inpaint_stopped_event.clear()
        self._inpaint_workers_finished = 0
        self._is_t2s_mode = True

        results = []

        inpaint_tasks = [
            asyncio.create_task(self._inpaint_worker(i))
            for i in range(self.inpaint_concurrency)
        ]
        tasks = [
            asyncio.create_task(self._detection_ocr_worker(file_paths, configs)),
            asyncio.create_task(self._translation_worker_t2s()),
            asyncio.create_task(self._render_worker(results)),
            *inpaint_tasks,
        ]

        was_cancelled = False

        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            logger.warning('[T2S专属流水线] 任务被取消')
            was_cancelled = True
            self.stop_workers = True
        except Exception as e:
            logger.error(f"[T2S专属流水线] 错误: {e}")
            logger.error(traceback.format_exc())
            self.stop_workers = True
            raise
        finally:
            self.stop_workers = True
            self.executor.shutdown(wait=True)

        if self.has_critical_error and not was_cancelled:
            error_msg = self.critical_error_msg or '未知错误'
            logger.error(f"[T2S专属流水线] 处理失败: {error_msg}")
            if self.critical_error_exception:
                raise self.critical_error_exception
            raise RuntimeError(f"T2S专属流水线处理失败: {error_msg}")

        elapsed = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        success_count = self.stats['rendering']
        fail_count = len(self.failed_images)

        if was_cancelled:
            processed_images = set()
            for ctx in results:
                if hasattr(ctx, 'image_name') and ctx.image_name:
                    processed_images.add(ctx.image_name)
            for img_name, _ in self.failed_images:
                processed_images.add(img_name)

            for img_path in self.all_file_paths:
                if img_path not in processed_images:
                    self.failed_images.append((img_path, '用户取消'))

            fail_count = len(self.failed_images)
            logger.warning(
                f"[T2S专属流水线] 已处理 {success_count} 张，未完成 {fail_count} 张"
            )
        else:
            logger.info("[T2S专属流水线] 完成")

        logger.info(f"  总耗时: {elapsed:.2f}秒")
        logger.info(f"  成功: {success_count} 张，失败: {fail_count} 张")
        if self.total_images > 0:
            logger.info(f"  平均速度: {elapsed / self.total_images:.2f}秒/张")
        logger.info(
            f"  处理统计: 检测+OCR={self.stats['detection_ocr']}, "
            f"翻译={self.stats['translation']}, 修复={self.stats['inpaint']}, "
            f"渲染={self.stats['rendering']}"
        )

        if self.failed_images:
            if was_cancelled:
                logger.warning(f"[T2S专属流水线] 以下 {fail_count} 张图片未完成:")
            else:
                logger.warning(f"[T2S专属流水线] 以下 {fail_count} 张图片处理失败:")
            for img_name, error_msg in self.failed_images:
                short_name = os.path.basename(img_name)
                short_error = error_msg[:100] + '...' if len(error_msg) > 100 else error_msg
                logger.warning(f"  - {short_name}: {short_error}")

            for img_name, error_msg in self.failed_images:
                failed_ctx = Context()
                failed_ctx.image_name = img_name
                failed_ctx.translation_error = error_msg
                failed_ctx.success = False
                results.append(failed_ctx)

        return results
