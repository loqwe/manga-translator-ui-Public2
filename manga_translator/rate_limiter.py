"""
速率限制器模块
控制翻译并发数和 RPM（每分钟请求数）

四层防护：
1. 指数退避 - 遇到 429 后暂停所有请求
2. 预热控制 - 并发扩展后渐进式启动新请求
3. 并发控制 - 限制同时发出的请求数
4. 令牌桶 - 控制每分钟总请求数，允许突发
"""
import asyncio
import logging
import threading
import time
from typing import Dict

logger = logging.getLogger('manga_translator')


class RateLimiter:
    """
    智能速率限制器：令牌桶 + 并发控制 + 指数退避兆底
    
    使用示例：
        limiter = RateLimiter(concurrency=3, rpm=60)
        async with limiter:
            result = await make_api_request()
        # 如果遇到 429 错误，调用：
        limiter.report_rate_limit_error()
    
    线程安全设计：
    - 使用普通 Semaphore 而非 BoundedSemaphore
    - 通过 _acquired_tokens 集合跟踪每个任务的获取状态
    - 确保每个 acquire 只能被 release 一次
    """
    
    def __init__(self, concurrency: int = 1, rpm: int = 0, burst: int = None, expand_window_sec: int = 180):
        """
        初始化速率限制器
        
        Args:
            concurrency: 最大并发请求数
            rpm: 每分钟最大请求数，0 表示不限制
            burst: 突发容量，默认等于 rpm（允许一开始快速发送 rpm 个请求）
            expand_window_sec: 扩展槽事件的检测窗口（秒）
        """
        self._concurrency = concurrency
        self._rpm = rpm
        self._expand_window_sec = max(0, int(expand_window_sec))
        self._last_expand_ts = 0.0
        
        # 令牌桶状态（仅当 rpm > 0 时启用）
        if rpm > 0:
            self._burst = burst if burst is not None else rpm
            self._tokens = float(self._burst)  # 当前令牌数
            self._max_tokens = float(self._burst)  # 桶容量
            self._refill_rate = rpm / 60.0  # 每秒补充的令牌数
            self._last_refill = time.time()
            self._token_lock_async = asyncio.Lock()  # 异步锁保护令牌操作
        else:
            self._burst = 0
            self._tokens = 0
            self._max_tokens = 0
            self._refill_rate = 0
            self._last_refill = 0
            self._token_lock_async = None
        
        # 指数退避状态
        self._backoff_until = 0  # 退避结束时间
        self._consecutive_errors = 0  # 连续错误次数
        self._max_backoff = 60  # 最大退避时间（秒）
        
        # 使用普通 Semaphore，配合 _acquired_tokens 实现安全释放
        self._semaphore = asyncio.Semaphore(concurrency)
        # 跟踪已获取的 token，确保每个 acquire 只被 release 一次
        self._acquired_tokens: set = set()
        self._token_counter = 0
        self._token_lock = threading.Lock()
        # 任务到 token 的映射，用于上下文管理器
        self._task_tokens: Dict[int, int] = {}
        # 并发计数器
        self._active_count = 0
        
        # 预热控制（并发扩展后渐进式启动）
        self._warmup_pending = 0  # 等待预热的请求数
        self._warmup_lock = asyncio.Lock()  # 预热锁
        
        # 日志初始化信息
        if rpm > 0:
            logger.info(f"[速率限制] 初始化: 并发数={concurrency}, RPM={rpm}, 突发容量={self._burst}")
        else:
            logger.debug(f"[速率限制] 初始化: 并发数={concurrency}, RPM=不限制")
    
    @property
    def concurrency(self) -> int:
        """获取当前并发限制"""
        return self._concurrency
    
    @property
    def rpm(self) -> int:
        """获取当前 RPM 限制"""
        return self._rpm
    
    def _refill_tokens(self):
        """补充令牌（内部方法）"""
        if self._rpm <= 0:
            return
        
        now = time.time()
        elapsed = now - self._last_refill
        
        # 按时间比例补充令牌
        new_tokens = elapsed * self._refill_rate
        self._tokens = min(self._max_tokens, self._tokens + new_tokens)
        self._last_refill = now
    
    async def _wait_for_token(self):
        """等待获取令牌（仅当 rpm > 0 时调用）"""
        if self._rpm <= 0:
            return
        
        async with self._token_lock_async:
            while True:
                self._refill_tokens()
                
                if self._tokens >= 1:
                    self._tokens -= 1
                    return
                
                # 计算需要等待多久才能获得 1 个令牌
                wait_time = (1 - self._tokens) / self._refill_rate
                # 最多等 0.5s 后重试，避免长时间阻塞
                actual_wait = min(wait_time, 0.5)
                logger.debug(f"[速率限制] 等待令牌: {actual_wait:.2f}s (当前令牌: {self._tokens:.2f})")
                
                # 释放异步锁后等待，允许其他任务补充令牌
                await asyncio.sleep(actual_wait)
    
    def report_rate_limit_error(self):
        """
        报告遇到了 429 限速错误
        
        触发指数退避：2^n 秒，最大 60 秒
        所有新请求都会等待退避结束
        """
        self._consecutive_errors += 1
        backoff_time = min(self._max_backoff, 2 ** self._consecutive_errors)
        self._backoff_until = time.time() + backoff_time
        logger.warning(f"[速率限制] 触发 429 限速，退避 {backoff_time}s (连续错误: {self._consecutive_errors})")
    
    def report_success(self):
        """
        报告请求成功
        
        重置连续错误计数器
        """
        if self._consecutive_errors > 0:
            logger.debug(f"[速率限制] 请求成功，重置错误计数器 ({self._consecutive_errors} -> 0)")
            self._consecutive_errors = 0
    
    async def _wait_for_backoff(self):
        """等待退避结束（如果正在退避中）"""
        now = time.time()
        if now < self._backoff_until:
            wait_time = self._backoff_until - now
            logger.info(f"[速率限制] 退避中，等待 {wait_time:.1f}s")
            await asyncio.sleep(wait_time)
    
    def expand_concurrency(self, new_concurrency: int, warmup: bool = True):
        """
        扩展并发数（只能增加，不能减少）
        
        通过多次调用 release() 来增加 Semaphore 的槽位数。
        正在 async with 中的任务不受影响。
        
        Args:
            new_concurrency: 新的并发限制（必须大于当前值）
            warmup: 是否启用预热模式（渐进式启动新请求）
        """
        if new_concurrency <= self._concurrency:
            return
        
        # 计算需要增加的槽位数
        slots_to_add = new_concurrency - self._concurrency
        
        # 启用预热模式：新请求每隔 2 秒启动一个
        if warmup and slots_to_add > 1:
            self._warmup_pending = slots_to_add - 1  # 第一个可以立即开始
            logger.info(f"[速率限制] 启用预热模式，{slots_to_add} 个新槽位将每隔 2s 启动")
        
        # 通过 release() 增加槽位
        for _ in range(slots_to_add):
            self._semaphore.release()
        
        old_concurrency = self._concurrency
        self._concurrency = new_concurrency
        self._last_expand_ts = time.time()
        logger.info(f"[速率限制] 并发数扩展: {old_concurrency} -> {new_concurrency}")
        logger.debug(f"[速率限制] 扩展槽标记: ts={self._last_expand_ts:.3f}, window={self._expand_window_sec}s")
    
    def update_rpm(self, new_rpm: int):
        """
        更新 RPM 限制
        
        Args:
            new_rpm: 新的 RPM 限制，0 表示不限制
        """
        old_rpm = self._rpm
        self._rpm = new_rpm
        
        if new_rpm > 0:
            self._burst = new_rpm
            self._max_tokens = float(new_rpm)
            self._refill_rate = new_rpm / 60.0
            if self._token_lock_async is None:
                self._token_lock_async = asyncio.Lock()
            # 保持当前令牌数，但不超过新的最大值
            self._tokens = min(self._tokens, self._max_tokens) if self._tokens > 0 else float(new_rpm)
            self._last_refill = time.time()
            logger.info(f"[速率限制] RPM 更新: {old_rpm} -> {new_rpm}")
        else:
            logger.info(f"[速率限制] RPM 更新: {old_rpm} -> 不限制")
    
    async def _wait_for_warmup(self):
        """
        预热等待：并发扩展后，新请求每隔 2 秒启动一个
        """
        if self._warmup_pending <= 0:
            return
        
        async with self._warmup_lock:
            # 检查是否还需要预热
            if self._warmup_pending <= 0:
                return
            
            # 消耗一个预热槽位
            self._warmup_pending -= 1
            remaining = self._warmup_pending
            
            # 等待 2 秒后启动
            logger.info(f"[速率限制] 预热: 等待 2s 后启动（剩余 {remaining} 个）")
            await asyncio.sleep(2)
            
            # 预热完成
            if remaining <= 0:
                logger.info("[速率限制] 预热完成，恢复正常并发")
    
    async def acquire(self) -> int:
        """
        获取请求许可（四层检查）
        
        Returns:
            token_id: 用于 release 时校验
        """
        # 第 1 层：检查是否在退避期
        await self._wait_for_backoff()
        
        # 第 2 层：预热控制（并发扩展后渐进启动）
        await self._wait_for_warmup()
        
        # 第 3 层：并发控制
        await self._semaphore.acquire()
        
        # 第 4 层：令牌桶（仅当 rpm > 0 时）
        await self._wait_for_token()
        
        # 生成并记录 token
        with self._token_lock:
            self._token_counter += 1
            token_id = self._token_counter
            self._acquired_tokens.add(token_id)
            self._active_count += 1
        
        return token_id
    
    async def acquire_token_only(self):
        """
        只获取 RPM 令牌，不占用并发槽位
        
        用于拆分重试场景：数量不匹配/安全限制错误后的单张重试
        只受 RPM 限制，不受并发数限制
        """
        # 第 1 层：检查是否在退避期
        await self._wait_for_backoff()
        
        # 第 2 层：令牌桶（仅当 rpm > 0 时）
        await self._wait_for_token()
    
    def release(self, token_id: int = None):
        """
        释放请求许可
        
        Args:
            token_id: acquire 返回的 token，用于防止重复释放
        """
        with self._token_lock:
            # 如果没有 token_id，尝试释放任意一个（向后兼容）
            if token_id is None:
                if self._acquired_tokens:
                    token_id = next(iter(self._acquired_tokens))
                else:
                    # 没有已获取的 token，跳过释放
                    return
            
            # 检查 token 是否有效
            if token_id not in self._acquired_tokens:
                # token 不存在，可能已经释放过，跳过
                return
            
            # 移除 token 并释放 semaphore
            self._acquired_tokens.discard(token_id)
            self._active_count = max(0, self._active_count - 1)
        
        # 在锁外释放 semaphore
        self._semaphore.release()
    
    async def __aenter__(self):
        """异步上下文管理器入口"""
        token_id = await self.acquire()
        # 使用任务 ID 关联 token
        task = asyncio.current_task()
        task_id = id(task) if task else id(asyncio.get_event_loop())
        with self._token_lock:
            self._task_tokens[task_id] = token_id
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文管理器出口"""
        task = asyncio.current_task()
        task_id = id(task) if task else id(asyncio.get_event_loop())
        with self._token_lock:
            token_id = self._task_tokens.pop(task_id, None)
        if token_id is not None:
            self.release(token_id)
        return False
    
    def get_last_expand_ts(self) -> float:
        """获取最近一次扩展槽事件时间戳（epoch 秒）"""
        return float(self._last_expand_ts or 0.0)
    
    def get_expand_window_sec(self) -> int:
        """获取扩展槽检测窗口（秒）"""
        return int(self._expand_window_sec)
    
    def is_within_expand_window(self, ts: float = None) -> bool:
        """判断给定时间戳是否落在扩展槽事件窗口内"""
        if not self._last_expand_ts or self._expand_window_sec <= 0:
            return False
        now_ts = float(ts if ts is not None else time.time())
        return abs(now_ts - self._last_expand_ts) <= self._expand_window_sec
    
    def get_stats(self) -> dict:
        """获取当前状态统计"""
        stats = {
            'concurrency_limit': self._concurrency,
            'active_requests': self._active_count,
            'rpm_limit': self._rpm,
            'consecutive_errors': self._consecutive_errors
        }
        if self._rpm > 0:
            self._refill_tokens()  # 更新令牌状态
            stats['tokens_available'] = round(self._tokens, 2)
            stats['max_tokens'] = self._max_tokens
        return stats
