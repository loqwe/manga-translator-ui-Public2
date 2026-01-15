"""
术语表缓存模块
支持并发翻译时的线程安全术语缓存和批量写入

解决的问题：
1. 术语重复：多个并发请求同时发现同一术语 → 内存中去重
2. 文件冲突：多个请求同时写入同一文件 → 延迟批量写入
3. 术语不同步：可接受的权衡，最终结果正确

使用方式：
- 翻译时：cache.add_terms(file_path, terms) 缓存到内存
- 翻译完成后：cache.flush_all() 批量写入文件
"""

import os
import sys
import sqlite3
import threading
import logging
import json
from datetime import datetime
from typing import Dict, List, Set, Optional

logger = logging.getLogger('manga_translator')


class GlossaryCache:
    """
    线程安全的术语缓存（单例模式）
    
    支持：
    - 并发写入（线程安全）
    - 内存去重（避免重复术语）
    - 批量写入（减少IO）
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """单例模式：确保全局只有一个缓存实例"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._init()
        return cls._instance
    
    def _init(self):
        """初始化缓存数据结构"""
        # 缓存：file_path -> terms 列表
        self._cache: Dict[str, List[dict]] = {}
        # 已见术语：file_path -> 已见过的去重Key集合（用于去重）
        # Key 结构："{work_name}::{category}::{original}"（work_name/category 可能为空）
        self._seen: Dict[str, Set[str]] = {}
        # 缓存操作锁
        self._cache_lock = threading.Lock()
        # 统计信息
        self._stats = {
            'added': 0,      # 添加的术语数
            'duplicated': 0,  # 去重跳过的术语数
            'flushed': 0     # 写入文件的术语数
        }
    
    def add_terms(self, file_path: str, terms: List[dict]) -> int:
        """
        添加术语到缓存（线程安全）
        
        Args:
            file_path: 提示词文件路径
            terms: 术语列表 [{"original": "...", "translation": "...", "category": "..."}]
        
        Returns:
            实际添加的术语数（去重后）
        """
        if not terms or not file_path:
            return 0
        
        added_count = 0
        
        with self._cache_lock:
            # 初始化该文件的缓存
            if file_path not in self._cache:
                self._cache[file_path] = []
                self._seen[file_path] = set()
            
            for term in terms:
                original = term.get("original", "")
                if not original:
                    continue

                work_name = (term.get("work_name") or "").strip()
                raw_work_name = (term.get("raw_work_name") or "").strip()
                category = str(term.get("category") or "").strip().lower()

                # work_name 为空时，用 raw_work_name 区分不同“无作品区”来源，避免互相去重覆盖
                work_key = work_name or (f"unassigned:{raw_work_name}" if raw_work_name else "")
                dedup_key = f"{work_key}::{category}::{original}"
                
                # 内存中去重：已存在则跳过
                if dedup_key in self._seen[file_path]:
                    self._stats['duplicated'] += 1
                    logger.debug(f"[术语缓存] 跳过重复术语: {dedup_key}")
                    continue
                
                # 添加到缓存
                self._cache[file_path].append(term)
                self._seen[file_path].add(dedup_key)
                added_count += 1
                self._stats['added'] += 1
                work_display = work_name or (f"unassigned:{raw_work_name}" if raw_work_name else "N/A")
                logger.debug(f"[术语缓存] 添加术语: {original} -> {term.get('translation', '')} (work={work_display})")
        
        if added_count > 0:
            logger.info(f"[术语缓存] 缓存了 {added_count} 个新术语 (目标: {file_path})")
        
        return added_count
    
    def flush_to_file(self, file_path: str) -> bool:
        """
        将指定文件的缓存术语写入文件
        
        Args:
            file_path: 提示词文件路径
        
        Returns:
            是否成功写入
        """
        # 原子操作：取出缓存并清空
        with self._cache_lock:
            terms = self._cache.pop(file_path, [])
            self._seen.pop(file_path, None)
        
        if not terms:
            return False
        
        # 调用原有的合并函数写入文件
        from .common import merge_glossary_to_file
        success = merge_glossary_to_file(file_path, terms)
        
        if success:
            self._stats['flushed'] += len(terms)
            logger.info(f"[术语缓存] 成功写入 {len(terms)} 个术语到: {file_path}")
        else:
            # success=False 可能是因为术语已存在，改为 debug 级别
            logger.debug(f"[术语缓存] 未写入（术语已存在或无效）: {file_path}")
            # 写入失败时，把术语放回缓存（下次重试）
            with self._cache_lock:
                if file_path not in self._cache:
                    self._cache[file_path] = []
                    self._seen[file_path] = set()
                for term in terms:
                    original = term.get("original", "")
                    if original and original not in self._seen[file_path]:
                        self._cache[file_path].append(term)
                        self._seen[file_path].add(original)
        
        return success
    
    def flush_all(self) -> int:
        """
        写入所有缓存的术语到对应文件
        
        Returns:
            成功写入的术语总数
        """
        # 获取所有需要写入的文件路径
        with self._cache_lock:
            file_paths = list(self._cache.keys())
        
        if not file_paths:
            return 0
        
        total_flushed = 0
        for path in file_paths:
            # 获取该文件的术语数量（用于统计）
            with self._cache_lock:
                count = len(self._cache.get(path, []))
            
            if self.flush_to_file(path):
                total_flushed += count
        
        if total_flushed > 0:
            logger.info(f"[术语缓存] 批量写入完成: {total_flushed} 个术语")
        
        return total_flushed
    
    def get_stats(self) -> dict:
        """获取缓存统计信息"""
        with self._cache_lock:
            pending = sum(len(terms) for terms in self._cache.values())
        
        return {
            'pending': pending,           # 待写入的术语数
            'added': self._stats['added'],       # 总添加数
            'duplicated': self._stats['duplicated'],  # 去重跳过数
            'flushed': self._stats['flushed']     # 已写入数
        }
    
    def clear(self):
        """清空缓存（用于测试或重置）"""
        with self._cache_lock:
            self._cache.clear()
            self._seen.clear()
        logger.debug("[术语缓存] 缓存已清空")
    
    def get_pending_count(self, file_path: Optional[str] = None) -> int:
        """
        获取待写入的术语数量
        
        Args:
            file_path: 指定文件路径，None 表示所有文件
        
        Returns:
            待写入的术语数量
        """
        with self._cache_lock:
            if file_path:
                return len(self._cache.get(file_path, []))
            else:
                return sum(len(terms) for terms in self._cache.values())


# 全局缓存实例（方便直接使用）
_glossary_cache: Optional[GlossaryCache] = None

# 全局并发数设置（用于决定是否使用缓存）
_translation_concurrency: int = 1

# 延迟刷新状态：基于小批量计数
_completed_batch_count: int = 0  # 已完成的小批量数量
_delayed_flush_lock = threading.Lock()


def set_translation_concurrency(concurrency: int, batch_size: int = 1):
    """
    设置翻译并发数（用于决定术语写入策略）
    
    - 并发 = 1：直接写入文件（无需缓存）
    - 并发 > 1：缓存后延迟写入（每偶数个小批量完成后刷新）
    
    延迟刷新策略（串行发送时）：
    - 每完成偶数个小批量（2、4、6...）刷新一次术语
    - 例如：批次1完成→缓存，批次2完成→刷新，批次3完成→缓存，批次4完成→刷新
    """
    global _translation_concurrency, _completed_batch_count
    _translation_concurrency = max(1, concurrency)
    # 重置延迟刷新状态
    with _delayed_flush_lock:
        _completed_batch_count = 0
    logger.debug(f"[术语缓存] 设置并发数: {_translation_concurrency}，刷新策略: 每偶数个小批量刷新")


def get_translation_concurrency() -> int:
    """获取当前翻译并发数"""
    return _translation_concurrency


def get_glossary_cache() -> GlossaryCache:
    """获取全局术语缓存实例"""
    global _glossary_cache
    if _glossary_cache is None:
        _glossary_cache = GlossaryCache()
    return _glossary_cache


def on_batch_translation_complete() -> int:
    """
    小批量翻译完成时调用，检查是否需要刷新术语
    
    刷新策略：每完成偶数个小批量（2、4、6...）刷新一次术语
    
    Returns:
        刷新的术语数量
    """
    global _completed_batch_count
    
    if _translation_concurrency <= 1:
        # 并发=1时不需要延迟刷新（已直接写入）
        return 0
    
    with _delayed_flush_lock:
        _completed_batch_count += 1
        
        # 每完成偶数个小批量刷新一次
        if _completed_batch_count % 2 == 0:
            cache = get_glossary_cache()
            flushed = cache.flush_all()
            if flushed > 0:
                logger.info(f"[术语缓存] 第 {_completed_batch_count} 个小批量完成，刷新 {flushed} 个术语")
            return flushed
    
    return 0


def reset_delayed_flush_state():
    """重置延迟刷新状态（新任务开始时调用）"""
    global _completed_batch_count
    with _delayed_flush_lock:
        _completed_batch_count = 0
    logger.debug("[术语缓存] 延迟刷新状态已重置")


def _write_glossary_to_db(terms: List[dict], tag: str = "术语提取"):
    """将术语记录写入 SQLite 数据库（用于术语日志对话框读取）"""
    if not terms:
        return
    
    # Determine result directory
    result_dir = os.getenv('MANGA_TRANSLATOR_RESULT_DIR')
    if not result_dir:
        try:
            if getattr(sys, 'frozen', False):
                result_dir = os.path.join(os.path.dirname(sys.executable), '_internal', 'result')
            else:
                from ..utils import BASE_PATH
                result_dir = os.path.join(BASE_PATH, 'result')
        except Exception:
            return
    
    try:
        os.makedirs(result_dir, exist_ok=True)
    except Exception:
        return
    
    db_path = os.path.join(result_dir, 'glossary_log.db')
    
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=1.5, isolation_level=None)
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS glossary_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT,
                tag TEXT,
                original TEXT,
                translation TEXT,
                category TEXT,
                work TEXT,
                raw_work TEXT,
                level TEXT DEFAULT 'INFO',
                message TEXT,
                extra_json TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_log_ts ON glossary_log(ts DESC);"
        )
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_glossary_log_work ON glossary_log(work);"
        )
        
        ts = datetime.now().isoformat(timespec='milliseconds')
        
        for term in terms:
            if not isinstance(term, dict):
                continue
            
            original = str(term.get('original') or '').strip()
            translation = str(term.get('translation') or '').strip()
            if not original or not translation:
                continue
            
            category = str(term.get('category') or '').strip()
            work = str(term.get('work_name') or term.get('work') or '').strip()
            raw_work = str(term.get('raw_work_name') or '').strip()
            
            message = f"{original} -> {translation}"
            if category:
                message += f" ({category})"
            
            extra_json = None
            try:
                extra_json = json.dumps(term, ensure_ascii=False)
            except Exception:
                extra_json = None
            
            cur.execute(
                """
                INSERT INTO glossary_log (
                    ts, tag, original, translation, category, work, raw_work, level, message, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (ts, tag, original, translation, category, work, raw_work, 'INFO', message, extra_json)
            )
    except Exception as e:
        logger.debug(f"[术语日志] 数据库写入失败: {e}")
    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


def add_glossary_terms(file_path: str, terms: List[dict]) -> int:
    """
    添加术语（自动根据并发数决定写入策略）
    
    - 并发 = 1：直接写入文件
    - 并发 > 1：缓存，等待下一批次翻译时写入
    
    Args:
        file_path: 提示词文件路径
        terms: 术语列表
    
    Returns:
        添加/写入的术语数
    """
    if not terms or not file_path:
        return 0
    
    # 无论并发数如何，都写入数据库（用于术语日志查看）
    try:
        _write_glossary_to_db(terms, tag="术语提取")
    except Exception:
        pass
    
    if _translation_concurrency <= 1:
        # 并发=1：直接写入文件（无需缓存）
        from .common import merge_glossary_to_file
        success = merge_glossary_to_file(file_path, terms)
        if success:
            logger.info(f"[术语] 直接写入 {len(terms)} 个术语到: {file_path}")
            return len(terms)
        else:
            logger.debug(f"[术语] 术语已存在或无效: {file_path}")
            return 0
    else:
        # 并发>1：缓存，等待延迟刷新
        cache = get_glossary_cache()
        return cache.add_terms(file_path, terms)
