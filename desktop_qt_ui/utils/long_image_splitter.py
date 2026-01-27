#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
长图拆分模块 - 本地模式

使用 YOLO 气泡检测 + 图像处理确定最佳切割点，
将长图拆分为多个适合阅读的图片段。

工作流程:
1. YOLO 检测气泡/文本区域 → 得到"禁切区"
2. 在目标高度附近扫描，找颜色变化最小的水平线
3. 排除禁切区，输出最终切点
4. 按切点切割图片并保存
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2
from PIL import Image

# Debug 目录配置（从模块位置推算项目根目录）
_MODULE_DIR = Path(__file__).resolve().parent  # desktop_qt_ui/utils
_PROJECT_ROOT = _MODULE_DIR.parent.parent       # 项目根目录
DEBUG_ROOT = _PROJECT_ROOT / "debug" / "split"


# 命名配置
NAMING_PRESETS = {
    "序号_源文件名": "{index}_{source}",      # 001_原文件名.jpg
    "序号": "{index}",                         # 001.jpg
    "源文件名_序号": "{source}_{index}",      # 原文件名_001.jpg
    "序号_宽x高": "{index}_{width}x{height}",  # 001_800x2600.jpg
}


def format_output_name(
    pattern: str,
    index: int,
    source_name: str,
    ext: str,
    width: int = 0,
    height: int = 0,
    index_digits: int = 1
) -> str:
    """
    根据命名模式生成输出文件名
    
    Args:
        pattern: 命名模式，如 "{index}_{source}"
        index: 序号
        source_name: 源文件名（不含扩展名）
        ext: 扩展名（含.）
        width: 图片宽度
        height: 图片高度
        index_digits: 序号位数
        
    Returns:
        格式化后的文件名
    """
    formatted_index = str(index).zfill(index_digits)
    
    replacements = {
        "index": formatted_index,
        "source": source_name,
        "width": str(width),
        "height": str(height),
    }
    
    try:
        name = pattern.format(**replacements)
    except KeyError:
        # 回退到默认模式
        name = f"{formatted_index}_{source_name}"
    
    return f"{name}{ext}"


@dataclass
class SplitResult:
    """拆分结果"""
    output_files: List[str]  # 输出文件路径列表
    cuts: List[int]  # 切割点 Y 坐标列表
    total_height: int  # 原图总高度
    
    @property
    def segment_count(self) -> int:
        """拆分后的片段数量"""
        return len(self.cuts) + 1


class LocalSplitter:
    """
    本地图片拆分器，使用 YOLO 气泡检测 + 图像处理确定切割点。
    
    工作流程:
    1. YOLO 检测气泡/文本区域 → 得到"禁切区"
    2. 在目标高度附近扫描，找颜色变化最小的水平线
    3. 排除禁切区，输出最终切点
    """
    
    def __init__(
        self,
        target_height: int = 2600,
        buffer_range: int = 300,
        min_segment_height: int = 1000,
        conf_threshold: float = 0.3,
        naming_pattern: str = "{index}_{source}",
        index_digits: int = 1,
        index_start: int = 0
    ):
        """
        初始化拆分器
        
        Args:
            target_height: 目标片段高度（像素）
            buffer_range: 搜索缓冲范围，在目标高度 ± 此范围内搜索最佳切点
            min_segment_height: 最小片段高度，防止切出过小的片段
            conf_threshold: YOLO 检测置信度阈值
            naming_pattern: 命名模式，支持 {index}, {source}, {width}, {height}
            index_digits: 序号位数
            index_start: 起始序号（全局序号从该值开始）
        """
        self.logger = logging.getLogger(__name__)
        self.target_height = target_height
        self.buffer_range = buffer_range
        self.min_segment_height = min_segment_height
        self.conf_threshold = conf_threshold
        self.naming_pattern = naming_pattern
        self.index_digits = index_digits
        self.index_start = index_start
        
        self._yolo_detector = None
        self._yolo_loaded = False
        
        # Debug 目录（每次处理会创建带时间戳的子目录）
        self.debug_enabled = True
        self.debug_dir: Optional[Path] = None
    
    async def load_yolo(self) -> None:
        """加载 YOLO 模型"""
        if self._yolo_loaded:
            return
        
        try:
            # 尝试导入翻译项目中的 YOLO OBB 检测器
            from manga_translator.detection.yolo_obb import YOLOOBBDetector
            
            self.logger.info("加载 YOLO 气泡检测模型...")
            self._yolo_detector = YOLOOBBDetector()
            await self._yolo_detector._load('cpu')  # 使用 CPU 模式，更稳定
            self._yolo_loaded = True
            self.logger.info("YOLO 气泡检测模型加载完成")
            
        except ImportError as e:
            self.logger.warning(f"无法导入 YOLO 检测器: {e}")
            self._yolo_loaded = False
        except Exception as e:
            self.logger.error(f"YOLO 模型加载失败: {e}")
            self._yolo_loaded = False
    
    def _init_debug_dir(self):
        """初始化本次处理的 debug 目录"""
        if not self.debug_enabled:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = DEBUG_ROOT / timestamp
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[调试] 输出目录: {self.debug_dir}")
    
    def _save_debug_data(
        self,
        image_path: Path,
        img: np.ndarray,
        forbidden_zones: List[Tuple[int, int]],
        cuts: List[int],
        bubbles: List = None
    ):
        """
        保存调试数据和可视化图片
        
        输出:
        - {image_name}_debug.jpg: 标注了禁切区和切割线的图片
        - {image_name}_data.json: 详细的分析数据
        """
        if not self.debug_enabled or self.debug_dir is None:
            return
        
        image_path = Path(image_path)
        base_name = image_path.stem
        
        try:
            height, width = img.shape[:2]
            
            # 1. 保存标注图片
            annotated = self._draw_debug_image(img.copy(), forbidden_zones, cuts)
            debug_img_path = self.debug_dir / f"{base_name}_debug.jpg"
            cv2.imwrite(str(debug_img_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
            
            # 2. 保存 JSON 数据
            debug_data = {
                "image": image_path.name,
                "image_path": str(image_path),
                "image_size": {"width": width, "height": height},
                "params": {
                    "target_height": self.target_height,
                    "buffer_range": self.buffer_range,
                    "min_segment_height": self.min_segment_height,
                    "conf_threshold": self.conf_threshold
                },
                "forbidden_zones": [
                    {"y_min": y_min, "y_max": y_max}
                    for y_min, y_max in forbidden_zones
                ],
                "cuts": cuts,
                "segment_count": len(cuts) + 1,
                "segment_heights": self._calc_segment_heights(cuts, height),
                "yolo_enabled": self._yolo_loaded
            }
            
            # 如果有 YOLO 检测结果，添加详细信息
            if bubbles:
                debug_data["bubbles"] = [
                    {
                        "class": getattr(b, 'class_name', 'unknown'),
                        "score": round(getattr(b, 'score', 0), 3),
                        "y_range": [getattr(b, 'y_min', 0), getattr(b, 'y_max', 0)]
                    }
                    for b in bubbles
                ]
            
            json_path = self.debug_dir / f"{base_name}_data.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(debug_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"[调试] 已保存: {base_name}_debug.jpg, {base_name}_data.json")
            
        except Exception as e:
            self.logger.warning(f"[调试] 保存失败: {e}")
    
    def _draw_debug_image(
        self,
        img: np.ndarray,
        forbidden_zones: List[Tuple[int, int]],
        cuts: List[int]
    ) -> np.ndarray:
        """绘制调试可视化图片"""
        height, width = img.shape[:2]
        
        # 1. 绘制禁切区（红色半透明矩形）
        overlay = img.copy()
        for y_min, y_max in forbidden_zones:
            cv2.rectangle(overlay, (0, y_min), (width, y_max), (0, 0, 255), -1)
        img = cv2.addWeighted(overlay, 0.3, img, 0.7, 0)
        
        # 2. 绘制禁切区边框（红色虚线）
        for y_min, y_max in forbidden_zones:
            # 上边界
            for x in range(0, width, 20):
                cv2.line(img, (x, y_min), (min(x + 10, width), y_min), (0, 0, 255), 2)
            # 下边界
            for x in range(0, width, 20):
                cv2.line(img, (x, y_max), (min(x + 10, width), y_max), (0, 0, 255), 2)
            # 标签
            cv2.putText(img, f"forbidden {y_min}-{y_max}", (10, y_min + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 3. 绘制切割线（蓝色实线）
        for i, y in enumerate(cuts):
            cv2.line(img, (0, y), (width, y), (255, 0, 0), 3)
            # 标签
            cv2.putText(img, f"CUT {i+1}: Y={y}", (width - 200, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 4. 绘制目标高度参考线（绿色虚线）
        y = self.target_height
        while y < height:
            for x in range(0, width, 40):
                cv2.line(img, (x, y), (min(x + 20, width), y), (0, 255, 0), 1)
            cv2.putText(img, f"target {y}", (10, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y += self.target_height
        
        # 5. 添加图例
        legend_y = 30
        cv2.putText(img, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.line(img, (80, legend_y - 5), (120, legend_y - 5), (255, 0, 0), 3)
        cv2.putText(img, "Cut Line", (125, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.rectangle(img, (200, legend_y - 10), (220, legend_y), (0, 0, 255), -1)
        cv2.putText(img, "Forbidden", (225, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.line(img, (300, legend_y - 5), (340, legend_y - 5), (0, 255, 0), 1)
        cv2.putText(img, "Target", (345, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def _calc_segment_heights(self, cuts: List[int], total_height: int) -> List[int]:
        """计算每个片段的高度"""
        all_points = [0] + sorted(cuts) + [total_height]
        heights = []
        for i in range(len(all_points) - 1):
            heights.append(all_points[i + 1] - all_points[i])
        return heights
    
    async def analyze(self, image_path: Path) -> Tuple[List[int], int]:
        """
        分析图片，返回切割点列表
        
        Args:
            image_path: 图片路径
            
        Returns:
            (cuts, total_height): 切割点 Y 坐标列表和图片总高度
        """
        # 读取图片
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        height, width = img.shape[:2]
        
        # 如果图片不够高，不需要切割
        if height <= self.target_height + self.buffer_range:
            self.logger.info(f"图片高度 {height}px 不需要切割")
            return [], height
        
        # 检测气泡区域（禁切区）
        forbidden_zones = []
        bubbles = None
        if self._yolo_loaded and self._yolo_detector is not None:
            forbidden_zones, bubbles = await self._detect_bubbles_with_details(img)
            self.logger.info(f"检测到 {len(forbidden_zones)} 个禁切区域")
        
        # 计算候选切割点
        cuts = self._find_cut_points(img, height, forbidden_zones)
        
        self.logger.info(f"分析完成: {len(cuts)} 个切割点 -> {cuts}")
        
        # 保存调试数据
        self._save_debug_data(image_path, img, forbidden_zones, cuts, bubbles)
        
        return cuts, height
    
    async def _detect_bubbles(self, img: np.ndarray) -> List[Tuple[int, int]]:
        """
        检测图片中的气泡/文本区域
        
        Returns:
            禁切区列表 [(y_min, y_max), ...]
        """
        zones, _ = await self._detect_bubbles_with_details(img)
        return zones
    
    async def _detect_bubbles_with_details(self, img: np.ndarray) -> Tuple[List[Tuple[int, int]], List]:
        """
        检测图片中的气泡/文本区域，并返回详细信息
        
        Returns:
            (forbidden_zones, bubbles): 禁切区列表和原始检测结果
        """
        if not self._yolo_loaded or self._yolo_detector is None:
            return [], None
        
        try:
            # 使用 YOLO 检测
            textlines, _, _ = await self._yolo_detector._infer(
                img,
                detect_size=640,
                text_threshold=self.conf_threshold,
                box_threshold=0.3,
                unclip_ratio=1.5,
                verbose=False
            )
            
            # 提取 Y 坐标范围
            forbidden_zones = []
            for quad in textlines:
                pts = quad.pts
                y_min = int(pts[:, 1].min())
                y_max = int(pts[:, 1].max())
                forbidden_zones.append((y_min, y_max))
            
            return forbidden_zones, textlines
            
        except Exception as e:
            self.logger.warning(f"YOLO 检测失败: {e}")
            return [], None
    
    def _find_cut_points(
        self,
        img: np.ndarray,
        height: int,
        forbidden_zones: List[Tuple[int, int]]
    ) -> List[int]:
        """
        找到所有切割点
        
        Args:
            img: 图片数组
            height: 图片高度
            forbidden_zones: 禁切区列表 [(y_min, y_max), ...]
            
        Returns:
            切割点 Y 坐标列表
        """
        cuts = []
        current_y = 0
        
        while True:
            # 下一个理想切割位置
            ideal_y = current_y + self.target_height
            
            # 如果剩余高度不够，停止
            if ideal_y >= height - self.min_segment_height:
                break
            
            # 在缓冲范围内扫描找最佳切割点
            search_start = max(current_y + self.min_segment_height, ideal_y - self.buffer_range)
            search_end = min(height - self.min_segment_height, ideal_y + self.buffer_range)
            
            if search_start >= search_end:
                break
            
            # 找最佳切割点
            best_y = self._find_best_cut_in_range(img, search_start, search_end, forbidden_zones)
            
            if best_y is None:
                # 找不到安全的切割点，使用理想位置
                best_y = ideal_y
                self.logger.warning(f"无法找到安全切割点，使用理想位置 Y={best_y}")
            
            cuts.append(best_y)
            current_y = best_y
        
        return cuts
    
    def _find_best_cut_in_range(
        self,
        img: np.ndarray,
        start_y: int,
        end_y: int,
        forbidden_zones: List[Tuple[int, int]],
        margin: int = 50
    ) -> Optional[int]:
        """
        在指定范围内找最佳切割点
        
        使用行扫描法：计算每一行的颜色方差，选择方差最小（最平滑）的行。
        同时避开禁切区。
        
        Args:
            img: 图片数组
            start_y: 搜索起始 Y
            end_y: 搜索结束 Y
            forbidden_zones: 禁切区列表
            margin: 禁切区边距
            
        Returns:
            最佳切割点 Y 坐标，如果找不到返回 None
        """
        height, width = img.shape[:2]
        
        # 转为灰度图计算
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        
        best_y = None
        best_score = float('inf')
        
        for y in range(start_y, end_y):
            # 检查是否在禁切区内
            if self._is_in_forbidden_zone(y, forbidden_zones, margin):
                continue
            
            # 计算这一行的"平滑度"分数
            # 越平滑（方差越小）越适合切割
            row = gray[y, :]
            
            # 1. 行内方差（越小越平滑）
            row_variance = np.var(row)
            
            # 2. 与上下行的差异（越小越说明是连续的纯色区域）
            if y > 0 and y < height - 1:
                row_above = gray[y - 1, :]
                row_below = gray[y + 1, :]
                vertical_diff = np.mean(np.abs(row.astype(float) - row_above.astype(float))) + \
                               np.mean(np.abs(row.astype(float) - row_below.astype(float)))
            else:
                vertical_diff = 0
            
            # 综合分数（越低越好）
            score = row_variance + vertical_diff * 2
            
            if score < best_score:
                best_score = score
                best_y = y
        
        return best_y
    
    def _is_in_forbidden_zone(
        self,
        y: int,
        forbidden_zones: List[Tuple[int, int]],
        margin: int
    ) -> bool:
        """检查 Y 坐标是否在禁切区内（包含边距）"""
        for y_min, y_max in forbidden_zones:
            if (y_min - margin) <= y <= (y_max + margin):
                return True
        return False
    
    def split_image(
        self,
        image_path: Path,
        cuts: List[int],
        output_dir: Optional[Path] = None,
        quality: int = 95,
        start_index: int = 0
    ) -> List[str]:
        """
        按切割点拆分图片
        
        Args:
            image_path: 原图路径
            cuts: 切割点 Y 坐标列表
            output_dir: 输出目录，默认为原图所在目录
            quality: JPEG 质量 (1-100)
            
        Returns:
            输出文件路径列表
        """
        image_path = Path(image_path)
        
        if output_dir is None:
            output_dir = image_path.parent
        else:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 读取图片
        with Image.open(image_path) as img:
            width, height = img.size
            
            # 构建所有切割点（包含头尾）
            all_points = [0] + sorted(cuts) + [height]
            
            output_files = []
            original_name = image_path.stem
            original_ext = image_path.suffix.lower()
            
            current_index = start_index
            
            for i in range(len(all_points) - 1):
                start_y = all_points[i]
                end_y = all_points[i + 1]
                
                if end_y <= start_y:
                    continue
                
                # 裁剪
                segment = img.crop((0, start_y, width, end_y))
                segment_height = end_y - start_y
                
                # 使用命名模式生成输出文件名（全局序号）
                output_name = format_output_name(
                    pattern=self.naming_pattern,
                    index=current_index,
                    source_name=original_name,
                    ext=original_ext,
                    width=width,
                    height=segment_height,
                    index_digits=self.index_digits
                )
                output_path = output_dir / output_name
                
                # 保存
                if original_ext in ['.jpg', '.jpeg']:
                    segment.save(output_path, 'JPEG', quality=quality)
                elif original_ext == '.png':
                    segment.save(output_path, 'PNG', optimize=True)
                elif original_ext == '.webp':
                    segment.save(output_path, 'WEBP', quality=quality)
                else:
                    # 默认保存为 JPEG
                    output_name = format_output_name(
                        pattern=self.naming_pattern,
                        index=current_index,
                        source_name=original_name,
                        ext=".jpg",
                        width=width,
                        height=segment_height,
                        index_digits=self.index_digits
                    )
                    output_path = output_dir / output_name
                    segment.save(output_path, 'JPEG', quality=quality)
                
                output_files.append(str(output_path))
                self.logger.debug(f"保存片段 {current_index}: Y={start_y}-{end_y} -> {output_name}")
                current_index += 1
        
        return output_files
    
    async def split(
        self,
        image_path: Path,
        output_dir: Optional[Path] = None,
        quality: int = 95,
        delete_original: bool = True,
        start_index: Optional[int] = None
    ) -> SplitResult:
        """
        分析并拆分图片（完整流程）
        
        Args:
            image_path: 图片路径
            output_dir: 输出目录
            quality: JPEG 质量
            delete_original: 拆分成功后是否删除原图
            
        Returns:
            SplitResult 拆分结果
        """
        image_path = Path(image_path)
        
        # 初始化调试目录
        if self.debug_dir is None:
            self._init_debug_dir()
        
        # 加载 YOLO（如果尚未加载）
        if not self._yolo_loaded:
            await self.load_yolo()
        
        # 分析获取切割点
        cuts, total_height = await self.analyze(image_path)
        
        # 如果不需要切割
        if not cuts:
            return SplitResult(
                output_files=[str(image_path)],
                cuts=[],
                total_height=total_height
            )
        
        # 执行切割
        if start_index is None:
            start_index = self.index_start
        output_files = self.split_image(
            image_path,
            cuts,
            output_dir,
            quality,
            start_index=start_index
        )
        
        # 删除原图
        if delete_original and output_files:
            try:
                os.remove(image_path)
                self.logger.info(f"已删除原图: {image_path.name}")
            except Exception as e:
                self.logger.warning(f"删除原图失败: {e}")
        
        return SplitResult(
            output_files=output_files,
            cuts=cuts,
            total_height=total_height
        )


# 同步版本的辅助函数（用于不支持异步的场景）
def split_image_sync(
    image_path: str,
    target_height: int = 2600,
    buffer_range: int = 300,
    min_segment_height: int = 1000,
    output_dir: Optional[str] = None,
    quality: int = 95,
    delete_original: bool = True,
    naming_pattern: str = "{index}_{source}",
    index_digits: int = 1,
    index_start: int = 0
) -> SplitResult:
    """
    同步版本的图片拆分函数（不使用 YOLO 检测）
    
    使用纯图像处理方法找到平滑的切割线。
    
    Args:
        image_path: 图片路径
        target_height: 目标片段高度
        buffer_range: 搜索缓冲范围
        min_segment_height: 最小片段高度
        output_dir: 输出目录
        quality: JPEG 质量
        delete_original: 是否删除原图
        naming_pattern: 命名模式
        
    Returns:
        SplitResult 拆分结果
    """
    image_path = Path(image_path)
    output_dir = Path(output_dir) if output_dir else image_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取图片
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")
    
    height, width = img.shape[:2]
    
    # 如果图片不够高，不需要切割
    if height <= target_height + buffer_range:
        return SplitResult(
            output_files=[str(image_path)],
            cuts=[],
            total_height=height
        )
    
    # 转为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 计算切割点
    cuts = []
    current_y = 0
    
    while True:
        ideal_y = current_y + target_height
        
        if ideal_y >= height - min_segment_height:
            break
        
        search_start = max(current_y + min_segment_height, ideal_y - buffer_range)
        search_end = min(height - min_segment_height, ideal_y + buffer_range)
        
        if search_start >= search_end:
            break
        
        # 找最平滑的行
        best_y = ideal_y
        best_score = float('inf')
        
        for y in range(search_start, search_end):
            row = gray[y, :]
            row_variance = np.var(row)
            
            if y > 0 and y < height - 1:
                row_above = gray[y - 1, :]
                row_below = gray[y + 1, :]
                vertical_diff = np.mean(np.abs(row.astype(float) - row_above.astype(float))) + \
                               np.mean(np.abs(row.astype(float) - row_below.astype(float)))
            else:
                vertical_diff = 0
            
            score = row_variance + vertical_diff * 2
            
            if score < best_score:
                best_score = score
                best_y = y
        
        cuts.append(best_y)
        current_y = best_y
    
    # 拆分图片
    with Image.open(image_path) as pil_img:
        all_points = [0] + sorted(cuts) + [height]
        output_files = []
        original_name = image_path.stem
        original_ext = image_path.suffix.lower()
        
        for i in range(len(all_points) - 1):
            start_y = all_points[i]
            end_y = all_points[i + 1]
            
            if end_y <= start_y:
                continue
            
            segment = pil_img.crop((0, start_y, width, end_y))
            segment_height = end_y - start_y
            
            # 使用命名模式生成文件名（全局序号）
            current_index = index_start + i
            output_name = format_output_name(
                pattern=naming_pattern,
                index=current_index,
                source_name=original_name,
                ext=original_ext,
                width=width,
                height=segment_height,
                index_digits=index_digits
            )
            output_path = output_dir / output_name
            
            if original_ext in ['.jpg', '.jpeg']:
                segment.save(output_path, 'JPEG', quality=quality)
            elif original_ext == '.png':
                segment.save(output_path, 'PNG', optimize=True)
            elif original_ext == '.webp':
                segment.save(output_path, 'WEBP', quality=quality)
            else:
                output_name = format_output_name(
                    pattern=naming_pattern,
                    index=current_index,
                    source_name=original_name,
                    ext=".jpg",
                    width=width,
                    height=segment_height,
                    index_digits=index_digits
                )
                output_path = output_dir / output_name
                segment.save(output_path, 'JPEG', quality=quality)
            
            output_files.append(str(output_path))
    
    # 删除原图
    if delete_original and output_files:
        try:
            os.remove(image_path)
        except Exception:
            pass
    
    return SplitResult(
        output_files=output_files,
        cuts=cuts,
        total_height=height
    )
