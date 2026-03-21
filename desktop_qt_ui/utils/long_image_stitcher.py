#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Long image stitcher module - streaming mode with YOLO boundary detection.

Workflow:
1. Collect and sort chapter images by filename number
2. Stream-stitch images until approaching max_height
3. At boundary: YOLO-check seam safety (forward 1 / backward 1 / force cut)
4. Save each stitched group with configurable naming
"""

import os
import re
import shutil
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime
from statistics import median

import numpy as np
import cv2
from PIL import Image

from .long_image_splitter import (
    NAMING_PRESETS,
    format_output_name,
    _imread_unicode,
)

# Debug directory
_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent.parent
DEBUG_ROOT = _PROJECT_ROOT / "debug" / "stitch"
_DEBUG_TS_DIR_RE = re.compile(r"^\d{8}_\d{6}$")

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.bmp'}


@dataclass
class StitchResult:
    """Result of stitching a chapter."""
    output_files: List[str]
    group_count: int
    source_count: int  # number of original images consumed


@dataclass
class _ImageInfo:
    """Metadata for a single source image."""
    path: str
    width: int
    height: int
    number: int  # extracted sequence number from filename
    filename: str


class LongImageStitcher:
    """
    Streaming long-image stitcher with YOLO seam-safety detection.

    Strategy:
    - Accumulate images until cumulative height reaches trigger_height
    - At trigger: check current boundary (YOLO), try forward +1, backward -1
    - Cut at the first safe boundary found; force-cut if all unsafe
    """

    def __init__(
        self,
        max_height: int = 10000,
        trigger_ratio: float = 0.85,
        margin: int = 0,
        naming_pattern: str = "{index}_{source}",
        index_digits: int = 3,
        index_start: int = 1,
        reset_index_per_chapter: bool = True,
        conf_threshold: float = 0.3,
        seam_region: int = 200,
    ):
        self.max_height = max_height
        self.trigger_height = int(max_height * trigger_ratio)
        self.margin = margin
        self.naming_pattern = naming_pattern
        self.index_digits = index_digits
        self.index_start = index_start
        self.reset_index_per_chapter = reset_index_per_chapter
        self.conf_threshold = conf_threshold
        self.seam_region = seam_region

        self.logger = logging.getLogger("LongImageStitcher")

        self._yolo_detector = None
        self._yolo_loaded = False

        # Debug
        self.debug_enabled = True
        self.debug_dir: Optional[Path] = None

    # ------------------------------------------------------------------
    # YOLO loading
    # ------------------------------------------------------------------

    async def load_yolo(self) -> None:
        """Load YOLO OBB model for seam detection."""
        if self._yolo_loaded:
            return
        try:
            from manga_translator.detection.yolo_obb import YOLOOBBDetector
            self.logger.info("Loading YOLO bubble detector for stitch...")
            self._yolo_detector = YOLOOBBDetector()
            device = YOLOOBBDetector.get_preferred_device()
            await self._yolo_detector.load(device)
            self._yolo_loaded = True
            active_device = 'cuda' if getattr(self._yolo_detector, 'using_cuda', False) else getattr(self._yolo_detector, 'device', device)
            self.logger.info(f"YOLO bubble detector loaded on {active_device}")
        except ImportError as e:
            self.logger.warning(f"Cannot import YOLO detector: {e}")
            self._yolo_loaded = False
        except Exception as e:
            self.logger.error(f"YOLO model load failed: {e}")
            self._yolo_loaded = False

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _init_debug_dir(self):
        if not self.debug_enabled:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_dir = DEBUG_ROOT / timestamp
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"[debug] output dir: {self.debug_dir}")
        self._cleanup_old_debug_dirs(keep_last=10)

    def _cleanup_old_debug_dirs(self, keep_last: int = 10) -> None:
        try:
            if not DEBUG_ROOT.exists():
                return
            candidates = [
                p for p in DEBUG_ROOT.iterdir()
                if p.is_dir() and _DEBUG_TS_DIR_RE.match(p.name)
            ]
            candidates.sort(key=lambda x: x.name)
            extra = len(candidates) - keep_last
            if extra <= 0:
                return
            for old_dir in candidates[:extra]:
                shutil.rmtree(old_dir, ignore_errors=True)
        except Exception as e:
            self.logger.warning(f"[debug] cleanup failed: {e}")

    def _save_debug_data(
        self,
        chapter_name: str,
        groups: List[List[_ImageInfo]],
        decisions: List[dict],
    ):
        """Save grouping decisions as JSON for debugging."""
        if not self.debug_enabled or self.debug_dir is None:
            return
        try:
            data = {
                "chapter": chapter_name,
                "max_height": self.max_height,
                "trigger_height": self.trigger_height,
                "group_count": len(groups),
                "groups": [],
                "decisions": decisions,
            }
            for i, group in enumerate(groups):
                total_h = sum(info.height for info in group)
                data["groups"].append({
                    "index": i,
                    "image_count": len(group),
                    "total_height": total_h,
                    "images": [info.filename for info in group],
                })

            out_path = self.debug_dir / f"{chapter_name}_groups.json"
            with open(out_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.warning(f"[debug] save failed: {e}")

    # ------------------------------------------------------------------
    # Image collection
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_number(filepath: str) -> int:
        match = re.search(r'(\d+)', os.path.splitext(os.path.basename(filepath))[0])
        return int(match.group(1)) if match else 0

    def collect_images(self, chapter_path: str) -> List[_ImageInfo]:
        """Collect and sort all images in a chapter directory."""
        image_files = []
        for f in sorted(os.listdir(chapter_path)):
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXTENSIONS:
                continue
            # Skip already-stitched files
            if f.lower().startswith('seg'):
                return []
            image_files.append(os.path.join(chapter_path, f))

        if not image_files:
            return []

        result = []
        for path in image_files:
            try:
                with Image.open(path) as img:
                    w, h = img.size
                result.append(_ImageInfo(
                    path=path,
                    width=w,
                    height=h,
                    number=self._extract_number(path),
                    filename=os.path.basename(path),
                ))
            except Exception as e:
                self.logger.warning(f"Cannot read image: {os.path.basename(path)} - {e}")

        result.sort(key=lambda x: x.number)
        return result

    # ------------------------------------------------------------------
    # YOLO seam safety check
    # ------------------------------------------------------------------

    async def _is_safe_boundary(self, img_a_path: str, img_b_path: str) -> bool:
        """
        Check if the seam between two images is safe to cut.

        Crops bottom of img_a and top of img_b, stitches into a seam patch,
        runs YOLO to detect if any bubble crosses the seam line.

        Returns True if safe (no bubble crosses), False if unsafe.
        Falls back to True (safe) if YOLO is not loaded.
        """
        if not self._yolo_loaded or self._yolo_detector is None:
            return True

        try:
            img_a = _imread_unicode(img_a_path)
            img_b = _imread_unicode(img_b_path)
            if img_a is None or img_b is None:
                return True

            h_a, w_a = img_a.shape[:2]
            h_b, w_b = img_b.shape[:2]

            # Crop seam regions
            region = self.seam_region
            bottom_a = img_a[max(0, h_a - region):h_a, :]
            top_b = img_b[0:min(region, h_b), :]

            # Unify width (pad narrower one with white)
            max_w = max(w_a, w_b)
            if bottom_a.shape[1] < max_w:
                bottom_a = cv2.copyMakeBorder(
                    bottom_a, 0, 0, 0, max_w - w_a,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )
            if top_b.shape[1] < max_w:
                top_b = cv2.copyMakeBorder(
                    top_b, 0, 0, 0, max_w - w_b,
                    cv2.BORDER_CONSTANT, value=(255, 255, 255)
                )

            # Stitch seam patch
            seam_patch = np.vstack([bottom_a, top_b])
            seam_line_y = bottom_a.shape[0]

            # YOLO inference
            textlines, _, _ = await self._yolo_detector._infer(
                seam_patch,
                detect_size=640,
                text_threshold=self.conf_threshold,
                box_threshold=0.3,
                unclip_ratio=1.5,
                verbose=False,
            )

            # Check if any detection crosses the seam line
            for quad in textlines:
                y_min = int(quad.pts[:, 1].min())
                y_max = int(quad.pts[:, 1].max())
                if y_min < seam_line_y and y_max > seam_line_y:
                    self.logger.info(
                        f"[seam] UNSAFE: bubble crosses seam "
                        f"(y={y_min}-{y_max}, seam={seam_line_y})"
                    )
                    return False

            return True

        except Exception as e:
            self.logger.warning(f"[seam] YOLO check failed: {e}, treating as safe")
            return True

    # ------------------------------------------------------------------
    # Core: streaming grouping with forward/backward search
    # ------------------------------------------------------------------

    async def compute_groups(
        self,
        images: List[_ImageInfo],
        progress_callback=None,
    ) -> Tuple[List[List[_ImageInfo]], List[dict]]:
        """
        Stream-group images with YOLO boundary detection.

        Strategy at trigger:
          1. Check current boundary (img[n-1] / img[n])
          2. If unsafe, try forward: check (img[n] / img[n+1])
          3. If still unsafe, try backward: check (img[n-2] / img[n-1])
          4. If all unsafe, force-cut at current boundary

        Returns:
            (groups, decisions): grouped image lists and decision log
        """
        if not images:
            return [], []

        groups = []
        decisions = []
        current_group: List[_ImageInfo] = []
        current_height = 0

        i = 0
        while i < len(images):
            img = images[i]

            # Still in safe zone: keep accumulating
            if current_height + img.height <= self.trigger_height or not current_group:
                current_group.append(img)
                current_height += img.height
                i += 1
                continue

            # --- Approaching limit: run boundary search ---
            decision = {
                "trigger_at_index": i,
                "current_height": current_height,
                "next_image": img.filename,
                "checks": [],
                "result": None,
            }

            # Candidate 1: cut here (between current_group[-1] and img)
            safe_here = await self._is_safe_boundary(
                current_group[-1].path, img.path
            )
            decision["checks"].append({
                "position": "current",
                "between": [current_group[-1].filename, img.filename],
                "safe": safe_here,
            })

            if safe_here:
                # Clean cut at current boundary
                decision["result"] = "cut_current"
                decisions.append(decision)
                groups.append(current_group)
                if progress_callback:
                    progress_callback(
                        f"[stitch] Group {len(groups)}: "
                        f"{len(current_group)} images, {current_height}px"
                    )
                current_group = [img]
                current_height = img.height
                i += 1
                continue

            # Candidate 2: extend forward (include img, check img/img_next)
            if i + 1 < len(images):
                safe_forward = await self._is_safe_boundary(
                    img.path, images[i + 1].path
                )
                decision["checks"].append({
                    "position": "forward",
                    "between": [img.filename, images[i + 1].filename],
                    "safe": safe_forward,
                })

                if safe_forward:
                    # Extend by one, then cut
                    current_group.append(img)
                    current_height += img.height
                    i += 1
                    decision["result"] = "cut_forward"
                    decisions.append(decision)
                    groups.append(current_group)
                    if progress_callback:
                        progress_callback(
                            f"[stitch] Group {len(groups)}: "
                            f"{len(current_group)} images, {current_height}px "
                            f"(extended forward)"
                        )
                    current_group = [images[i]] if i < len(images) else []
                    current_height = images[i].height if i < len(images) else 0
                    i += 1
                    continue

            # Candidate 3: retreat backward (cut between group[-2] and group[-1])
            if len(current_group) >= 2:
                safe_backward = await self._is_safe_boundary(
                    current_group[-2].path, current_group[-1].path
                )
                decision["checks"].append({
                    "position": "backward",
                    "between": [current_group[-2].filename, current_group[-1].filename],
                    "safe": safe_backward,
                })

                if safe_backward:
                    # Pop last image from current group, cut, restart
                    popped = current_group.pop()
                    popped_height = popped.height
                    current_height -= popped_height
                    decision["result"] = "cut_backward"
                    decisions.append(decision)
                    groups.append(current_group)
                    if progress_callback:
                        progress_callback(
                            f"[stitch] Group {len(groups)}: "
                            f"{len(current_group)} images, {current_height}px "
                            f"(retreated backward)"
                        )
                    # Restart with popped image + current img
                    current_group = [popped, img]
                    current_height = popped_height + img.height
                    i += 1
                    continue

            # All candidates unsafe: force cut at current boundary
            decision["result"] = "force_cut"
            decisions.append(decision)
            groups.append(current_group)
            if progress_callback:
                progress_callback(
                    f"[stitch] Group {len(groups)}: "
                    f"{len(current_group)} images, {current_height}px "
                    f"(force cut)"
                )
            current_group = [img]
            current_height = img.height
            i += 1

        # Flush remaining
        if current_group:
            groups.append(current_group)
            if progress_callback:
                progress_callback(
                    f"[stitch] Group {len(groups)}: "
                    f"{len(current_group)} images, {current_height}px (final)"
                )

        return groups, decisions

    # ------------------------------------------------------------------
    # Stitch a single group into one image
    # ------------------------------------------------------------------

    def stitch_group_and_save(
        self,
        group: List[_ImageInfo],
        output_dir: str,
        group_index: int,
        global_index: int,
    ) -> Tuple[Optional[str], int]:
        """
        Stitch a group of images into a single long image and save.

        Args:
            group: list of _ImageInfo
            output_dir: directory to save the result
            group_index: 0-based group index
            global_index: current global naming index

        Returns:
            (output_path, next_global_index)
        """
        if not group:
            return None, global_index

        try:
            # Load all images
            pil_images = []
            for info in group:
                try:
                    img = Image.open(info.path)
                    pil_images.append((img, info))
                except Exception as e:
                    self.logger.warning(f"Cannot open {info.filename}: {e}")

            if not pil_images:
                return None, global_index

            # Compute canvas size
            max_width = max(img.width for img, _ in pil_images)
            total_height = (
                sum(img.height for img, _ in pil_images)
                + self.margin * max(0, len(pil_images) - 1)
            )

            # Determine output format from majority of source images
            ext_counts: Dict[str, int] = {}
            for _, info in pil_images:
                ext = os.path.splitext(info.filename)[1].lower()
                ext_counts[ext] = ext_counts.get(ext, 0) + 1
            majority_ext = max(ext_counts, key=ext_counts.get)
            if majority_ext not in {'.jpg', '.jpeg', '.png', '.webp'}:
                majority_ext = '.jpg'

            # Create canvas and paste
            canvas = Image.new('RGB', (max_width, total_height), (255, 255, 255))
            y_offset = 0
            for img, info in pil_images:
                x_offset = (max_width - img.width) // 2
                canvas.paste(img.convert('RGB'), (x_offset, y_offset))
                y_offset += img.height + self.margin

            # Generate filename using shared naming system.
            start_source_name = os.path.splitext(group[0].filename)[0]
            end_source_name = os.path.splitext(group[-1].filename)[0]
            if len(group) == 1 or start_source_name == end_source_name:
                source_name = start_source_name
            else:
                source_name = f"{start_source_name}-{end_source_name}"
            output_name = format_output_name(
                pattern=self.naming_pattern,
                index=global_index,
                source_name=source_name,
                ext=majority_ext,
                width=max_width,
                height=total_height,
                index_digits=self.index_digits,
            )
            output_path = os.path.join(output_dir, output_name)

            # Save (WebP max dimension is 16383px, fallback to JPEG if exceeded)
            WEBP_MAX_DIM = 16383
            if majority_ext == '.webp' and (total_height > WEBP_MAX_DIM or max_width > WEBP_MAX_DIM):
                self.logger.info(
                    f"Image {max_width}x{total_height} exceeds WebP limit, fallback to JPEG"
                )
                majority_ext = '.jpg'
                output_name = os.path.splitext(output_name)[0] + '.jpg'
                output_path = os.path.join(output_dir, output_name)

            if majority_ext in ('.jpg', '.jpeg'):
                canvas.save(output_path, 'JPEG', quality=95, optimize=True)
            elif majority_ext == '.png':
                canvas.save(output_path, 'PNG', optimize=True)
            elif majority_ext == '.webp':
                canvas.save(output_path, 'WEBP', quality=95)

            # Close PIL images
            for img, _ in pil_images:
                img.close()
            canvas.close()

            self.logger.info(
                f"Saved group {group_index}: {len(group)} images -> "
                f"{output_name} ({max_width}x{total_height})"
            )
            return output_path, global_index + 1

        except Exception as e:
            self.logger.error(f"Stitch group {group_index} failed: {e}")
            return None, global_index

    # ------------------------------------------------------------------
    # Full chapter stitch pipeline
    # ------------------------------------------------------------------

    async def stitch_chapter(
        self,
        chapter_path: str,
        delete_originals: bool = True,
        progress_callback=None,
    ) -> StitchResult:
        """
        Full pipeline: collect -> group -> stitch -> save -> cleanup.

        Args:
            chapter_path: path to chapter directory
            delete_originals: whether to delete source images after stitching
            progress_callback: callable(str) for progress messages

        Returns:
            StitchResult
        """
        chapter_name = os.path.basename(chapter_path)

        # Initialize debug dir (once per batch)
        if self.debug_dir is None:
            self._init_debug_dir()

        # Load YOLO if not loaded
        if not self._yolo_loaded:
            if progress_callback:
                progress_callback(f"[stitch] Loading YOLO model...")
            await self.load_yolo()

        # Collect images
        images = self.collect_images(chapter_path)
        if not images:
            return StitchResult(output_files=[], group_count=0, source_count=0)

        if len(images) <= 1:
            # Single image, nothing to stitch
            return StitchResult(
                output_files=[images[0].path],
                group_count=1,
                source_count=1,
            )

        if progress_callback:
            progress_callback(
                f"[stitch] {chapter_name}: {len(images)} images, "
                f"max_height={self.max_height}px"
            )

        median_height = int(median(info.height for info in images))
        if median_height > self.trigger_height:
            skip_message = (
                f"[stitch] 跳过 {chapter_name}: 全章图片高度中位数为 "
                f"{median_height}px，高于触发阈值 {self.trigger_height}px"
            )
            self.logger.info(skip_message)
            if progress_callback:
                progress_callback(skip_message)
            return StitchResult(
                output_files=[],
                group_count=0,
                source_count=len(images),
            )

        # Compute groups
        groups, decisions = await self.compute_groups(images, progress_callback)

        single_group_count = sum(1 for group in groups if len(group) == 1)
        multi_group_count = sum(1 for group in groups if len(group) > 1)
        if single_group_count > 0 and multi_group_count > 0:
            skip_message = (
                f"[stitch] 跳过 {chapter_name}: 检测到混合分组，"
                f"单图组 {single_group_count} 个，多图组 {multi_group_count} 个"
            )
            self.logger.info(skip_message)
            if progress_callback:
                progress_callback(skip_message)
            self._save_debug_data(chapter_name, groups, decisions)
            return StitchResult(
                output_files=[],
                group_count=0,
                source_count=len(images),
            )

        # Save debug data
        self._save_debug_data(chapter_name, groups, decisions)

        # Stitch each group
        output_files = []
        successful_original_paths = set()
        global_index = self.index_start

        for gi, group in enumerate(groups):
            result_path, global_index = self.stitch_group_and_save(
                group, chapter_path, gi, global_index
            )
            if result_path:
                output_files.append(result_path)
                for info in group:
                    successful_original_paths.add(info.path)

        # Delete originals only for groups that were saved successfully.
        if delete_originals and output_files:
            for path in successful_original_paths:
                # Don't delete if it's also an output (single-image group edge case)
                if path not in output_files:
                    try:
                        os.remove(path)
                    except Exception as e:
                        self.logger.warning(f"Cannot delete {path}: {e}")

        return StitchResult(
            output_files=output_files,
            group_count=len(groups),
            source_count=len(images),
        )
