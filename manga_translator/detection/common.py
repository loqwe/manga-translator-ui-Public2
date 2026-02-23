from abc import abstractmethod
from typing import List, Tuple
from collections import Counter
import numpy as np
import cv2

from ..utils import InfererModule, ModelWrapper, Quadrilateral


class CommonDetector(InfererModule):

    async def detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float, unclip_ratio: float,
                     invert: bool, gamma_correct: bool, rotate: bool, auto_rotate: bool = False, verbose: bool = False, min_box_area_ratio: float = 0.0009, result_path_fn=None):
        '''
        Returns textblock list and text mask.
        '''

        # Apply filters
        img_h, img_w = image.shape[:2]
        orig_image = image.copy()
        minimum_image_size = 400
        # Automatically add border if image too small (instead of simply resizing due to them more likely containing large fonts)
        add_border = min(img_w, img_h) < minimum_image_size
        if rotate:
            self.logger.debug('Adding rotation')
            image = self._add_rotation(image)
        if add_border:
            self.logger.debug('Adding border')
            image = self._add_border(image, minimum_image_size)
        if invert:
            self.logger.debug('Adding inversion')
            image = self._add_inversion(image)
        if gamma_correct:
            self.logger.debug('Adding gamma correction')
            image = self._add_gamma_correction(image)
        # if True:
        #     self.logger.debug('Adding histogram equalization')
        #     image = self._add_histogram_equalization(image)

        # cv2.imwrite('histogram.png', image)
        # cv2.waitKey(0)

        # Run detection
        textlines, raw_mask, mask = await self._detect(image, detect_size, text_threshold, box_threshold, unclip_ratio, verbose, result_path_fn)
        # 面积过滤已移至文本行合并后进行（基于合并后的大框）

        # Remove filters
        if add_border:
            textlines, raw_mask, mask = self._remove_border(image, img_w, img_h, textlines, raw_mask, mask)
        if auto_rotate:
            # Rotate if horizontal aspect ratios are prevalent to potentially improve detection
            if len(textlines) > 0:
                orientations = ['h' if txtln.aspect_ratio > 1 else 'v' for txtln in textlines]
                majority_orientation = Counter(orientations).most_common(1)[0][0]
            else:
                majority_orientation = 'h'
            if majority_orientation == 'h':
                self.logger.info('Rerunning detection with 90° rotation')
                return await self.detect(orig_image, detect_size, text_threshold, box_threshold, unclip_ratio, invert, gamma_correct,
                                         rotate=(not rotate), auto_rotate=False, verbose=verbose, result_path_fn=result_path_fn)
        if rotate:
            textlines, raw_mask, mask = self._remove_rotation(textlines, raw_mask, mask, img_w, img_h)

        return textlines, raw_mask, mask

    @abstractmethod
    async def _detect(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                      unclip_ratio: float, verbose: bool = False, result_path_fn=None) -> Tuple[List[Quadrilateral], np.ndarray, np.ndarray]:
        pass

    def _add_border(self, image: np.ndarray, target_side_length: int):
        old_h, old_w = image.shape[:2]
        new_w = new_h = max(old_w, old_h, target_side_length)
        new_image = np.zeros([new_h, new_w, 3]).astype(np.uint8)
        # new_image[:] = np.array([255, 255, 255], np.uint8)
        x, y = 0, 0
        # x, y = (new_h - old_h) // 2, (new_w - old_w) // 2
        new_image[y:y+old_h, x:x+old_w] = image
        return new_image

    def _remove_border(self, image: np.ndarray, old_w: int, old_h: int, textlines: List[Quadrilateral], raw_mask, mask):
        new_h, new_w = image.shape[:2]
        raw_mask = self._resize_and_crop_mask(raw_mask, new_w, new_h, old_w, old_h)
        mask = self._resize_and_crop_mask(mask, new_w, new_h, old_w, old_h)

        # Filter out regions within the border and clamp the points of the remaining regions
        new_textlines = []
        for txtln in textlines:
            if txtln.xyxy[0] >= old_w and txtln.xyxy[1] >= old_h:
                continue
            points = txtln.pts
            points[:,0] = np.clip(points[:,0], 0, old_w)
            points[:,1] = np.clip(points[:,1], 0, old_h)
            new_txtln = Quadrilateral(points, txtln.text, txtln.prob)
            new_textlines.append(new_txtln)
        return new_textlines, raw_mask, mask

    def _resize_and_crop_mask(self, mask, resize_w: int, resize_h: int, crop_w: int, crop_h: int):
        if mask is None:
            return None

        if isinstance(mask, tuple):
            return tuple(self._resize_and_crop_mask(m, resize_w, resize_h, crop_w, crop_h) for m in mask)

        if isinstance(mask, list):
            return [self._resize_and_crop_mask(m, resize_w, resize_h, crop_w, crop_h) for m in mask]

        if not isinstance(mask, np.ndarray):
            self.logger.warning(f'Unexpected mask type in _remove_border: {type(mask)}')
            return mask

        if mask.size == 0 or len(mask.shape) < 2:
            self.logger.warning(f'Invalid mask shape in _remove_border: {mask.shape}')
            return mask

        resized = cv2.resize(mask, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
        return resized[:crop_h, :crop_w]

    def _add_rotation(self, image: np.ndarray):
        return np.rot90(image, k=-1)

    def _remove_rotation(self, textlines, raw_mask, mask, img_w, img_h):
        raw_mask = np.ascontiguousarray(np.rot90(raw_mask))
        
        # mask 可能是 tuple（包含多个调试图片）或单个数组
        if mask is not None:
            if isinstance(mask, tuple):
                # 如果是 tuple，对每个元素分别旋转
                rotated_masks = []
                for m in mask:
                    if m is not None and hasattr(m, 'shape'):
                        rotated_masks.append(np.ascontiguousarray(np.rot90(m).astype(np.uint8)))
                    else:
                        rotated_masks.append(m)
                mask = tuple(rotated_masks)
            else:
                # 单个数组
                mask = np.ascontiguousarray(np.rot90(mask).astype(np.uint8))

        for i, txtln in enumerate(textlines):
            rotated_pts = txtln.pts[:,[1,0]]
            rotated_pts[:,1] = -rotated_pts[:,1] + img_h
            textlines[i] = Quadrilateral(rotated_pts, txtln.text, txtln.prob)
        return textlines, raw_mask, mask

    def _add_inversion(self, image: np.ndarray):
        return cv2.bitwise_not(image)

    def _add_gamma_correction(self, image: np.ndarray):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mid = 0.5
        mean = np.mean(gray)
        gamma = np.log(mid * 255) / np.log(mean)
        img_gamma = np.power(image, gamma).clip(0,255).astype(np.uint8)
        return img_gamma

    def _add_histogram_equalization(self, image: np.ndarray):
        img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)

        # equalize the histogram of the Y channel
        img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])

        # convert the YUV image back to RGB format
        img_output = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
        return img_output


class OfflineDetector(CommonDetector, ModelWrapper):
    _MODEL_SUB_DIR = 'detection'

    async def _detect(self, *args, **kwargs):
        return await self.infer(*args, **kwargs)

    @abstractmethod
    async def _infer(self, image: np.ndarray, detect_size: int, text_threshold: float, box_threshold: float,
                       unclip_ratio: float, verbose: bool = False, result_path_fn=None):
        pass
