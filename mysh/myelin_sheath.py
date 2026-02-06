import json
import math
from typing import List, Literal
from typing import Dict, Optional, Tuple, Self

import cv2
import numpy as np
import torch
from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm


class Mask(BaseModel):
    mask: List[List[float]] = Field(description="2d array with numbers between 0 and 1 that represents mask")

    @field_validator('mask')
    def check_mask(cls, value):
        value_array = np.array(value)
        if value_array.min() < 0 or value_array.max() > 1:
            raise ValueError('mask values must be between 0 and 1')
        return value_array.tolist()

    def __repr__(self):
        shape = np.array(self.mask).shape
        return f'{self.__class__.__name__}(mask=<List[List[float]] with shape {shape}>)'

    def to_numpy(self):
        return np.array(self.mask)

    def to_tensor(self):
        return torch.tensor(self.mask)


class MyelinSheath(BaseModel):
    offset_top: int = Field(description="Offset from the top of the picture")
    offset_left: int = Field(description="Offset from the left of the picture")
    myelin_mask: Mask = Field(description="Mask of the myelin sheath")
    axon_mask: Mask = Field(description="Mask of the axon")
    confidence: Optional[float] = Field(description="Number that represents the confidence in given mask (optional)")

    def __init__(self, **data):
        super().__init__(**data)
        self._cache = {}

    def _get_binary_mask(self, mask_type: Literal['axon', 'myelin']) -> np.ndarray:
        """
        Get cached binary mask or create and cache it.

        Args:
            mask_type: Either 'axon' or 'myelin'

        Returns:
            Binary uint8 mask
        """
        cache_key = f'binary_{mask_type}'
        if cache_key not in self._cache:
            if mask_type == 'axon':
                mask = self.axon_mask.to_numpy()
            else:  # myelin
                mask = self.myelin_mask.to_numpy()
            self._cache[cache_key] = (mask > 0.5).astype(np.uint8)
        return self._cache[cache_key]

    def _get_combined_binary_mask(self) -> np.ndarray:
        """Get cached combined binary mask or create and cache it."""
        if 'binary_combined' not in self._cache:
            axon_mask = self._get_binary_mask('axon')
            myelin_mask = self._get_binary_mask('myelin')
            self._cache['binary_combined'] = cv2.bitwise_or(axon_mask, myelin_mask)
        return self._cache['binary_combined']

    def _calculate_shape_metrics(self, mask_type: str) -> Dict[str, float]:
        """
        Calculate and cache shape metrics for a given mask type.

        Args:
            mask_type: 'axon', 'myelin', or 'combined'

        Returns:
            Dictionary containing area, perimeter and circularity
        """
        cache_key = f'metrics_{mask_type}'
        if cache_key not in self._cache:
            if mask_type == 'combined':
                mask = self._get_combined_binary_mask()
            else:
                mask = self._get_binary_mask(mask_type)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            if not contours:
                self._cache[cache_key] = {"area": 0.0, "perimeter": 0.0, "circularity": 0.0}
            else:
                contour = max(contours, key=cv2.contourArea)

                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0

                self._cache[cache_key] = {
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "circularity": float(circularity)
                }

        return self._cache[cache_key]

    @property
    def g_ratio(self) -> float:
        """
        Calculate and cache g-ratio.

        Returns:
            G-ratio value
        """
        if 'g_ratio' not in self._cache:
            axon_metrics = self._calculate_shape_metrics('axon')
            total_metrics = self._calculate_shape_metrics('combined')

            axon_diameter = 2 * math.sqrt(axon_metrics["area"] / np.pi)
            total_diameter = 2 * math.sqrt(total_metrics["area"] / np.pi)

            self._cache['g_ratio'] = float(axon_diameter / total_diameter if total_diameter > 0 else 0)

        return self._cache['g_ratio']

    @property
    def axon_metrics(self) -> Dict[str, float]:
        """Get cached axon metrics."""
        return self._calculate_shape_metrics('axon')

    @property
    def myelin_metrics(self) -> Dict[str, float]:
        """Get cached myelin metrics."""
        return self._calculate_shape_metrics('myelin')

    @property
    def mnf_area(self) -> float:
        """Get cached MNF area."""
        if 'mnf_area' not in self._cache:
            total_metrics = self._calculate_shape_metrics('combined')
            self._cache['mnf_area'] = float(total_metrics["area"])
        return self._cache['mnf_area']

    def calculate_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Calculate all metrics at once (uses cached values).

        Returns:
            Dictionary containing all metrics
        """
        return {
            "g_ratio": self.g_ratio,
            "axon_metrics": self.axon_metrics,
            "myelin_metrics": self.myelin_metrics,
            "mnf_area": self.mnf_area
        }

    def clear_cache(self):
        """Clear all cached values if masks are modified."""
        self._cache.clear()

    def calculate_iou(
            self,
            other: 'MyelinSheath',
            mask_type: Literal['myelin', 'axon'] = 'myelin'
    ) -> float:
        """
        Calculate IoU between this instance and another MyelinSheath instance.

        Args:
            other: Another MyelinSheath instance to compare with
            mask_type: Which mask to use for comparison ('myelin' or 'axon')

        Returns:
            float: IoU score between 0 and 1
        """
        mask1 = self.myelin_mask.to_numpy() if mask_type == 'myelin' else self.axon_mask.to_numpy()
        mask2 = other.myelin_mask.to_numpy() if mask_type == 'myelin' else other.axon_mask.to_numpy()

        x1_start = max(self.offset_left, other.offset_left)
        y1_start = max(self.offset_top, other.offset_top)
        x2_end = min(self.offset_left + mask1.shape[1], other.offset_left + mask2.shape[1])
        y2_end = min(self.offset_top + mask1.shape[0], other.offset_top + mask2.shape[0])

        if x1_start >= x2_end or y1_start >= y2_end:
            return 0.0

        local1_x_start = x1_start - self.offset_left
        local1_y_start = y1_start - self.offset_top
        local1_x_end = x2_end - self.offset_left
        local1_y_end = y2_end - self.offset_top

        local2_x_start = x1_start - other.offset_left
        local2_y_start = y1_start - other.offset_top
        local2_x_end = x2_end - other.offset_left
        local2_y_end = y2_end - other.offset_top

        region1 = mask1[local1_y_start:local1_y_end, local1_x_start:local1_x_end]
        region2 = mask2[local2_y_start:local2_y_end, local2_x_start:local2_x_end]

        intersection = np.logical_and(region1, region2).sum()
        if intersection == 0:
            return 0.0

        union = np.logical_or(region1, region2).sum()
        return float(intersection) / union

    def prune(self) -> Self:
        """
        Creates a new MyelinSheath object with masks cropped to remove unnecessary zero padding.
        Returns a new MyelinSheath instance with updated offsets and pruned masks.

        Returns:
            MyelinSheath: A new instance with pruned masks and updated offsets
        """

        def find_mask_bounds(mask: np.ndarray) -> Tuple[int, int, int, int]:
            """Find the minimum bounding box containing non-zero elements."""
            rows = np.any(mask > 0.5, axis=1)
            cols = np.any(mask > 0.5, axis=0)

            if not np.any(rows) or not np.any(cols):
                return 0, 0, mask.shape[0], mask.shape[1]

            rmin, rmax = np.where(rows)[0][[0, -1]]
            cmin, cmax = np.where(cols)[0][[0, -1]]

            # Add a 1-pixel border to avoid edge artifacts
            rmin = max(0, rmin - 1)
            cmin = max(0, cmin - 1)
            rmax = min(mask.shape[0] - 1, rmax + 1)
            cmax = min(mask.shape[1] - 1, cmax + 1)

            return rmin, cmin, rmax + 1, cmax + 1

        myelin_array = np.array(self.myelin_mask.mask)
        axon_array = np.array(self.axon_mask.mask)

        myelin_bounds = find_mask_bounds(myelin_array)
        axon_bounds = find_mask_bounds(axon_array)

        top = min(myelin_bounds[0], axon_bounds[0])
        left = min(myelin_bounds[1], axon_bounds[1])
        bottom = max(myelin_bounds[2], axon_bounds[2])
        right = max(myelin_bounds[3], axon_bounds[3])

        cropped_myelin = myelin_array[top:bottom, left:right]
        cropped_axon = axon_array[top:bottom, left:right]

        return MyelinSheath(
            offset_top=self.offset_top + top,
            offset_left=self.offset_left + left,
            myelin_mask=Mask(mask=cropped_myelin.tolist()),
            axon_mask=Mask(mask=cropped_axon.tolist()),
            confidence=self.confidence
        )


def get_mask_bounds(mask: np.ndarray, offset: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Get the bounds of non-zero elements in the mask."""
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return (
        offset[0] + rmin,  # top
        offset[1] + cmin,  # left
        offset[0] + rmax,  # bottom
        offset[1] + cmax  # right
    )


def quick_intersection_check(bounds1: Tuple[int, int, int, int],
                             bounds2: Tuple[int, int, int, int]) -> bool:
    """Quickly check if two bounding boxes intersect."""
    return not (bounds1[3] < bounds2[1] or  # right1 < left2
                bounds1[1] > bounds2[3] or  # left1 > right2
                bounds1[2] < bounds2[0] or  # bottom1 < top2
                bounds1[0] > bounds2[2])  # top1 > bottom2


def count_overlapping_pixels(mask1: np.ndarray, mask2: np.ndarray,
                             offset1: Tuple[int, int], offset2: Tuple[int, int],
                             bounds1: Tuple[int, int, int, int],
                             bounds2: Tuple[int, int, int, int]) -> int:
    """Count the number of overlapping pixels between two masks."""
    intersect_top = max(bounds1[0], bounds2[0])
    intersect_left = max(bounds1[1], bounds2[1])
    intersect_bottom = min(bounds1[2], bounds2[2])
    intersect_right = min(bounds1[3], bounds2[3])

    if intersect_right < intersect_left or intersect_bottom < intersect_top:
        return 0

    pos1_top = intersect_top - offset1[0]
    pos1_left = intersect_left - offset1[1]
    pos2_top = intersect_top - offset2[0]
    pos2_left = intersect_left - offset2[1]

    height = intersect_bottom - intersect_top + 1
    width = intersect_right - intersect_left + 1

    region1 = mask1[pos1_top:pos1_top + height, pos1_left:pos1_left + width]
    region2 = mask2[pos2_top:pos2_top + height, pos2_left:pos2_left + width]

    return np.sum(np.logical_and(region1 > 0.5, region2 > 0.5))


def deduplicate_masks(predicted_myelin_sheaths: List[MyelinSheath],
                      min_overlap_pixels: int = 20) -> Tuple[List[MyelinSheath], List[MyelinSheath]]:
    """
    Deduplicate myelin sheaths by removing smaller overlapping masks.

    Args:
        predicted_myelin_sheaths: List of detected myelin sheaths
        min_overlap_pixels: Minimum number of overlapping pixels to consider as duplicate

    Returns:
        Tuple of (unique_sheaths, duplicated_sheaths)
    """
    if not predicted_myelin_sheaths:
        return [], []

    mask_data = []
    for sheath in predicted_myelin_sheaths:
        mask = sheath.myelin_mask.to_numpy()
        offset = (sheath.offset_top, sheath.offset_left)
        bounds = get_mask_bounds(mask, offset)
        area = np.sum(mask > 0.5)  # Count non-zero pixels
        mask_data.append((mask, offset, bounds, area, sheath))

    mask_data.sort(key=lambda x: x[3], reverse=True)

    unique_sheaths = []
    duplicated_sheaths = []
    seen_data = []

    for current_data in tqdm(mask_data, disable=True):
        current_mask, current_offset, current_bounds, current_area, current_sheath = current_data

        is_duplicate = False
        for seen_mask, seen_offset, seen_bounds, seen_area, _ in seen_data:
            if quick_intersection_check(current_bounds, seen_bounds):
                overlap_pixels = count_overlapping_pixels(
                    current_mask, seen_mask,
                    current_offset, seen_offset,
                    current_bounds, seen_bounds
                )

                if overlap_pixels >= min_overlap_pixels:
                    is_duplicate = True
                    duplicated_sheaths.append(current_sheath)
                    break

        if not is_duplicate:
            seen_data.append(current_data)
            unique_sheaths.append(current_sheath)

    return unique_sheaths, duplicated_sheaths


def json_to_myelin_sheaths(json_path: str) -> List[MyelinSheath]:
    """
    Convert ground truth JSON data to a list of MyelinSheath objects.

    Args:
        json_path (str): Path to the JSON file containing ground truth data

    Returns:
        List[MyelinSheath]: List of MyelinSheath objects created from the JSON data
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    myelin_sheaths = []

    for axon in data['axons']:
        axon_contour = np.array(axon['axon_contour_xy'], dtype=np.int32)
        myelin_contour = np.array(axon['myelin_contour_xy'], dtype=np.int32)

        x_min = min(min(axon_contour[:, 0]), min(myelin_contour[:, 0]))
        y_min = min(min(axon_contour[:, 1]), min(myelin_contour[:, 1]))
        x_max = max(max(axon_contour[:, 0]), max(myelin_contour[:, 0]))
        y_max = max(max(axon_contour[:, 1]), max(myelin_contour[:, 1]))

        # Add padding to ensure the contours aren't at the edge
        padding = 2
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(1024, x_max + padding)
        y_max = min(1024, y_max + padding)

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        axon_mask = np.zeros((height, width), dtype=np.float32)
        myelin_mask = np.zeros((height, width), dtype=np.float32)

        local_axon_contour = axon_contour - np.array([x_min, y_min])
        local_myelin_contour = myelin_contour - np.array([x_min, y_min])

        cv2.fillPoly(axon_mask, [local_axon_contour], 1.0)
        cv2.fillPoly(myelin_mask, [local_myelin_contour], 1.0)

        # Remove axon area from myelin mask to ensure they don't overlap
        myelin_mask[axon_mask > 0] = 0

        myelin_sheath = MyelinSheath(
            offset_top=y_min,
            offset_left=x_min,
            myelin_mask=Mask(mask=myelin_mask.tolist()),
            axon_mask=Mask(mask=axon_mask.tolist()),
            confidence=1.0  # Ground truth data has full confidence
        )

        myelin_sheaths.append(myelin_sheath)

    return myelin_sheaths
