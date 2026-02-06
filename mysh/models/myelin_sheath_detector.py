from abc import ABC, abstractmethod
from typing import List

import numpy as np

from mysh.myelin_sheath import MyelinSheath


class MyelinSheathDetector(ABC):
    @abstractmethod
    def detect_myelin_sheaths(self,
                              image: np.ndarray,
                              **kwargs) -> List[MyelinSheath]:
        raise NotImplementedError


def create_myelin_mask(img_height, img_width, myelin_sheaths: List[MyelinSheath]) -> np.ndarray:
    """
    Create a mask for myelin sheaths from a list of MyelinSheath objects.

    :param img_height: Height of the output mask.
    :param img_width: Width of the output mask.
    :param myelin_sheaths: List of MyelinSheath objects.
    :return: A binary mask with myelin sheaths marked as 1.
    """
    mask_layer = np.zeros((img_height, img_width), dtype=bool)
    for sheath in myelin_sheaths:
        mask = sheath.myelin_mask.to_numpy().astype(int)
        offset_top = sheath.offset_top
        offset_left = sheath.offset_left

        mask_height, mask_width = mask.shape
        if offset_top + mask_height <= img_height and offset_left + mask_width <= img_width:
            mask_layer[offset_top:offset_top + mask_height, offset_left:offset_left + mask_width] |= mask.astype(bool)

    return mask_layer.astype(int)