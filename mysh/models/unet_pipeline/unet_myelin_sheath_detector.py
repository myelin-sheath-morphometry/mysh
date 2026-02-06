import os
from typing import Tuple, List

import cv2
import numpy as np
from sympy.printing.pytorch import torch
from tqdm import tqdm

from mysh.logger import get_default_logger
from mysh.myelin_sheath import MyelinSheath, Mask, deduplicate_masks
from mysh.models.myelin_sheath_detector import MyelinSheathDetector
from mysh.models.unet import UNet, batch_inference
from mysh.models.unet_pipeline.config import DEFAULT_MYELIN_MODEL_PATH, DEFAULT_AXON_MODEL_PATH


logger = get_default_logger(__name__)

class UnetMyelinSheathDetector(MyelinSheathDetector):

    def __init__(self,
                 axon_model_path: str = DEFAULT_AXON_MODEL_PATH,
                 myelin_model_path: str = DEFAULT_MYELIN_MODEL_PATH):
        device = os.environ.get('TORCH_DEVICE', 'mps')
        self.axon_model = UNet.load_from_checkpoint(axon_model_path).to(device)
        self.myelin_model = UNet.load_from_checkpoint(myelin_model_path).to(device)

        logger.info(f"Loaded axon model: {self.axon_model.device}")
        logger.info(f"Loaded myelin model: {self.myelin_model.device}")

    def detect_myelin_sheaths(self,
                              image: np.ndarray,
                              axon_threshold: float = 0.5,
                              myelin_threshold: float = 0.5,
                              inference_batch_size: int = 16,
                              min_axon_size_pixels: int = 5,
                              disable_progress_bar: bool = True,
                              progress_callback=None):
        small_images, offsets = self.split_image(image)
        masks = batch_inference(small_images, self.axon_model, inference_batch_size)
        masks = masks.numpy()

        detected_myelins: List[MyelinSheath] = []

        for idx, (small_image, offset, mask) in tqdm(
                enumerate(
                    zip(small_images, offsets, masks)
                ),
                total=len(masks),
                disable=disable_progress_bar
        ):
            offset_top, offset_left = offset
            component_masks = self.separate_components(mask[0] > axon_threshold, min_axon_size_pixels)

            if not component_masks:
                continue

            myelins_to_detect = torch.stack([
                torch.cat([small_image, torch.tensor(component).unsqueeze(0)])
                for component in component_masks
            ])

            myelin_masks = batch_inference(myelins_to_detect, self.myelin_model, inference_batch_size)
            detected_myelins.extend(
                self.detect_myelins(
                    myelin_masks[:, 0, :, :].numpy(),
                    np.array(component_masks),
                    offset_top,
                    offset_left,
                    myelin_threshold
                )
            )

            if progress_callback:
                progress_callback(
                    idx + 1,
                    len(masks),
                    f"Processed crop {idx + 1} / {len(masks)}"
                )

        unique_masks, _ = deduplicate_masks(detected_myelins)
        return unique_masks

    @staticmethod
    def split_image(image: np.ndarray,
                    crop_size: Tuple[int, int] = (256, 256),
                    overlap: int = 128) -> Tuple[torch.Tensor, List[Tuple[int, int]]]:
        """
        Split a large image into overlapping crops for processing.

        Parameters:
            image (np.ndarray): Image of size (3, H, W)
            crop_size (Tuple[int, int]): Size of each crop (height, width)
            overlap (int): Overlap between adjacent crops in pixels

        Returns:
            Tuple[torch.Tensor, List[Tuple[int, int]]]:
                - Tensor of crops of shape (N, 3, crop_height, crop_width)
                - List of (top, left) offsets for each crop
        """
        _, height, width = image.shape
        stride = crop_size[0] - overlap

        crops = []
        offsets = []

        for top in range(0, height - crop_size[0] + 1, stride):
            for left in range(0, width - crop_size[1] + 1, stride):
                crop = image[:, top:top + crop_size[0], left:left + crop_size[1]]
                crops.append(crop)
                offsets.append((top, left))

        # Handle edge cases for right and bottom borders
        if width % stride != 0:
            for top in range(0, height - crop_size[0] + 1, stride):
                left = width - crop_size[1]
                crop = image[:, top:top + crop_size[0], left:left + crop_size[1]]
                crops.append(crop)
                offsets.append((top, left))

        if height % stride != 0:
            for left in range(0, width - crop_size[1] + 1, stride):
                top = height - crop_size[0]
                crop = image[:, top:top + crop_size[0], left:left + crop_size[1]]
                crops.append(crop)
                offsets.append((top, left))

        # Handle bottom-right corner
        if height % stride != 0 and width % stride != 0:
            top = height - crop_size[0]
            left = width - crop_size[1]
            crop = image[:, top:top + crop_size[0], left:left + crop_size[1]]
            crops.append(crop)
            offsets.append((top, left))

        return torch.stack([torch.tensor(crop).float() / 255.0 for crop in crops]), offsets

    @staticmethod
    def separate_components(binary_mask: np.ndarray, min_axon_size_pixels: int) -> List[np.ndarray]:
        """
        Separates a binary mask into individual component masks using OpenCV.

        Parameters:
            binary_mask (np.ndarray): Binary mask of shape (H, W)
            min_axon_size_pixels (int): Minimum size in pixels for a component to be considered valid

        Returns:
            List[np.ndarray]: List of binary masks, each containing a single component
        """
        if binary_mask.dtype != np.uint8:
            binary_mask = binary_mask.astype(np.uint8)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask)

        component_masks = []
        for i in range(1, num_labels):
            component_mask = np.zeros_like(binary_mask, dtype=np.uint8)
            component_mask[labels == i] = 1

            if component_mask.sum() > min_axon_size_pixels:
                component_masks.append(component_mask)

        return component_masks

    @staticmethod
    def detect_myelins(myelin_masks: np.ndarray,
                       axon_masks: np.ndarray,
                       offset_top: int,
                       offset_left: int,
                       threshold: float = 0.5) -> List[MyelinSheath]:
        """
        Split myelin masks array into list of MyelinSheath objects, including both myelin and axon masks.

        Parameters:
            myelin_masks (np.ndarray): Array of shape (M, 256, 256) containing M masks of individual myelin sheaths
            axon_masks (np.ndarray): Array of shape (M, 256, 256) containing M masks of corresponding axons
            offset_top (int): Offset from top of the original image
            offset_left (int): Offset from left of the original image
            threshold (float): Threshold for making mask binary

        Returns:
            List[MyelinSheath]: List of detected myelin sheaths with their corresponding axons
        """
        detected_sheaths = []

        for myelin_mask, axon_mask in zip(myelin_masks, axon_masks):
            binary_myelin = (myelin_mask > threshold).astype(np.float32)
            binary_axon = axon_mask.astype(np.float32)  # Axon mask should already be binary

            if not (np.any(binary_myelin) and np.any(binary_axon)):
                continue

            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_myelin.astype(np.uint8))

            if num_labels < 2:  # Skip if no components found (1 includes background)
                continue

            component_areas = stats[1:, cv2.CC_STAT_AREA]  # Skip background
            largest_component_idx = np.argmax(component_areas) + 1

            myelin_component_mask = np.zeros_like(binary_myelin)
            myelin_component_mask[labels == largest_component_idx] = 1.0

            confidence = float(np.mean(myelin_mask[labels == largest_component_idx]))

            sheath = MyelinSheath(
                offset_top=offset_top,
                offset_left=offset_left,
                myelin_mask=Mask(mask=myelin_component_mask.tolist()),
                axon_mask=Mask(mask=binary_axon.tolist()),
                confidence=confidence
            ).prune()

            detected_sheaths.append(sheath)

        return detected_sheaths