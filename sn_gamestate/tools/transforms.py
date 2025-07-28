from __future__ import division, print_function, absolute_import

import cv2
import torch
import numpy as np
from albumentations import (
    Resize, Compose, Normalize, ColorJitter, HorizontalFlip, CoarseDropout, RandomCrop, PadIfNeeded
)
from albumentations.pytorch import ToTensorV2
from prtreid.data.masks_transforms import masks_preprocess_all, AddBackgroundMask, ResizeMasks, PermuteMasksDim, \
    RemoveBackgroundMask
from prtreid.data.data_augmentation import RandomOcclusion
from torchvision import transforms as T


class NpToTensor(object):
    def __call__(self, masks):
        assert isinstance(masks, np.ndarray)
        return torch.as_tensor(masks)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TorchNormalizeBatch(object):
    def __init__(self, mean, std, max_pixel_value=255.0):
        self.mean = mean
        self.std = std
        self.max_pixel_value = max_pixel_value

    def __call__(self, img):
        """
        Args:
            img (torch.Tensor): Tensor image of size (B, C, H, W).
        Returns:
            torch.Tensor: Normalized Tensor image.
        """
        mean = torch.as_tensor(self.mean, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        std = torch.as_tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)
        normalized_img = (img - mean * self.max_pixel_value) / std
        return normalized_img

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1}, max_pixel_value={2})'.format(self.mean, self.std, self.max_pixel_value)


def build_transforms(
    height,
    width,
    config,
    mask_scale=4,
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    remove_background_mask=False,
    masks_preprocess = 'none',
    softmax_weight = 0,
    mask_filtering_threshold = 0.3,
    background_computation_strategy = 'threshold',
    **kwargs
):
    """Builds train and test transform functions.

    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to model training.
            Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values. Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation values. Default is
            ImageNet standard deviation values.
    """
    transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        norm_mean = [0.485, 0.456, 0.406] # imagenet mean
        norm_std = [0.229, 0.224, 0.225] # imagenet std
    normalize = Normalize(mean=norm_mean, std=norm_std)

    print('Building test transforms ...')
    print('+ resize to {}x{}'.format(height, width))
    print('+ to torch tensor of range [0, 1]')
    print('+ normalization (mean={}, std={})'.format(norm_mean, norm_std))

    transform_te = [
        Resize(height, width),
        normalize,
        ToTensorV2()
    ]

    transform_te += [PermuteMasksDim()]

    if remove_background_mask:  # ISP masks
        print('+ use remove background mask')
        # remove background before performing other transforms
        transform_te = [RemoveBackgroundMask()] + transform_te

        # Derive background mask from all foreground masks once other tasks have been performed
        print('+ use add background mask')
        transform_te += [AddBackgroundMask('sum')]
    else:  # Pifpaf confidence based masks
        if masks_preprocess != 'none':
            print('+ masks preprocess = {}'.format(masks_preprocess))
            masks_preprocess_transform = masks_preprocess_all[masks_preprocess]
            # mask grouping as first transform to reduce tensor size asap and speed up other transforms
            transform_te = [masks_preprocess_transform()] + transform_te

        print('+ use add background mask')
        transform_te += [AddBackgroundMask(background_computation_strategy, softmax_weight, mask_filtering_threshold)]

    transform_te += [ResizeMasks(height, width, mask_scale)]
    transform_te = Compose(transform_te, is_check_shapes=False)

    return transform_te
