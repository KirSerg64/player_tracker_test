from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import logging
import onnxruntime

from prtreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from sn_gamestate.tools.transforms import build_transforms

log = logging.getLogger(__name__)


class FeatureExtractor(object):
    """A simple API for feature extraction.

    """

    def __init__(
        self,
        cfg,
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        num_classes=1,
        verbose=True,
        model=None
    ):
        if verbose:
            num_params, flops = compute_model_complexity(
                model, cfg
            )
            print('Model: {}'.format(cfg.model.name))
            print('- params: {:,}'.format(num_params))
            print('- flops: {:,}'.format(flops))

        # if model_path:
        #     assert check_isfile(model_path), \
        #         "PRTreID model weights not found at '{}'".format(model_path)
        #     load_pretrained_weights(model, model_path)

        # Build transform functions
        preprocess = build_transforms(
            image_size[0],
            image_size[1],
            cfg,
            norm_mean=pixel_mean,
            norm_std=pixel_std,
            masks_preprocess=cfg.model.bpbreid.masks.preprocess,
            softmax_weight=cfg.model.bpbreid.masks.softmax_weight,
            background_computation_strategy=cfg.model.bpbreid.masks.background_computation_strategy,
            mask_filtering_threshold=cfg.model.bpbreid.masks.mask_filtering_threshold,
        )

        to_pil = T.ToPILImage()

        self._onnx_providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                self._onnx_providers.insert(0, "CUDAExecutionProvider")
            else:  # Only log warning if CUDA was requested but unavailable
                log.warning("Failed to start ONNX Runtime with CUDA. Using CPU...")
                # device = "cpu"

        log.info(f"Using ONNX Runtime {self._onnx_providers[0]}")
        self._onnx_path = model_path
        self._onnx_path = self._onnx_path.replace(".pth.tar", ".onnx")
        # if not Path(self._onnx_path).is_file():
        #     raise FileNotFoundError(f"ONNX model file not found at {self._onnx_path}.")
        # Class attributes
        self.model = onnxruntime.InferenceSession(self._onnx_path, providers=self._onnx_providers)
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input, external_parts_masks=None):
        if isinstance(input, list):
            images = []
            masks = None

            for i, element in enumerate(input):
                transf_args = {}
                if external_parts_masks is not None:
                    transf_args['mask'] = external_parts_masks[i].transpose(1, 2, 0)
                transf_args['image'] = element
                result = self.preprocess(**transf_args)
                images.append(result['image'])
                if external_parts_masks is not None:
                    masks.append(result['mask'])

            images = torch.stack(images, dim=0)
            images = images.to(self.device)
            if external_parts_masks is not None:
                masks = torch.stack(masks, dim=0)
                masks = masks.to(self.device)

        elif isinstance(input, np.ndarray):
            image = input
            transf_args = {}
            if external_parts_masks is not None:
                transf_args['mask'] = external_parts_masks.transpose(1, 2, 0)
            transf_args['image'] = image
            result = self.preprocess(**transf_args)
            images = result['image'].unsqueeze(0).to(self.device)
            masks = result['mask'].unsqueeze(0).to(self.device)

        elif isinstance(input, torch.Tensor):
            images = input.cpu().detach().numpy()
            masks = None           
        else:
            raise NotImplementedError

        features = self.model.run(None, {"input": images})

        return features
