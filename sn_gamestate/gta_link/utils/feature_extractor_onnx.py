from __future__ import absolute_import
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
import onnxruntime

from torchreid.utils import (
    check_isfile, load_pretrained_weights, compute_model_complexity
)
from torchreid.models import build_model

import logging
log = logging.getLogger(__name__)


class FeatureExtractorOnnx(object):
    """A simple API for feature extraction.

    FeatureExtractor can be used like a python function, which
    accepts input of the following types:
        - a list of strings (image paths)
        - a list of numpy.ndarray each with shape (H, W, C)
        - a single string (image path)
        - a single numpy.ndarray with shape (H, W, C)
        - a torch.Tensor with shape (B, C, H, W) or (C, H, W)

    Returned is a torch tensor with shape (B, D) where D is the
    feature dimension.

    Args:
        model_name (str): model name.
        model_path (str): path to model weights.
        image_size (sequence or int): image height and width.
        pixel_mean (list): pixel mean for normalization.
        pixel_std (list): pixel std for normalization.
        pixel_norm (bool): whether to normalize pixels.
        device (str): 'cpu' or 'cuda' (could be specific gpu devices).
        verbose (bool): show model details.

    Examples::

        from torchreid.utils import FeatureExtractor

        extractor = FeatureExtractor(
            model_name='osnet_x1_0',
            model_path='a/b/c/model.pth.tar',
            device='cuda'
        )

        image_list = [
            'a/b/c/image001.jpg',
            'a/b/c/image002.jpg',
            'a/b/c/image003.jpg',
            'a/b/c/image004.jpg',
            'a/b/c/image005.jpg'
        ]

        features = extractor(image_list)
        print(features.shape) # output (5, 512)
    """

    def __init__(
        self,
        model_name='',
        model_path='',
        image_size=(256, 128),
        pixel_mean=[0.485, 0.456, 0.406],
        pixel_std=[0.229, 0.224, 0.225],
        pixel_norm=True,
        device='cuda',
        verbose=False
    ):
        # Build model
        self.session = None
        if model_path and check_isfile(model_path):
            providers = ["CPUExecutionProvider"]
            if device == "cuda":
                if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                    providers.insert(0, "CUDAExecutionProvider")
                else:  # Only log warning if CUDA was requested but unavailable
                    log.warning("Failed to start ONNX Runtime with CUDA. Using CPU...")
                    device = torch.device("cpu")
            log.info(f"Using ONNX Runtime {providers[0]}")
            self.session = onnxruntime.InferenceSession(model_path, providers=providers)            
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

        assert self.session is not None, "Model path is invalid or model file does not exist."
        # if verbose:
        #     num_params, flops = compute_model_complexity(
        #         model, (1, 3, image_size[0], image_size[1])
        #     )
        #     print('Model: {}'.format(model_name))
        #     print('- params: {:,}'.format(num_params))
        #     print('- flops: {:,}'.format(flops))            

        # Build transform functions
        transforms = []
        transforms += [T.Resize(image_size)]
        transforms += [T.ToTensor()]
        if pixel_norm:
            transforms += [T.Normalize(mean=pixel_mean, std=pixel_std)]
        preprocess = T.Compose(transforms)

        to_pil = T.ToPILImage()

        # Class attributes
        self.preprocess = preprocess
        self.to_pil = to_pil
        self.device = device

    def __call__(self, input):
        if isinstance(input, (list, tuple)):
            if isinstance(input[0], np.ndarray):
                input_batch = []
                for img in input:
                    img_tensor = self.transform(img).unsqueeze(0)
                    input_batch.append(img_tensor)
                input_batch = torch.cat(input_batch, 0)
            elif isinstance(input[0], torch.Tensor):
                input_batch = torch.stack(input)
            else:
                raise TypeError(f"Unsupported image type: {type(input[0])}")
        elif isinstance(input, np.ndarray):
            input_batch = self.to_pil(input)
            input_batch = self.preprocess(input_batch)
            input_batch = input_batch.unsqueeze(0).to(self.device)
        elif isinstance(input, torch.Tensor):
            if input.dim() == 3:
                input = input.unsqueeze(0)
            input_batch = input.to(self.device)
        else:
            log.warning(f"Unsupported input type: {type(input)}")      # debug line
            raise NotImplementedError

        input_batch = input_batch.numpy()

        features = self.session.run([self.output_name], {self.input_name: input_batch})[0]

        return features
