import numpy as np
import pandas as pd
import torch
import logging

from omegaconf import OmegaConf
from yacs.config import CfgNode as CN

from tracklab.pipeline import DetectionLevelModule
# FIXME this should be removed and use KeypointsSeriesAccessor and KeypointsFrameAccessor
from tracklab.utils.coordinates import rescale_keypoints
from tracklab.utils.collate import default_collate
# from sn_gamestate.reid.prtreid_dataset import ReidDataset
from prtreid.scripts.main import build_config, build_torchreid_model_engine
from prtreid.tools.feature_extractor import FeatureExtractor
from prtreid.utils.imagetools import (
    build_gaussian_heatmaps,
)
from tracklab.utils.collate import Unbatchable

import tracklab
from pathlib import Path
import onnxruntime


import prtreid
from torch.nn import functional as F
from prtreid.data.masks_transforms import (
    CocoToSixBodyMasks,
    masks_preprocess_transforms,
)
from prtreid.utils.tools import extract_test_embeddings
from prtreid.data.datasets import configure_dataset_class

from prtreid.scripts.default_config import engine_run_kwargs

from tracklab.utils.download import download_file
log = logging.getLogger(__name__)


class PRTONNXReId(DetectionLevelModule):
    collate_fn = default_collate
    input_columns = ["bbox_ltwh"]
    output_columns = ["embeddings", "visibility_scores", "body_masks", "role_detection", "role_confidence"]
    forget_columns = ["embeddings", "body_masks"]
    role_mapping = {'ball': 0, 'goalkeeper': 1, 'other': 2, 'player': 3, 'referee': 4, None: -1}

    def __init__(
        self,
        cfg,
        tracking_dataset,
        dataset,
        device,
        save_path,
        job_id,
        use_keypoints_visibility_scores_for_reid,
        training_enabled,
        batch_size,
    ):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device

        self.use_keypoints_visibility_scores_for_reid = (
            use_keypoints_visibility_scores_for_reid
        )
        self.cfg = CN(OmegaConf.to_container(cfg, resolve=True))
        self.inverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
        # set parts information (number of parts K and each part name),
        # depending on the original loaded masks size or the transformation applied:
        self.cfg.data.save_dir = save_path
        self.cfg.project.job_id = job_id
        self.cfg.use_gpu = torch.cuda.is_available()
        self.cfg = build_config(config=self.cfg)
        self.test_embeddings = self.cfg.model.bpbreid.test_embeddings
        # Register the PoseTrack21ReID dataset to Torchreid that will be instantiated when building Torchreid engine.
        self.training_enabled = training_enabled
        self.feature_extractor = None

        cuda = isinstance(self.device, torch.device) and torch.cuda.is_available() and self.device.type != "cpu"  # use CUDA
        providers = ["CPUExecutionProvider"]
        if cuda:
            if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                providers.insert(0, "CUDAExecutionProvider")
            else:  # Only log warning if CUDA was requested but unavailable
                log.warning("Failed to start ONNX Runtime with CUDA. Using CPU...")
                device = torch.device("cpu")
                cuda = False
        log.info(f"Using ONNX Runtime {providers[0]}")
        ONNX_PATH = self.cfg.model.load_weights
        ONNX_PATH = ONNX_PATH.replace(".pth.tar", ".onnx")
        self.feature_extractor = onnxruntime.InferenceSession(ONNX_PATH, providers=providers)

    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r]
        crop = Unbatchable([crop])
        batch = {
            "img": crop,
        }

        return batch

    @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None
        reid_result = self.feature_extractor(
            im_crops, external_parts_masks=external_parts_masks
        )
        embeddings, visibility_scores, body_masks, _, role_cls_scores = extract_test_embeddings(
            reid_result, self.test_embeddings
        )
        
        role_scores_ = []
        role_scores_.append(role_cls_scores['globl'].cpu() if role_cls_scores is not None else None)
        role_scores_ = torch.cat(role_scores_, 0) if role_scores_[0] is not None else None
        roles = [torch.argmax(i).item() for i in role_scores_]
        roles = [self.inverse_role_mapping[index] for index in roles]
        role_confidence = [torch.max(i).item() for i in role_scores_]

        embeddings = embeddings.cpu().detach().numpy()
        visibility_scores = visibility_scores.cpu().detach().numpy()
        body_masks = body_masks.cpu().detach().numpy()

        if self.use_keypoints_visibility_scores_for_reid:
            kp_visibility_scores = batch["visibility_scores"].numpy()
            if visibility_scores.shape[1] > kp_visibility_scores.shape[1]:
                kp_visibility_scores = np.concatenate(
                    [np.ones((visibility_scores.shape[0], 1)), kp_visibility_scores],
                    axis=1,
                )
            visibility_scores = np.float32(kp_visibility_scores)

        reid_df = pd.DataFrame(
            {
                "embeddings": list(embeddings),
                "visibility_scores": list(visibility_scores),
                "body_masks": list(body_masks),
                "role_detection": roles,
                "role_confidence": role_confidence,
            },
            index=detections.index,
        )
        return reid_df

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))
