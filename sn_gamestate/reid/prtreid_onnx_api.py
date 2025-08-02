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
from sn_gamestate.tools.feature_extractor_onnx import FeatureExtractor
from prtreid.utils.imagetools import (
    build_gaussian_heatmaps,
)
from tracklab.utils.collate import Unbatchable

from albumentations import (
    Resize, Compose, Normalize
)
from albumentations.pytorch import ToTensorV2

from torch.nn import functional as F
from prtreid.data.masks_transforms import (
    CocoToSixBodyMasks,
    masks_preprocess_transforms,
)
from prtreid.utils.constants import bn_correspondants
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

            # ONNX model returns a list of numpy arrays, reconstruct the dictionary format
    output_names = ['globl', 'backg', 'foreg', 'conct', 'parts', 'bn_globl', 'bn_backg', 'bn_foreg', 'bn_conct', 
                    'bn_parts', 'globl1', 'backg1', 'foreg1', 'conct1', 'parts1', 'globl2', 'backg2', 'foreg2', 
                    'conct2', 'parts2', 'globl3', 'backg3', 'foreg3', 'conct3', 'globl4', 'backg4', 'foreg4', 
                    'conct4', 'parts4', 'all_tensor', 'globl5', 'backg5', 'foreg5', 'conct5', 'parts5']

    CROP_HEIGHT = 256
    CROP_WIDTH = 128
    NORM_MEAN = (0.485, 0.456, 0.406), # imagenet mean
    NORM_STD = (0.229, 0.224, 0.225), # imagenet std

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
        self.model = None
        # Build transform functions
        normalize = Normalize(
            mean=(0.485, 0.456, 0.406), # imagenet mean
            std=(0.229, 0.224, 0.225), # imagenet std
            max_pixel_value=255.0,
            always_apply=True,
        )
        self._image_transfrom = Compose([
            Resize(self.CROP_HEIGHT, self.CROP_WIDTH),
            normalize,
            ToTensorV2(),
        ])


    @torch.no_grad()
    def preprocess(
        self, image, detection: pd.Series, metadata: pd.Series
    ):  # Tensor RGB (1, 3, H, W)
        # mask_w, mask_h = 32, 64
        l, t, r, b = detection.bbox.ltrb(
            image_shape=(image.shape[1], image.shape[0]), rounded=True
        )
        crop = image[t:b, l:r].copy()
        crop = self._image_transfrom(**{'image': crop})['image']

        batch = {
            "img": crop,
        }

        return batch

    # @torch.no_grad()
    def process(self, batch, detections: pd.DataFrame, metadatas: pd.DataFrame):
        im_crops = batch["img"]
        # im_crops = [im_crop.cpu().detach().numpy() for im_crop in im_crops]
        if "masks" in batch:
            external_parts_masks = batch["masks"]
            external_parts_masks = external_parts_masks.cpu().detach().numpy()
        else:
            external_parts_masks = None

        if self.feature_extractor is None:
            self.feature_extractor = FeatureExtractor(
                self.cfg,
                model_path=self.cfg.model.load_weights,
                device=self.device,
                image_size=(self.cfg.data.height, self.cfg.data.width),
                model=self.model,
                verbose=False,  # FIXME
            )

        reid_result = self.feature_extractor(
            im_crops, external_parts_masks=external_parts_masks
        )

        (embeddings, 
         visibility_scores, 
         body_masks, 
         _, 
         role_cls_scores,
         ) = self.extract_embeddings(
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

    def extract_embeddings(self, model_output, test_embeddings):
        
        # Map the first few arrays to base component names
        embeddings = {}
        visibility_scores = {}
        parts_masks = {}
        
        for i, name in enumerate(self.output_names[:10]):  # Only process first 10 base components
            if i < len(model_output):
                tensor = torch.from_numpy(model_output[i])
                base_name = ''.join([c for c in name if not c.isdigit()])
                
                if base_name not in embeddings:
                    embeddings[base_name] = tensor
                    # Create visibility scores with same batch size
                    batch_size = tensor.shape[0]
                    visibility_scores[base_name] = torch.ones(batch_size, 1) if len(tensor.shape) < 3 else torch.ones(tensor.shape[:2])
                    # Create parts masks 
                    parts_masks[base_name] = tensor if len(tensor.shape) == 4 else torch.ones(batch_size, 1, 64, 32)
        
        # Create role classification scores for the calling code
        role_cls_score = {key: torch.ones(embeddings[key].shape[0], 5) for key in embeddings.keys()}
        
        # Process test embeddings
        embeddings_list = []
        visibility_scores_list = []
        embeddings_masks_list = []

        for test_emb in test_embeddings:
            embds = embeddings[test_emb]
            embeddings_list.append(embds if len(embds.shape) == 3 else embds.unsqueeze(1))
            
            lookup_key = bn_correspondants.get(test_emb, test_emb)
            vis_scores = visibility_scores[lookup_key]
            visibility_scores_list.append(vis_scores if len(vis_scores.shape) == 2 else vis_scores.unsqueeze(1))
            
            pt_masks = parts_masks[lookup_key]
            embeddings_masks_list.append(pt_masks if len(pt_masks.shape) == 4 else pt_masks.unsqueeze(1))

        embeddings = torch.cat(embeddings_list, dim=1)
        visibility_scores = torch.cat(visibility_scores_list, dim=1)
        embeddings_masks = torch.cat(embeddings_masks_list, dim=1)

        return embeddings, visibility_scores, embeddings_masks, None, role_cls_score

    def train(self):
        self.engine, self.model = build_torchreid_model_engine(self.cfg)
        self.engine.run(**engine_run_kwargs(self.cfg))
