"""
GenerateTracklets API - Tracklab VideoLevelModule wrapper

This module wraps the generate_tracklets functionality from sn_gamestate.gta_link.generate_tracklets
to work as a tracklab VideoLevelModule that can be integrated into the video processing pipeline.
"""

import os
import logging
from pathlib import Path
from typing import Any, Dict
import pickle
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm

from tracklab.pipeline.videolevel_module import VideoLevelModule
from tracklab.utils.cv2 import cv2_load_image

from sn_gamestate.gta_link.utils.feature_extractor_onnx import FeatureExtractorOnnx
from sn_gamestate.gta_link.Tracklet import Tracklet


log = logging.getLogger(__name__)


class GenerateTracklets(VideoLevelModule):
    """
    VideoLevelModule for generating tracklets with embeddings from tracking detections.
    
    This module takes tracking detections DataFrame and generates tracklet embeddings
    using a feature extractor model. The output is a dictionary of Tracklet objects
    that can be passed to RefineTracklets for further processing.
    """
    
    input_columns = ["track_id", "bbox_ltwh", "bbox_conf", "image_id"]
    output_columns = ["tracklets_dict"]
    
    def __init__(self, model_path: str, device: str = 'auto', optimal_batch_size: int = 8, enable_batch_padding: bool = True, **kwargs):
        """
        Initialize GenerateTracklets module.
        
        Args:
            model_path (str): Path to the feature extractor ONNX model
            device (str): Device to use ('cuda', 'cpu', or 'auto')
            optimal_batch_size (int): Optimal batch size for ONNX inference (default: 8)
            enable_batch_padding (bool): Enable padding to optimal batch sizes (default: True)
            **kwargs: Additional arguments
        """
        super().__init__()
        
        # Determine device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        self.model_path = model_path
        self.optimal_batch_size = optimal_batch_size
        self.enable_batch_padding = enable_batch_padding
        
        # Initialize feature extractor with batch optimization
        self.extractor = FeatureExtractorOnnx(
            model_name='osnet_x1_0',
            model_path=model_path,
            device=self.device,
            optimal_batch_size=optimal_batch_size,
            enable_batch_padding=enable_batch_padding
        )
        
        self.to_pil = T.ToPILImage()

        # Image transforms for cropped detections
        self.val_transforms = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        log.info(f"GenerateTracklets initialized with device: {self.device}, "
                f"optimal_batch_size: {optimal_batch_size}, batch_padding: {enable_batch_padding}")
    
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Process tracking detections to generate tracklets with embeddings.
        
        Args:
            detections (pd.DataFrame): DataFrame containing tracking detections with columns:
                - track_id: Track identifier
                - bbox_ltwh: Bounding box in [left, top, width, height] format
                - bbox_conf: Detection confidence
                - image_id: Frame identifier
                - (other columns will be preserved)
            metadatas (pd.DataFrame): Metadata for each frame (contains image file paths)
                
        Returns:
            pd.DataFrame: Modified detections DataFrame with added 'tracklets_dict' column
        """
        
        if detections.empty:
            log.warning("No detections provided to GenerateTracklets")
            detections['tracklets_dict'] = None
            return detections
            
        if 'track_id' not in detections.columns:
            log.warning("No track_id column found in detections")
            detections['tracklets_dict'] = None
            return detections
            
        log.info(f"Processing {len(detections)} detections across {len(detections.image_id.unique())} frames")
        
        # Initialize tracklets dictionary
        seq_tracks = {}
        
        # Group by frame for efficient processing
        frames_data = {}
        for frame_id in detections.image_id.unique():
            frame_detections = detections[detections.image_id == frame_id]
            
            # Get corresponding metadata for this frame
            if frame_id in metadatas.index:
                frame_metadata = metadatas.loc[frame_id]
                if 'file_path' in frame_metadata:
                    frames_data[frame_id] = {
                        'detections': frame_detections,
                        'image_path': frame_metadata['file_path']
                    }
                else:
                    log.warning(f"No file_path found in metadata for frame {frame_id}")
            else:
                log.warning(f"No metadata found for frame {frame_id}")
        
        # Process each frame
        for frame_id in tqdm(sorted(frames_data.keys()), desc="Generating tracklets"):
            frame_data = frames_data[frame_id]
            frame_detections = frame_data['detections']
            image_path = frame_data['image_path']
               
            # Load image
            try:
                img = cv2_load_image(image_path)
            except Exception as e:
                log.error(f"Failed to load image {image_path}: {e}")
                continue
            
            # Prepare batch for feature extraction
            input_batch = None
            tid2idx = {}
            
            # Process each detection in this frame
            for idx, detection in frame_detections.iterrows():
                track_id = detection['track_id']
                
                # Skip invalid track IDs
                if pd.isna(track_id) or track_id < 0:
                    continue
                
                bbox_ltwh = detection['bbox_ltwh'].tolist()
                bbox_conf = detection['bbox_conf'].item()
                
                # Extract bounding box coordinates
                if hasattr(bbox_ltwh, '__len__') and len(bbox_ltwh) >= 4:
                    l, t, w, h = bbox_ltwh[:4]
                else:
                    log.warning(f"Invalid bbox format for detection {idx}: {bbox_ltwh}")
                    continue
                
                # Update tracklet with detection info
                if track_id not in seq_tracks:
                    seq_tracks[track_id] = Tracklet(track_id, frame_id, bbox_conf, [l, t, w, h])
                else:
                    seq_tracks[track_id].append_det(frame_id, bbox_conf, [l, t, w, h])
                
                tid2idx[track_id] = len(tid2idx)  # Index for batch processing
                
                # Crop and transform image
                try:
                    # Ensure bbox coordinates are within image bounds
                    img_h, img_w = img.shape[:2]
                    l = int(max(0, min(l, img_w - 1)))
                    t = int(max(0, min(t, img_h - 1)))
                    r = int(max(l + 1, min(l + w, img_w)))
                    b = int(max(t + 1, min(t + h, img_h)))

                    # Crop and transform
                    crop = self.to_pil(img[t:b, l:r].copy())
                    crop_tensor = self.val_transforms(crop).unsqueeze(0)
                    
                    if input_batch is None:
                        input_batch = crop_tensor
                    else:
                        input_batch = torch.cat([input_batch, crop_tensor], dim=0)
                        
                except Exception as e:
                    log.warning(f"Failed to crop detection {idx}: {e}")
                    continue
            
            # Extract features if we have valid crops
            if input_batch is not None and len(tid2idx) > 0:
                try:                    
                    features = self.extractor(input_batch)
                    
                    # Convert to numpy if needed
                    if isinstance(features, torch.Tensor):
                        features = features.cpu().detach().numpy()
                    
                    # Normalize and add features to tracklets
                    for tid, batch_idx in tid2idx.items():
                        if batch_idx < len(features):
                            feat = features[batch_idx]
                            feat = feat / np.linalg.norm(feat)  # L2 normalize
                            seq_tracks[tid].append_feat(feat)
                        
                except Exception as e:
                    log.error(f"Feature extraction failed for frame {frame_id}: {e}")
            else:
                log.debug(f"No valid detections to process in frame {frame_id}")
        
        # Add tracklets dictionary to the detections DataFrame
        # Since this is video-level processing, we add it as a single value that all rows will share
        tracklets_dict = seq_tracks if seq_tracks else {}
        
        # For video-level modules, we typically store the result in the first row or all rows
        # Here we'll store it in a way that can be accessed by the next module
        if not detections.empty:
            # Use at[] method to avoid Series conversion issues
            detections['tracklets_dict'] = None  # Initialize column
            detections.at[detections.index[0], 'tracklets_dict'] = tracklets_dict
        else:
            # If no detections, create a minimal result DataFrame
            detections = pd.DataFrame({'tracklets_dict': [tracklets_dict]})
        
        log.info(f"Generated {len(tracklets_dict)} tracklets with embeddings")
        
        return detections
    
    def save_tracklets(self, tracklets_dict: Dict, output_path: str):
        """
        Save tracklets dictionary to pickle file.
        
        Args:
            tracklets_dict: Dictionary of Tracklet objects
            output_path: Path to save the pickle file
        """
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(tracklets_dict, f)
            log.info(f"Tracklets saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save tracklets to {output_path}: {e}")


def create_generate_tracklets_module(
    model_path: str, 
    device: str = 'auto', 
    optimal_batch_size: int = 8, 
    enable_batch_padding: bool = True
) -> GenerateTracklets:
    """
    Factory function to create GenerateTracklets module.
    
    Args:
        model_path: Path to feature extractor model
        device: Device to use for processing
        optimal_batch_size: Optimal batch size for ONNX inference (powers of 2 recommended)
        enable_batch_padding: Enable padding to optimal batch sizes for better performance
        
    Returns:
        GenerateTracklets: Configured module instance
    """
    return GenerateTracklets(
        model_path=model_path, 
        device=device,
        optimal_batch_size=optimal_batch_size,
        enable_batch_padding=enable_batch_padding
    )
