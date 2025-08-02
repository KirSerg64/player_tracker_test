"""
RefineTracklets API - Tracklab VideoLevelModule wrapper

This module wraps the refine_tracklets functionality from sn_gamestate.gta_link.refine_tracklets
to work as a tracklab VideoLevelModule that can be integrated into the video processing pipeline.
"""

import logging
from typing import Any, Dict, Optional
import pickle
import numpy as np
import pandas as pd
from collections import defaultdict

from tracklab.pipeline.videolevel_module import VideoLevelModule
from sn_gamestate.gta_link.Tracklet import Tracklet
# Import refinement functions with fallbacks
from sn_gamestate.gta_link.refine_tracklets import (
    find_consecutive_segments,
    query_subtracks,
    get_distance_matrix,
    detect_id_switch,
    get_spatial_constraints
)

from sn_gamestate.gta_link.refine_tracklets_batched import (
    split_tracklets,
    merge_tracklets,
    merge_tracklets_batched
)


log = logging.getLogger(__name__)


class RefineTracklets(VideoLevelModule):
    """
    VideoLevelModule for refining and merging tracklets using advanced algorithms.
    
    This module takes a dictionary of Tracklet objects (typically from GenerateTracklets)
    and applies splitting and merging algorithms to improve tracking consistency.
    """
    
    input_columns = ["tracklets_dict"]
    output_columns = ["refined_tracklets_dict", "track_id_refined"]
    
    def __init__(
        self,
        use_split: bool = True,
        eps: float = 0.5,
        min_samples: int = 3,
        max_k: int = 10,
        min_len: int = 30,
        merge_dist_thres: float = 0.3,
        spatial_factor: float = 1.0,
        batch_size: int = 50,
        use_batched_merge: bool = True,
        mapping_strategy: str = "sequential",  # "similarity" or "sequential"
        return_refined_detections: bool = True,  # If True, return refined tracklets as detections
        **kwargs
    ):
        """
        Initialize RefineTracklets module.
        
        Args:
            use_split (bool): Whether to use tracklet splitting
            eps (float): DBSCAN epsilon parameter for splitting
            min_samples (int): DBSCAN minimum samples parameter
            max_k (int): Maximum number of clusters for splitting
            min_len (int): Minimum tracklet length threshold
            merge_dist_thres (float): Distance threshold for merging
            spatial_factor (float): Spatial constraint factor
            batch_size (int): Batch size for processing
            mapping_strategy (str): Track ID mapping strategy - "similarity" or "sequential"
            return_refined_detections (bool): If True, return refined tracklets as detections DataFrame instead of original detections
            **kwargs: Additional arguments
        """
        super().__init__()
        
        self.use_split = use_split
        self.eps = eps
        self.min_samples = min_samples
        self.max_k = max_k
        self.min_len = min_len
        self.merge_dist_thres = merge_dist_thres
        self.spatial_factor = spatial_factor
        self.batch_size = batch_size
        self.use_batched_merge = use_batched_merge
        self.mapping_strategy = mapping_strategy
        self.return_refined_detections = return_refined_detections

        log.info(f"RefineTracklets initialized with use_split={use_split}, "
                f"eps={eps}, min_samples={min_samples}, merge_dist_thres={merge_dist_thres}, "
                f"mapping_strategy={mapping_strategy}, return_refined_detections={return_refined_detections}")
    
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Process tracklets dictionary to refine and merge tracklets.
        
        Args:
            detections (pd.DataFrame): DataFrame containing 'tracklets_dict' column
            metadatas (pd.DataFrame): Metadata (not used in this module)
            
        Returns:
            pd.DataFrame: Detections with refined tracklet information
        """
        
        if detections.empty:
            log.warning("No detections provided to RefineTracklets")
            return detections
            
        if 'tracklets_dict' not in detections.columns:
            log.warning("No tracklets_dict column found in detections")
            detections['refined_tracklets_dict'] = None
            detections['track_id_refined'] = detections.get('track_id', None)
            return detections
        
        # Get tracklets dictionary from the first non-null entry
        tracklets_dict = None
        for idx, row in detections.iterrows():
            if row['tracklets_dict'] is not None:
                tracklets_dict = row['tracklets_dict']
                break
        
        if not tracklets_dict:
            log.warning("No tracklets dictionary found in detections")
            detections['refined_tracklets_dict'] = None
            detections['track_id_refined'] = detections.get('track_id', None)
            return detections
        
        log.info(f"Refining {len(tracklets_dict)} tracklets")
        
        try:
            # Step 1: Get spatial constraints
            max_x_range, max_y_range = self._get_spatial_constraints(tracklets_dict)
            
            # Step 2: Split tracklets if enabled
            if self.use_split:
                log.info(f"Splitting tracklets - before: {len(tracklets_dict)}")
                split_tracklets = self._split_tracklets(tracklets_dict)
                log.info(f"After splitting: {len(split_tracklets)}")
            else:
                split_tracklets = tracklets_dict
            
            # Step 3: Merge tracklets
            log.info(f"Merging tracklets - before: {len(split_tracklets)}")
            if self.use_batched_merge:
                refined_tracklets = self._merge_tracklets_batched(
                    split_tracklets, max_x_range, max_y_range
                )
            else:
                refined_tracklets = self._merge_tracklets(
                    split_tracklets, max_x_range, max_y_range
                )
            log.info(f"After merging: {len(refined_tracklets)}")
            
            # Step 4: Create mapping from old track IDs to new track IDs
            # if self.mapping_strategy == "sequential":
            #     track_id_mapping = self._create_sequential_track_id_mapping(tracklets_dict, refined_tracklets)
            # else:  # "similarity" strategy
            #     track_id_mapping = self._create_track_id_mapping(tracklets_dict, refined_tracklets)
            
            # Step 5: Update detections with refined track IDs
            # detections = self._update_detections_with_refined_ids(detections, track_id_mapping)
            
            # Step 6: Add refined tracklets dictionary
            # Initialize column with None values
            detections['refined_tracklets_dict'] = [None] * len(detections)
            
            # Set the refined tracklets dictionary in the first row only
            if not detections.empty:
                first_idx = detections.index[0]
                detections.at[first_idx, 'refined_tracklets_dict'] = refined_tracklets
            
            log.info(f"Tracklet refinement completed: {len(tracklets_dict)} -> {len(refined_tracklets)}")
            
        except Exception as e:
            log.error(f"Error during tracklet refinement: {e}")
            # Fallback: return original data
            detections['refined_tracklets_dict'] = [tracklets_dict if i == 0 else None for i in range(len(detections))]
            detections['track_id_refined'] = detections.get('track_id', None)
        
        # If return_refined_detections flag is set, convert refined tracklets to detections format
        if self.return_refined_detections:
            try:
                refined_tracklets_dict = None
                for idx, row in detections.iterrows():
                    if row.get('refined_tracklets_dict') is not None:
                        refined_tracklets_dict = row['refined_tracklets_dict']
                        break
                
                if refined_tracklets_dict:
                    refined_detections = self._convert_tracklets_to_detections(
                        refined_tracklets_dict, detections, metadatas
                    )
                    log.info(f"Converted refined tracklets to detections: {len(refined_detections)} rows")
                    return refined_detections
                else:
                    log.warning("No refined tracklets found, returning original detections")
            except Exception as e:
                log.error(f"Error converting refined tracklets to detections: {e}")
                log.info("Falling back to returning original detections")
        
        return detections
    
    def _get_spatial_constraints(self, tracklets_dict: Dict) -> tuple:
        """Get spatial constraints for merging."""
        try:
            max_x_range, max_y_range = get_spatial_constraints(tracklets_dict, self.spatial_factor)
            log.debug(f"Spatial constraints: x_range={max_x_range}, y_range={max_y_range}")
            return max_x_range, max_y_range
        except Exception as e:
            log.warning(f"Failed to compute spatial constraints: {e}")
            return 1000.0, 1000.0  # Default values
    
    def _split_tracklets(self, tracklets_dict: Dict) -> Dict:
        """Split tracklets using DBSCAN clustering."""
        try:
            return split_tracklets(
                tracklets_dict,
                eps=self.eps,
                max_k=self.max_k,
                min_samples=self.min_samples,
                len_thres=self.min_len
            )
        except Exception as e:
            log.error(f"Error during tracklet splitting: {e}")
            return tracklets_dict
    
    def _merge_tracklets_batched(self, tracklets_dict: Dict, max_x_range: float, max_y_range: float) -> Dict:
        """Merge tracklets in batches."""
        try:
            # Use the batched merging function from refine_tracklets_batched
            return merge_tracklets_batched(
                tracklets_dict,
                seq2Dist={},  # Empty dict as we're not using it for visualization
                batch_size=self.batch_size,
                max_x_range=max_x_range,
                max_y_range=max_y_range,
                merge_dist_thres=self.merge_dist_thres
            )
            
        except Exception as e:
            log.error(f"Error during tracklet merging: {e}")
            return tracklets_dict
    
    def _merge_tracklets(self, tracklets_dict: Dict, max_x_range: float, max_y_range: float) -> Dict:
        """Merge tracklets using distance threshold."""
        try:
            # Use the simple merge function from refine_tracklets_batched
            return merge_tracklets(
                tracklets_dict,
                merge_dist_thres=self.merge_dist_thres,
                max_x_range=max_x_range,
                max_y_range=max_y_range
            )
        except Exception as e:
            log.error(f"Error in merge_tracklets: {e}")
            return tracklets_dict
    
    def _create_sequential_track_id_mapping(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Create mapping from original track IDs to sequential IDs (1.0, 2.0, 3.0, ...).
        
        This is a simple mapping strategy that assigns sequential floating-point IDs
        starting from 1.0 to all original tracklets, regardless of the refined tracklets.
        
        Args:
            original_tracklets (Dict): Dictionary of original tracklets
            refined_tracklets (Dict): Dictionary of refined tracklets (not used in this strategy)
            
        Returns:
            Dict: Mapping from original track IDs to sequential IDs
        """
        mapping = {}
        
        try:
            # Sort original tracklet IDs for consistent ordering
            sorted_orig_ids = sorted(refined_tracklets.keys())
            
            # Assign sequential IDs starting from 1.0
            for i, orig_id in enumerate(sorted_orig_ids, start=1):
                mapping[orig_id] = float(i)
            
            log.debug(f"Created sequential mapping: {len(mapping)} tracklets -> IDs 1.0 to {len(mapping)}.0")
            
        except Exception as e:
            log.error(f"Error creating sequential track ID mapping: {e}")
            # Fallback: identity mapping
            mapping = {orig_id: orig_id for orig_id in original_tracklets.keys()}
        
        return mapping
    
    def _create_track_id_mapping(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Create mapping from original track IDs to refined track IDs.
        
        Uses a more sophisticated approach considering:
        1. Temporal overlap
        2. Spatial similarity (bbox overlap)
        3. Feature similarity (if available)
        4. One-to-one assignment with conflict resolution
        """
        mapping = {}
        
        try:
            # Calculate similarity matrix between original and refined tracklets
            similarity_matrix = self._calculate_tracklet_similarity_matrix(
                original_tracklets, refined_tracklets
            )
            
            # Use Hungarian algorithm or greedy assignment with conflict resolution
            mapping = self._assign_tracklets_with_conflicts(
                original_tracklets, refined_tracklets, similarity_matrix
            )
            
            log.debug(f"Created tracklet mapping with {len(mapping)} assignments")
            
        except Exception as e:
            log.error(f"Error creating track ID mapping: {e}")
            # Fallback: identity mapping
            mapping = {orig_id: orig_id for orig_id in original_tracklets.keys()}
        
        return mapping
    
    def _calculate_tracklet_similarity_matrix(self, original_tracklets: Dict, refined_tracklets: Dict) -> Dict:
        """
        Calculate similarity matrix between original and refined tracklets.
        
        Returns:
            Dict with structure: {(orig_id, refined_id): similarity_score}
        """
        similarity_matrix = {}
        
        for orig_id, orig_tracklet in original_tracklets.items():
            for refined_id, refined_tracklet in refined_tracklets.items():
                similarity = self._calculate_tracklet_similarity(orig_tracklet, refined_tracklet)
                similarity_matrix[(orig_id, refined_id)] = similarity
        
        return similarity_matrix
    
    def _calculate_tracklet_similarity(self, tracklet1, tracklet2) -> float:
        """
        Calculate similarity between two tracklets using multiple factors.
        
        Returns:
            float: Similarity score between 0 and 1
        """
        # 1. Temporal overlap (most important)
        times1 = set(tracklet1.times) if hasattr(tracklet1, 'times') else set()
        times2 = set(tracklet2.times) if hasattr(tracklet2, 'times') else set()
        
        if not times1 or not times2:
            return 0.0
        
        temporal_overlap = len(times1.intersection(times2))
        temporal_union = len(times1.union(times2))
        temporal_score = temporal_overlap / temporal_union if temporal_union > 0 else 0.0
        
        # If no temporal overlap, similarity is 0
        if temporal_overlap == 0:
            return 0.0
        
        # 2. Spatial similarity (bbox overlap in overlapping frames)
        spatial_score = self._calculate_spatial_similarity(tracklet1, tracklet2, times1.intersection(times2))
        
        # 3. Feature similarity (if features are available)
        feature_score = self._calculate_feature_similarity(tracklet1, tracklet2)
        
        # Weighted combination
        weights = {
            'temporal': 0.5,
            'spatial': 0.3,
            'feature': 0.2
        }
        
        total_score = (
            weights['temporal'] * temporal_score +
            weights['spatial'] * spatial_score +
            weights['feature'] * feature_score
        )
        
        return total_score
    
    def _calculate_spatial_similarity(self, tracklet1, tracklet2, common_times) -> float:
        """Calculate spatial similarity based on bbox overlap in common frames."""
        if not common_times:
            return 0.0
        
        try:
            bbox_overlaps = []
            
            for time in common_times:
                # Find bboxes for this time in both tracklets
                bbox1 = self._get_bbox_at_time(tracklet1, time)
                bbox2 = self._get_bbox_at_time(tracklet2, time)
                
                if bbox1 is not None and bbox2 is not None:
                    overlap = self._calculate_bbox_iou(bbox1, bbox2)
                    bbox_overlaps.append(overlap)
            
            return np.mean(bbox_overlaps) if bbox_overlaps else 0.0
            
        except Exception as e:
            log.debug(f"Error calculating spatial similarity: {e}")
            return 0.0
    
    def _get_bbox_at_time(self, tracklet, time):
        """Get bbox for tracklet at specific time."""
        try:
            if hasattr(tracklet, 'times') and hasattr(tracklet, 'bboxes'):
                if time in tracklet.times:
                    idx = tracklet.times.index(time)
                    if idx < len(tracklet.bboxes):
                        return tracklet.bboxes[idx]
        except:
            pass
        return None
    
    def _calculate_bbox_iou(self, bbox1, bbox2) -> float:
        """Calculate IoU between two bboxes in [l, t, w, h] format."""
        try:
            # Convert to [x1, y1, x2, y2]
            l1, t1, w1, h1 = bbox1[:4]
            l2, t2, w2, h2 = bbox2[:4]
            
            x1_1, y1_1, x2_1, y2_1 = l1, t1, l1 + w1, t1 + h1
            x1_2, y1_2, x2_2, y2_2 = l2, t2, l2 + w2, t2 + h2
            
            # Calculate intersection
            xi1 = max(x1_1, x1_2)
            yi1 = max(y1_1, y1_2)
            xi2 = min(x2_1, x2_2)
            yi2 = min(y2_1, y2_2)
            
            if xi2 <= xi1 or yi2 <= yi1:
                return 0.0
            
            intersection = (xi2 - xi1) * (yi2 - yi1)
            
            # Calculate union
            area1 = w1 * h1
            area2 = w2 * h2
            union = area1 + area2 - intersection
            
            return intersection / union if union > 0 else 0.0
            
        except:
            return 0.0
    
    def _calculate_feature_similarity(self, tracklet1, tracklet2) -> float:
        """Calculate feature similarity using cosine similarity."""
        try:
            if hasattr(tracklet1, 'features') and hasattr(tracklet2, 'features'):
                features1 = tracklet1.features
                features2 = tracklet2.features
                
                if features1 and features2:
                    # Use average feature vectors
                    avg_feat1 = np.mean(features1, axis=0)
                    avg_feat2 = np.mean(features2, axis=0)
                    
                    # Cosine similarity
                    dot_product = np.dot(avg_feat1, avg_feat2)
                    norm1 = np.linalg.norm(avg_feat1)
                    norm2 = np.linalg.norm(avg_feat2)
                    
                    if norm1 > 0 and norm2 > 0:
                        return dot_product / (norm1 * norm2)
            
            return 0.0
            
        except Exception as e:
            log.debug(f"Error calculating feature similarity: {e}")
            return 0.0
    
    def _assign_tracklets_with_conflicts(self, original_tracklets: Dict, refined_tracklets: Dict, 
                                       similarity_matrix: Dict) -> Dict:
        """
        Assign tracklets resolving conflicts using a greedy approach with thresholding.
        
        Alternative approaches:
        1. Hungarian algorithm for optimal assignment
        2. Greedy with conflict resolution (implemented here)
        3. Many-to-one mapping for merged tracklets
        """
        mapping = {}
        used_refined_ids = set()
        min_similarity_threshold = 0.1  # Minimum similarity to create mapping
        
        # Sort original tracklets by their best similarity score (descending)
        orig_scores = {}
        for orig_id in original_tracklets.keys():
            best_score = max(
                similarity_matrix.get((orig_id, ref_id), 0.0) 
                for ref_id in refined_tracklets.keys()
            )
            orig_scores[orig_id] = best_score
        
        sorted_orig_ids = sorted(orig_scores.keys(), key=lambda x: orig_scores[x], reverse=True)
        
        # Assign each original tracklet to best available refined tracklet
        for orig_id in sorted_orig_ids:
            best_refined_id = None
            best_similarity = min_similarity_threshold
            
            for refined_id in refined_tracklets.keys():
                similarity = similarity_matrix.get((orig_id, refined_id), 0.0)
                
                # For one-to-one mapping, skip already used refined tracklets
                # Comment out next line to allow many-to-one mapping
                if refined_id in used_refined_ids:
                    continue
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_refined_id = refined_id
            
            if best_refined_id is not None:
                mapping[orig_id] = best_refined_id
                used_refined_ids.add(best_refined_id)
            else:
                # No good match found, keep original ID
                mapping[orig_id] = orig_id
        
        return mapping
    
    def _update_detections_with_refined_ids(self, detections: pd.DataFrame, track_id_mapping: Dict) -> pd.DataFrame:
        """Update detections DataFrame with refined track IDs."""
        
        # Initialize track_id_refined column
        if 'track_id' in detections.columns:
            detections = detections.copy()  # Create a copy to avoid SettingWithCopyWarning
            detections['track_id_refined'] = detections['track_id'].copy()
        else:
            detections = detections.copy()
            detections['track_id_refined'] = None
            return detections
        
        if track_id_mapping:
            try:
                # Apply mapping using proper pandas assignment
                def map_track_id(track_id):
                    if pd.isna(track_id):
                        return track_id
                    return track_id_mapping.get(track_id, track_id)
                
                detections['track_id_refined'] = detections['track_id'].apply(map_track_id)
                log.debug(f"Updated {len(detections)} detections with refined track IDs")
                
            except Exception as e:
                log.error(f"Error updating track IDs: {e}")
                # Fallback: keep original track IDs
                detections['track_id_refined'] = detections['track_id'].copy()
        
        return detections
    
    def _convert_tracklets_to_detections(self, refined_tracklets: Dict, original_detections: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Convert refined tracklets dictionary to detections DataFrame format.
        
        Args:
            refined_tracklets (Dict): Dictionary of refined Tracklet objects
            original_detections (pd.DataFrame): Original detections for reference structure
            metadatas (pd.DataFrame): Metadata for image/frame information
            
        Returns:
            pd.DataFrame: Refined tracklets converted to detections format
        """
        
        refined_detections_list = []
        
        try:
            # Create a mapping from frame to image_id for metadata lookup
            frame_to_image_id = {}
            if not metadatas.empty and 'frame' in metadatas.columns and 'id' in metadatas.columns:
                frame_to_image_id = dict(zip(metadatas['frame'], metadatas['id']))
            
            # Extract all detection data from refined tracklets
            for track_id, tracklet in refined_tracklets.items():
                if hasattr(tracklet, 'times') and hasattr(tracklet, 'bboxes'):
                    for i, (time, bbox) in enumerate(zip(tracklet.times, tracklet.bboxes)):
                        # Create detection row
                        detection_row = {
                            'track_id': track_id,
                            'track_id_refined': track_id,  # For refined tracklets, these are the same
                            'bbox_ltwh': bbox,
                            'image_id': frame_to_image_id.get(time, time),  # Use metadata mapping or default to time
                            'bbox_conf': 0.9,  # Default confidence for refined tracklets
                            'video_id': 0,  # Default video ID
                            'category_id': 1,  # Default category (person)
                        }
                        
                        # Add feature embeddings if available
                        if hasattr(tracklet, 'features') and tracklet.features and i < len(tracklet.features):
                            detection_row['embeddings'] = tracklet.features[i]
                        else:
                            detection_row['embeddings'] = None
                        
                        # Copy common columns from original detections (for compatibility)
                        for col in original_detections.columns:
                            if col not in detection_row:
                                # Set default values for missing columns
                                if col in ['body_masks', 'role_confidence', 'role_detection', 'visibility_scores']:
                                    detection_row[col] = None
                                elif col in ['age', 'costs', 'hints', 'matched_with', 'state', 'time_since_update']:
                                    detection_row[col] = 0
                                elif col in ['track_bbox_kf_ltwh', 'track_bbox_pred_kf_ltwh']:
                                    detection_row[col] = bbox  # Use tracklet bbox as default
                                elif col == 'tracklets_dict':
                                    detection_row[col] = None  # Will be set for first row only
                                elif col == 'refined_tracklets_dict':
                                    detection_row[col] = None  # Will be set for first row only
                                else:
                                    detection_row[col] = None
                        
                        refined_detections_list.append(detection_row)
            
            # Create DataFrame from detections list
            if refined_detections_list:
                refined_detections = pd.DataFrame(refined_detections_list)
                
                # Set refined_tracklets_dict in the first row only (following original pattern)
                if not refined_detections.empty:
                    first_idx = refined_detections.index[0]
                    refined_detections.at[first_idx, 'refined_tracklets_dict'] = refined_tracklets
                
                # Sort by image_id and track_id for consistency
                refined_detections = refined_detections.sort_values(['image_id', 'track_id']).reset_index(drop=True)
                
                log.debug(f"Converted {len(refined_tracklets)} tracklets to {len(refined_detections)} detection rows")
                return refined_detections
            else:
                log.warning("No detections generated from refined tracklets")
                # Return empty DataFrame with same structure as original
                empty_df = original_detections.iloc[:0].copy()  # Empty with same columns
                return empty_df
                
        except Exception as e:
            log.error(f"Error in tracklet to detections conversion: {e}")
            # Fallback: return original detections
            return original_detections.copy()
    
    def save_refined_tracklets(self, refined_tracklets: Dict, output_path: str):
        """
        Save refined tracklets dictionary to pickle file.
        
        Args:
            refined_tracklets: Dictionary of refined Tracklet objects
            output_path: Path to save the pickle file
        """
        try:
            import os
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'wb') as f:
                pickle.dump(refined_tracklets, f)
            log.info(f"Refined tracklets saved to {output_path}")
        except Exception as e:
            log.error(f"Failed to save refined tracklets to {output_path}: {e}")


class TrackletVisualizationEngine(VideoLevelModule):
    """
    Additional module for visualizing refined tracklets.
    This can be used as a final step to visualize the results.
    """
    
    input_columns = ["refined_tracklets_dict", "track_id_refined"]
    output_columns = ["visualization_ready"]
    
    def __init__(self, **kwargs):
        super().__init__()
        
    def process(self, detections: pd.DataFrame, metadatas: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare refined tracklets for visualization.
        
        This is a placeholder for visualization preparation logic.
        You can extend this to create visualization-ready data structures.
        """
        
        detections['visualization_ready'] = True
        
        if 'refined_tracklets_dict' in detections.columns:
            refined_tracklets = None
            for idx, row in detections.iterrows():
                if row['refined_tracklets_dict'] is not None:
                    refined_tracklets = row['refined_tracklets_dict']
                    break
            
            if refined_tracklets:
                log.info(f"Prepared {len(refined_tracklets)} refined tracklets for visualization")
        
        return detections


def create_refine_tracklets_module(
    use_split: bool = True,
    eps: float = 0.5,
    min_samples: int = 3,
    max_k: int = 10,
    min_len: int = 30,
    merge_dist_thres: float = 0.3,
    spatial_factor: float = 1.0,
    batch_size: int = 50,
    mapping_strategy: str = "similarity",
    return_refined_detections: bool = False
) -> RefineTracklets:
    """
    Factory function to create RefineTracklets module.
    
    Args:
        use_split: Whether to use tracklet splitting
        eps: DBSCAN epsilon parameter
        min_samples: DBSCAN minimum samples parameter
        max_k: Maximum clusters for splitting
        min_len: Minimum tracklet length
        merge_dist_thres: Merging distance threshold
        spatial_factor: Spatial constraint factor
        batch_size: Processing batch size
        mapping_strategy: Track ID mapping strategy - "similarity" or "sequential"
        return_refined_detections: If True, return refined tracklets as detections DataFrame
        
    Returns:
        RefineTracklets: Configured module instance
    """
    return RefineTracklets(
        use_split=use_split,
        eps=eps,
        min_samples=min_samples,
        max_k=max_k,
        min_len=min_len,
        merge_dist_thres=merge_dist_thres,
        spatial_factor=spatial_factor,
        batch_size=batch_size,
        mapping_strategy=mapping_strategy,
        return_refined_detections=return_refined_detections
    )
