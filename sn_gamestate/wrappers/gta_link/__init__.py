"""
GTA Link wrapper modules for tracklab integration

This module provides tracklab-compatible wrappers for the GTA link functionality:
- GenerateTracklets: Creates tracklet embeddings from tracking detections
- RefineTracklets: Refines and merges tracklets using advanced algorithms
"""

from .generate_tracklets_api import GenerateTracklets, create_generate_tracklets_module
from .refine_tracklets_api import RefineTracklets, TrackletVisualizationEngine, create_refine_tracklets_module

__all__ = [
    'GenerateTracklets',
    'RefineTracklets', 
    'TrackletVisualizationEngine',
    'create_generate_tracklets_module',
    'create_refine_tracklets_module'
]
