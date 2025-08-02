# GTA Link Tracklab Integration

This directory contains tracklab-compatible wrappers for the GTA Link functionality, providing advanced tracklet generation and refinement capabilities.

## Overview

The GTA Link integration consists of three main components:

1. **GenerateTracklets** - Generates tracklet embeddings from tracking detections
2. **RefineTracklets** - Refines and merges tracklets using advanced algorithms  
3. **TrackletVisualizationEngine** - Prepares refined tracklets for visualization

## Modules

### GenerateTracklets

**Purpose**: Creates tracklet embeddings from tracking detections using a feature extractor model.

**Input**: 
- Tracking detections DataFrame with columns: `track_id`, `bbox_ltwh`, `bbox_conf`, `image_id`
- Frame metadata with image file paths

**Output**:
- Modified detections DataFrame with `tracklets_dict` column containing Tracklet objects

**Configuration**:
```yaml
generate_tracklets:
  _target_: sn_gamestate.wrappers.gta_link.generate_tracklets_api.GenerateTracklets
  model_path: ${model_dir}/gta_link/feature_extractor_osnet_x1_0.onnx
  device: auto  # 'cuda', 'cpu', or 'auto'
```

### RefineTracklets

**Purpose**: Refines and merges tracklets using splitting and merging algorithms to improve tracking consistency.

**Input**:
- Detections DataFrame with `tracklets_dict` column

**Output**:
- Modified detections with `refined_tracklets_dict` and `track_id_refined` columns

**Configuration**:
```yaml
refine_tracklets:
  _target_: sn_gamestate.wrappers.gta_link.refine_tracklets_api.RefineTracklets
  use_split: true           # Enable tracklet splitting
  eps: 0.5                  # DBSCAN epsilon for splitting
  min_samples: 3            # DBSCAN minimum samples
  max_k: 10                 # Maximum clusters for splitting
  min_len: 30               # Minimum tracklet length
  merge_dist_thres: 0.3     # Distance threshold for merging
  spatial_factor: 1.0       # Spatial constraint factor
  batch_size: 50            # Processing batch size
```

### TrackletVisualizationEngine

**Purpose**: Prepares refined tracklets for visualization (extensible for custom visualization needs).

**Configuration**:
```yaml
tracklet_vis:
  _target_: sn_gamestate.wrappers.gta_link.refine_tracklets_api.TrackletVisualizationEngine
```

## Usage Example

### 1. Complete Pipeline Configuration

Create a configuration file (e.g., `my_gta_pipeline.yaml`):

```yaml
modules:
  # Standard detection and tracking
  detector:
    _target_: sn_gamestate.wrappers.bbox_detector.yolo_ultralytics_api.YOLOUltralytics
    model_path: ${model_dir}/yolo/yolo11m.pt
    cfg:
      min_confidence: 0.4
      device: ${device}

  reid_tracker:
    _target_: tracklab.wrappers.PRTReIDStrongSORT
    cfg:
      model_path: ${model_dir}/reid/prtreid-soccernet-baseline.onnx
      device: ${device}

  # GTA Link modules
  generate_tracklets:
    _target_: sn_gamestate.wrappers.gta_link.generate_tracklets_api.GenerateTracklets
    model_path: ${model_dir}/gta_link/feature_extractor_osnet_x1_0.onnx
    device: ${device}

  refine_tracklets:
    _target_: sn_gamestate.wrappers.gta_link.refine_tracklets_api.RefineTracklets
    use_split: true
    eps: 0.5
    merge_dist_thres: 0.3

  tracklet_vis:
    _target_: sn_gamestate.wrappers.gta_link.refine_tracklets_api.TrackletVisualizationEngine
```

### 2. Run the Pipeline

```bash
python main.py --config-name=my_gta_pipeline
```

### 3. Programmatic Usage

```python
from sn_gamestate.wrappers.gta_link.generate_tracklets_api import GenerateTracklets
from sn_gamestate.wrappers.gta_link.refine_tracklets_api import RefineTracklets

# Initialize modules
generate_module = GenerateTracklets(
    model_path="path/to/feature_extractor.onnx",
    device="cuda"
)

refine_module = RefineTracklets(
    use_split=True,
    eps=0.5,
    merge_dist_thres=0.3
)

# Process detections
detections_with_tracklets = generate_module.process(detections, metadatas)
refined_detections = refine_module.process(detections_with_tracklets, metadatas)
```

## Data Flow

The typical data flow through the GTA Link modules:

```
Input Detections (with track_id, bbox_ltwh, etc.)
        ↓
GenerateTracklets
    • Extracts features from detection crops
    • Creates Tracklet objects with embeddings
    • Adds 'tracklets_dict' column
        ↓
RefineTracklets  
    • Splits tracklets based on embedding clusters
    • Merges similar tracklets using distance thresholds
    • Updates track IDs to refined versions
    • Adds 'refined_tracklets_dict' and 'track_id_refined'
        ↓
TrackletVisualizationEngine
    • Prepares data for visualization
    • Can be extended for custom visualization needs
        ↓
Final Output (refined tracking results)
```

## Requirements

### Model Files

You need the following model files:

1. **Feature Extractor**: `feature_extractor_osnet_x1_0.onnx`
   - Place in: `pretrained_models/gta_link/`
   - Used by GenerateTracklets for embedding extraction

### Dependencies

The modules use functions from:
- `sn_gamestate.gta_link.generate_tracklets`
- `sn_gamestate.gta_link.refine_tracklets`
- `sn_gamestate.gta_link.refine_tracklets_batched`
- `sn_gamestate.gta_link.Tracklet`
- `sn_gamestate.gta_link.utils.feature_extractor_onnx`

## Testing

Run the test script to verify the modules work correctly:

```bash
python test_gta_link_modules.py
```

This will run tests with synthetic data to verify the module structure and basic functionality.

## Configuration Parameters

### GenerateTracklets Parameters

- `model_path`: Path to the ONNX feature extractor model
- `device`: Processing device ('cuda', 'cpu', or 'auto')

### RefineTracklets Parameters

- `use_split`: Enable/disable tracklet splitting (default: true)
- `eps`: DBSCAN epsilon parameter for splitting (default: 0.5)
- `min_samples`: DBSCAN minimum samples (default: 3)
- `max_k`: Maximum clusters for splitting (default: 10)
- `min_len`: Minimum tracklet length threshold (default: 30)
- `merge_dist_thres`: Distance threshold for merging (default: 0.3)
- `spatial_factor`: Spatial constraint factor (default: 1.0)
- `batch_size`: Processing batch size (default: 50)

## Advanced Usage

### Custom Feature Extractors

You can extend GenerateTracklets to use different feature extractors:

```python
class CustomGenerateTracklets(GenerateTracklets):
    def __init__(self, custom_model_path, **kwargs):
        # Initialize with custom model
        super().__init__(model_path=custom_model_path, **kwargs)
        
    def extract_features(self, crops):
        # Custom feature extraction logic
        pass
```

### Custom Refinement Algorithms

You can extend RefineTracklets with custom refinement logic:

```python
class CustomRefineTracklets(RefineTracklets):
    def _split_tracklets(self, tracklets_dict):
        # Custom splitting logic
        pass
        
    def _merge_tracklets(self, tracklets_dict, max_x_range, max_y_range):
        # Custom merging logic  
        pass
```

## Troubleshooting

### Common Issues

1. **Missing model file**: Ensure `feature_extractor_osnet_x1_0.onnx` is in the correct path
2. **CUDA out of memory**: Reduce batch size or use CPU processing
3. **Import errors**: Check that all dependencies are installed
4. **Empty tracklets**: Verify input detections have valid track_id values

### Debug Mode

Enable detailed logging:

```python
import logging
logging.getLogger('sn_gamestate.wrappers.gta_link').setLevel(logging.DEBUG)
```

## Performance Notes

- **GenerateTracklets**: Processing time depends on number of detections and feature extraction model
- **RefineTracklets**: Processing time depends on number of tracklets and complexity of splitting/merging
- **Memory usage**: Large videos may require batch processing to avoid memory issues

## Future Extensions

Potential enhancements:

1. **Multi-camera support**: Extend for multi-camera tracklet association
2. **Online processing**: Adapt for real-time/streaming video processing  
3. **Custom distance metrics**: Add support for different tracklet similarity measures
4. **Advanced visualization**: Enhanced visualization and analysis tools
5. **Export formats**: Support for different output formats (MOT, COCO, etc.)
