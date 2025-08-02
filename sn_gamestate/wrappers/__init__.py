# Import modules with error handling to avoid breaking the main package
try:
    from .dataset.external_video import ExternalVideo
except ImportError:
    pass  # ExternalVideo requires cv2 which may not be available

# Import GTA Link modules
try:
    from .gta_link import GenerateTracklets, RefineTracklets, TrackletVisualizationEngine
except ImportError:
    pass  # GTA Link modules may have dependency issues