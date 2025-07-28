import logging
from typing import Any, Optional

import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from tracklab.callbacks import Callback

from sn_gamestate.utils.save_tracklets_MOT import save_tracklab_df_to_mot
import os

log = logging.getLogger(__name__)


class ResultsSaver(Callback):

    def on_video_loop_end(
        self,
        engine: "TrackingEngine",
        video_metadata: pd.Series,
        video_idx: int,
        detections: pd.DataFrame,
        image_pred: pd.DataFrame,
    ):
        # Get the video file name from video_metadata
        video_path = video_metadata.iloc[video_idx]["name"]
        dir_name, file_name = os.path.split(video_path)
        base_name, _ = os.path.splitext(file_name)
        # Create MOT output directory
        output_path = os.path.join(dir_name,  "outputs")
        save_tracklab_df_to_mot(detections, video_idx, output_path)
        log.info(f"Saved MOT results to {output_path}")