import os
from pathlib import Path
import cv2
import logging
import pandas as pd

from tracklab.visualization.visualization_engine import VisualizationEngine
from tracklab.callbacks import Progressbar
from tracklab.utils.cv2 import cv2_load_image


from sn_gamestate.utils.frame_extractor import frame_generator  # Make sure this path is correct

log = logging.getLogger(__name__)


class VisualizationEngineCustom(VisualizationEngine):
    def __init__(self, *args, **kwargs):
        # Just pass all arguments to the parent VisualizationEngine
        super().__init__(*args, **kwargs)

    def on_video_loop_end(self, engine, video_metadata, video_idx, detections, image_pred):
        """
        Visualize only the frames that were processed, using a generator to extract them efficiently.
        """
        if not (self.save_videos or self.save_images):
            return

        progress = engine.callbacks.get("progress", Progressbar(dummy=True))
        tracker_state = engine.tracker_state

        # Get all processed frame IDs for this video
        # processed_ids = list(detections["image_id"].unique())
        # video_path = video_metadata.iloc[video_idx]["name"]
        # video_width = video_metadata.iloc[video_idx]["width"]
        # video_height = video_metadata.iloc[video_idx]["height"]
        video_path = video_metadata["name"]
        video_width = video_metadata["width"]
        video_height = video_metadata["height"]
        video_dir, video_name = os.path.split(video_path)
        save_dir = Path(video_dir) / "outputs"

        # Prepare video writer if needed
        video_writer = None
        if self.save_videos:
            filepath = save_dir / "videos_res" / f"{video_name}.mp4"
            filepath.parent.mkdir(parents=True, exist_ok=True)
            video_writer = cv2.VideoWriter(
                str(filepath),
                cv2.VideoWriter_fourcc(*"mp4v"),
                float(self.video_fps),
                (video_width, video_height),
            )

        progress.init_progress_bar("vis", "Visualization", len(image_pred.values))

        # Use the frame_generator to yield frames by processed_ids
        image_global_id = 0
        mot_annotations = []
        for idx, image_pred_row in image_pred.iterrows():# frame_generator(video_path, processed_ids):
            # Prepare detection and prediction data for this frame
            image_id = image_pred_row['id']
            detections_pred = detections[detections.image_id == image_id] if len(detections) else None
            # image_pred_row = image_pred.loc[image_id] if image_pred is not None and image_id in image_pred.index else None
            frame = cv2_load_image(image_pred_row['file_path'])  # Use cv2_load_image to load the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert to RGB if needed
            # Save original image if required
            if self.save_images:
                filepath = save_dir / f"seq_{video_idx}" / "img1" / f"{image_global_id:06d}.jpg"               
                filepath.parent.mkdir(parents=True, exist_ok=True)
                assert cv2.imwrite(str(filepath), frame)
            # Prepare MOT annotations for this frame
                if detections_pred is not None and not detections_pred.empty:
                    if 'track_id' not in detections_pred.columns:
                        detections_pred['track_id'] = -1
                    mot_annotations.extend(detections_pred[['track_id_refined', 'bbox_ltwh', 'bbox_conf']].apply(
                        lambda x: (
                            f"{image_global_id},{int(x['track_id_refined'])},{x['bbox_ltwh'][0]:.2f},"
                            f"{x['bbox_ltwh'][1]:.2f},{x['bbox_ltwh'][2]:.2f},{x['bbox_ltwh'][3]:.2f},"
                            f"{x['bbox_conf']:.5f},-1,-1,-1\n"
                        ),
                        axis=1
                    ).tolist())
                image_global_id += 1
   
            # Draw frame using visualizers
            for visualizer in self.visualizers.values():
                try:
                    visualizer.draw_frame(frame, detections_pred, pd.DataFrame([]), image_pred_row, pd.DataFrame([]))
                except Exception as e:
                    log.warning(f"Visualizer {visualizer} raised error : {e} during drawing.")

            # Write to video if required
            if self.save_videos and video_writer is not None:
                video_writer.write(frame)

            progress.on_module_step_end(None, "vis", None, None)

        # Save all tracklet records into a single file named seq_{video_idx}.txt
        mot_annotation_path = save_dir / f"seq_{video_idx}.txt"
        with open(mot_annotation_path, 'w') as f:
            for row in mot_annotations:
                f.write(row)

        if video_writer is not None:
            video_writer.release()
        progress.on_module_end(None, "vis", None)