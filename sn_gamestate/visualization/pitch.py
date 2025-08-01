import cv2
import numpy as np
from pathlib import Path

from tracklab.utils.cv2 import draw_text
from tracklab.visualization import ImageVisualizer

# from sn_calibration_baseline.soccerpitch import SoccerPitch

import logging

log = logging.getLogger(__name__)

pitch_file = Path(__file__).parent / "Radar.png"


class Pitch(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        draw_pitch(image, detections_pred, detections_gt, image_pred)

class Radar(ImageVisualizer):
    def draw_frame(self, image, detections_pred, detections_gt, image_pred, image_gt):
        for detection, group in zip([detections_pred, detections_gt], ["Predictions", "Ground Truth"]):
            if detection is not None and "bbox_pitch" in detection:
                draw_radar_view(image, detection, group=group)

def draw_pitch(
    patch,
    detections_pred,
    detections_gt,
    image_pred,
    line_thickness=3,
):
    # Draw the lines on the image pitch
    if "lines" in image_pred:
        image_height, image_width, _ = patch.shape
        for name, line in image_pred["lines"].items():
            if name == "Circle central" and len(line) > 4:
                points = np.array([(int(p["x"] * image_width), int(p["y"]*image_height)) for p in line])
                ellipse = cv2.fitEllipse(points)
                # cv2.ellipse(patch, ellipse, color=SoccerPitch.palette[name], thickness=line_thickness)
            # else:
            #     for j in np.arange(len(line)-1):
            #         cv2.line(
            #             patch,
            #             (int(line[j]["x"] * image_width), int(line[j]["y"] * image_height)),
            #             (int(line[j+1]["x"] * image_width), int(line[j+1]["y"] * image_height)),
            #             color=SoccerPitch.palette[name],
            #             thickness=line_thickness,  # TODO : make this a parameter
            #         )

def draw_radar_view(patch, detections, scale=4, delta=32, group="Ground Truth"):
    pitch_width = 105 + 2 * 10  # pitch size + 2 * margin
    pitch_height = 68 + 2 * 5  # pitch size + 2 * margin
    sign = -1 if group == "Ground Truth" else +1
    y_delta = 3
    radar_center_x = int(1920/2 - pitch_width * scale / 2 * sign - delta * sign)
    radar_center_y = int(1080 - pitch_height * scale / 2 - y_delta)
    radar_top_x = int(radar_center_x - pitch_width * scale / 2)
    radar_top_y = int(1080 - pitch_height * scale - y_delta)
    radar_width = int(pitch_width * scale)
    radar_height = int(pitch_height * scale)
    if pitch_file is not None:
        radar_img = cv2.resize(cv2.imread(str(pitch_file)), (pitch_width * scale, pitch_height * scale))
        cv2.line(radar_img, (0, 0), (0, radar_img.shape[0]), thickness=6, color=(0, 0, 255))
        cv2.line(radar_img, (radar_img.shape[1], 0), (radar_img.shape[1], radar_img.shape[0]), thickness=6, color=(255, 0, 0))
    else:
        radar_img = np.ones((pitch_height * scale, pitch_width * scale, 3)) * 255

    alpha = 0.3
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,:] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width, :], 1-alpha, radar_img, alpha, 0.0)
    patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :] = cv2.addWeighted(patch[radar_top_y:radar_top_y + radar_height, radar_top_x:radar_top_x + radar_width,
    :], 1-alpha, radar_img, alpha, 0.0)
    draw_text(
        patch,
        group,
        (radar_center_x, radar_top_y - 5),
        0, 1, 1,
        color_txt=(255, 255, 255),
        color_bg=None,
        alignH="c",
        alignV="t",
    )
    for name, detection in detections.iterrows():
        if "role" in detection and detection.role == "ball":
            continue
        if "role" in detection and "team" in detection:
            color = (0, 0, 255) if detection.team == "left" else (255, 0, 0)
        else:
            color = (0, 0, 0)
        bbox_name = "bbox_pitch"
        if not isinstance(detection[bbox_name], dict):
            continue
        x_middle = np.clip(detection[bbox_name]["x_bottom_middle"], -10000, 10000)
        y_middle = np.clip(detection[bbox_name]["y_bottom_middle"], -10000, 10000)
        cat = None
        if "jersey_number" in detection and detection.jersey_number is not None:
            if "role" in detection and detection.role == "player":
                if isinstance(detection.jersey_number, float) and np.isnan(detection.jersey_number):
                    cat = None
                else:
                    cat = f"{int(detection.jersey_number)}"

        if "role" in detection:
            if detection.role == "goalkeeper":
                cat = "GK"
            elif detection.role == "referee":
                cat = "RE"
                color = (238, 210, 2)
            elif detection.role == "other":
                cat = "OT"
                color = (0, 255, 0)
        if cat is not None:
            draw_text(
                patch,
                cat,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                1,
                0.2*scale,
                1,
                color_txt=color,
                color_bg=None,
                alignH="c",
                alignV="b",
            )
        else:
            cv2.circle(
                patch,
                (radar_center_x + int(x_middle * scale),
                 radar_center_y + int(y_middle * scale)),
                scale,
                color=color,
                thickness=-1
            )
