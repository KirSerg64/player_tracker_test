import logging
from typing import Any
from xml.parsers.expat import model

import torch
import pandas as pd
import supervision as sv


from ultralytics import YOLO

from tracklab.pipeline.imagelevel_module import ImageLevelModule
from tracklab.utils.coordinates import ltrb_to_ltwh

log = logging.getLogger(__name__)


def collate_fn(batch):
    idxs = [b[0] for b in batch]
    images = [b["image"] for _, b in batch]
    shapes = [b["shape"] for _, b in batch]
    return idxs, (images, shapes)


class YOLOOnnx(ImageLevelModule):
    collate_fn = collate_fn
    input_columns = []
    output_columns = [
        "image_id",
        "video_id",
        "category_id",
        "bbox_ltwh",
        "bbox_conf",
    ]

    def __init__(self, cfg, device, batch_size, **kwargs):
        super().__init__(batch_size)
        self.cfg = cfg
        self.device = device
        self.model = YOLO(cfg.path_to_checkpoint)
        # self.model.to(device)
        self.id = 0
        self.class_map = {cls: id for id, cls in enumerate(cfg.classes)}
        self.classes_to_detect = tuple([
            self.class_map['player'], 
            self.class_map['goalkeeper'], 
            self.class_map['referee'],
        ])
        self.use_slicer = cfg.get("use_slicer", False)
        self.slicer = sv.InferenceSlicer(
            callback=self.callback, thread_workers=4,
        )

    # @torch.no_grad()
    def callback(self, image_slice) -> sv.Detections:
        results = self.model.predict(image_slice, device=self.device)[0]
        return sv.Detections.from_ultralytics(results)

    @torch.no_grad()
    def preprocess(self, image, detections, metadata: pd.Series):
        return {
            "image": image,
            "shape": (image.shape[1], image.shape[0]),
        }

    @torch.no_grad()
    def process(self, batch: Any, detections: pd.DataFrame, metadatas: pd.DataFrame):
        images, shapes = batch
        if self.use_slicer:
            results_by_image = []
            for img in images:
                sliced_results = self.slicer(img)
                results_by_image.append(sliced_results)
            # results_by_image = sv.Detections.merge(results_by_image)
        else:
            # with torch.no_grad(): 
            #     results_by_image = self.model(images, verbose=False)
            # <class 'ultralytics.engine.results.Results'>
            results_by_image = self.model.predict(images, device=self.device)
            results_by_image = [sv.Detections.from_ultralytics(res) for res in results_by_image]
        detections = []
        for results, shape, (_, metadata) in zip(
            results_by_image, shapes, metadatas.iterrows()
        ):
            for xyxy, _, conf, class_id, _, _ in results:
                # check for `player` class
                if conf >= self.cfg.min_confidence and class_id in self.classes_to_detect:
                    detections.append(
                        pd.Series(
                            dict(
                                image_id=metadata.name,
                                bbox_ltwh=ltrb_to_ltwh(xyxy, shape),
                                bbox_conf=conf,
                                video_id=metadata.video_id,
                                category_id=1,  # `person` class in posetrack
                            ),
                            name=self.id,
                        )
                    )
                    self.id += 1
        return detections