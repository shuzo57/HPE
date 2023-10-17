from typing import Union

import cv2
import numpy as np
from mmdet.apis import inference_detector
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.structures.pose_data_sample import PoseDataSample

from .utils import process_mmdet_results


class PoseEstimator:
    def __init__(self, pose_estimator, detecor):
        self.pose_estimator = pose_estimator
        self.detector = detecor

    def predict(self, img: Union[str, np.ndarray]) -> list:
        if isinstance(img, str):
            img = self._read_img(img)

        # object detection
        scope = self.detector.cfg.get("default_scope", "mmdet")
        if scope is not None:
            init_default_scope(scope)
        mmdet_results = inference_detector(self.detector, img)

        # extract person information from detection results
        person_boxes = process_mmdet_results(mmdet_results)

        # pose estimation
        scope = self.pose_estimator.cfg.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)
        mmpose_results = inference_topdown(
            self.pose_estimator, img, person_boxes
        )

        return mmpose_results

    def get_keypoints(self, mmpose_result: PoseDataSample) -> np.ndarray:
        return mmpose_result.pred_instances.keypoints[0]

    def get_visibles(self, mmpose_result: PoseDataSample) -> np.ndarray:
        return mmpose_result.pred_instances.visibles[0]

    def _read_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
