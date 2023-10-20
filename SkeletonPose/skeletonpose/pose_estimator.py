from enum import Enum
from typing import Union

import cv2
import numpy as np
from mmdet.apis import inference_detector
from mmengine.registry import init_default_scope
from segment_anything import SamPredictor

from mmpose.apis import inference_topdown
from mmpose.structures.pose_data_sample import PoseDataSample

from .config import COCO_CATEGORY
from .utils import get_id_by_name, process_mmdet_results


class BackgroundMethod(Enum):
    RAW = 1
    SHUFFLE_BACKGROUND = 2
    SHUFFLED_PIXEL = 3
    MEAN_COLOR_BACKGROUND = 4


class PoseEstimator:
    def __init__(
        self,
        pose_estimator,
        detecor,
        sam=None,
        class_id: Union[str, int] = 0,
        class_threshold: float = 0.50,
    ):
        self.pose_estimator = pose_estimator
        self.detector = detecor
        self.sam_predicter = SamPredictor(sam) if sam is not None else None
        self.class_id = (
            class_id
            if isinstance(class_id, int)
            else get_id_by_name(class_id, COCO_CATEGORY)
        )
        self.class_threshold = class_threshold
        self.background_method = BackgroundMethod.RAW
        self.processed_img = None

    def predict(
        self,
        img: Union[str, np.ndarray],
        method: BackgroundMethod = BackgroundMethod.RAW,
        kernel_size: int = 5,
    ) -> list:
        if isinstance(img, str):
            img = self._read_img(img)

        # object detection
        scope = self.detector.cfg.get("default_scope", "mmdet")
        if scope is not None:
            init_default_scope(scope)
        mmdet_results = inference_detector(self.detector, img)

        # extract class information from detection results
        class_boxes = process_mmdet_results(
            mmdet_results,
            class_id=self.class_id,
            class_threshold=self.class_threshold,
        )

        # use sam to segment person
        if method != BackgroundMethod.RAW and self.sam_predicter is not None:
            self.sam_predicter.set_image(img)

            masks = []
            for class_box in class_boxes:
                mask, _, _ = self.sam_predicter.predict(
                    point_coords=None,
                    point_labels=None,
                    box=class_box[None, :],
                    multimask_output=False,
                )
                masks.append(mask[0])
            mask = np.logical_or.reduce(masks)
            dilated_mask = self._dilate_mask(mask, kernel_size=kernel_size)
            img = self.apply_background_method(img, dilated_mask, method)
            self.processed_img = img  # for visualization
        else:
            self.processed_img = None  # reset processed_img

        # pose estimation
        scope = self.pose_estimator.cfg.get("default_scope", "mmpose")
        if scope is not None:
            init_default_scope(scope)
        mmpose_results = inference_topdown(
            self.pose_estimator, img, class_boxes
        )
        return mmpose_results

    def get_keypoints(self, mmpose_result: PoseDataSample) -> np.ndarray:
        return mmpose_result.pred_instances.keypoints[0]

    def get_keypoint_scores(self, mmpose_result: PoseDataSample) -> np.ndarray:
        return mmpose_result.pred_instances.keypoint_scores[0]

    def get_bbox(self, mmpose_result: PoseDataSample) -> np.ndarray:
        return mmpose_result.pred_instances.bboxes[0]

    def get_img_shape(self, mmpose_result: PoseDataSample) -> tuple:
        return mmpose_result.img_shape

    def get_keypoints_xyn(self, mmpose_result: PoseDataSample) -> np.ndarray:
        keypoints = self.get_keypoints(mmpose_result)
        img_width, img_height = self.get_img_shape(mmpose_result)
        keypoints_xyn = self._convert_to_xywhn(
            keypoints, img_width, img_height
        )
        return keypoints_xyn

    def get_processed_img(self) -> np.ndarray:
        return self.processed_img

    def apply_background_method(
        self, img: np.ndarray, mask: np.ndarray, method: BackgroundMethod
    ) -> np.ndarray:
        if method == BackgroundMethod.RAW:
            return img
        elif method == BackgroundMethod.SHUFFLE_BACKGROUND:
            return self._shuffle_background(img, mask)
        elif method == BackgroundMethod.SHUFFLED_PIXEL:
            return self._shuffled_pixel(img, mask)
        elif method == BackgroundMethod.MEAN_COLOR_BACKGROUND:
            return self._mean_color_background(img, mask)
        else:
            raise ValueError("Invalid background method")

    def _read_img(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _convert_to_xywhn(self, keypoints, img_width, img_height):
        normalized_keypoints = keypoints.copy()
        normalized_keypoints[:, 0] = keypoints[:, 0] / img_width
        normalized_keypoints[:, 1] = keypoints[:, 1] / img_height
        return normalized_keypoints

    def _dilate_mask(
        self, mask: np.ndarray, kernel_size: int = 5
    ) -> np.ndarray:
        if mask.ndim != 2:
            raise ValueError("mask must be 2 dimension")
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        dilated_mask = cv2.dilate(mask.astype(np.uint8), kernel)
        return dilated_mask.astype(bool)

    def _shuffle_background(
        self, img: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        background_img = img.copy()
        shuffled_img = background_img.reshape(-1, 3)
        np.random.shuffle(shuffled_img)
        background_img_shuffled = shuffled_img.reshape(img.shape)
        return np.where(mask[:, :, None], img, background_img_shuffled)

    def _shuffled_pixel(self, img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return_img = img.copy()
        background_img = img.copy()
        for i in range(background_img.shape[0]):
            for j in range(background_img.shape[1]):
                if not mask[i, j]:
                    np.random.shuffle(background_img[i, j])
        return_img[~mask] = background_img[~mask]
        return return_img

    def _mean_color_background(
        self, img: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        mean_color = img.mean(axis=(0, 1)).astype(np.uint8)
        return np.where(mask[:, :, None], img, mean_color)
