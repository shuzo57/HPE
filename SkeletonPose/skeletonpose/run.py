import os

import cv2
import pandas as pd
import torch
from mmdet.apis import init_detector
from skeletonpose import PoseEstimator

from mmpose.apis import init_model

from .config import CONFIDENCE_FILE, KEYPOINTS_17, POSITION_FILE
from .utils import set_dirctory_structure
from .visualizer import plot_keypoints


def Run(
    input_path: str,
    output_path: str,
    pose_config: str,
    pose_checkpoint: str,
    det_config: str,
    det_checkpoint: str,
):
    data_dir, img_dir = set_dirctory_structure(input_path, output_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    pose_model = init_model(pose_config, pose_checkpoint, device=device)
    det_model = init_detector(det_config, det_checkpoint, device=device)

    pe = PoseEstimator(pose_model, det_model)

    position_columns = [
        f"{keypoint}_{coord}"
        for keypoint in KEYPOINTS_17
        for coord in ["x", "y"]
    ]
    position_df = pd.DataFrame(columns=position_columns)
    confidence_df = pd.DataFrame(columns=KEYPOINTS_17)

    cap = cv2.VideoCapture(input_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"\r frame: {int(cap.get(cv2.CAP_PROP_POS_FRAMES))}", end="")

        mmpose_results = pe.predict(frame)
        keypoints = pe.get_keypoints(mmpose_results[0])
        bbox = pe.get_bbox(mmpose_results[0])
        # keypoints_xyn = pe.get_keypoints_xyn(mmpose_results[0])
        keypoints_scores = pe.get_keypoint_scores(mmpose_results[0])

        # position_df.loc[len(position_df)] = keypoints_xyn.reshape(-1)
        position_df.loc[len(position_df)] = keypoints.reshape(-1)
        confidence_df.loc[len(confidence_df)] = keypoints_scores

        drew_keypoints_img = plot_keypoints(frame, keypoints, bbox)
        cv2.imwrite(
            os.path.join(
                img_dir, f"{int(cap.get(cv2.CAP_PROP_POS_FRAMES))}.jpg"
            ),
            cv2.cvtColor(drew_keypoints_img, cv2.COLOR_RGB2BGR),
        )

    position_df.to_csv(os.path.join(data_dir, POSITION_FILE), index=False)
    confidence_df.to_csv(os.path.join(data_dir, CONFIDENCE_FILE), index=False)
    print("\nDone.")
