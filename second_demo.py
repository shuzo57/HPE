import warnings

import cv2
import torch
from mmdet.apis import inference_detector, init_detector
from mmdet.visualization import DetLocalVisualizer
from mmengine.registry import init_default_scope

from config import DET_CHECKPOINT, DET_CONFIG, POSE_CHECKPOINT, POSE_CONFIG
from mmpose.apis import inference_topdown, init_model
from mmpose.structures import merge_data_samples
from mmpose.visualization import PoseLocalVisualizer
from utils import process_mmdet_results

warnings.filterwarnings("ignore")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device:", device)

pose_model = init_model(POSE_CONFIG, POSE_CHECKPOINT, device=device)
det_model = init_detector(DET_CONFIG, DET_CHECKPOINT, device=device)

img_path = "examples/img1.jpg"
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# object detection
scope = det_model.cfg.get("default_scope", "mmdet")
if scope is not None:
    init_default_scope(scope)
mmdet_results = inference_detector(det_model, img_path)

# visualization
det_vis = DetLocalVisualizer.get_instance(
    name="mmdet_tutorial",
    vis_backends=[dict(type="LocalVisBackend")],
    save_dir="./examples",
)
det_vis.add_datasample(
    name="det_img1",
    image=img,
    data_sample=mmdet_results,
)

# extract person information from detection results
person_results = process_mmdet_results(mmdet_results)

# pose estimation
scope = pose_model.cfg.get("default_scope", "mmpose")
if scope is not None:
    init_default_scope(scope)
mmpose_results = inference_topdown(pose_model, img_path, person_results)
data_samples = merge_data_samples(mmpose_results)
"""
Please chcek the output of warning:
mmengine - WARNING - The current default scope "mmdet" is not "mmpose",
`init_default_scope` will force set the currentdefault scope to "mmpose".
"""

pose_vis = PoseLocalVisualizer.get_instance(
    name="mmpose_tutorial",
    vis_backends=[dict(type="LocalVisBackend")],
    save_dir="./examples",
)
pose_vis.add_datasample(
    name="pose_img1",
    image=img,
    data_sample=mmpose_results[0],
)

# URL:
# https://github.com/open-mmlab/mmpose/blob/main/demo/MMPose_Tutorial.ipynb
