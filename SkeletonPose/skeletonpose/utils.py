import os

import numpy as np
from mmdet.structures.det_data_sample import DetDataSample

from .config import DATA_DIR, IMG_DIR


def process_mmdet_results(
    mmdet_results: DetDataSample,
    class_id: int = 0,
    class_threshold: float = 0.30,
) -> np.ndarray:
    pred_instances = mmdet_results.pred_instances

    class_results = []
    for box, label, score in zip(
        pred_instances.bboxes,
        pred_instances.labels,
        pred_instances.scores,
    ):
        if label == class_id and score > class_threshold:
            box = box.cpu().numpy()
            class_results.append(box)
        if score < class_threshold:
            break

    return np.array(class_results)


def set_dirctory_structure(input_path: str, output_path: str) -> tuple:
    input_name = get_input_name(input_path)
    data_dir = os.path.join(output_path, DATA_DIR, input_name)
    img_dir = os.path.join(output_path, IMG_DIR, input_name)

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    return data_dir, img_dir


def get_id_by_name(name: str, categories: dict) -> int:
    if name in categories:
        return categories[name]
    else:
        raise ValueError(f"Name {name} not found in categories dict.")


def get_input_name(input_path: str) -> str:
    return os.path.splitext(os.path.basename(input_path))[0]
