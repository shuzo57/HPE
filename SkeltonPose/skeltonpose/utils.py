import numpy as np
from mmdet.structures.det_data_sample import DetDataSample


def process_mmdet_results(
    mmdet_results: DetDataSample,
    class_id: int = 0,
    class_threshold: float = 0.65,
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
