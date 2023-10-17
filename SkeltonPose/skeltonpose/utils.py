import numpy as np
from mmdet.structures.det_data_sample import DetDataSample


def process_mmdet_results(
    mmdet_results: DetDataSample,
    person_id: int = 0,
    person_threshold: float = 0.65,
) -> np.ndarray:
    pred_instances = mmdet_results.pred_instances

    person_results = []
    for box, label, score in zip(
        pred_instances.bboxes,
        pred_instances.labels,
        pred_instances.scores,
    ):
        if label == person_id and score > person_threshold:
            box = box.cpu().numpy()
            person_results.append(box)
        if score < person_threshold:
            break

    return np.array(person_results)
