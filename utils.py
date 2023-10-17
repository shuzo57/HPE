import mmcv
import numpy as np
from mmdet.apis import inference_detector
from mmdet.structures.det_data_sample import DetDataSample
from mmengine.registry import init_default_scope

from mmpose.apis import inference_topdown
from mmpose.evaluation.functional import nms
from mmpose.structures import merge_data_samples


def process_mmdet_results(
    mmdet_results: DetDataSample,
    person_id: int = 0,
    person_threshold: float = 0.8,
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


def visualize_img(
    img_path, detector, pose_estimator, visualizer, show_interval, out_file
):
    """Visualize predicted keypoints (and heatmaps) of one image."""

    # predict bbox
    scope = detector.cfg.get("default_scope", "mmdet")
    if scope is not None:
        init_default_scope(scope)
    detect_result = inference_detector(detector, img_path)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate(
        (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1
    )
    bboxes = bboxes[
        np.logical_and(pred_instance.labels == 0, pred_instance.scores > 0.3)
    ]
    bboxes = bboxes[nms(bboxes, 0.3)][:, :4]

    # predict keypoints
    pose_results = inference_topdown(pose_estimator, img_path, bboxes)
    data_samples = merge_data_samples(pose_results)

    # show the results
    img = mmcv.imread(img_path, channel_order="rgb")

    visualizer.add_datasample(
        "result",
        img,
        data_sample=data_samples,
        draw_gt=False,
        draw_heatmap=True,
        draw_bbox=True,
        show=False,
        wait_time=show_interval,
        out_file=out_file,
        kpt_thr=0.3,
    )
