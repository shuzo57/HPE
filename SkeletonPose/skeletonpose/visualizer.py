import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import CONNECTIONS_17, KEYPOINTS_17


def plot_keypoints(
    img: np.ndarray,
    keypoints: np.ndarray = None,
    bbox: np.ndarray = None,
    point_color: tuple = (128, 128, 128),
    left_point_color: tuple = (0, 165, 255),
    right_point_color: tuple = (255, 255, 0),
    point_size: int = 5,
    line_color: tuple = (255, 0, 0),
    linewidth: int = 2,
    box_color: tuple = (0, 0, 255),
) -> np.ndarray:
    return_img = img.copy()
    if bbox is not None:
        cv2.rectangle(
            return_img,
            (int(bbox[0]), int(bbox[1])),
            (int(bbox[2]), int(bbox[3])),
            box_color,
            linewidth,
        )

    if keypoints is not None:
        for start, end in CONNECTIONS_17.values():
            kp1 = keypoints[KEYPOINTS_17.index(start)]
            kp2 = keypoints[KEYPOINTS_17.index(end)]
            cv2.line(
                return_img,
                (int(kp1[0]), int(kp1[1])),
                (int(kp2[0]), int(kp2[1])),
                line_color,
                linewidth,
            )

        for name, (x, y) in zip(KEYPOINTS_17, keypoints):
            if name.startswith("LEFT"):
                cv2.circle(
                    return_img,
                    (int(x), int(y)),
                    point_size,
                    left_point_color,
                    -1,
                )
            elif name.startswith("RIGHT"):
                cv2.circle(
                    return_img,
                    (int(x), int(y)),
                    point_size,
                    right_point_color,
                    -1,
                )
            else:
                cv2.circle(
                    return_img, (int(x), int(y)), point_size, point_color, -1
                )

    return return_img


def img_show(
    img: np.ndarray,
    figure_size: tuple = (8, 8),
    rgb: bool = False,
    save_path: str = "",
) -> None:
    """
    Displays the given image using matplotlib.

    Parameters
    ----------
    img : np.ndarray
        Image array.
    figure_size : tuple, optional
        Size of the displayed image (default is (6, 6)).
    rgb : bool, optional
        RGB order option (default is False).

    Returns
    -------
    None
    """
    if rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    _, ax = plt.subplots(figsize=figure_size)
    ax.imshow(img)
    ax.axis("off")
    if save_path != "":
        plt.savefig(save_path)
    plt.show()
