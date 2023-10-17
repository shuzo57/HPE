import cv2
import matplotlib.pyplot as plt
import numpy as np

from .config import CONNECTIONS_17, KEYPOINTS_17


def plot_keypoints(
    img: np.ndarray,
    keypoints_data: np.ndarray,
    point_color: tuple = (0, 255, 0),
    point_size: int = 5,
    line_color: tuple = (255, 0, 0),
    linewidth: int = 2,
) -> np.ndarray:
    for start, end in CONNECTIONS_17.values():
        kp1 = keypoints_data[KEYPOINTS_17.index(start)]
        kp2 = keypoints_data[KEYPOINTS_17.index(end)]
        cv2.line(
            img,
            (int(kp1[0]), int(kp1[1])),
            (int(kp2[0]), int(kp2[1])),
            line_color,
            linewidth,
        )

    for x, y in keypoints_data:
        cv2.circle(img, (int(x), int(y)), point_size, point_color, -1)

    return img


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
