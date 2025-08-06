from logging import warning
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import cv2

from mmdet3d.structures import Box3DMode, get_box_type, Det3DDataSample
from mmdet3d.visualization.vis_utils import proj_camera_bbox3d_to_img
                        
from torch.utils.data import Dataset

bbox_palettes = [
    (255, 158, 0),  # Orange
    (255, 99, 71),  # Tomato
    (255, 140, 0),  # Darkorange
    (255, 127, 80),  # Coral
    (233, 150, 70),  # Darksalmon
    (220, 20, 60),  # Crimson
    (255, 61, 99),  # Red
    (0, 0, 230),  # Blue
    (47, 79, 79),  # Darkslategrey
    (112, 128, 144),  # Slategrey
]

def draw_bboxes(
    image: np.ndarray,
    result: Det3DDataSample,
    alpha: Union[int, float] = 0.4,
    line_widths: Union[int, float, List[Union[int, float]]] = 2):
    """Set the image to draw.

    Args:
        image (np.ndarray): The image to draw.
    """
    assert image is not None
    image = image.astype('uint8')

    bboxes_3d = result.pred_instances_3d.bboxes_3d 
    labels_3d = result.pred_instances_3d.labels_3d
    input_meta = result.metainfo

    corners_2d = proj_camera_bbox3d_to_img(bboxes_3d, input_meta)

    # Color
    img_size = image.shape[:2][::-1]  # (width, height)
    edge_colors = [bbox_palettes[label][::-1] for label in labels_3d]
    for color in edge_colors:
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        color = [channel / 255 for channel in color]

    valid_point_idx = (corners_2d[..., 0] >= 0) & \
        (corners_2d[..., 0] <= img_size[0]) & \
        (corners_2d[..., 1] >= 0) & (corners_2d[..., 1] <= img_size[1])
    valid_bbox_idx = valid_point_idx.sum(axis=-1) >= 4
    corners_2d = corners_2d[valid_bbox_idx]
    filter_edge_colors = []
    filter_edge_colors_norm = []
    for i, color in enumerate(edge_colors):
        if valid_bbox_idx[i]:
            filter_edge_colors.append(color)
    edge_colors = filter_edge_colors

    overlay = image.copy()
    for color, corner_2d in zip(edge_colors, corners_2d):
        corner_2d = corner_2d.astype(np.int32)
        front = corner_2d[4:]
        rear = corner_2d[:4]
        cv2.fillPoly(overlay, [front], color=color)
        cv2.polylines(overlay, [rear], isClosed=True, color=color, thickness=line_widths)
        for i in range(4):
            cv2.line(
                overlay, 
                tuple(front[i]), 
                tuple(rear[i]), color=color, 
                thickness=line_widths)

    drawed_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return drawed_image