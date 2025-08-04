import cv2
import numpy as np
from typing import Optional, Tuple

def resizeWithCropFactor(img, size, intrinsic=None):
    h1, w1 = img.shape[:2]
    w2, h2 = size
    if h1 == h2 and w1 == w2:
        return img, intrinsic 
    r1 = w1 / h1
    r2 = w2 / h2
    if r1 >= r2:
        changed_width = int(r2 * h1)
        offset_pxl_pos = int((w1 - changed_width) / 2)
        img = img[:, offset_pxl_pos : offset_pxl_pos + changed_width]
        ratio = h2 / h1
    elif r1 < r2:
        changed_height = int(w1 / r2)
        offset_pxl_pos = int((h1 - changed_height) / 2)
        img = img[offset_pxl_pos : offset_pxl_pos + changed_height, :]
        ratio = w2 / w1

    if intrinsic is not None:
        cx = w2 / 2
        cy = h2 / 2

        new_intrinsic = intrinsic.copy()
        new_intrinsic[0, -1] = cx
        new_intrinsic[1, -1] = cy
        new_intrinsic[0, 0] *= ratio
        new_intrinsic[1, 1] *= ratio
        return cv2.resize(img, (w2, h2)), new_intrinsic

    return cv2.resize(img, (w2, h2)), None

def put_text_lines(image, list_texts, position, color=(0, 255, 0), **kwargs):
    font_scale = kwargs.get('font_scale', 0.8)
    thickness = kwargs.get('thickness', 2)
    y_offset = kwargs.get('y_offset', 0)
    drawed_image = image.copy()
    if isinstance(list_texts, str):
        list_texts = [list_texts]
    for line in list_texts:
        text_size = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        cv2.putText(
            drawed_image, 
            line, 
            (position[0], position[1] + y_offset),
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            color, 
            thickness)
        y_offset += text_size[1] + 10
    return drawed_image

def draw_frame_axes(
    image, 
    cam_K, 
    distortion_coeff, 
    rvecs, 
    tvecs,
    axis_length=0.06,
    markers_corners=None,
    draw_detected_markers_boxes=False):

    annotated_image = image.copy()

    for rvec, tvec in zip(rvecs, tvecs):
        annotated_image = cv2.drawFrameAxes(
            annotated_image, 
            cam_K, 
            distortion_coeff,
            rvec, 
            tvec, 
            axis_length)

    if draw_detected_markers_boxes and markers_corners is not None:
        annotated_image = cv2.aruco.drawDetectedMarkers(annotated_image, markers_corners) 

    return annotated_image
