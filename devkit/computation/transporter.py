import cv2
import numpy as np

class Transporter:
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.ar_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

    def estimate_arucos_poses(
        self, 
        image,
        intrinsic,
        distortion_coef,
        marker_sizes,
        aruco_corners,
        correct_rot_mat=np.eye(3)):
        """
        Estimate the pose of ArUco markers in the image.
        Parameters:
            image: Input image containing ArUco markers.
            intrinsic: Camera intrinsic matrix.
            distortion_coef: Distortion coefficients of the camera.
            marker_sizes: Size of the ArUco marker in meters.
            aruco_corners: Detected corners of the ArUco markers.
            correct_rot_mat: Rotation matrix to correct the orientation of the markers.
        Returns:
            rvecs: List of rotation vectors for each detected marker.
            tvecs: List of translation vectors for each detected marker.
        """

        rvecs, tvecs = [], [] 
    
        for i in range(len(aruco_corners)):
            marker_size = marker_sizes[i]
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(
                aruco_corners[i], 
                marker_size, 
                intrinsic, 
                distortion_coef)
            r_mat, _ = cv2.Rodrigues(rvec)
    
            # The aruco is rotated about z-axis, we need to adjust the rotation matrix. If 
            # it is pasted correctly, just multiply with identity matrix.
            r_mat_correct = r_mat @ correct_rot_mat
            r_vec_correct, _ = cv2.Rodrigues(r_mat_correct)
    
            rvecs.append(r_vec_correct)
            tvecs.append(tvec)
                 
        return rvecs, tvecs

    def detect_aruco_markers_by_id(
        self, 
        image,
        selected_aruco_ids,
        BGR_format=True):
        """
        Detect ArUco markers in the image by their IDs.
        Parameters:
            image: Input image containing ArUco markers.
            selected_aruco_ids: List of ArUco marker IDs to detect.
            BGR_format: If True, the input image is in BGR format, otherwise RGB.
        Returns:
            boxes: List of bounding boxes for detected markers.
            centers: List of centers of the detected markers.
            detected_corners: List of corners for the detected markers.
            detected_aruco_ids: List of IDs of the detected markers.
        """

        boxes = []
        centers = []
        detected_corners = []
        detected_aruco_ids = []
        gray_img = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)

        corners, ids, _ = self.ar_detector.detectMarkers(gray_img)

        if ids is None:
            return boxes, centers, detected_corners, detected_aruco_ids
            
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in selected_aruco_ids:
                x_coords = corners[i][0][:, 0]
                y_coords = corners[i][0][:, 1]
                x1, y1 = int(np.min(x_coords)), int(np.min(y_coords))
                x2, y2 = int(np.max(x_coords)), int(np.max(y_coords))
                center = ((x1+x2)//2, (y1+y2)//2)
                boxes.append(((x1, y1, x2, y2), f'ArUco_{marker_id}', 1.0))
                centers.append(center)
                detected_corners.append(corners[i])
                detected_aruco_ids.append(marker_id)
                
        return boxes, centers, detected_corners, detected_aruco_ids

    def draw_aruco_poses(
        self,
        image, 
        markers_corners, 
        matrix_coefficients,
        distortion_coefficients,
        rvecs,
        tvecs):
    
        annotated_image = image.copy()
    
        for rvec, tvec in zip(rvecs, tvecs):
            annotated_image = cv2.drawFrameAxes(
                annotated_image, 
                matrix_coefficients, 
                distortion_coefficients, 
                rvec, 
                tvec, 
                length=0.06) 
    
        return annotated_image