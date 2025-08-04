import numpy as np
import json

aruco_id_by_side = {
    'bottom': [2],
    'right': [6, 9],
    'left': [5, 4],
    'front': [7, 8]
}

aruco_size_by_id = {
    2: 0.049,  # Bottom marker
    6: 0.046,  # Right markers
    9: 0.046,  # Right markers 
    7: 0.046,  # Front markers
    8: 0.046,  # Front markers
    5: 0.046,  # Left markers
    4: 0.042   # Left markers
}

guilder_aruco_2_center_mat_dict = {
    "bottom_aruco_pose_2_global_pose": [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.058],  
        [0.0, 0.0, 1.0, -0.204],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_6_2_global_pose": [
        [0.48946482, -0.06568191, -0.8695459, 0.04617943],
        [-0.01581092, 0.99632695, -0.08415835, -0.11587574],
        [0.87187969, 0.05494087, 0.4866285, -0.15147996],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_9_2_global_pose": [
        [0.49003224, -0.0593383, -0.86968234, -0.20230754],
        [-0.01229018, 0.99711097, -0.07495778, -0.11712953],
        [0.87161766, 0.04742028, 0.48788725, -0.14085418],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_5_2_global_pose": [
        [0.48948049, 0.06616372, 0.86950055, 0.21364427],
        [-0.01231153, 0.99754233, -0.06897622, -0.09399025],
        [-0.87192733, 0.02305763, 0.48909209, -0.15406877],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_4_2_global_pose": [
        [0.5453138, 0.04945392, 0.83677187, -0.08650763],
        [-0.01416501, 0.99865923, -0.04979044, -0.10088804],
        [-0.83811229, 0.01529853, 0.54528318, -0.16317482],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_7_2_global_pose": [
        [-1.0, 0.0, 0.0, 0.155],
        [0.0, 1.0, 0.0, -0.107],
        [0.0, 0.0, -1.0, -0.244],
        [0.0, 0.0, 0.0, 1.0]],
    "aruco_id_8_2_global_pose": [
        [-1.0, 0.0, 0.0, -0.11],
        [0.0, 1.0, 0.0, -0.107],
        [0.0, 0.0, -1.0, -0.244],
        [0.0, 0.0, 0.0, 1.0]]
}

aruco_2_center_mat = {
    2: np.array(guilder_aruco_2_center_mat_dict['bottom_aruco_pose_2_global_pose'], dtype=np.float64),
    6: np.array(guilder_aruco_2_center_mat_dict['aruco_id_6_2_global_pose'], dtype=np.float64),
    9: np.array(guilder_aruco_2_center_mat_dict['aruco_id_9_2_global_pose'], dtype=np.float64),
    5: np.array(guilder_aruco_2_center_mat_dict['aruco_id_5_2_global_pose'], dtype=np.float64),
    4: np.array(guilder_aruco_2_center_mat_dict['aruco_id_4_2_global_pose'], dtype=np.float64),
    7: np.array(guilder_aruco_2_center_mat_dict['aruco_id_7_2_global_pose'], dtype=np.float64),
    8: np.array(guilder_aruco_2_center_mat_dict['aruco_id_8_2_global_pose'], dtype=np.float64)    
}

aruco_ids_to_detect = list(aruco_2_center_mat.keys())