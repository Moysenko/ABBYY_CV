import cv2
import numpy as np
from numba import njit
from time import time


PERMISSIBLE_INACCURACY = 2.5
MAX_CORNERS = 1000
QUALITY_LEVEL = 1e-3
MIN_CORNERS_DISTANCE = 2


def find_key_points(img, method):
    start = time()

    if method == "shi-thomasi":
        key_points = [
            cv2.KeyPoint(x=f[0][0], y=f[0][1], _size=20) 
            for f in cv2.goodFeaturesToTrack(
                img, MAX_CORNERS, QUALITY_LEVEL, MIN_CORNERS_DISTANCE)
        ]
    elif method == "sift":
        key_points = cv2.SIFT_create().detect(img)
    else:  # orb
        key_points = cv2.ORB_create().detect(img)
    
    return key_points, time() - start


@njit
def get_distances(first, second):
    N, M = first.shape[0], second.shape[0]
    distances = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            distances[i, j] = np.sqrt(np.sum((first[i] - second[j])**2))
    return distances


def repeatability_by_motion(img_list, method):
    initial_key_points, total_time = find_key_points(img_list[0], method)
    initial_key_points = cv2.KeyPoint_convert(initial_key_points)
    
    result = np.zeros((len(img_list) - 1, initial_key_points.shape[0]))
    total_key_points = len(initial_key_points)

    for k, img in enumerate(img_list[1:]):
        key_points, current_execution_time = find_key_points(img, method)
        
        total_time += current_execution_time
        total_key_points += len(key_points)

        phase_shift, _ = cv2.phaseCorrelate(np.float32(img_list[0]), np.float32(img))
        key_points = cv2.KeyPoint_convert(key_points) - phase_shift

        distances = get_distances(initial_key_points, key_points).min(axis=1)
        result[k, distances < PERMISSIBLE_INACCURACY] = 1
    return result, total_time / total_key_points
