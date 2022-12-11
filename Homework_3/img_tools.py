from math import sin, cos, atan2, floor, ceil
from skimage.transform import rescale
import numpy as np
from enum import Enum, auto

def to_brightness(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def make_height_power_of_2(img):
    nearest_power = 1
    while nearest_power < img.shape[1]:
        nearest_power *= 2
    return rescale(img, nearest_power / img.shape[1])

class InterpolationType(Enum):
    NEAREST_NEIGHBOUR = auto()
    BILINEAR = auto()

def contains_coordinates(img, x, y):
    return 0 <= x < img.shape[0] and 0 <= y < img.shape[1]

def get_unbounded(img, x, y):
    if not contains_coordinates(img, x, y):
        return np.zeros_like(img[0][0])
    else:
        return img[x][y]

def _get_bilinear_interpolated(img, x, y):
    x1, x2 = floor(x), ceil(x)
    y1, y2 = floor(y), ceil(y)
    R1 = (x2 - x) * get_unbounded(img, x1, y1) + (x - x1) * get_unbounded(img, x2, y1)
    R2 = (x2 - x) * get_unbounded(img, x1, y2) + (x - x1) * get_unbounded(img, x2, y2)
    return (y2 - y) * R1 + (y - y1) * R2

def get_pixel(img, x, y, interpolation_type):
    match interpolation_type:
        case InterpolationType.NEAREST_NEIGHBOUR:
            return get_unbounded(img, round(x), round(y))
        case InterpolationType.BILINEAR:
            return _get_bilinear_interpolated(img, x, y)
        case _:
            raise AttributeError("Invalid interpolation type")

def rotate_by_shift(img, shift: int, interpolation_type: InterpolationType):
    angle = -atan2(shift, img.shape[1])
    print(f"Rotating by angle {-angle}")
    center = np.array(img.shape[:2]) / 2
    rotate_matrix = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    
    result = np.zeros_like(img)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            from_x, from_y = rotate_matrix @ (np.array([x, y]) - center) + center
            result[x][y] = get_pixel(img, from_x, from_y, interpolation_type)

    return result