import numpy as np
from functools import cache

def _shift(img, shift_x, shift_y):
    if shift_x < 0:
        return np.flip(_shift(np.flip(img, axis=0), -shift_x, shift_y), axis=0)
    if shift_y < 0:
        return np.flip(_shift(np.flip(img, axis=1), shift_x, -shift_y), axis=1)

    shifted_img = np.zeros_like(img)
    shifted_img[shift_x:, shift_y:] = img[:img.shape[0] - shift_x, :img.shape[1] - shift_y]
    return shifted_img

def _axes_gradient(img, dx, dy):
    if dx == 0:
        return _axes_gradient(img.T, dy, 0).T
    if dx < 0:
        return np.flip(_axes_gradient(np.flip(img, axis=0), -dx, 0), axis=0)

    gradient = np.zeros_like(img)
    for shift_x in (0, 1):
        for shift_y in (-1, 0, 1):
            k = 1 if shift_y == 0 else 2
            to = _shift(img, 1 + shift_x, shift_y)
            from_ = _shift(img, -1 + shift_x, shift_y)
            gradient += np.abs(to - from_) / k

    return gradient

def _diagonal_gradient(img, dx, dy):
    if dx < 0:
        return np.flip(_diagonal_gradient(np.flip(img, axis=0), -dx, dy), axis=0)
    if dy < 0:
        return np.flip(_diagonal_gradient(np.flip(img, axis=1), dx, -dy), axis=1)

    gradient = np.zeros_like(img)
    for shift_x in (0, 1):
        for shift_y in (0, 1):
            to = _shift(img, 1 + shift_x, 1 + shift_y)
            from_ = _shift(img, -1 + shift_x, -1 + shift_y)
            gradient += np.abs(to - from_)
    return gradient

def _directions(img):
    gradients = []
    for delta in ((-1, 0), (0, -1), (1, 0), (0, 1)):
        gradients.append(_axes_gradient(img, *delta))
    for delta in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        gradients.append(_diagonal_gradient(img, *delta))
    gradients = np.array(gradients)

    k1 = 1.5
    k2 = 0.5
    grad_min = np.min(gradients, axis=0)
    grad_max = np.max(gradients, axis=0)
    threshold = k1 * grad_min + k2 * (grad_max + grad_min)
    return gradients <= threshold

def _axes_sum_rgb(img, dx, dy):
    if dx == 0:
        axes = list(range(0, len(img.shape)))
        axes[0], axes[1] = axes[1], axes[0]
        return np.transpose(_axes_sum_rgb(np.transpose(img, axes=axes), dy, 0), axes=axes)
    if dx < 0:
        return np.flip(_axes_sum_rgb(np.flip(img, axis=0), -dx, 0), axis=0)

    axes_sum = _shift(img, 2, 0)
    for shift_x in (0, 1):
        for shift_y in (-1, 0, 1):
            axes_sum += _shift(img, shift_x, shift_y)
    return axes_sum

def _diagonal_sum_rgb(img, dx, dy):
    if dx < 0:
        return np.flip(_diagonal_sum_rgb(np.flip(img, axis=0), -dx, dy), axis=0)
    if dy < 0:
        return np.flip(_diagonal_sum_rgb(np.flip(img, axis=1), dx, -dy), axis=1)

    diagonal_sum = np.zeros_like(img)
    for shift in ((0, 0), (1, 0), (0, 1), (1, 1), (2, 1), (1, 2), (2, 2)):
        diagonal_sum += _shift(img, *shift) 
    return diagonal_sum 

def _directions_sum_rgb(img):
    directions_sum_rgb = []
    for delta in ((-1, 0), (0, -1), (1, 0), (0, 1)):
        directions_sum_rgb.append(_axes_sum_rgb(img, *delta))
    for delta in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        directions_sum_rgb.append(_diagonal_sum_rgb(img, *delta))    
    return np.array(directions_sum_rgb)

@cache
def _shaped_bayer_pattern(N, M):
    bayer_pattern = np.full((N, M, 3), 0, dtype=np.float64)
    for x in range(N):
        for y in range(M):
            if (x + y) % 2 == 1:
                bayer_pattern[x][y][1] = 1
            elif x % 2 == 0:
                bayer_pattern[x][y][0] = 1
            else:
                bayer_pattern[x][y][2] = 1
    return bayer_pattern
    
def _bayer_pattern(img):
    return _shaped_bayer_pattern(*img.shape)

def _average_directions_rgb(img):
    bayer_pattern = _bayer_pattern(img)
    rgb_img = bayer_pattern * img[..., np.newaxis]
    directions_count = _directions_sum_rgb(bayer_pattern)
    return _directions_sum_rgb(rgb_img) * (directions_count != 0) / np.maximum(directions_count, 1.)
    
def _rgb_sum(img):
    directions = _directions(img).astype(np.float64)
    average_directions_rgb = _average_directions_rgb(img)
    directions_average_rgb = np.sum(average_directions_rgb * directions[..., np.newaxis], axis=0)
    return directions_average_rgb / np.sum(directions, axis=0)[..., np.newaxis]

def _rgb_deltas(img):
    rgb_sum = _rgb_sum(img)
    return rgb_sum - np.sum(_bayer_pattern(img) * rgb_sum, axis=-1)[..., np.newaxis]

def VNG(img):
    img = img.astype(np.float64)
    rgb_deltas = _rgb_deltas(img)
    vng_img = img[..., np.newaxis] + rgb_deltas
    return np.clip(vng_img, 0, 255).astype(np.uint8)
