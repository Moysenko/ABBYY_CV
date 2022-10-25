import numpy as np

def to_brightness(img):
    return 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]

def PSNR(orig, generated):
    MSE = ((generated.astype(np.float64) - orig.astype(np.float64)) ** 2).mean()
    return 10 * np.log10((generated.astype(np.float64) ** 2).max() / MSE)