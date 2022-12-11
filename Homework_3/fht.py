from skimage import img_as_ubyte
from skimage.util import img_as_float
from skimage.io import imread, imsave
from functools import lru_cache
import numpy as np
import img_tools
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt


class FHT:
  def __init__(self, img_name):
    self.source = img_as_float(img_as_ubyte(imread(img_name)))

  def process(self):
    self._img = img_tools.to_brightness(self.source)
    old_recursion_limit = sys.getrecursionlimit()
    sys.setrecursionlimit(10000)

    N, M = self._img.shape
    MAX_SHIFT = min(N, M) // 2
    max_variance, target_shift = 0, 0
    variances = []

    print("Calculating variances")
    for shift in tqdm(range(-MAX_SHIFT, MAX_SHIFT + 1)):
      variance = self._get_brightness_variance(shift)
      variances.append(variance)
      if variance > max_variance:
        max_variance = variance
        target_shift = shift

      if shift % 300 == 0 and shift != -MAX_SHIFT:
        plt.clf()
        plt.plot(np.arange(-MAX_SHIFT, shift + 1), variances)
        plt.savefig('out/variances.png')

    print(f"Finished calculating, optimal shift is {target_shift}")
    sys.setrecursionlimit(old_recursion_limit)

    return img_tools.rotate_by_shift(
      self.source, target_shift, 
      img_tools.InterpolationType.BILINEAR
    )

  def _get_brightness_variance(self, shift):
    line_brightness = []
    N, M = self._img.shape 
    for x in range(min(-shift, 0), max(N - shift, N) + 1):
      stats = self._brightness_statistics(x, 0, shift, M)
      if stats[1] != 0:
        line_brightness.append(stats[0] / stats[1])
    return np.var(line_brightness) if line_brightness else 0

  @lru_cache(10**8)
  def _brightness_statistics(self, x, y, shift_x, shift_y):
    """
    returns (brightness sum, amount of featured cells)
    of cells from <x, y> to <x + shift_x, y + shift_y> not inclusive
    """

    sign = lambda x: 1 if x >= 0 else -1
    if not img_tools.contains_coordinates(self._img, x, y) and \
        not img_tools.contains_coordinates(self._img, x + shift_x - sign(shift_x), y + shift_y - 1):
      return np.array([0, 0])   
    
    if abs(shift_x) <= 1:
      res = np.zeros(2)
      for y_i in range(y, y + shift_y):
        if img_tools.contains_coordinates(self._img, x, y_i):
          res += np.array([self._img[x][y_i], 1])
      return res

    shift_x2 = sign(shift_x) * (abs(shift_x) // 2)
    shift_y2 = shift_y // 2
    return self._brightness_statistics(x, y, shift_x2, shift_y2) +\
      self._brightness_statistics(x + shift_x2, y + shift_y2, shift_x - shift_x2, shift_y - shift_y2)