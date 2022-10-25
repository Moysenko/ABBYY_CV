from skimage import img_as_ubyte
from skimage.io import imread, imsave
from VNR import VNG
from helpers import to_brightness, PSNR
import time

cfa = img_as_ubyte(imread('./CFA.bmp'))

start_time = time.time()
generated = VNG(cfa)
execution_time = time.time() - start_time

imsave('result.png', img_as_ubyte(generated))

megapixels = cfa.shape[0] * cfa.shape[1] / 10**6
print(f'Execution speed: {execution_time / megapixels :.3f}(sec/MP)')

orig = img_as_ubyte(imread('./Original.bmp'))
print('PSNR =', PSNR(to_brightness(orig), to_brightness(generated)))