from fht import FHT
from skimage.io import imsave

sample_num = input()
fht = FHT(f'img/{sample_num}.jpg')
result = fht.process()

imsave(f'out/{sample_num}_rotated.jpg', result)