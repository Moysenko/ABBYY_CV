import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

import repeatability_core


def read_images(images_folder):
    img_list = []
    for img_name in os.listdir(images_folder):
        img_list.append(
            cv2.imread(
                os.path.join(images_folder, img_name), 
                cv2.IMREAD_GRAYSCALE
            )
        )
    return img_list


img_list = read_images('input')

for method in ["shi-thomasi", "sift", "orb"]:
    method_result, execution_time = repeatability_core.repeatability_by_motion(
        img_list, method)

    N_images = len(img_list) - 1
    prefix_repeatabilities = [
        method_result[:i].mean() 
        for i in range(1, N_images + 1)
    ]
    plt.plot(np.arange(1, N_images + 1), prefix_repeatabilities)
    plt.title(f'Method: {method}')
    plt.xlabel('number of photos')
    plt.ylabel('repeatability')
    plt.savefig(f'output/{method}.jpg')
    plt.clf()

    print(f'Method {method} execution time is {execution_time * 1000} ms, '
          f'repeatability = {prefix_repeatabilities[-1]}')