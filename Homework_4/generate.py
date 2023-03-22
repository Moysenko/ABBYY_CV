import numpy as np
from skimage.transform import ProjectiveTransform
from skimage.io import imread, imsave
from skimage.util import img_as_float
from skimage.filters import gaussian
from skimage import img_as_ubyte
from skimage.transform import rescale, resize
from os import walk


BACKGROUND_SCALE = 1.5
IRL_DOC_SHAPE = np.array([29.7, 21])

BACKGROUND_PROPERTIES = {
    'pupitr.png': {
        'corners_coordinates': [[78, 311], [62, 713], [343, 370], [320, 785],],
        'irl_shape': [34, 50]
    },
    'stol.jpg': {
        'corners_coordinates': [[567, 387], [842, 1357], [2040, -420], [2310, 510],],
        'irl_shape': [120, 60]
    },
    'beliy_stol.jpeg': {
        'corners_coordinates': [[942, 368], [237, 2748], [2140, 693], [1228, 3342],],
        'irl_shape': [60, 120]
    }
}

def center_and_normalize_points(points):
    pointsh = np.row_stack([points.T, np.ones((points.shape[0]), )])
    matrix = np.zeros((3, 3))

    C = points.mean(axis=0)
    N = np.sqrt(2) / np.sqrt(np.sum((points - C) ** 2, axis=1)).mean()
    
    matrix[0, 0] = N
    matrix[1, 1] = N
    matrix[2, 2] = 1
    matrix[0:2, 2] = -N * C

    return matrix, (matrix @ pointsh)[:2].T

def find_homography(src_keypoints, dest_keypoints):
    src_matrix, src = center_and_normalize_points(src_keypoints)
    dest_matrix, dest = center_and_normalize_points(dest_keypoints)

    A = np.zeros((2 * src.shape[0], 9))
    A[::2, 2] = -1
    A[1::2, 5] = -1
    A[::2, 0:2] = -src
    A[1::2, 3:5] = -src
    A[::2, 8] = dest[:, 0]
    A[1::2, 8] = dest[:, 1]
    A[::2, 6:8] = src * dest[:, 0:1]
    A[1::2, 6:8] = src * dest[:, 1:2]
    
    _, _, vh = np.linalg.svd(A)

    return np.linalg.inv(dest_matrix) @ vh[-1].reshape(3, 3) @ src_matrix

def get_doc_background_corners(doc, background_shape):
    h, w, _ = doc.shape
    scale = background_shape / IRL_DOC_SHAPE

    h_new, w_new = h * scale[0], w * scale[1]
    x_c, y_c = h // 2, w // 2
    
    table_x_1, table_x_2 = x_c - h_new // 2, x_c + h_new // 2
    table_y_1, table_y_2 = y_c - w_new // 2, y_c + w_new // 2

    return np.array([[x, y] for x in (table_x_1, table_x_2) for y in (table_y_1, table_y_2)])

def overlap(doc, background, inversed_transform, doc_corners):
    x_from, y_from = doc_corners.min(axis=0)
    x_to, y_to = doc_corners.max(axis=0)

    for x in range(x_from, x_to):
        for y in range(y_from, y_to):
            xt, yt = inversed_transform([x, y])[0]
            x1, x2 = int(np.floor(xt)), int(np.ceil(xt))
            y1, y2 = int(np.floor(yt)), int(np.ceil(yt))
            if x1 < 0 or y1 < 0 or x2 >= doc.shape[0] or y2 >= doc.shape[1] or\
                    x >= background.shape[0] or y >= background.shape[1]:
                continue
            background[x][y] = doc[x1][y1] * (x2 - xt) * (y2 - yt) +\
                                doc[x1][y2] * (x2 - xt) * (yt - y1) +\
                                doc[x2][y1] * (xt - x1) * (y2 - yt) +\
                                doc[x2][y2] * (xt - x1) * (yt - y1)
    return background

def apply_texture(doc, texture):
    texture = resize(texture, doc.shape[:2])
    return 0.7*doc + 0.3*texture

def doc_preprocessing(doc, texture):
    doc = rescale(doc, 0.8, channel_axis=-1)
    doc = apply_texture(doc, texture)
    return doc

def doc_postprocessing(doc):
    doc = rescale(rescale(doc, 0.9, channel_axis=-1), 1.1, channel_axis=-1)
    doc = gaussian(doc, sigma=0.7)
    return doc

def get_doc(doc_name, texture_name):
    doc = img_as_float(img_as_ubyte(imread(f'./Documents/{doc_name}')))
    texture = img_as_float(img_as_ubyte(imread(f'./Textures/{texture_name}')))
    doc = doc_preprocessing(doc, texture)
    return doc

def get_background_with_properties(background_name):
    background = img_as_float(img_as_ubyte(imread(f'./Backgrounds/{background_name}')))
    background = rescale(background, BACKGROUND_SCALE, channel_axis=-1)
    return background, BACKGROUND_PROPERTIES[background_name]

def get_projective_transform(doc, properties):
    doc_corners = get_doc_background_corners(doc, np.array(properties['irl_shape']))
    background_corners = np.array(properties['corners_coordinates']) * BACKGROUND_SCALE
    projective_tranform = ProjectiveTransform(find_homography(doc_corners, background_corners))  
    return projective_tranform

def photoshop(doc_name, texture_name, background_name):
    doc = get_doc(doc_name, texture_name)
    background, properties = get_background_with_properties(background_name)

    projective_transform = get_projective_transform(doc, properties)
    h, w, _ = doc.shape
    doc_coords = projective_transform(np.array([[0, 0], [h, 0],
                                                [h, w], [0, w]]))
    doc = overlap(doc, background, projective_transform.inverse, np.round(doc_coords).astype(int))

    doc = doc_postprocessing(doc)
    return doc

def main():
    get_files_list = lambda dirname: list(set(next(walk(dirname))[2]) - set(['.DS_Store']))
    filename = lambda full_name: full_name.split(".")[0]


    for background_name in get_files_list('./Backgrounds'):
        for doc_name in get_files_list('./Documents'):
            for texture_name in get_files_list('./Textures'):
                print(f'Processing document {doc_name}, texture {texture_name}, background {background_name}')
                result = photoshop(doc_name, texture_name, background_name)
                img_name = '-'.join([filename(doc_name), filename(texture_name), filename(background_name)])
                imsave(f'./Results/{img_name}.png', result)
                print('Done')

main()