import os
import cv2
import numpy as np

def create_dir(dirname):
    if not os.path.exists(dirname):
        print("Creating %s"%dirname)
        os.makedirs(dirname)
    else:
        pass

def create_muldir(*args):
    for dirname in args:
        create_dir(dirname)

def concat_images(images):
    h = images[0].shape[0]

    for i in range(len(images)):
        if i == 0:
            continue
        img = images[i]
        img = cv2.resize(img, dsize = (img.shape[1] * h / img.shape[0], h), interpolation = cv2.INTER_CUBIC)
        images[i] = img
    
    return np.concatenate(images, axis=1)

def concat_images_in_dir(fnames, result_fname):
    images = []
    for f in fnames:
        images.append(cv2.imread(f))
    image = concat_images(images)
    cv2.imwrite(result_fname, image)
