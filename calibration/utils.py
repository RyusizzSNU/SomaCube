import os
import cv2
import numpy as np
import pygame
from pyquaternion import Quaternion

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

def find_images_in_dir(name):
    images = []
    i = 0
    while(True):
        try:
            with open('%s_%d.jpg'%(name, i)) as f:
                images.append(cv2.imread('%s_%d.jpg'%(name, i)))
                i += 1
        except IOError:
            break
    return images


def generate_objpoints(num, target_w, target_h, target_size):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((target_w * target_h, 3), np.float32)
    objp[:, :2] = np.mgrid[0:target_w, 0:target_h].T.reshape(-1, 2)
    objp = objp * target_size
    objp = np.tile(np.expand_dims(objp, 0), [num, 1, 1])
    return objp

def rotate_image(img, deg):
    # rotate image deg degree, counterclockwise
    if deg == 0:
        return img

    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    return cv2.warpAffine(img, M, (rows, cols))


def find_imagepoints_from_images(images, w, h, square=True, show_imgs=0, rotate_angle=0):
    # Arrays to store object points and image points from all the images.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    read_imgs = []
    points_dict = {}

    for i in range(len(images)):
        img = images[i]
        img = rotate_image(img, rotate_angle)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        if square:
            ret, corners = cv2.findChessboardCorners(img, (w, h), None)
        else:
            ret, corners = cv2.findCirclesGrid(img, (w, h), None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            points_dict[i] = corners
            read_imgs.append(img)
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (w, h), corners, ret)
            
            if show_imgs != 0:
                cv2.imshow('img', img)
                cv2.waitKey(0)
    print("succeeded to find points in %d imgs", len(read_imgs))
    return read_imgs, points_dict

def quit_pressed():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

def show_image(picture):
    surface = pygame.display.get_surface()
    surface.blit(picture, (0, 0))
    pygame.display.flip()

def show(fname):
    picture = pygame.image.load(fname)
    show_image(picture)

def rot_trans_to_rigid(rot, trans):
    rot = np.concatenate([rot, np.expand_dims(trans, 1)], 1)
    return np.concatenate([rot, [[0., 0., 0., 1.]]], 0)

def quat_trans_to_rigid(quat, trans):
    # quat : quaternion of shape (4)
    # trans : translation vector of shape (3)
    rot = Quaternion(quat).rotation_matrix
    return rot_trans_to_rigid(rot, trans)

def axis_angle_trans_to_rigid(axis, deg, trans):
    rot = Quaternion(axis=axis, degrees=deg).rotation_matrix
    return rot_trans_to_rigid(rot, trans) 
