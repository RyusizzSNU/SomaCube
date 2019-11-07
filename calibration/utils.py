import os
import cv2
import numpy as np
import pickle
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
    return cv2.warpAffine(img, M, (cols, rows))


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

def load_static_C2B(calib='static'):
    with open('results/extrinsic/%s/mtx.pkl'%calib, 'rb') as f:
        C2B = pickle.load(f)[0]
        print 'static C2B from %s calibration:\n'%calib, C2B
    return C2B

def load_static_C2I():
    with open('results/intrinsic/static/mtx.pkl', 'rb') as f:
        C2I = pickle.load(f)
        print 'static C2I :\n', C2I
    return C2I

def load_wrist_C2W(calib='wrist'):
    with open('results/extrinsic/%s/mtx.pkl'%calib, 'rb') as f:
        if calib == 'both':
            C2W = pickle.load(f)[1]
            print 'wrist C2W from both calibration:\n', C2W
        elif calib == 'wrist':
            C2W = pickle.load(f)[0]
            print 'wrist C2W from wrist calibration:\n', C2W
    return C2W

def load_wrist_C2I():
    with open('results/intrinsic/wrist/mtx.pkl', 'rb') as f:
        C2I = pickle.load(f)
        print 'wrist C2I :\n', C2I
    return C2I

def static_image_to_base(p_I, C2I, C2B):
    p_I = [[p_I[0]], [p_I[1]], [1]]
    p_C = np.matmul(np.linalg.inv(C2I), p_I)
    p_B = np.matmul(C2B, np.concatenate((p_C, [[1]]), axis=0))

    o_C = [[0], [0], [0], [1]]
    o_B = np.matmul(C2B, o_C)

    p = o_B - (o_B[2] / (o_B[2] - p_B[2])) * (o_B - p_B)
    return p

def wrist_image_to_base(p_I, C2I, C2W, W2B):
    p_I = [[p_I[0]], [p_I[1]], [1]]
    p_C = np.matmul(np.linalg.inv(C2I), p_I)
    p_W = np.matmul(C2W, np.concatenate((p_C, [[1]]), axis=0))
    p_B = np.matmul(W2B, p_W)

    o_C = [[0], [0], [0], [1]]
    o_W = np.matmul(C2W, o_C)
    o_B = np.matmul(W2B, o_W)

    p = o_B - (o_B[2] / (o_B[2] - p_B[2])) * (o_B - p_B)
    return p

def base_to_static_image(p_B, C2I, C2B):
    p_B = [[p_B[0]], [p_B[1]], [p_B[2]], [1]]
    p_C = np.matmul(np.linalg.inv(C2B), p_B)
    p_C = p_C[:3, :]
    p_I = np.squeeze(np.matmul(C2I, p_C))
    p_I = p_I / p_I[2]
    return p_I[:2]

def base_to_wrist_image(p_B, C2I, C2W, W2B):
    p_B = [[p_B[0]], [p_B[1]], [p_B[2]], [1]]
    p_W = np.matmul(np.linalg.inv(W2B), p_B)
    p_C = np.matmul(np.linalg.inv(C2W), p_W)
    p_C = p_C[:3, :]
    p_I = np.squeeze(np.matmul(C2I, p_C))
    p_I = p_I / p_I[2]
    return p_I[:2]
