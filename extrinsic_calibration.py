import os
import argparse
import cv2
import numpy as np
import pickle
import utils
from pyquaternion import Quaternion
from scipy.optimize import minimize

parser = argparse.ArgumentParser()
parser.add_argument('--cam', type=str, help='cameras to calibrate. one of [\'static\', \'wrist\', \'both\']')
parser.add_argument('--h', type=int, default=8, help="number of rows of the target")
parser.add_argument('--w', type=int, default=6, help="number of cols of the target")
parser.add_argument('--data_dir', type=str, default='/home/tidyboy/calibration/ext_data/')
parser.add_argument('--show_imgs', type=int, default=1)
parser.add_argument('--method', type=str, default='nelder-mead', help='which optimization method to use')
args = parser.parse_args() 

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def rotate_image(img, deg):
    # rotate image deg degree, counterclockwise
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    return cv2.warpAffine(img, M, (rows, cols))

s_imgs_needed = (args.cam == 'both' or args.cam == 'static')
w_imgs_needed = (args.cam == 'both' or args.cam == 'wrist')
s_imgs = []
w_imgs = []
w_points = []
s_points = []
W2Bs = [] 

def find_points_from_images(cam):
    i = 0
    imgs = []
    points_dict = {}
    while(True):
        try:
            with open(args.data_dir + '/images/%s_%d.jpg'%(cam, i)) as f:
                pass
            imgs.append(cv2.imread(args.data_dir + '/images/%s_%d.jpg'%(cam, i)))
        except IOError:
            break
        i += 1
    for i in range(len(imgs)):
        img = imgs[i]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(img, (args.h, args.w), None)
        ret, corners = cv2.findCirclesGrid(img, (args.h, args.w), None)

        if ret:
            corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
            points_dict[i] = np.squeeze(corners)
           
            img = cv2.drawChessboardCorners(img, (args.h, args.w), corners, ret)
            if args.show_imgs != 0:
                cv2.imshow('img', img)
                cv2.waitKey(0)
    return points_dict

def intersection_dict_by_indices(dicts):
    result = []
    for i in range(len(dicts)):
        result.append([])
    for key in dicts[0]:
        all_dict_have_this_key = True
        for i in range(len(dicts)):
            if not (key in dicts[i]):
                all_dict_have_this_key = False
        if all_dict_have_this_key:
            for i in range(len(dicts)):
                result[i].append(dicts[i][key])
            with open(args.data_dir + '/poses/%d.yaml'%key, 'rb') as f:
                W2Bs.append(pickle.load(f))
    return result
    

if s_imgs_needed:
    s_points = find_points_from_images('static')
    if w_imgs_needed:
        w_points = find_points_from_images('wrist')
        s_points, w_points = intersection_dict_by_indices([s_points, w_points])
    else:
        s_points = intersection_dict_by_indices([s_points])[0]
elif w_imgs_needed:
    w_points = find_points_from_images('wrist')
    w_points = intersection_dict_by_indices([w_points])[0]


'''for j in range(i):
    if s_imgs_needed:
        s_img = s_imgs[j]
        s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
        s_ret, s_corners = cv2.findChessboardCorners(s_img, (args.h, args.w), None)
    if w_imgs_needed:
        w_img = w_imgs[j]
        #w_img = rotate_image(w_img, 90)
        w_img = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)
        w_ret, w_corners = cv2.findChessboardCorners(w_img, (args.h, args.w), None)

    if (not s_imgs_needed or s_ret) and (not w_imgs_needed or w_ret):
        if s_imgs_needed:
            s_corners = cv2.cornerSubPix(s_img, s_corners, (11, 11), (-1, -1), criteria)
            s_points.append(np.squeeze(s_corners))
        if w_imgs_needed:
            w_corners = cv2.cornerSubPix(w_img, w_corners, (11, 11), (-1, -1), criteria)
            w_points.append(np.squeeze(w_corners))
        with open(args.data_dir + '/poses/%d.yaml'%j, 'rb') as f:
            W2Bs.append(pickle.load(f))

        s_img = cv2.drawChessboardCorners(s_img, (args.h, args.w), s_corners, s_ret)
        w_img = cv2.drawChessboardCorners(w_img, (args.h, args.w), w_corners, w_ret)

        s_img_shape = s_img.shape
        w_img = cv2.resize(w_img, dsize = (w_img.shape[1] * s_img.shape[0] / w_img.shape[0], s_img.shape[0]), interpolation=cv2.INTER_CUBIC)
        img = np.concatenate((s_img, w_img), axis=1)
        if args.show_imgs != 0:
            cv2.imshow('img', img)
            cv2.waitKey(0)'''

b = len(W2Bs)
print 'found chessboard corners in %d images'%b
n = args.h * args.w
if s_imgs_needed:
    s_points = np.array(s_points) # b * n * 2
    s_p_I = np.expand_dims(
            np.concatenate((s_points, np.ones([b, n, 1])), \
                axis=2\
            ), 3\
        ) # b * n * 3 * 1
if w_imgs_needed:
    w_points = np.array(w_points) # b * n * 2
    w_p_I = np.expand_dims(
            np.concatenate((w_points, np.ones([b, n, 1])), \
                axis=2\
            ), 3\
        )
W2Bs = np.array(W2Bs) # b * 4 * 4

target_points= np.zeros((args.h*args.w, 3), np.float32)
target_points[:,:2] = np.mgrid[0:args.h,0:args.w].T.reshape(-1,2)
target_points = target_points * 0.025


with open('results/intrinsic/static/mtx.pkl', 'rb') as f:
    s_C2I = pickle.load(f)
with open('results/intrinsic/wrist/mtx.pkl', 'rb') as f:
    w_C2I = pickle.load(f)
#w_C2I = np.matmul(np.array([[0, 1, 0], [-1, 0, 640], [0, 0, 1]]), w_C2I)

# initial guess
s_C2B_quat_0 = Quaternion(matrix=np.array[[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
s_C2B_trans_0 = np.array([-0.44, -0.1, 1.1])
w_C2W_quat_0 = Quaternion(matrix=np.array[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
w_C2W_trans_0 = np.array([-0.02, -0.09, 0.02])
T2B_quat_0 = Quaternion(matrix=np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])).q
T2B_trans_0 = np.array([-0.37, -0.31, 0.])
w_T2W_quat_0 = Quaternion(matrix=np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])).q
w_T2W_trans_0 = np.array([0.05, -0.05, -0.01])
#s_C2B_0 = np.array([[1., -0., -0., -0.44], [0., -1., 0., -0.1], [0., 0., -1., 1.1], [0., 0., 0., 1.]])
#w_C2W_0 = np.array([[1., 0., 0., -0.02], [0., 1., 0., -0.09], [0., 0., 1., 0.02], [0., 0., 0., 1.]])

step = 0

def quat_trans_to_rot(quat, trans):
    # quat : quaternion of shape (4)
    # trans : translation vector of shape (3)
    rot = Quaternion(quat).rotation_matrix                  # 3 * 3
    rot = np.concatenate([rot, np.expand_dims(trans, 1)], 1)  # 3 * 4
    return np.concatenate([rot, [[0., 0., 0., 1.]]], 0)       # 4 * 4

def static_loss(x):
    # x : concatenation of s_C2B_quat, s_C2B_trans, w_T2W_quat, w_T2W_trans. shape : (14)
    global step
    step += 1
    s_C2B = quat_trans_to_rot(x[0:4], x[4:7])               # 4 * 4
    w_T2W = quat_trans_to_rot(x[7:11], x[11:14])            # 4 * 4

    p_T = np.reshape(target_points, [-1, 3])
    p_T = np.concatenate([p_T, np.ones([n, 1])], 1)         # n * 4
    p_T = np.expand_dims(p_T, 2)                            # n * 4 * 1
    p = np.matmul(w_T2W, p_T)                               # n * 4 * 1
    p = np.matmul(np.expand_dims(W2Bs, 1), p)               # b * n * 4 * 1

    p = np.matmul(np.linalg.inv(s_C2B), p)                  # b * n * 4 * 1
    p = p[:, :, :3, :]                                      # b * n * 3 * 1
    p = np.matmul(s_C2I, p)                                 # b * n * 3 * 1
    p = p / np.expand_dims(p[:, :, 2, :], axis=2)

    l = np.mean(np.square(p - s_p_I))
    if step % 1000 == 0:
        print 'step%d : %f'%(step, l)
    return l

def wrist_loss(x):
    # x : concatenation of w_C2W_quat, w_C2W_trans, T2B_quat, T2B_trans. shape : (14)
    global step
    step += 1
    w_C2W = quat_trans_to_rot(x[0:4], x[4:7])               # 4 * 4
    T2B = quat_trans_to_rot(x[7:11], x[11:14])              # 4 * 4

    p_T = np.reshape(target_points, [-1, 3])
    p_T = np.concatenate([p_T, np.ones([n, 1])], 1)         # n * 4
    p_T = np.expand_dims(p_T, 2)                            # n * 4 * 1
    p = np.matmul(T2B, p_T)                                 # n * 4 * 1

    p = np.matmul(np.expand_dims(np.linalg.inv(W2Bs), 1), p)    # b * n * 4 * 1
    p = np.matmul(np.linalg.inv(w_C2W), p)                  # b * n * 4 * 1
    p = p[:, :, :3, :]                                      # b * n * 3 * 1
    p = np.matmul(w_C2I, p)                                 # b * n * 3 * 1
    p = p  / np.expand_dims(p[:, :, 2, :], axis=2)

    l = np.mean(np.square(p - w_p_I))
    if step % 1000 == 0:
        print 'step%d : %f'%(step, l)
    return l

def repr_loss(x):
    # x : concatenation of s_C2B_quat, s_C2B_trans, w_C2W_quat, w_C2B_trans. shape : (14)
    global step
    step += 1

    s_C2B = quat_trans_to_rot(x[0:4], x[4:7])               # 4 * 4
    w_C2W = quat_trans_to_rot(x[7:11], x[11:14])            # 4 * 4

    w_C2B = np.matmul(W2Bs, w_C2W)                          # b * 4 * 4
    w_I2C = np.linalg.inv(w_C2I)                            # 3 * 3

    p = np.matmul(w_I2C, w_p_I)                             # b * n * 3 * 1
    p = np.concatenate((p, np.ones([b, n, 1, 1])), axis=2)  # b * n * 4 * 1
    p = np.matmul(np.expand_dims(w_C2B, 1), p)              # b * n * 4 * 1
    
    p = np.matmul(np.linalg.inv(s_C2B), p)                  # b * n * 4 * 1
    p = p[:, :, :3, :]                                      # b * n * 3 * 1
    p = np.matmul(s_C2I, p)                                 # b * n * 3 * 1
    p = p / np.expand_dims(p[:, :, 2, :], axis=2)
    
    l = np.mean(np.square(p - s_p_I))
    if step % 1000 == 0:
        print 'step%d : %f'%(step, l)
    return l

if args.cam == 'both':
    x = minimize(repr_loss, np.concatenate([s_C2B_quat_0, s_C2B_trans_0, w_C2W_quat_0, w_C2W_trans_0]), method = args.method, options={'xtol':1e-9, 'maxiter' : 1000000, 'maxfev' : 64000000, 'disp' : True}).x
elif args.cam == 'wrist':
    x = minimize(wrist_loss, np.concatenate([w_C2W_quat_0, w_C2W_trans_0, T2B_quat_0, T2B_trans_0]), method = args.method, options={'xtol':1e-9, 'maxiter' : 1000000, 'maxfev' : 64000000, 'disp' : True}).x
elif args.cam == 'static':
    x = minimize(static_loss, np.concatenate([s_C2B_quat_0, s_C2B_trans_0, w_T2W_quat_0, w_T2W_trans_0]), method = args.method, options={'xtol':1e-9, 'maxiter' : 1000000, 'maxfev' : 64000000, 'disp' : True}).x

print(x)
x0 = quat_trans_to_rot(x[0:4], x[4:7])
x1 = quat_trans_to_rot(x[7:11], x[11:14])
print x0
print x1
utils.create_muldir('results', 'results/extrinsic/', 'results/extrinsic/%s'%args.cam)
with open('results/extrinsic/%s/mtx.pkl'%args.cam, 'w') as f:
    pickle.dump(np.stack([x0, x1], 0), f)

