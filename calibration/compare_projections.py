import os
import time
import datetime
import cv2
import numpy as np
import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--m', type=int, default=8, help="number of rows of the target")
parser.add_argument('--n', type=int, default=6, help="number of cols of the target")
args = parser.parse_args() 

def rotate_image(img, deg):
    # rotate image deg degree, counterclockwise
    rows, cols = img.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), deg, 1)
    return cv2.warpAffine(img, M, (rows, cols))

def I2B_with_z_constraint(p, C2I, C2B, z=0): 
    # p : np array of shape (2)
    # C2I : Camera to Image matrix(same with intrinsic parameter matrix
    # B2C : Base to Camera matrix

    p = [[p[0]], [p[1]], [1]]
    I2C = np.linalg.inv(C2I)
    o_C = [[0], [0], [0]]
    p_C = np.matmul(I2C, p)
    o_B = np.matmul(C2B, np.concatenate((o_C, [[1]]), axis=0))
    p_B = np.matmul(C2B, np.concatenate((p_C, [[1]]), axis=0))

    return o_B - ((o_B[2] - z)/(o_B[2] - p_B[2])) * (o_B - p_B)

def B2I(p, C2I, C2B):
    # p : np array of shape (4, 1)
    B2C = np.linalg.inv(C2B)
    p = np.matmul(C2I, np.matmul(B2C, p)[:3, :])
    p = p / p[2][0]
    return p

starttime = datetime.datetime.now()
nexttime = starttime + datetime.timedelta(1)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

with open('results/intrinsic/static/mtx.pkl', 'rb') as f:
    s_C2I = pickle.load(f)
with open('results/intrinsic/wrist/mtx.pkl', 'rb') as f:
    w_C2I = pickle.load(f)
w_C2I = np.matmul(np.array([[0, 1, 0], [-1, 0, 640], [0, 0, 1]]), w_C2I)

with open('results/extrinsic/static/mtx.pkl', 'rb') as f:
    s_C2B = pickle.load(f)[0]
with open('results/extrinsic/wrist/mtx.pkl', 'rb') as f:
    w_C2W = pickle.load(f)[0]
        
img_id = 0
while True:
    try:
        with open('ext_data/poses/%d.yaml'%img_id, 'rb') as f:
            w_W2B = pickle.load(f)
        s_img = cv2.imread('ext_data/images/static_%d.jpg'%img_id)
        w_img = cv2.imread('ext_data/images/wrist_%d.jpg'%img_id)
    except IOError:
        break
    w_C2B = np.matmul(w_W2B, w_C2W)

    img_id += 1
    w_img = rotate_image(w_img, 90)
   
    s_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2GRAY)
    w_img = cv2.cvtColor(w_img, cv2.COLOR_BGR2GRAY)

    s_ret, s_corners = cv2.findChessboardCorners(s_img, (args.m, args.n), None)
    w_ret, w_corners = cv2.findChessboardCorners(w_img, (args.m, args.n), None)
    
    if s_ret == True and w_ret == True:
        s_corners = cv2.cornerSubPix(s_img, s_corners, (11,11), (-1,-1), criteria)
        w_corners = cv2.cornerSubPix(w_img, w_corners, (11,11), (-1,-1), criteria)

        s_img_w_reprojected = np.copy(s_img)
        
        s_img = cv2.drawChessboardCorners(s_img, (args.m, args.n), s_corners, s_ret)
        w_img = cv2.drawChessboardCorners(w_img, (args.m, args.n), w_corners, w_ret)
        
        sum_err = 0
        for i in range(len(s_corners)):
            s_p_I = s_corners[i][0]
            w_p_I = w_corners[i][0]
            print '----------------point %d---------------'%i
            print 's_p_I:\n', s_p_I, '\nw_p_I:\n', w_p_I
            s_p_B = I2B_with_z_constraint(s_p_I, s_C2I, s_C2B)
            w_p_B = I2B_with_z_constraint(w_p_I, w_C2I, w_C2B)
            print 's_p_B:\n', s_p_B, '\nw_p_B:\n', w_p_B
            err = np.linalg.norm(s_p_B - w_p_B)
            print '\nerr :', err
            sum_err += err

            w_p_sI = B2I(w_p_B, s_C2I, s_C2B)
            s_img_w_reprojected = cv2.circle(s_img_w_reprojected, (int(w_p_sI[0][0]), int(w_p_sI[1][0])), 2, (255, 0, 0), -1)

        print '\navg_err :', sum_err / len(s_corners)
    
    s_img_shape = s_img.shape
    w_img = cv2.resize(w_img, dsize = (w_img.shape[1] * s_img.shape[0] / w_img.shape[0], s_img.shape[0]), interpolation=cv2.INTER_CUBIC) 
    img = np.concatenate((s_img, w_img), axis=1)

    cv2.imshow('img',img)
    cv2.imshow('img_reproj', s_img_w_reprojected)
    cv2.waitKey(0)
