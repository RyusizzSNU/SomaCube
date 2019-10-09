import time
import datetime
import numpy as np
import cv2
import os
import glob
import pickle
import argparse
import utils
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, help="directory to save results")
parser.add_argument('--cam_type', type=str, help="Camera type. one of realsense and sony")
parser.add_argument('--m', type=int, default=8, help="number of rows of the target")
parser.add_argument('--n', type=int, default=6, help="number of cols of the target")
parser.add_argument('--size', type=float, default=0.025, help="size of each lattice of target")
parser.add_argument('--num_pic', type=int, default=50, help="number of pictures to take")
parser.add_argument('--period', type=float, default=0.5, help="period between which pictures would be taken")


args = parser.parse_args()

filename = 'int_cal_pic'

cam = cam_tool(args.cam_type)
starttime = datetime.datetime.now()
nexttime = starttime + datetime.timedelta(seconds = 5)

t_list = []

for i in range(args.num_pic):
    while True:
        t = datetime.datetime.now()
        if t > nexttime:
            t_list.append(t)
            break

    cam.capture(filename + str(i) + '.jpg')
    cam.show(filename + str(i) + '.jpg')
    nexttime = nexttime + datetime.timedelta(seconds = args.period)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((args.m*args.n,3), np.float32)
objp[:,:2] = np.mgrid[0:args.m,0:args.n].T.reshape(-1,2)
objp = objp * args.size

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob(filename + '*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (args.m, args.n),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (args.m, args.n), corners,ret)
print("success :", len(objpoints))

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

utils.create_muldir('results', 'results/intrinsic', 'results/intrinsic/%s'%args.result_dir)

pickle.dump(np.array(ret), open('results/intrinsic/%s/ret.pkl'%args.result_dir, 'wb'))
pickle.dump(np.array(mtx), open('results/intrinsic/%s/mtx.pkl'%args.result_dir, 'wb'))
pickle.dump(np.array(dist), open('results/intrinsic/%s/dist.pkl'%args.result_dir, 'wb'))
pickle.dump(np.array(rvecs), open('results/intrinsic/%s/rvecs.pkl'%args.result_dir, 'wb'))
pickle.dump(np.array(tvecs), open('results/intrinsic/%s/tvecs.pkl'%args.result_dir, 'wb'))

cam.exit()
