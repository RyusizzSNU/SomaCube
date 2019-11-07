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
parser.add_argument('--cam', type=str, help="Camera to calibrate. one of 'static' and 'wrist'")
parser.add_argument('--static_cam_model', type=str, default='sony')
parser.add_argument('--wrist_cam_model', type=str, default='rs')
parser.add_argument('--w', type=int, default=9, help="number of cols of the target")
parser.add_argument('--h', type=int, default=6, help="number of rows of the target")
parser.add_argument('--size', type=float, default=0.0228, help="size of each lattice of target(meter)")
parser.add_argument('--num_pic', type=int, default=50, help="number of pictures to take")
parser.add_argument('--period', type=float, default=0.5, help="period between which pictures would be taken")

args = parser.parse_args()

filename = 'int_cal_pic'

if args.cam == 'static':
    cam = cam_tool(args.static_cam_model)
elif args.cam == 'wrist':
    cam = cam_tool(args.wrist_cam_model)
nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 5)

t_list = []

for i in range(args.num_pic):
    while True:
        t = datetime.datetime.now()
        if t > nexttime:
            t_list.append(t)
            break

    cam.capture('%s_%d.jpg'%(filename, i))
    utils.show('%s_%d.jpg'%(filename, i))
    nexttime = nexttime + datetime.timedelta(seconds = args.period)

images = utils.find_images_in_dir(filename)
imgs, imgpoints = utils.find_imagepoints_from_images(images, args.w, args.h)   # 2d points in image plane.
objpoints = utils.generate_objpoints(len(imgpoints), args.w, args.h, args.size) # 3d point in real world space

imgpoints = np.array([imgpoints[i] for i in imgpoints])

cv2.destroyAllWindows()

print (imgs[0].shape[1], imgs[0].shape[0])

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (imgs[0].shape[1], imgs[0].shape[0]), None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

utils.create_muldir('results', 'results/intrinsic', 'results/intrinsic/%s'%args.cam)

pickle.dump(np.array(ret), open('results/intrinsic/%s/ret.pkl'%args.cam, 'wb'))
pickle.dump(np.array(mtx), open('results/intrinsic/%s/mtx.pkl'%args.cam, 'wb'))
pickle.dump(np.array(dist), open('results/intrinsic/%s/dist.pkl'%args.cam, 'wb'))
pickle.dump(np.array(rvecs), open('results/intrinsic/%s/rvecs.pkl'%args.cam, 'wb'))
pickle.dump(np.array(tvecs), open('results/intrinsic/%s/tvecs.pkl'%args.cam, 'wb'))

cam.exit()

for img_fname in glob.glob('%s_*.jpg'%filename):
    os.remove(img_fname)
