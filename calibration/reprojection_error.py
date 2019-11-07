import os
import urx
import pygame
import numpy as np
import pickle
import cv2
import argparse
import utils
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--cam', type=str, help='camera to calculate reprojection error')
parser.add_argument('--calib', type=str, help='calibration method to evaluate')
parser.add_argument('--w', type=int, default=9, help="number of cols of the target")
parser.add_argument('--h', type=int, default=6, help="number of rows of the target")
parser.add_argument('--size', type=float, default=0.0228, help='size of each lattice of target')
parser.add_argument('--static_cam_model', type=str, default='sony')
parser.add_argument('--wrist_cam_model', type=str, default='rs')
parser.add_argument('--proj_to_image', type=int, default=1)
args = parser.parse_args()

if args.cam == 'static':
    cam = cam_tool(args.static_cam_model)
elif args.cam == 'wrist':
    cam = cam_tool(args.wrist_cam_model)

cam.capture('preview.jpg')
img = cv2.imread('preview.jpg')
_, imgpoints = utils.find_imagepoints_from_images([img], args.w, args.h, show_imgs=1)
imgpoints = np.squeeze(imgpoints[0])

robot = urx.Robot('192.168.1.109')
W2B = np.array(robot.get_pose().array)

center = np.array([-0.45165, -0.1984, 0])
objpoints = utils.generate_objpoints(1, args.w, args.h, args.size) # 1 * (w*h) * 3
objpoints = objpoints * np.array([1., -1., 1.]) + center + np.array([args.w - 1, 0., 0.], dtype=float) * -args.size
objpoints = np.squeeze(objpoints, 0)

err_sum = 0
if args.cam == 'static':
    C2B = utils.load_static_C2B(calib=args.calib)
    C2I = utils.load_static_C2I()
    if args.proj_to_image != 0:
        for i in range(len(objpoints)):
            p_B = objpoints[i]
            p = utils.base_to_static_image(p_B, C2I, C2B)

            img = cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            err = np.sqrt(np.sum(np.square(p - imgpoints[i])))
            
            print p, imgpoints[i], err
            err_sum += err
        cv2.imshow('img', img)
        cv2.waitKey(0)
    else:
        for i in range(len(imgpoints)):
            p_I = imgpoints[i]
            p = utils.static_image_to_base(p_I, C2I, C2B)
            p = np.squeeze(p)[:3]

            err = np.sqrt(np.sum(np.square(p - objpoints[i])))

            print p, objpoints[i], err
            err_sum += err

elif args.cam == 'wrist':
    C2W = utils.load_wrist_C2W(calib=args.calib)
    C2I = utils.load_wrist_C2I()

    if args.proj_to_image != 0:
        for i in range(len(objpoints)):
            p_B = objpoints[i]
            p = utils.base_to_wrist_image(p_B, C2I, C2W, W2B)

            img = cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            err = np.sqrt(np.sum(np.square(p - imgpoints[i])))

            print p, imgpoints[i], err
            err_sum += err
        cv2.imshow('img', img)
        cv2.waitKey(0)
    else:
        err_sum = 0
        for i in range(len(imgpoints)):
            p_I = imgpoints[i]
            p = utils.wrist_image_to_base(p_I, C2I, C2W, W2B)
            p = np.squeeze(p)[:3]

            err = np.sqrt(np.sum(np.square(p - objpoints[i])))

            print p, objpoints[i], err
            err_sum += err
            
print err_sum / len(imgpoints)

robot.close()
