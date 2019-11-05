import pygame
import datetime
import os
import urx
import numpy as np
import pickle
import cv2
import argparse
import utils
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--w', type=int, default=9, help="number of cols of the target")
parser.add_argument('--h', type=int, default=6, help="number of rows of the target")
parser.add_argument('--size', type=float, default=0.0228, help="size of each lattice of target")
parser.add_argument('--auto', type=int, default=0)
args = parser.parse_args() 

starttime = None
nexttime = None

def wait():
    while True:
        if args.auto != 0:
            t = datetime.datetime.now()
            if t > nexttime:
                return
        else:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    return

robot = urx.Robot("192.168.1.109")

cam = cam_tool('rs')
cam.capture('preview.jpg')
cam.show('preview.jpg')

_, imgpoints = utils.find_imagepoints_from_images([cv2.imread('preview.jpg')], args.w, args.h, show_imgs=1)
imgpoints = np.squeeze(imgpoints[0])

img = cv2.imread('preview.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

with open('results/extrinsic/wrist/mtx.pkl', 'rb') as f:
    C2W = pickle.load(f)[0]
    print 'C2W :\n', C2W
with open('results/intrinsic/wrist/mtx.pkl', 'rb') as f:
    C2I = pickle.load(f)
    print 'C2I :\n', C2I
W2B = np.array(robot.get_pose().array)
i = 0
nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 5)

while True:
    wait()

    p_I = imgpoints[i]
    p_I = [[p_I[0]], [p_I[1]], [1]]
    p_C = np.matmul(np.linalg.inv(C2I), p_I)
    p_W = np.matmul(C2W, np.concatenate((p_C, [[1]]), axis=0))
    p_B = np.matmul(W2B, p_W)

    o_C = [[0], [0], [0], [1]]
    o_W = np.matmul(C2W, o_C)
    o_B = np.matmul(W2B, o_W)

    p = o_B - (o_B[2] / (o_B[2] - p_B[2])) * (o_B - p_B)

    pose = np.array([[0., 1., 0., p[0]], [1., 0., 0., p[1]], [0., 0., -1., 0.19], [0., 0., 0., 1.]])
    robot.movex('movel', pose, acc= 0.1, vel=0.2)
    i += 1
    nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 1)
