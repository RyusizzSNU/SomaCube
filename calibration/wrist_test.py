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

for i in range(len(imgpoints)):
    wait()

    p_I = imgpoints[i]
    p = utils.wrist_image_to_base(p_I, C2I, C2W, W2B)

    pose = np.array([[0., 1., 0., p[0]], [1., 0., 0., p[1]], [0., 0., -1., 0.19], [0., 0., 0., 1.]])
    robot.movex('movel', pose, acc= 0.1, vel=0.2)
    nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 1)
