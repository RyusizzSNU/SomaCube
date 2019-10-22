import pygame
import datetime
import os
import urx
import numpy as np
import pickle
import cv2
import argparse
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--h', type=int, default=9, help="number of rows of the target")
parser.add_argument('--w', type=int, default=6, help="number of cols of the target")
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

cam = cam_tool('sony')
cam.capture('preview.jpg')
cam.show('preview.jpg')

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
img = cv2.imread('preview.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(img, (args.h, args.w), None)
if ret:
    corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
    img = cv2.drawChessboardCorners(img, (args.h, args.w), corners, ret)
    cv2.imshow('img', img)
    cv2.waitKey(0)

with open('results/extrinsic/static/mtx.pkl', 'rb') as f:
    C2B = pickle.load(f)[0]
    print 'C2B :\n', C2B
with open('results/intrinsic/static/mtx.pkl', 'rb') as f:
    C2I = pickle.load(f)
    print 'C2I :\n', C2I

i = 0
nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 5)

while True:
    wait()

    p_I = corners[i][0]
    p_I = [[p_I[0]], [p_I[1]], [1]]
    p_C = np.matmul(np.linalg.inv(C2I), p_I)
    p_B = np.matmul(C2B, np.concatenate((p_C, [[1]]), axis=0))

    o_C = [[0], [0], [0], [1]]
    o_B = np.matmul(C2B, o_C)

    p = o_B - (o_B[2] / (o_B[2] - p_B[2])) * (o_B - p_B)

    pose = np.array([[0., 1., 0., p[0]], [1., 0., 0., p[1]], [0., 0., -1., 0.19], [0., 0., 0., 1.]])
    robot.movex('movel', pose, acc= 0.1, vel=0.2)
    i += 1
    nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 1)
