import pygame
import os
import urx
import pickle
import numpy as np
import argparse
from cam_tool import cam_tool
from pyquaternion import Quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--cam_type', type=str, help="Camera type. one of realsense and sony")
args = parser.parse_args()

current_dir = '/home/tidyboy/calibration/'
data_dir = '/home/tidyboy/ARAICodes/catkin_ws/src/robot_cal_tools/rct_examples/data/target_6x7_0/'

filename = 'ext_cal_pic'
cam = cam_tool(args.cam_type)

robot = urx.Robot("192.168.1.109")

pic_id = 0
while(True):
    cam.capture(filename + '.jpg')
    cam.show(filename + '.jpg')

    speed = [0, 0, 0, 0, 0, 0]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            break
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                speed[0] = -0.1
            if event.key == pygame.K_RIGHT:
                speed[0] = 0.1
            if event.key == pygame.K_DOWN:
                speed[1] = -0.1
            if event.key == pygame.K_UP:
                speed[1] = 0.1
            if event.key == pygame.K_s:
                speed[2] = -0.1
            if event.key == pygame.K_w:
                speed[2] = 0.1
            if event.key == pygame.K_q:
                speed[3] = -0.1
            if event.key == pygame.K_e:
                speed[3] = 0.1
            if event.key == pygame.K_h:
                speed[4] = -0.1
            if event.key == pygame.K_k:
                speed[4] = 0.1
            if event.key == pygame.K_y:
                speed[5] = -0.1
            if event.key == pygame.K_i:
                speed[5] = 0.1
            if event.key == pygame.K_SPACE:
                cam.capture(str(pic_id) + '.jpg')
                pos = robot.get_pose()
                pos = np.array(pos.array)
                q = Quaternion(matrix = pos[0:3, 0:3])
                v = pos[:, 3]
                with open(str(pic_id) + '.yaml', 'w') as f:
                    f.write('x: %f\ny: %f\nz: %f\nqx: %f\nqy: %f\nqz: %f\nqw: %f'\
                            %(v[0], v[1], v[2], q[1], q[2], q[3], q[0]))

                os.rename(current_dir + str(pic_id) + '.jpg',\
                        data_dir + '/images/' + str(pic_id) + '.jpg')
                os.rename(current_dir + str(pic_id) + '.yaml',\
                        data_dir + '/poses/' + str(pic_id) + '.yaml')
                pic_id += 1
                
        robot.speedl(speed, acc=0.1, min_time=2)

robot.close()

