import pygame
import datetime
import os
import urx
import pickle
import numpy as np
import argparse
import utils
from cam_tool import cam_tool
from pyquaternion import Quaternion

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/home/tidyboy/calibration/ext_data/')
args = parser.parse_args()

utils.create_muldir(args.data_dir, args.data_dir + '/images', args.data_dir + '/poses')
#data_dir = '/home/tidyboy/ARAICodes/catkin_ws/src/robot_cal_tools/rct_examples/data/target_6x7_0/'


static_cam = cam_tool('sony')
wrist_cam = cam_tool('rs')

robot = urx.Robot("192.168.1.109")

pic_id = 0
initialized = False

starttime = datetime.datetime.now()
nexttime = starttime + datetime.timedelta(seconds = 5)
'''
for i in range(10):
    while True:
        t = datetime.datetime.now()
        if t > nexttime:
            break

    static_cam.capture('static_view.jpg')
    wrist_cam.capture('wrist_view.jpg')
   
    concat_view = utils.concat_images_in_dir(['static_view.jpg', 'wrist_view.jpg'], 'concat_view.jpg')

    if not initialized:
        pygame.display.set_mode(pygame.image.load('concat_view.jpg').get_size())
        initialized = True
    wrist_cam.show('concat_view.jpg')

    static_cam.capture(args.data_dir + '/images/static_%d.jpg'%pic_id)
    wrist_cam.capture(args.data_dir + '/images/wrist_%d.jpg'%pic_id)
    pos = robot.get_pose()
    pos = np.array(pos.array)
    with open(args.data_dir + '/poses/%d.yaml'%pic_id, 'w') as f:
        pickle.dump(pos, f)
    pic_id += 1

    nexttime = nexttime + datetime.timedelta(seconds = 0.5)
'''
while True:
    static_cam.capture('static_view.jpg')
    wrist_cam.capture('wrist_view.jpg')
   
    concat_view = utils.concat_images_in_dir(['static_view.jpg', 'wrist_view.jpg'], 'concat_view.jpg')

    if not initialized:
        pygame.display.set_mode(pygame.image.load('concat_view.jpg').get_size())
        initialized = True
    wrist_cam.show('concat_view.jpg')

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
                static_cam.capture(args.data_dir + '/images/static_%d.jpg'%pic_id)
                wrist_cam.capture(args.data_dir + '/images/wrist_%d.jpg'%pic_id)
                pos = robot.get_pose()
                pos = np.array(pos.array)
                with open(args.data_dir + '/poses/%d.yaml'%pic_id, 'w') as f:
                    pickle.dump(pos, f)
                pic_id += 1
                
    robot.speedl(speed, acc=0.1, min_time=2)

robot.close()

