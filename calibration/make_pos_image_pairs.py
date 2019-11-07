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
parser.add_argument('--data_dir', type=str, default='ext_data/')
parser.add_argument('--cam', type=str, help="cameras to make data pairs with. one of ['static', 'wrist', 'both']")
parser.add_argument('--static_cam_model', type=str, default='sony')
parser.add_argument('--wrist_cam_model', type=str, default='rs')
parser.add_argument('--auto', type=int, default = 0)
parser.add_argument('--num_pic', type=int, default=50, help="number of pictures to take")
parser.add_argument('--period', type=float, default=0.5, help="period between which pictures would be taken")


args = parser.parse_args()

utils.create_muldir(args.data_dir, args.data_dir + '/images', args.data_dir + '/poses',\
        args.data_dir + '/images/' + args.cam, args.data_dir + '/poses/' + args.cam)

#data_dir = '/home/tidyboy/ARAICodes/catkin_ws/src/robot_cal_tools/rct_examples/data/target_6x7_0/'

static_on = (args.cam == 'static' or args.cam == 'both')
wrist_on = (args.cam == 'wrist' or args.cam == 'both')

if static_on:
    static_cam = cam_tool(args.static_cam_model)
if wrist_on:
    wrist_cam = cam_tool(args.wrist_cam_model)

robot = urx.Robot("192.168.1.109")

pic_id = 0
initialized = False

nexttime = datetime.datetime.now() + datetime.timedelta(seconds = 5)

def wait():
    while True:
        if args.auto != 0:
            t = datetime.datetime.now()
            if t > nexttime:
                return
        else:
            return

def record():
    global pic_id
    if static_on:
        static_cam.capture(args.data_dir + '/images/%s/static_%d.jpg'%(args.cam, pic_id))
    if wrist_on:
        wrist_cam.capture(args.data_dir + '/images/%s/wrist_%d.jpg'%(args.cam, pic_id))
    pos = np.array(robot.get_pose().array)
    with open(args.data_dir + '/poses/%s/%d.yaml'%(args.cam, pic_id), 'w') as f:
        pickle.dump(pos, f)
    pic_id += 1

while not utils.quit_pressed():
    wait()
    if static_on:
        static_cam.capture('static_view.jpg')
    if wrist_on:
        wrist_cam.capture('wrist_view.jpg')
    
    view_fname = ''
    if args.cam == 'both':
        view_fname = 'concat_view.jpg'
        concat_view = utils.concat_images_in_dir(['static_view.jpg', 'wrist_view.jpg'], view_fname)
    elif args.cam == 'static':
        view_fname = 'static_view.jpg'
    elif args.cam == 'wrist':
        view_fname = 'wrist_view.jpg'

    if not initialized:
        pygame.display.set_mode(pygame.image.load(view_fname).get_size())
        initialized = True

    utils.show(view_fname)
    if args.auto == 0: 
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
                    record()
                    
        robot.speedl(speed, acc=0.1, min_time=2)
    else:
        record()
        if(pic_id >= args.num_pic):
            break
        nexttime = datetime.datetime.now() + datetime.timedelta(seconds = args.period)

robot.close()

