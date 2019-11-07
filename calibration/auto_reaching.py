import urx
import cv2
import math
import numpy as np
import utils

from PIL import Image, ImageFile, ImageFilter

import robot_control
from cam_tool import cam_tool

import grippoint

size = 0.024
H0 = 0.20
H1 = H0 + size + 0.005
H2 = H1 + size

robot = robot_control.Robot('192.168.1.109')
cam = cam_tool('sony')

cam.capture('view.jpg')

grip = grippoint.find_grip_points('view.jpg') # 7 * 2
C2B = utils.load_static_C2B()
C2I = utils.load_static_C2I()
print grip, C2B, C2I

poses = []
for p_I in grip:
    p = utils.static_image_to_base(p_I, C2I, C2B)
    poses.append(p)
    print p

o_x = poses[4][0]
o_y = poses[4][1]

robot.move_b([poses[3][0], poses[3][1], H2])
robot.release()
robot.move_b([poses[3][0], poses[3][1], H0])
robot.grip()
robot.move_b([poses[3][0], poses[3][1], H2])
robot.move_b([o_x + size, o_y - size, H2])
robot.move_b([o_x + size, o_y - size, H1])
robot.release()
robot.move_b([o_x + size, o_y - size, H2])

robot.move_b([poses[2][0], poses[2][1], H2])
robot.move_b([poses[2][0], poses[2][1], H0])
robot.grip()
robot.move_b([poses[2][0], poses[2][1], H2])
robot.move_b([o_x + size*3, o_y, H2])
robot.move_b([o_x + size*3, o_y, H1])
robot.release()
robot.move_b([o_x + size*3, o_y, H2])

temp_pos = np.array([poses[0][0] - size, poses[0][1] + size, 0])
robot.move_b(temp_pos + [0, 0, H2])
robot.move_b(temp_pos + [0, 0, H0])
robot.grip()
robot.move_b(temp_pos + [0, 0, H2])
robot.move_b([o_x, o_y - size, H2])
robot.move_b([o_x, o_y - size, H1])
robot.release()
robot.move_b([o_x, o_y - size, H2])

robot.move_b([poses[6][0], poses[6][1], H2])
robot.move_b([poses[6][0], poses[6][1], H0])
robot.grip()
robot.move_b([poses[6][0], poses[6][1], H2])
robot.move_b([o_x + size*4, o_y - size*2, H2])
robot.move_b([o_x + size*4, o_y - size*2, H0])
robot.release()
robot.move_b([o_x + size*4, o_y - size*2, H2])

robot.move_b([poses[1][0], poses[1][1], H2])
robot.move_b([poses[1][0], poses[1][1], H1])
robot.grip()
robot.move_b([poses[1][0], poses[1][1], H2])
robot.move_b([o_x + size*4, o_y - size*3, H2])
robot.move_b([o_x + size*4, o_y - size*3, H1])
robot.release()
robot.move_b([o_x + size*4, o_y - size*3, H2])

robot.move_b([poses[5][0], poses[5][1], H2])
robot.move_b([poses[5][0], poses[5][1], H0])
robot.grip()
robot.move_b([poses[5][0], poses[5][1], H2])
robot.move_b([o_x + size*1.5, o_y - size*3, H2])
robot.move_b([o_x + size*1.5, o_y - size*3, H1])
robot.release()
robot.move_b([o_x + size*1.5, o_y - size*3, H2])

