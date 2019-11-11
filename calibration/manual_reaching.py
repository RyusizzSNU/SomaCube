import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
from pyquaternion import Quaternion
import numpy as np
import pickle
import utils

from time import sleep
import datetime

center = np.array([-0.4531, -0.20, 0.])
H1 = 0.30
H0 = 0.20
size = 0.024
H0_ = H0 + size + 0.005
G0 = 96
G1 = 255

robot = urx.Robot('192.168.1.109')
gripper = Robotiq_Two_Finger_Gripper(robot)

def l2b(x):
    # lattice corrdinates to base coordinates
    return center + np.array([float(x[0]) * size, float(x[1]) * size, float(x[2])])

def l2rigid(x, rot):
    #rot1 = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    #rot2 = Quaternion(axis=[0., 0., 1.], degrees=48)
    #rot = np.matmul(rot1, rot2)
    return utils.rot_trans_to_rigid(rot, l2b(x))

def moveto(x):
    #rot1 = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
    #rot2 = Quaternion(axis=[0., 0., 1.], degrees=48)
    #rot = np.matmul(rot1, rot2)
    print 'move starts at %s'%datetime.datetime.now()
    rot = np.array([[0.7313537, 0.68199836, 0.], [0.68199836, -0.7313537, 0.], [0., 0., -1.]])
    robot.movex('movel', l2rigid(x, rot), acc=0.5, vel=0.05)
    print 'move ends at %s'%datetime.datetime.now()
    

def grip():
    # gripper.gripper_action(255)
    gripper.close_gripper()
    sleep(2)

def release():
    # gripper.gripper_action(96)
    gripper.open_gripper()
    sleep(2)

moveto([0, -1, H1])
release()
moveto([0, -1, H0])
grip()
moveto([-3, -1, H0])
release()
moveto([-3, -1, H0])

moveto([5, 1, H1])
moveto([5, 1, H0_])
grip()
moveto([0, 1, H0_])
release()
moveto([0, 1, H1])

moveto([-4, -3, H1])
moveto([-4, -3, H0])
grip()
moveto([-4, -3, H0_])
moveto([-4, -1, H0_])
release()
moveto([-4, -1, H1])

moveto([1, -5, H1])
moveto([1, -5, H0])
grip()
moveto([0, -2, H0])
release()
moveto([0, -2, H1])

moveto([5, -4, H1])
moveto([5, -4, H0_])
grip()
moveto([0, -4, H0_])
moveto([0, -3, H0_])
release()
moveto([0, -3, H1])

moveto([-3, -7, H1])
moveto([-3, -7, H0])
grip()
moveto([-3, -7, H0_])
moveto([-2, -2, H0_])
release()
moveto([-2, -2, H1])

robot.close()
