import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper
import numpy as np
import utils
from pyquaternion import Quaternion
from time import sleep
import datetime

class Robot:
    def __init__(self, ip):
        self.robot = urx.Robot('192.168.1.109')
        self.gripper = Robotiq_Two_Finger_Gripper(self.robot)
        
        self.initial_position = np.array(self.robot.get_pose().array) # 4 * 4

    def set_coordinates(self, center, size, H0):
        self.center = center
        self.size = size
        self.H0 = H0

    def l2b(self, x):
        # lattice coordinates to base coordinates
        if x[2] == 0:
            h = self.H0
        if x[2] == 1:
            h = self.H0 + self.size + 0.005
        if x[2] == 2:
            h = self.H0 + self.size * 2 + 0.005

        return self.center + np.array([float(x[0]) * self.size, float(x[1]) * self.size, h])

    def l2rigid(self, x, rot):
        # lattic coordinates to rigid matrix
        return utils.rot_trans_to_rigid(rot, self.l2b(x))

    def moveto(self, x):
        #rot1 = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
        #rot2 = Quaternion(axis=[0., 0., 1.], degrees=48)
        #rot = np.matmul(rot1, rot2)
        print 'move starts at %s'%datetime.datetime.now()
        rot = np.array([[0.7313537, 0.68199836, 0.], [0.68199836, -0.7313537, 0.], [0., 0., -1.]])
        self.robot.movex('movel', self.l2rigid(x, rot), acc=0.5, vel=0.05)
        print 'move ends at %s'%datetime.datetime.now()


    def grip(self):
        self.gripper.gripper_action(255)
        #self.gripper.close_gripper()
        sleep(2)

    def release(self):
        self.gripper.gripper_action(96)
        #self.gripper.open_gripper()
        sleep(2)

    def close(self):
        self.robot.close()
