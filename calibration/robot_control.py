import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

class Robot:
    def __init__(self, ip):
        self.robot = urx.Robot('192.168.1.109')
        self.gripper = Robotiq_Two_Finger_Gripper(robot)

