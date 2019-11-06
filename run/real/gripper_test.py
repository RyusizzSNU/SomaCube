import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

robot = urx.Robot('192.168.1.109')
gripper = Robotiq_Two_Finger_Gripper(robot)

gripper.gripper_action(96)
gripper.close_gripper()

