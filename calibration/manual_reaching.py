import numpy as np
import pickle
import utils
import robot_control

center = np.array([-0.4531, -0.20, 0.])
H0 = 0.20
size = 0.024

robot = robot_control.Robot('192.168.1.109')
robot.set_coordinates(center, size, H0)

robot.move_l([0, -1, 2])
robot.release()
robot.move_l([0, -1, 0])
robot.grip()
robot.move_l([-3, -1, 0])
robot.release()
robot.move_l([-3, -1, 2])

robot.move_l([5, 1, 2])
robot.move_l([5, 1, 1])
robot.grip()
robot.move_l([0, 1, 1])
robot.release()
robot.move_l([0, 1, 2])

robot.move_l([-4, -3, 2])
robot.move_l([-4, -3, 0])
robot.grip()
robot.move_l([-4, -3, 1])
robot.move_l([-4, -1, 1])
robot.release()
robot.move_l([-4, -1, 2])

robot.move_l([1, -5, 2])
robot.move_l([1, -5, 0])
robot.grip()
robot.move_l([0, -2, 0])
robot.release()
robot.move_l([0, -2, 2])

robot.move_l([5, -4, 2])
robot.move_l([5, -4, 1])
robot.grip()
robot.move_l([0, -4, 1])
robot.move_l([0, -3, 1])
robot.release()
robot.move_l([0, -3, 2])

robot.move_l([-3, -7, 2])
robot.move_l([-3, -7, 0])
robot.grip()
robot.move_l([-3, -7, 1])
robot.move_l([-2, -2, 1])
robot.release()
robot.move_l([-2, -2, 2])

robot.close()
