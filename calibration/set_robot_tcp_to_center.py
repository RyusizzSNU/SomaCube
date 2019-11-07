import urx
import utils
import numpy as np
robot = urx.Robot('192.168.1.109')
rot = np.array([[0., 1., 0.], [1., 0., 0.], [0., 0., -1.]])
trans = np.array([-0.45165, -0.1984, 0.20])
r = utils.rot_trans_to_rigid(rot, trans, vel=0.05)
robot.movex('movel', r)
robot.close()
