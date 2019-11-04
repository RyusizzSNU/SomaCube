import argparse
import utils
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--cam_model', type=str, help="Camera model. one of rs(realsense) and sony")
args = parser.parse_args()

cam = cam_tool(args.cam_model)

while not utils.quit_pressed():
    filename = 'preview.jpg'
    cam.capture(filename)
    utils.show(filename)
