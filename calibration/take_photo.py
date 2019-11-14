import piggyphoto, pygame
import argparse
import utils
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--cam_model', type=str, help="Camera model")
parser.add_argument('--file_name', type=str, help="file name to save picture")
parser.add_argument('--depth', type=int, default=0, help="to use depth map")
args = parser.parse_args()

cam = cam_tool(args.cam_model)

i = 0
filename = args.file_name
while not utils.quit_pressed():
    cam.capture(filename + '.jpg', args.depth)
    utils.show(filename + '.jpg')

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            cam.capture(filename + '_' + str(i) + '.jpg', args.depth)
            i += 1

