import pygame
import argparse
from cam_tool import cam_tool

parser = argparse.ArgumentParser()
parser.add_argument('--cam_type', type=str, help="Camera type. one of realsense and sony")
args = parser.parse_args()


def quit_pressed():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

cam = cam_tool(args.cam_type)

while not quit_pressed():
    filename = 'preview.jpg'
    cam.capture(filename)
    cam.show(filename)

