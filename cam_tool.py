import piggyphoto
import pyrealsense2 as rs
import os
import cv2
import pygame
import numpy as np

class cam_tool:
    def __init__(self, cam_type):
        self.cam_type = cam_type
        if cam_type == 'sony':
            self.C = piggyphoto.camera()
            self.C.leave_locked()
            self.capture('temp.jpg')
        elif cam_type == 'rs':
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(config)
            self.capture('temp.jpg')
        
        picture = pygame.image.load('temp.jpg')
        pygame.display.set_mode(picture.get_size())
        self.main_surface = pygame.display.get_surface()
        
        os.remove('temp.jpg')

    def capture(self, fn):
        if self.cam_type == 'sony':
            self.C.capture_preview(fn)
        elif self.cam_type == 'rs':
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return
            im = np.asanyarray(color_frame.get_data())
            cv2.imwrite(fn, im)

    def show(self, file):
        picture = pygame.image.load(file)
        self.main_surface.blit(picture, (0, 0))
        pygame.display.flip()

    def exit(self):
        if self.cam_type == 'rs':
            self.pipeline.stop()
