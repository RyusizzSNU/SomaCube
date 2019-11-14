import piggyphoto
import pyrealsense2 as rs
import os
import cv2
import pygame
import numpy as np

class cam_tool:
    def __init__(self, cam_type):
        self.cam_type = cam_type
        print 'Initializing %s...'%cam_type
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

    def capture(self, path=None, depth=False):
        tempname = 'temp.jpg'
        if path == tempname:
            tempname = 'temp2.jpg'

        if self.cam_type == 'sony':
            if path == None:
                return self.C.capture_preview()
            else:
                self.C.capture_preview(tempname)
        elif self.cam_type == 'rs':
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return
            if depth:
                depth_frame = frames.get_depth_frame()
                if not depth_frame:
                    return

            color_image = np.asanyarray(color_frame.get_data())
            if depth:
                depth_image = np.asanyarray(depth_frame.get_data())
                #depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
                depth_image = np.tile(np.expand_dims(cv2.convertScaleAbs(depth_image, alpha=0.03), 2), [1, 1, 3])
                image = np.hstack((color_image, depth_image))

            else:
                image = color_image

            if path == None:
                return image
            cv2.imwrite(tempname, image)

        os.rename(tempname, path)

    def show_image(self, picture):
        self.main_surface.blit(picture, (0, 0))
        pygame.display.flip()

    def show(self, file):
        picture = pygame.image.load(file)
        self.show_image(picture)

    def exit(self):
        if self.cam_type == 'rs':
            self.pipeline.stop()
