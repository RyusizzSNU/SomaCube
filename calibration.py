import piggyphoto, pygame
import os
import time
import datetime
import argparse
import cv2
import numpy as np

def quit_pressed():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False

def show(file):
    picture = pygame.image.load(file)
    main_surface.blit(picture, (0, 0))
    pygame.display.flip()

C = piggyphoto.camera()
C.leave_locked()
C.capture_preview('preview.jpg')

picture = pygame.image.load("preview.jpg")
pygame.display.set_mode(picture.get_size())
main_surface = pygame.display.get_surface()


w = 3
h = 3
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((w*h,3), np.float32)
objp[:,:2] = np.mgrid[0:w,0:h].T.reshape(-1, 2)

objpoints = []
imgpoints = []

while not quit_pressed():
    filename = 'preview.jpg'
    C.capture_preview(filename)

    img = cv2.imread(filename)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray_img, (w, h), None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        img = cv2.drawChessboardCorners(img, (w, h), corners2, ret)
        cv2.imshow('img', img)
        
    else:
        print('could not find chessboard corners')
    show(filename)
    input()
