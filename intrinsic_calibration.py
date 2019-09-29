import piggyphoto
import pygame
import os
import time
import datetime
import argparse
import numpy as np
import cv2
import glob
import pickle

def show(file):
    picture = pygame.image.load(file)
    main_surface.blit(picture, (0, 0))
    pygame.display.flip()

C = piggyphoto.camera()
C.leave_locked()
C.capture_preview('preview.jpg')

picture = pygame.image.load('preview.jpg')
pygame.display.set_mode(picture.get_size())
main_surface = pygame.display.get_surface()


period = 0.5
starttime = datetime.datetime.now()
nexttime = starttime + datetime.timedelta(seconds = period)

t_list = []

for i in range(50):
    while True:
        t = datetime.datetime.now()
        if t > nexttime:
            t_list.append(t)
            break

    filename = 'preview' + str(i) + '.jpg'
    C.capture_preview(filename)
    show(filename)
    nexttime = nexttime + datetime.timedelta(seconds = period)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

m = 8
n = 6

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((m*n,3), np.float32)
objp[:,:2] = np.mgrid[0:m,0:n].T.reshape(-1,2)
objp = objp * 2.1

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (m,n),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11), (-1,-1), criteria)
        imgpoints.append(corners)
        
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (m,n), corners,ret)
print("success :", len(objpoints))

cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print(ret)
print(mtx)
print(dist)
print(rvecs)
print(tvecs)

pickle.dump(np.array(ret), open('ret.pkl', 'wb'))
pickle.dump(np.array(mtx), open('mtx.pkl', 'wb'))
pickle.dump(np.array(dist), open('dist.pkl', 'wb'))
pickle.dump(np.array(rvecs), open('rvecs.pkl', 'wb'))
pickle.dump(np.array(tvecs), open('tvecs.pkl', 'wb'))

