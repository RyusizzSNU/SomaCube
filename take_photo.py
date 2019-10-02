import piggyphoto, pygame
import os
import time
import datetime
import argparse

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

i = 0
while not quit_pressed():
    filename = 'picture'
    C.capture_preview(filename + '.jpg')
    show(filename + '.jpg')

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            C.capture_preview(filename + str(i) + '.jpg')
            i += 1

