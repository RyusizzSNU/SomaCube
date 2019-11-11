import argparse
import os.path as osp
import piggyphoto, pygame
import os
import time
#python2

def quit_pressed():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                return True
    return False

def key_pressed():
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_PERIOD:
                return True
    return False

def show(main_surface, file_name):
    picture = pygame.image.load(file_name)
    main_surface.blit(picture, (0, 0))
    pygame.display.flip()

def screenshot(cam, file_name):
    cam.capture_preview(file_name)


def main():
    prs = argparse.ArgumentParser()
    prs.add_argument("-p", "--prefix_name", type=str, default='')
    prs.add_argument("-s", "--suffix_index", type=int, default=-1)
    prs.add_argument("-n", "--number_of_data", type=int, default=1)

    args = prs.parse_args()

    pygame.init()


    prefix = args.prefix_name
    if args.prefix_name == '':
        assert args.suffix_index >= 0


    dir_path = 'dataset'

    n = args.number_of_data
    #screen = pygame.display.set_mode((100, 100))

    cam = piggyphoto.camera()
    cam.leave_locked()
    cam.capture_preview('preview.jpg')
    picture = pygame.image.load("preview.jpg")
    pygame.display.set_mode(picture.get_size())
    main_surface = pygame.display.get_surface()
    os.remove('preview.jpg')
    if n <= 1:
        file_name = osp.join(dir_path, args.prefix_name)
        while not quit_pressed():
            cam.capture_preview(file_name)
            show(main_surface, file_name)
            #pygame.display.flip()
            #print(file_name)
    else:
        i = 0
        done = False
        while i < n and not done:
            key_pressed = False
            file_name = osp.join(dir_path, '{}_{:04d}.jpg'.format(args.prefix_name, i))
            while not(key_pressed):
                cam.capture_preview(file_name)
                show(main_surface, file_name)
                #pygame.display.flip()
                #print(file_name)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        done = True
                        key_pressed = True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            done = True
                            key_pressed = True
                        if event.key == pygame.K_PERIOD:
                            i += 1
                            key_pressed = True
            if done:
                os.remove(file_name)



if __name__ == '__main__':
    main()
