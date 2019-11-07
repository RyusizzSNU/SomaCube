import cv2
import numpy as np
import math

import matplotlib.pyplot as plt

from PIL import Image, ImageFile, ImageFilter
from torchvision import transforms
import numpy as np
import torch
import random
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import scipy.misc

from modeling.deeplab import *

ImageFile.LOAD_TRUNCATED_IMAGES = True

class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        
        
        
      

        return img
    
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample
        img = np.array(img).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return img

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))

        img = torch.from_numpy(img).float()

        return img
    
def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb
    
def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])

class DeeplabModule():
    def __init__(self):
        self.composed_transforms = transforms.Compose([
                                        FixScaleCrop(crop_size=1024),
                                        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                        ToTensor()
                                    ])
        self.model = DeepLab(num_classes=8,
                        backbone="resnet",
                        output_stride=16,
                        sync_bn=None,
                        freeze_bn=False)
        
        self.display_func = (lambda x:\
                                decode_seg_map_sequence(torch.max(x, 1)[1].detach().cpu().numpy(),\
                                                        dataset="coco")[0])
        self.model_init()
        self.load_checkpoint()
        
    def model_init(self):
        self.model.eval()
        self.model.cuda()
        
    def load_checkpoint(self):
        checkpoint = torch.load("model_best.pth.tar")
        self.model.load_state_dict(checkpoint['state_dict'])
    
    def run(self, input_img_str):
        img = Image.open(input_img_str).convert('RGB')
        img = self.composed_transforms(img).cuda()
        
        output = self.model(img.view(1, 3, 1024, 1024))
        
        #result = output
        result = self.display_func(output)
        
        return result
    
def restoreOrigin(sample):
    x_pad = np.zeros((3, 1024, 259)) # # of classes = 8, original size = (1542, 1024)
    
    img = sample.squeeze().detach().cpu().numpy()
    rest_img = np.concatenate((x_pad, img, x_pad), axis=2) #* 255 # denormalize
    print(rest_img.dtype)
    rest_img = np.transpose((rest_img * 255).astype(np.uint8), (1, 2, 0))
    print(rest_img.dtype, rest_img.shape)
#     k = scipy.misc.imresize(rest_img, (680, 1024), interp='bilinear')
    k = np.array(Image.fromarray(rest_img).resize((1024, 680)))
    
#     return np.transpose(k, (0, 2, 1))
    return k
    
def pose1(cx,cy,angle) : #block4_2
    
     for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if((area >2000)&(area<100400)) :

            pr=1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/4
            a = r
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
def pose2(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        
        if((area >2000)&(area<100400)) :

            pr=3
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = r
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
           
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
def pose3(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr=1.5
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = ((r/2)**2+r**2)**0.5
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)+pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point

def pose4(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr=1.5
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = ((r/2)**2+r**2)**0.5
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)-pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
def pose5(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr=1.5
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = ((r/2)**2+r**2)**0.5
           
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
          
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)+pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
def pose6(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr = 1
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
            
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/2
            a = r*((2)**0.5)/2
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)+pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
def pose7(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr=1.5
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
           
        
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = ((r/2)**2+r**2)**0.5
           
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)+pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
def pose8(cx,cy,angle) : #block4_2
     for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<100400)) :

            pr=1.5
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 255), 1)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img_color, [box], 0, (255, 255, 255), 1)
            
       
            cx= int(cx)
            cy =int(cy)
            center = (cx,cy)
            
            d = ((area*pr)**0.5)
            
            r= d/3
            a = ((r/2)**2+r**2)**0.5
            
        

        
            x =cx-(r)
            y =cy+(r/2)
            pa=math.atan(1/2)
            
    
            standardpoint=(x,y)
            
            ag = math.radians(-angle)+pa
            ##여기에 angle 값 넣으면됨
            ########
        
        
        
            rx =cx+a*math.cos(ag)
            ry =cy+a*math.sin(ag)
            rx =int(rx)
            ry =int(ry)
            rot_point = (rx,ry)
            
            return center,rot_point
            
lable_to_color = {0:[128, 0, 0],
                  1:[0, 128, 0],
                  2:[128, 128, 0],
                  3:[0, 0, 128],
                  4:[128, 0, 128],
                  5:[0, 128, 128],
                  6:[128, 128, 128]
                  }
def coco_thresholding(img, lbl):
    color = lable_to_color[lbl]
    mask = np.all(img == color, axis=-1).astype(np.uint8)*255
    return mask
img = Image.open("val_scatter_0008.jpg").convert('RGB')
plt.imshow(img)

deeplab = DeeplabModule()
out = deeplab.run("val_scatter_0008.jpg")
# plt.imshow(np.transpose(out, (1,2,0)))
img_mask = restoreOrigin(out)
plt.imshow(img_mask)
num = (7,)
print(num + img_mask.shape)


# mask = np.all(res == (128, 128, 128), axis=-1).astype(np.uint8)*255
blocks_mask = np.zeros(shape=((7,) + img_mask.shape[:-1]), dtype=np.uint8)
blocks_contours = [None]*7
blocks_hierarchies = [None]*7
for i in range(7):
    blocks_mask[i, :] = mask = coco_thresholding(img_mask, i)
    plt.imshow(mask)
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    blocks_contours[i] = contours
    blocks_hierarchies[i] = hierarchy
    
    
img = cv2.imread('val_scatter_0005.jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('val_scatter_0005.jpg', cv2.IMREAD_COLOR)
print(img_color.shape)

thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 351, 40)

contours, hierarchy = cv2.findContours(thr3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


def center(contours):
    center = []
    ag = []
    area = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if((area >2000)&(area<15000)) :

            x, y, w, h = cv2.boundingRect(cnt)
            #cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
            cx = (x*2+w)/2
            cy = (y*2+h)/2
            cx= int(cx)
            cy =int(cy)
            ct = area,(cx,cy)
            center.append(ct)
#             area.append(aread)

            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            #cv2.drawContours(img_color, [box], 0, (0, 0, 255), 1)
            a = box[3][0] - box[0][0]
            b = box[3][1] - box[0][1]
            #print(box)

            z = b / a
            angle = math.atan(z)
            s = np.rad2deg(angle)

            if (s<0) :
                s *=-1
                
            d = round(s,1)
            ag.append(d)
            
    return center,ag

blocks_cts = [None]*7
blocks_angles = [None]*7
img_centerp = img_mask.copy()
for i in range(7):
    ct, angle = center(blocks_contours[i])
    blocks_cts[i] = ct
    blocks_angles[i] = angle
#     print(ct)
#     print(angle)
    cv2.circle(img_centerp, ct[0][1], 0, (255, 255, 255), 5)
    
plt.imshow(img_centerp) 


lable_to_pose = {0: [1, 6],
                 1: [6],
                 2: [6],
                 3: [2, 3, 4],
                 4: [6],
                 5: [2, 5, 8],
                 6: [2, 7]}
def mk_ps_tp(angle, lable) :
#     blocks_pose_lists = [None]*7
#     for i in range(7):
#     for st in blocks_angles :
    poses = lable_to_pose[lable]
    template_lists = []
    for p in poses:
        templates = []
        for i in range (0,4) :
            d = angle+i*90
            d = str(d)
            templates.append("resize/pose{}/pose{}_{}.jpg".format(p, p, d))
        template_lists.append(templates)
    
    return template_lists

blocks_template_lists = [None]*7
for i in range(7):
    blocks_template_lists[i] = mk_ps_tp(blocks_angles[i][0], i)
print(blocks_template_lists)


# methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR', 'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
# methods = ['cv2.TM_SQDIFF', 'cv2.TM_CCORR_NORMED']
methods = ['cv2.TM_CCOEFF_NORMED']

for meth in methods:
    method = eval(meth)
    blocks_result = [None]*7
    blocks_pose = [None]*7
    blocks_p1 = [None]*7
    img_boxed = img_centerp.copy()
    for i in range(7) :
        template_lists = blocks_template_lists[i]
        template_lists = sum(template_lists, [])
        max_f = 0
        max_index = 0
        for j, temp in enumerate(template_lists):
            temp_img = cv2.imread(temp, cv2.IMREAD_GRAYSCALE) #TODO: 템플릿 크기조절 필요
            temp_img = np.array(Image.fromarray(temp_img).resize((194,194)))
            
            w,h = temp_img.shape[::-1]
            print(i, j, w, h)
    #         method = eval('cv2.TM_CCOEFF')

            res = cv2.matchTemplate(blocks_mask[i], temp_img, method)
            min_val,max_val,min_loc,max_loc = cv2.minMaxLoc(res)
                    #print(np.min(res), np.max(res))
#             print(max_val, np.max(res))
            blocks_result[i] = s = max_val

            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                    top_left = min_loc
            else:
                    top_left = max_loc
            
            if max_f < s:
                max_f = s
                max_index = j
                bottom_right = (top_left[0]+h, top_left[1]+w)
                blocks_p1[i] = (top_left, bottom_right)
                blocks_pose[i] = temp
#         print(i, max_index)

        cv2.rectangle(img_boxed, blocks_p1[i][0], blocks_p1[i][1], lable_to_color[i], 5)
        print(blocks_pose[i])
        temp_img = cv2.imread(blocks_pose[i], cv2.IMREAD_GRAYSCALE)
        temp_img = np.array(Image.fromarray(temp_img).resize((194,194)))

        plt.imshow(temp_img), plt.show()
        
        
#         plt.imshow(img_boxed)
    print(blocks_p1)

    plt.subplots(1, 3, figsize=(20, 10))
    plt.subplot(121), plt.title(meth), plt.imshow(res, cmap='gray'), plt.yticks([]), plt.xticks([])
    plt.subplot(122), plt.imshow(img_boxed)
#     plt.subplot(133), plt.imshow(cv2.imread(template_lists[max_index], cv2.IMREAD_GRAYSCALE))
    plt.show()
print(blocks_p1)


layer = np.ones_like(img_boxed)*128
img_overlayed = img_boxed.copy()
pl =np.zeros(7)
b= np.zeros(7)
for i in range(7) :
#     template_lists = blocks_template_lists[i]
#     max_index = blocks_pose[i]
#     template_lists = sum(template_lists, [])
    temp = blocks_pose[i]
    a = list(blocks_pose[i])
    p=a[17:18]
    pl[i]=int(p[0])
    agstr = a[19:-4]
    l = len(agstr)
    c=0
    if l == 5:
        b[i] = int(agstr[0])*100+int(agstr[1])*10+int(agstr[2])+int(agstr[4])*0.1
        c+=1
    elif l == 4 : 
        b[i] = int(agstr[0])*10+int(agstr[1])*1+int(agstr[3])*0.1
        c+=1
    else :
        b[i] = int(agstr[0])*1+int(agstr[2])*0.1
        c+=1
 
          
    temp_img = cv2.imread(temp, cv2.IMREAD_GRAYSCALE)
    temp_img = np.array(Image.fromarray(temp_img).resize((194, 194)))
#     print(temp_img.shape)
    layer[blocks_p1[i][0][1]:blocks_p1[i][1][1], blocks_p1[i][0][0]:blocks_p1[i][1][0], :] = np.tile(np.expand_dims(temp_img, axis=2), [1,1,3])
plt.figure(figsize=(15,12)), plt.imshow(layer), plt.show()




img_overlayed = cv2.addWeighted(img_overlayed, 0.5, layer, 0.5, 0)
grip = []
i=0
while i <7:
    img = cv2.imread(blocks_pose[i], cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(blocks_pose[i], cv2.IMREAD_COLOR)
    ret, origin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    cx = (blocks_p1[i][0][0]+blocks_p1[i][1][0])/2
    cy = (blocks_p1[i][0][1]+blocks_p1[i][1][1])/2
    if pl[i] == 1:
        center,rot_point= pose1(cx,cy,b[i])
        print(b[i])
        i+=1
        
    elif pl[i] == 2:
        center,rot_point= pose2(cx,cy,b[i])
        
        i+=1
    elif pl[i] == 3:
        center,rot_point= pose3(cx,cy,b[i])
        
        i+=1
    elif pl[i] == 4:
    
        center,rot_point= pose4(cx,cy,b[i])
        
        i+=1
    elif pl[i] == 5:
        center,rot_point= pose5(cx,cy,b[i])
       
        i+=1
    elif pl[i] == 6:
        center,rot_point= pose6(cx,cy,b[i])
        
        i+=1
    elif pl[i] == 7:
        center,rot_point= pose7(cx,cy,b[i])
        
        i+=1
    else:
        center,rot_point= pose8(cx,cy,b[i])
        i+=1
    cv2.circle(img_overlayed, rot_point, 1,(0,125, 255), 3)
    grip.append(rot_point)


print(grip)
plt.figure(figsize=(15,12)), plt.imshow(img_overlayed), plt.show()
