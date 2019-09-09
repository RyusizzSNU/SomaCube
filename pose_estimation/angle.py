import cv2
import numpy as np
import math

def ImageSobel():

    sobelx = cv2.Sobel(thr3, -1, 1, 0, ksize=3)
    sobely = cv2.Sobel(thr3, -1, 0, 1, ksize=3)

new_file = open('angle', 'a')
img = cv2.imread('block1_0000.jpg', cv2.IMREAD_GRAYSCALE)
img_color = cv2.imread('block1_0000.jpg', cv2.IMREAD_COLOR)
ret, origin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
i = 0
tem = ("template/tem%d.jpg" %i)


thr3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 351, 40)

contours, hierarchy = cv2.findContours(thr3, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
blur = cv2.GaussianBlur(thr3, (3, 3), 0)


for cnt in contours:
    area = cv2.contourArea(cnt)
    print(area)
    if(area >2000) :


        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 1)
        print(x)
        print(y)
        print(x+w)
        print(y+h)
        cropped = img_color[y-30 : y+h+30 , x-30 : x+w+30]

        img = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)




        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img_color, [box], 0, (0, 0, 255), 1)
        a = box[3][0] - box[0][0]
        b = box[3][1] - box[0][1]
        print(box)
        #cv2.circle(img_color, (cx, cy), 5, (0, 0, 255), 1)

        z = b / a
        angle = math.atan(z)
        s = np.rad2deg(angle)
        if (s<0) :
            s += 90
            print("angle : %f" % s)
            new_file.write("\nangle : %f" % s)
            new_file.close()

        else :
            print("angle : %f" % s)
            new_file.write("\nangle : %f" % s)
            new_file.close()


#cv2.imshow("result", img)

#cv2.waitKey(0)


cv2.imshow("result", img_color)


cv2.waitKey(0)

ImageSobel()

cv2.destroyAllWindows()

