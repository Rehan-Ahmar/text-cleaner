import os
import math
import cv2
import numpy as np

from deskew import Deskew

def process1(img):
    gray = cv2.bitwise_not(img)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print("Process1 angle: {:.3f}".format(angle))
    return rotated
    
def compute_skew(src):
    #load in grayscale:
    #src = cv2.imread(file_name,0)
    height, width = src.shape[0:2]
    #invert the colors of our image:
    cv2.bitwise_not(src, src)
    #Hough transform:
    minLineLength = width/2.0
    maxLineGap = 20
    lines = cv2.HoughLinesP(src,1,np.pi/180,100,minLineLength,maxLineGap)
    #calculate the angle between each line and the horizontal line:
    angle = 0.0
    nb_lines = len(lines)
    for line in lines:
        angle += math.atan2(line[0][3]*1.0 - line[0][1]*1.0,line[0][2]*1.0 - line[0][0]*1.0);
    angle /= nb_lines*1.0
    return angle* 180.0 / np.pi

def deskew_image(img, angle):
    #invert the colors of our image:
    cv2.bitwise_not(img, img)
    #compute the minimum bounding box:
    non_zero_pixels = cv2.findNonZero(img)
    center, wh, theta = cv2.minAreaRect(non_zero_pixels)
    root_mat = cv2.getRotationMatrix2D(center, angle, 1)
    rows, cols = img.shape
    rotated = cv2.warpAffine(img, root_mat, (cols, rows), flags=cv2.INTER_CUBIC)
    #Border removing:
    sizex = np.int0(wh[0])
    sizey = np.int0(wh[1])
    print(theta)
    if theta > -45 :
        temp = sizex
        sizex= sizey
        sizey= temp
    return cv2.getRectSubPix(rotated, (sizey,sizex), center)

def process_image(img_path, output_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    print('Skew correcting image: ', img_path)
    processed1 = process1(img)
    cv2.imwrite(os.path.join(output_path, 'processed1.png'), processed1)
    
    try:
        angle = compute_skew(img)
        print("deskew angle: {:.3f}".format(angle))
        out = deskew_image(img, angle)
        cv2.imwrite(os.path.join(output_path, 'processed2.png'), out)
    except:
        print('Failed')
    
    d = Deskew(input_file=img_path, display_image=False, output_file=os.path.join(output_path, 'corrected.png'), r_angle=0)
    d.run()

if __name__ == '__main__':
    start()