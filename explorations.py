import os
import math
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(img):
    equ = cv2.equalizeHist(img)
    return equ
    
def adaptive_histogram_equalization(img):
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)
    
def process1(img):
    # adjust contrast
    img = cv2.multiply(img, 1.2)
    # create a kernel for the erode() function
    kernel = np.ones((1, 1), np.uint8)
    # erode() the image to bolden the text
    img = cv2.erode(img, kernel, iterations=10)
    return img
    
def process2(img):
    # Convert to gray
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply dilation and erosion to remove some noise
    #kernel = np.ones((2, 2), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.erode(img, kernel, iterations=1)
    
    # Apply blurring
    #img = cv.blur(img,(5,5))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.medianBlur(img, 5)
    #img = cv.bilateralFilter(img,9,75,75)
    
    # Apply threshold to get binary image
    #ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #adaptive_mean_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #adaptive_gaussian_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return img
    
    
def process3(gray):
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    #cls = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    #return cls
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    close = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    div = np.float32(gray)/(close)
    res = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    return res
    
def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def sharpen_image(img):
    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened
    
def skew_correction(img):
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(img)
 
    # threshold the image, setting all foreground pixels to 255 and all background pixels to 0
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # grab the (x, y) coordinates of all pixel values that are greater than zero, then use these coordinates to
    # compute a rotated bounding box that contains all coordinates
    coords = np.column_stack(np.where(thresh > 0))
    print(coords)
    angle = cv2.minAreaRect(coords)[-1]
 
    # the `cv2.minAreaRect` function returns values in the range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)
    # otherwise, just take the inverse of the angle to make
    # it positive
    else:
        angle = -angle
    # rotate the image to deskew it
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    print("angle: {:.3f}".format(angle))
    cv2.imwrite('rotated.png', rotated)
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

def deskew(img, angle):
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

def crop_image(img, border_width):
    cropped = img[border_width:img.shape[0]-border_width, border_width:img.shape[1]-border_width]
    return cropped
    
def process_image(args):
    img = cv2.imread(args.infile, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('in.png', img)
    equ = histogram_equalization(img)
    #cv2.imwrite('equalized.png', equ)
    adp = adaptive_histogram_equalization(img)
    #cv2.imwrite('adaptive_equalized.png', adp)
    
    #processed2_img = process2(img)
    #cv2.imwrite('processed2_img.png', processed2_img)
    
    #processed2_equ = process2(equ)
    #cv2.imwrite('processed2_equ.png', processed2_equ)
    
    #processed3 = remove_noise_and_smooth(img)
    #cv2.imwrite('remove_noise.png', processed3)
    
    #out = skew_correction(img)
    angle = compute_skew(img)
    print("angle: {:.3f}".format(angle))
    out = deskew(img, angle)
    cv2.imwrite(args.outfile, out)

def start():
    parser = argparse.ArgumentParser(description='Text Cleaner')
    #parser.add_argument('-r', '--rotate', type=str, choices=['cw', 'ccw'], help='rotate image 90 degrees in direction specified; options: cw(clockwise), ccw(counterclockwise)')
    #parser.add_argument('-l', '--layout', type=str, choices=['p', 'l'], default='p', help='desired layout; options: p(portrait) or l(landscape).')
    #parser.add_argument('-c', '--cropoff', type=str, help='image cropping offsets after potential rotate 90; choices: one, two or four non-negative integer comma separated values.')
    #parser.add_argument('-g', '--grayscale', action='store_true', default=False, help='convert document to grayscale before enhancing.')
    parser.add_argument('-e', '--enhance', type=str, choices=['stretch', 'normalize'], help='enhance image brightness before cleaning.')
    parser.add_argument('-f', '--filtersize', type=int, default=15, help='size of filter used to clean background; integer>0')
    parser.add_argument('-o', '--offset', type=int, default=5, help='offset of filter in percent used to reduce noise; integer>=0')
    parser.add_argument('-u', '--unrotate', action='store_true', default=False, help='unrotate image; cannot unrotate more than about 5 degrees.')
    parser.add_argument('-th', '--threshold', type=int, help='text smoothing threshold; 0<=threshold<=100; nominal value is about 50; default is no smoothing.')
    parser.add_argument('-sh', '--sharpamt', type=float, default=0, help='sharpening amount in pixels; float>=0; nominal about 1; default=0')
    #parser.add_argument('-sa', '--saturation', type=int, default=200, help='color saturation expressed as percent; integer>=0; only applicable if -g not set; a value of 100 is no change; default=200(double saturation)')
    parser.add_argument('-a', '--adaptblur', type=float, default=0, help='alternate text smoothing using adaptive blur; float>=0; default=0 (no smoothing)')
    #parser.add_argument('-t', '--trim', action='store_true', default=False, help='trim background around outer part of image')
    #parser.add_argument('-p ', '--padamt', type=int, default=0, help='border pad amount around outer part of image; integer>=0')
    #parser.add_argument('-b', '--bgcolor', type=str, default='white', help='desired color for background.')
    #parser.add_argument('-F', '--fuzzval', type=int, default=10, help='fuzz value for determining bgcolor when bgcolor=image; integer>=0')
    #parser.add_argument('-i', '--invert', type=int, default=0, help='invert colors; choices: 1 or 2 for one-way or two-ways (input or input and output); default is no inversion')
    parser.add_argument('infile', type=str, help='Input file path')
    parser.add_argument('outfile', type=str, help='Output file path')
    args = parser.parse_args()
    print(args)
    process_image(args)

if __name__ == '__main__':
    start()