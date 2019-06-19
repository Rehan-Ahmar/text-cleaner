import os
import cv2
import numpy as np
from matplotlib import pyplot as plt

def histogram_equalization(img):
    equ = cv2.equalizeHist(img)
    return equ
    
def adaptive_histogram_equalization(img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(img)
    
def sharpen_image(img):
    # shapening kernel must equal to one
    kernel_sharpening = np.array([[-1,-1,-1], 
                                [-1, 9,-1],
                                [-1,-1,-1]])
    sharpened = cv2.filter2D(image, -1, kernel_sharpening)
    return sharpened
    
def alternate_process(img):
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    equ = histogram_equalization(img)
    #adp = adaptive_histogram_equalization(img)
    
    # Apply dilation and erosion
    #kernel = np.ones((2, 2), np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    img = cv2.dilate(img, kernel, iterations=10)
    img = cv2.erode(img, kernel, iterations=10)
    
    # Apply blurring
    #img = cv.blur(img,(5,5))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    #img = cv2.medianBlur(img, 5)
    #img = cv.bilateralFilter(img,9,75,75)
    
    # Apply thresholding
    #ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    #adaptive_mean_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    #adaptive_gaussian_thresholding = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
	
    #sharpened_img = sharpen_image(img)
    return img
    
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

def whiten_borders(img, border_width):
    mask = np.ones(img.shape,np.uint8)*255
    mask[border_width:img.shape[0]-border_width, border_width:img.shape[1]-border_width] \
         = img[border_width:img.shape[0]-border_width, border_width:img.shape[1]-border_width]
    return mask
    
def clean_image(img_path, output_path, border_width=50):
    print('Cleaning image: ', img_path)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(os.path.join(output_path, 'in.png'), img)
    
    #processed = alternate_process(img)
    #cv2.imwrite(os.path.join(output_path, 'alternate2.png'), processed)
    
    out = remove_noise_and_smooth(img)
    out = whiten_borders(out, border_width)
    out = cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(os.path.join(output_path, 'cleaned.png'), out)
    