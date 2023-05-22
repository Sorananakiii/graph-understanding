import cv2
import numpy as np



def read_image(path, resize=False, new_size=(256, 256)):
    image = cv2.imread(path)
    if resize :
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image, gray_image

def image_smoothening(img):
    IMAGE_SIZE = 1800
    BINARY_THREHOLD = 180
    
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3

def preprocessing(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    ims = image_smoothening(img)
    or_image = cv2.bitwise_or(ims, closing)
    return or_image