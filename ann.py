#%%
import cv2
import numpy as np
from matplotlib import pyplot as plt
cap = cv2.VideoCapture(0) 

def showimg(img, cmap='gray'):
    plt.imshow(img, cmap=cmap)
    plt.axis("off")
    plt.show()


def resize(img):
    target_size = (28, 28)

    # Get the original dimensions
    original_height, original_width = img.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = original_width / original_height

    # Determine the new size while maintaining the aspect ratio
    if aspect_ratio > 1:  # Wider than tall
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:  # Taller than wide or square
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    # Resize the image while maintaining the aspect ratio
    resized_image = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Calculate padding to be added
    pad_top = (target_size[1] - new_height) // 2
    pad_bottom = target_size[1] - new_height - pad_top
    pad_left = (target_size[0] - new_width) // 2
    pad_right = target_size[0] - new_width - pad_left

    # Add padding to the resized image to make it 28x28
    padded_image = cv2.copyMakeBorder(resized_image, pad_top, pad_bottom, pad_left, pad_right,
                                    borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return padded_image

#%%

def getSymbols(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply thresholding to create a binary image
    _, binary = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)

    # Find contours
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Minimum area threshold (adjust this value based on your needs)
    min_area = 150

    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > min_area]
    symbols_loc = [cv2.boundingRect(contour) for contour in filtered_contours]
    symbols_loc.sort(key= lambda x : x[0] )
    symbols = [resize(binary[y:y+h, x:x+w]) for x, y, w, h in symbols_loc]
    return symbols

#%%
image = cv2.imread(r'C:\Mathlab\my code\ANN\annpro\ann_project\img.jpg')
s=getSymbols(image)
count=1
for sym in s:
    showimg(sym)
    # cv2.imwrite(f'symbol{count}.png', sym)
    count+=1
