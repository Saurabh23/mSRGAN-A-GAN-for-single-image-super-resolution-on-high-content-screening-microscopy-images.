import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
import cv2
np.seterr(divide='ignore', invalid='ignore')

imred = cv2.imread("C:/Users/Saurabh/Desktop/cyto2017/classes/846_F11_2_red.tif")
imgreen = cv2.imread("C:/Users/Saurabh/Desktop/cyto2017/classes/846_F11_2_green.tif")
imblue = cv2.imread("C:/Users/Saurabh/Desktop/cyto2017/classes/846_F11_2_blue.tif")
imyellow = cv2.imread("C:/Users/Saurabh/Desktop/cyto2017/classes/846_F11_2_yellow.tif")

b,g,r = cv2.split(imred)
b1,g1,r1 = cv2.split(imblue)
b2,g2,r2 = cv2.split(imgreen)
b3,g3,r3 = cv2.split(imyellow)

#np.sum(imred[:,:,0] != imred[:,:,1] ) 

final_img = np.zeros_like(imred)
final_img[:,:,0] = b  # red
final_img[:,:,1] = b2 # b2 = green
final_img[:,:,2] = b1 # blue

cv2.imwrite("merged.png", final_img)