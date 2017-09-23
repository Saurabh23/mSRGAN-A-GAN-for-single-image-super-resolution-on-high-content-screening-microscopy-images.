import os
import cv2
#files = os.listdir("C:/Users/Saurabh/Desktop/cyto2017/classes/")
files = os.listdir("/storage/srgan-1-master/vgg19/cyto/train/")
imgfiles = [f for f in files if f.endswith(".tif")]
#path = "C:/Users/Saurabh/Desktop/cyto2017/classes/"
path = "/storage/srgan-1-master/vgg19/cyto/train/"

for i in range(0,len(imgfiles)):
	os.chdir("/storage/srgan-1-master/vgg19/cyto/train/")
    img = cv2.imread(path + imgfiles[i])
    res = cv2.resize(img,(96,96), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite(imgfiles[i]+".png",res)