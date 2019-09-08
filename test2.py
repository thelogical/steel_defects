import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from PIL import Image,ImageDraw
import math as m

#data = os.listdir('/root/Downloads/kaggle_project/decoded')

mask_map = np.zeros(256 * 1600).astype('uint8')
k = pd.read_csv('/root/Downloads/kaggle_project/train.csv')
imdata = k[k.Image.str.contains("70279ce5b.jpg_3")]
data = next(imdata.itertuples())[2].split(" ")
#print(data.to_string(index=False))

pixels,offsets = data[::2],data[1::2]
px = []
for i in range(len(pixels)):
    off = offsets[i]
    px.append(int(pixels[i])-1)
    for j in range(1,int(off)):
        px.append(int(pixels[i])+j-1)

mask_map[px] = 1
mask_map = mask_map.reshape(1600, 256).T
contours, _ = cv2.findContours(mask_map.copy(), 1, 1)
img = cv2.imread('/root/Downloads/kaggle_project/train/70279ce5b.jpg')
imag = Image.open('/root/Downloads/kaggle_project/train/70279ce5b.jpg')
#plt.imshow(mask_map)
#plt.show()
sorted_ctrs = sorted(contours, key=lambda ctr: cv2.boundingRect(ctr)[1])
for cont in contours:
    #rect = cv2.minAreaRect(cont)
    x, y, w, h = cv2.boundingRect(cont)
    #(x,y),(w,h),a = rect
    #box = cv2.boxPoints(rect)
    #print(rect)
    #box = np.int0(box)
    #print(box)
    draw = ImageDraw.Draw(imag)
    draw.rectangle([x,y,x+w,y+h],fill=None,outline=(255,0,0),width=2)
    #rect2 = cv2.drawContours(img.copy(), [box], 0, (0, 0, 255), 3)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 5)
    imag.show()
    #cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    plt.imshow(img)
    plt.show()