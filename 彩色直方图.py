import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
 
img = cv.imread('./da.jpg')
cv.imshow('Img', img)
 
plt.figure()
plt.title('Color Histogram')
plt.xlabel('level')
plt.ylabel('number of pixels')
colors = ('b', 'g', 'r')
for i,item in enumerate(colors):
     hist = cv.calcHist([img], [i], None, [256], [0,256])
     plt.plot(hist, color=item)
     plt.xlim([0,256])
 
plt.show()
 