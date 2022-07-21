import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
def histogram(grayfig):#绘制直方图
    x = grayfig.size[0]
    y = grayfig.size[1]
    ret = np.zeros(256)
    for i in range(x):  #遍历像素点获得灰度值
        for j in range(y):
            k = grayfig.getpixel((i,j))
            ret[k] = ret[k]+1
    for k in range(256):
        ret[k] = ret[k]/(x*y)
    return ret#返回包含各灰度值占比的数组
 
def histogram_sum(grayfig):#绘制累计直方图
    x = grayfig.size[0]
    y = grayfig.size[1]
    ret = np.zeros(256)
    for i in range(x):
        for j in range(y):
            k = grayfig.getpixel((i,j))
            ret[k] = ret[k]+1
    for k in range(1,256):
        ret[k] = ret[k]+ret[k-1]#累加
    for k in range(256):
        ret[k] = ret[k]/(x*y)
    return ret
 
im = Image.open('./da.jpg')#注意更改路径
im.show()
im_gray = im.convert('L')#获得灰度图
im_gray.show()
 
lenaGrayHist_1 = histogram(im_gray)
lenaGrayHist_2 = histogram_sum(im_gray)
 
plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.title("普通直方图")
plt.bar(range(256),lenaGrayHist_1,color='b')
plt.show()

plt.figure()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.title("累计直方图")
plt.bar(range(256),lenaGrayHist_2,color='y')
plt.show()

#show函数一次只能显示一个，先显示的是蓝色的正常灰度直方图，关掉以后显示的是黄色的累计灰度直方图