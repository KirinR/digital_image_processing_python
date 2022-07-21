import numpy as np
import cv2
import matplotlib.pyplot as plt

#线性变换 输入图像为I，宽W、高为H，输出图像为O
def calcGrayHist1(image):
 #灰度图像矩阵的高、宽
    rows, cols = image.shape
 #存储灰度直方图
    grayHist=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] +=1
    # 显示灰度直方图
    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
    plt.axis([0, 255, 0, y_maxValue])
    plt.ylabel("gray level")
    plt.ylabel("number or pixels")
    plt.title("原图")
    # 显示灰度直方图
    plt.show()

def calcGrayHist2(image):
 #灰度图像矩阵的高、宽
    rows, cols = image.shape
 #存储灰度直方图
    grayHist=np.zeros([256],np.uint64)
    for r in range(rows):
        for c in range(cols):
            grayHist[image[r][c]] +=1
    # 显示灰度直方图
    # 画出灰度直方图
    x_range = range(256)
    plt.plot(x_range, grayHist, 'r', linewidth=2, c='black')
    # 设置坐标轴的范围
    y_maxValue = np.max(grayHist)
    plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
    plt.axis([0, 255, 0, y_maxValue])
    plt.ylabel("gray level")
    plt.ylabel("number or pixels")
    plt.title("线性变化后")
    # 显示灰度直方图
    plt.show()
 




if __name__=="__main__":
 # 读图像
    I = cv2.imread("da.jpg", cv2.IMREAD_GRAYSCALE)
 #线性变换
    a=3
    O=float(a)*I
 #进行数据截断，大于255 的值要截断为255
    O[0>255]=255
 #数据类型转换
    O=np.round(O)
 #uint8类型
    O=O.astype(np.uint8)
 #显示原图和线性变换后的效果
    cv2.imshow("原图",I)
    cv2.imshow("线性变化后",O)
    calcGrayHist1(I)
    calcGrayHist2(O)
    cv2.waitKey(0)

    

#伽马变换
    #图像归一化
    fI=I/255.0
    gamma=0.3
    O1=np.power(fI,gamma)
    #显示原图和伽马变换
    cv2.imshow("原图",I)
    cv2.imshow("伽马变化后",O1)
    cv2.waitKey()
    