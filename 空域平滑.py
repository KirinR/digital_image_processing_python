import cv2
import numpy as np
import matplotlib.pyplot as plt
 #当使用的平滑模板尺寸增大时，对噪声的消除效果有所增强，
 #但是图像更模糊，可视细节逐渐减少，运算量逐步增大


#加椒盐噪声的函数
def saltPepper(image, salt, pepper):
    height = image.shape[0]
    width = image.shape[1]
    pertotal = salt + pepper    #总噪声占比
    noiseImage = image.copy()
    noiseNum = int(pertotal * height * width)
    for i in range(noiseNum):
        rows = np.random.randint(0, height-1)
        cols = np.random.randint(0,width-1)
        if(np.random.randint(0,100)<salt*100):
            noiseImage[rows,cols] = 255
        else:
            noiseImage[rows,cols] = 0
    return noiseImage
 
 
#显示函数
def matplotlib_multi_pic1(images):
    for i in range(len(images)):
        img = images[i]
        title = "("+str(i+1)+")"
        #行，列，索引
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(title,fontsize=10)
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show(imagess):
    for i in range(len(imagess)):
        img = imagess[i]
        title = "("+str(i+1)+")"
        #行，列，索引
        plt.subplot(2, 2, i+1)
        plt.imshow(img, cmap="gray")
        plt.title(title,fontsize=10)
        plt.xticks([])
        plt.yticks([])
    plt.show()
 
#邻域平均法
image = cv2.imread("da.jpg", cv2.IMREAD_GRAYSCALE)
imageNoise = saltPepper(image, 0.1, 0.1)
imageAver3 = cv2.blur(imageNoise, (3, 3))
imageAver5 = cv2.blur(imageNoise, (5, 5))
images = [image, imageNoise, imageAver3, imageAver5]
matplotlib_multi_pic1(images)

# 中值滤波 medianx 即模板为 x行x列
median3 = cv2.medianBlur(image, 3)
median5 = cv2.medianBlur(image, 5)
median7 = cv2.medianBlur(image, 7)
imagess = [image, imageNoise, median3, median5]
show(imagess)


#先显示邻域平均法，后显示中值滤波法