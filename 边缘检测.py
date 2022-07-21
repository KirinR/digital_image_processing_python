import cv2  
import numpy as np

#LoG算子的运行速度较慢，请耐心等待

#边缘强化
CRH = cv2.imread("da.jpg",0)  
CRH = CRH.astype('float')  
row, column = CRH.shape  
gradient = np.zeros((row, column))  
for x in range(row - 1):  
    for y in range(column - 1):  
        gx = abs(CRH[x + 1, y] - CRH[x, y])  
        gy = abs(CRH[x, y + 1] - CRH[x, y])  
        gradient[x, y] = gx + gy  
  
sharp = CRH + gradient     
sharp = np.where(sharp > 255, 255, sharp)  
sharp = np.where(sharp < 0, 0, sharp)  
# 数据类型变换  
gradient = gradient.astype('uint8')  
sharp = sharp.astype('uint8')  
cv2.imshow('边缘强化',gradient) 
cv2.waitKey(0)    



#Roberts算子   
img = cv2.imread("da.jpg",0)
img = cv2.merge([img,img,img])      
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# 2. Roberts算子
kernelx = np.array([[-1, 0], [0, 1]], dtype=int)  
kernely = np.array([[0, -1], [1, 0]], dtype=int)  
# 3. 卷积操作
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)  
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)  
# 4. 数据格式转换
grayImage = cv2.convertScaleAbs(grayImage)
absX = cv2.convertScaleAbs(x)  
absY = cv2.convertScaleAbs(y)  
Roberts = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)  
cv2.imshow('Roberts算子',Roberts) 
cv2.waitKey(0)    



#Sobel 算子
# 1. 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# 2. 求Sobel 算子
kernelx = cv2.Sobel(grayImage, cv2.CV_16S, 1, 0)  
kernely = cv2.Sobel(grayImage, cv2.CV_16S, 0, 1)
x = cv2.filter2D(grayImage, cv2.CV_16S, kernelx)  
y = cv2.filter2D(grayImage, cv2.CV_16S, kernely)    
# 3. 数据格式转换
absX = cv2.convertScaleAbs(x)  
absY = cv2.convertScaleAbs(y)  
# 4. 组合图像
Sobel = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
cv2.imshow('Sobel算子',Sobel) 
cv2.waitKey(0)    



#Laplacian 算子
# 1. 灰度化处理图像
grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
# 2. 高斯滤波
imagedo = cv2.GaussianBlur(grayImage,(5,5),0,0)
# 3. 拉普拉斯算法
dst = cv2.Laplacian(imagedo, cv2.CV_16S, ksize=3) 
# 4. 数据格式转换
Laplacian = cv2.convertScaleAbs(dst)
cv2.imshow('Laplacian',Laplacian) 
cv2.waitKey(0)    



#LoG 边缘算子
image = cv2.copyMakeBorder(img, 2, 2, 2, 2, borderType=cv2.BORDER_REPLICATE)
image = cv2.GaussianBlur(image,(3,3),0,0)
image1 = np.zeros(shape=image.shape,dtype=np.int16)
# 使用Numpy定义LoG算子
m1 = np.array([[0, 0, -1, 0, 0], [0, -1, -2, -1, 0], [-1, -2, 16, -2, -1], [0, -1, -2, -1 ,0], [0, 0, -1, 0, 0]])
#  卷积运算
# 为了使卷积对每个像素都进行运算，原图像的边缘像素要对准模板的中心。
# 由于图像边缘扩大了2像素，因此要从位置2到行(列)-2
rows = image.shape[0]
cols = image.shape[1]
for i in range(2, rows - 2):
    for j in range(2, cols - 2):
            image1[i, j] = np.sum(m1 * image[i - 2:i + 3, j - 2:j + 3, 1])   
image1 = cv2.convertScaleAbs(image1) 
cv2.imshow('LoG 边缘算子',image1) 
cv2.waitKey(0)



#Canny 算子
# 1. 高斯滤波
blur = cv2.GaussianBlur(img, (3, 3), 0) 
# 2. 灰度转换
grayImage = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)  
# 3. 求x，y方向的Sobel算子
gradx = cv2.Sobel(grayImage, cv2.CV_16SC1, 1, 0)  
grady = cv2.Sobel(grayImage, cv2.CV_16SC1, 0, 1)  
# 4. 使用Canny函数处理图像，x,y分别是3求出来的梯度，低阈值50，高阈值150
edge_output = cv2.Canny(gradx, grady, 50, 150)
cv2.imshow('Canny算子',edge_output) 
cv2.waitKey(0)