import base64
from configparser import NoSectionError
from doctest import NORMALIZE_WHITESPACE
from encodings.utf_8_sig import getregentry
from flask_cors import CORS
from flask import Flask,request
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import os
import random

app=Flask(__name__,static_folder='static')
CORS(app,resources=r'/*')
@app.route('/trans',methods=['POST'])
def transImg():
    file=request.json['img']
    try:
        data=str.split(file,',')[1]
        img_data=base64.urlsafe_b64decode(data+'='*(4-len(data)%4))
        img_data=np.frombuffer(img_data,np.uint8)
        img_data=cv2.imdecode(img_data,cv2.IMREAD_COLOR)
    #HSV   
        hsv=cv2.cvtColor(img_data, cv2.COLOR_BGR2HSV)
        h=hsv[:,:,0]
        s=hsv[:,:,1]
        v=hsv[:,:,2]
        cv2.imwrite("./static/h.jpg",h)  
        cv2.imwrite("./static/s.jpg", s)  
        cv2.imwrite("./static/v.jpg", v)



    #RGB
        
        b = img_data[:, :, 0]  
        g = img_data[:, :, 1]  
        r = img_data[:, :, 2]  
        cv2.imwrite('./static/b.jpg', b)  
        cv2.imwrite('./static/g.jpg', g)  
        cv2.imwrite('./static/r.jpg', r)



    #按倍数采样
        ratio = 2 # 设置采样比率
        # 设置采样后的图片大小
        image1 = np.zeros((int(img_data.shape[0] / ratio), int(img_data.shape[1] / ratio), img_data.shape[2]), dtype='int32')
        # 对图像进遍历
        for i in range(image1.shape[0]):
            for j in range(image1.shape[1]):
                for k in range(image1.shape[2]):
                    delta = img_data[i * ratio:(i + 1) * ratio, j * ratio:(j + 1) * ratio, k]  # 获取需要采样的图像块
                    image1[i, j, k] = np.mean(delta)  # 计算均值，并存入结果图像
        cv2.imwrite('./static/2倍.jpg', image1)  



    #灰度化
        row, col, channel = img_data.shape
        image_gray = np.zeros((row, col))
        for r in range(row):
            for l in range(col):
                image_gray[r, l] = 1 / 3 * img_data[r, l, 0] + 1 / 3 * img_data[r, l, 1] + 1 / 3 * img_data[r, l, 2]
        cv2.imwrite('./static/灰度化.jpg', image_gray)  



    #二值化普通法与OTSU法
        grey =cv2.cvtColor(img_data, cv2.COLOR_BGR2GRAY)
        thresh=125
        maxValue=255
        th,dst= cv2.threshold(grey,thresh,maxValue,cv2.THRESH_BINARY)
        cv2.imwrite('./static/普通二值化.jpg', dst)  
        th1,dst1= cv2.threshold(grey,thresh,maxValue,cv2.THRESH_OTSU)
        cv2.imwrite('./static/OTSU二值化.jpg', dst1)
        th2, dst2 = cv2.threshold(grey, 127, 255, cv2.THRESH_BINARY_INV)  
        cv2.imwrite('./static/反二进制二值化.jpg', dst2)#和普通二值化反过来
        th3, dst3 = cv2.threshold(grey, 127, 255, cv2.THRESH_TRUNC)
        cv2.imwrite('./static/截断阈值化.jpg', dst3)#大于127变为127，小于127不变
        th4, dst4 = cv2.threshold(grey, 127, 255, cv2.THRESH_TOZERO_INV)
        cv2.imwrite('./static/反阈值化为0.jpg', dst4)#大于127变为0，小于127不变
        th5, dst5 = cv2.threshold(grey, 127, 255, cv2.THRESH_TOZERO)
        cv2.imwrite('./static/阈值化为0.jpg', dst5)#大于127不变，小于127变为0

    # 平移
        # 构建移动矩阵,x轴左移 10 个像素，y轴下移 30 个
        height, width, channel = img_data.shape
        M = np.float32([[1, 0, 10], [0, 1, 30]])  # 构建移动矩阵,x轴左移10个像素，y轴下移30个  
        shifted = cv2.warpAffine(img_data, M, (width, height))  
        cv2.imwrite('./static/平移.jpg', shifted) 



    # 镜像（垂直方向左右镜像）
        

        mirrow1 = img_data[:,::-1,:]
        cv2.imwrite('./static/左右镜像.jpg', mirrow1) 

    # 镜像（水平方向上下镜像）
        mirrow2 = img_data[::-1,:,:]
        cv2.imwrite('./static/上下镜像.jpg', mirrow2)



    # 旋转
        
        #k的取值一般为1、2、3，分别表示`顺时针`旋转90度、180度、270度；
        #k也可以取负数-1、-2、-3。k取正数表示`逆时针`旋转，取负数表示顺时针旋转。
        # np.rot90(img, 1) 逆时针旋转90度
        image_rotate_fu90 = np.rot90(img_data, 1)
        # np.rot90(img, -1) 顺时针旋转90度
        image_rotate_90 = np.rot90(img_data, -1)
        cv2.imwrite('./static/顺时针旋转90度.jpg', image_rotate_90)
        cv2.imwrite('./static/逆时针旋转90度.jpg', image_rotate_fu90)

        M = cv2.getRotationMatrix2D((col/2, row/2), 45, 1)
        M1 = cv2.getRotationMatrix2D((col/2, row/2), -45, 1)
        image_rotate_fu45 = cv2.warpAffine(img_data, M, (col, row))
        image_rotate_45   = cv2.warpAffine(img_data, M1, (col, row))
        cv2.imwrite('./static/顺时针旋转45度.jpg', image_rotate_45)
        cv2.imwrite('./static/逆时针旋转45度.jpg', image_rotate_fu45)

        def rotate_bound(image, angle):
    
            (h, w) = image.shape[:2]
            (cX, cY) = (w / 2, h / 2)

            M2 = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M2[0, 0])
            sin = np.abs(M2[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

    
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            return cv2.warpAffine(image, M, (nW, nH))
        rotated_img1 = imutils.rotate_bound(img_data,-60)
        cv2.imwrite('./static/逆时针旋转60度补全版.jpg', rotated_img1)

        def rotate_bound(image, angle):
    
            (h, w) = image.shape[:2]
            (cX, cY) = (w / 2, h / 2)

            M2 = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
            cos = np.abs(M2[0, 0])
            sin = np.abs(M2[0, 1])
            nW = int((h * sin) + (w * cos))
            nH = int((h * cos) + (w * sin))

    
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            return cv2.warpAffine(image, M, (nW, nH))
        rotated_img2 = imutils.rotate_bound(img_data, 60)
        cv2.imwrite('./static/顺时针旋转60度补全版.jpg', rotated_img2)



    #重设像素
        rere = cv2.resize(img_data, (200,200))
        cv2.imwrite('./static/重设像素.jpg', rere)


    
    #插值（最邻近）
        emptyImage=np.zeros((800,800,channel),np.uint8)
        print(emptyImage)
        sh = 800/height
        sw = 800/width
        for i in range(800):
            for j in range(800):
                x=int(i/sh) #找出目标图像对应原图像最近的点
                y=int(j/sw)
                emptyImage[i,j] = img_data[x,y]
        cv2.imwrite('./static/最邻近插值.jpg', emptyImage)



    #插值（双线性）
        lena_x2 = cv2.resize(img_data, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_LINEAR) 
        cv2.imwrite('./static/双线性插值.jpg', lena_x2)



    #插值（双三次）
        lena_x2_cubic = cv2.resize(img_data, (0, 0), fx=2, fy=2, interpolation=cv2.INTER_CUBIC) 
        cv2.imwrite('./static/双三次插值.jpg', lena_x2_cubic)
    
    
    
    #添加噪声 
    # 椒盐噪声的阈值  
        prob = 0.2  
        thres = 1 - prob  
    # 遍历图像，获取叠加噪声后的图像 
        noise = np.zeros(img_data.shape, np.uint8) 
        for i in range(img_data.shape[0]):  
            for j in range(img_data.shape[1]):  
                rdn = random.random()  
                if rdn < prob:  
                # 添加胡椒噪声  
                    noise[i][j] = 0  
                elif rdn > thres:  
                # 添加食盐噪声  
                    noise[i][j] = 255  
                else:  
                # 不添加噪声  
                   noise[i][j] = img_data[i][j]  
        cv2.imwrite('./static/椒盐噪声.jpg', noise)

        # 将图片的像素值归一化，存入矩阵中  
        imagedo = np.array(img_data/255, dtype=float)  
        # 生成正态分布的噪声，其中0表示均值，0.1表示方差  
        noisedo = np.random.normal(0, 0.1, imagedo.shape)  
        # 将噪声叠加到图片上  
        Noise = imagedo + noisedo  
        # 将图像的归一化像素值控制在0和1之间，防止噪声越界  
        Noise = np.clip(Noise, 0.0, 1.0)  
        # 将图像的像素值恢复到0到255之间  
        Noise = np.uint8(Noise*255)
        cv2.imwrite('./static/高斯噪声.jpg', Noise)



        #均值滤波器
        output1 = np.zeros(grey.shape, np.uint8)
        # 遍历图像，进行均值滤波
        for i in range(grey.shape[0]):
            for j in range(grey.shape[1]):
                # 计算均值,完成对图片的几何均值滤波
                ji = 1.0
                ######### Begin #########
                for n in range(-1, 2):  
                # 防止越界  
                    if 0 <= i < grey.shape[0] and 0 <= j + n < grey.shape[1]:  
                # 像素值求和  
                        ji *= grey[i][j + n]  
                # 求均值，作为最终的像素值  
                output1[i][j] = pow(ji,1/3)

        cv2.imwrite('./static/均值滤波.jpg', output1)



        #排序统计类滤波器
        # 待输出的图片
        output2 = np.zeros(grey.shape, np.uint8)
        # 存储滤波器范围内的像素值  
        array = []  
        # 获取列表的中间值的函数  
        def get_max(array):  
            # 列表的长度  
            length = len(array)  
            # 对列表进行选择排序，获得有序的列表  
            for i in range(length):  
                for j in range(i + 1, length):  
                # 选择最大的值  
                    if array[j] > array[i]:  
                # 交换位置  
                        temp = array[j]  
                        array[j] = array[i]  
                        array[i] = temp  
            return array[0]

        for i in range(grey.shape[0]):
            for j in range(grey.shape[1]):
                # 清空滤波器内的像素值  
                array.clear()  
        # 遍历滤波器内的像素  
                for m in range(-1, 2):  
                    for n in range(-1, 2):  
                    # 防止越界  
                        if 0 <= i + m < grey.shape[0] and 0 <= j + n < grey.shape[1]:  
                        # 像素值加到列表中  
                            array.append(grey[i + m][j + n])
                        # 求max值，作为最终的像素值  
                output2[i][j] = get_max(array)
        cv2.imwrite('./static/排序统计滤波.jpg', output2)



        #选择性滤波器
        output3 = np.zeros(grey.shape, np.uint8)       
        # 遍历图像，进行均值滤波  
        array1 = []  
        # 带通的范围  
        min = 200
        for i in range(grey.shape[0]):  
            for j in range(grey.shape[1]):  
                # 滤波器内像素值的和  
                array.clear()  
            if min < grey[i][j]:  
                output3[i][j] = grey[i][j]  
            else:  
                output3[i][j] = 0
        cv2.imwrite('./static/选择滤波.jpg', output3)


        #腐蚀膨胀
        #Mat getStructuringElement(int shape, Size esize, Point anchor = Point(-1, -1));  
        # 第一个参数MORPH_RECT;  MORPH_CROSS;  MORPH_ELLIPSE;  分别对应上面我们提到的矩形结构元、交叉形结构元和椭圆形结构元。     
        #第二个参数表示结构元的大小，比如(5, 5)表示一个5X5大小的结构元。
        #第三个参数表示核心的位置，默认值是(-1, -1)，它表示图形的几何中心。
  


        src = img_data
        kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))  
        erosion = cv2.erode(src, kernel1)  
        cv2.imwrite('./static/腐蚀.jpg', erosion)

        dilation = cv2.dilate(src, kernel1)
        cv2.imwrite('./static/膨胀.jpg', dilation)



        #开闭运算
        # 交叉结构元
        kernel2 = cv2.getStructuringElement(cv2.MORPH_CROSS , (10, 10))  
        # 进行闭运算
        close = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel2)
        cv2.imwrite('./static/闭运算.jpg', close)
        
        # 进行开运算
        open = cv2.morphologyEx(src, cv2.MORPH_OPEN, kernel1)
        cv2.imwrite('./static/开运算.jpg', open)

        return 'ok'
    except:
        return 'err'
app.run()