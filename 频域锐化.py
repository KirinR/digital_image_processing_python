import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
 
 
def ideal_high_filter(img, D0):
    #生成一个理想高通滤波器
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 0 if d < D0 else 1
    return filter_img
 
 
def butterworth_high_filter(img, D0, rank):
    #生成一个巴特沃斯高通滤波器
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = 1 / (1 + (D0 / d) ** (2 * rank))
    return filter_img
 
 
def exp_high_filter(img, D0, rank):
    #生成一个指数高通滤波器
    h, w = img.shape[:2]
    filter_img = np.zeros((h, w))
    u = np.fix(h / 2)
    v = np.fix(w / 2)
    for i in range(h):
        for j in range(w):
            d = np.sqrt((i - u) ** 2 + (j - v) ** 2)
            filter_img[i, j] = np.exp((-1) * (D0 / d) ** rank)
    return filter_img
 
 
def filter_use(img, filter):
    #将图像img与滤波器filter结合，生成对应的滤波图像
    # 首先进行傅里叶变换
    f = np.fft.fft2(img)
    f_center = np.fft.fftshift(f)
    # 应用滤波器进行反变换
    S = np.multiply(f_center, filter)  # 频率相乘——l(u,v)*H(u,v)
    f_origin = np.fft.ifftshift(S)  # 将低频移动到原来的位置
    f_origin = np.fft.ifft2(f_origin)  # 使用ifft2进行傅里叶的逆变换
    f_origin = np.abs(f_origin)  # 设置区间
    f_origin = f_origin / np.max(f_origin.all())
    return f_origin
 
 
def DFT_show(img):
    #对传入的图像进行傅里叶变换，生成频域图像
    f = np.fft.fft2(img)  # 使用numpy进行傅里叶变换
    fshift = np.fft.fftshift(f)  # 把零频率分量移到中间
    result = np.log(1 + abs(fshift))
    return result
 
 
src = cv.imread("da.jpg", 0)
my_img = src.copy()
 
# 1.理想高通滤波
ideal_filter = ideal_high_filter(my_img, D0=40)  # 生成理想高通滤波器
ideal_img = filter_use(my_img, ideal_filter)  # 将滤波器应用到图像，生成理想高通滤波图像
fre_img = DFT_show(my_img)  # 原图的频域图像
fre_ideal_img = DFT_show(ideal_img)  # 理想高通滤波图像的频域图像
plt.figure(dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.subplot(221)
plt.title('原图')
plt.imshow(my_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(222)
plt.title("理想高通滤波图像")
plt.imshow(ideal_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(223)
plt.title('原图频域图')
plt.imshow(fre_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(224)
plt.title("理想高通滤波图像的频域图")
plt.imshow(fre_ideal_img, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
 
# 2.巴特沃斯高通滤波器
my_img = src.copy()
butterworth_filter = butterworth_high_filter(my_img, D0=40, rank=2)  # 生成Butterworth高通滤波器
butterworth_img = filter_use(my_img, butterworth_filter)  # 将滤波器应用到图像，生成Butterworth高通滤波图像
fre_butterworth_img = DFT_show(butterworth_img)  # Butterworth高通滤波图像的频域图像
plt.figure(dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.subplot(221)
plt.title('原图')
plt.imshow(my_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(222)
plt.title("巴特沃斯高通滤波图像")
plt.imshow(butterworth_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(223)
plt.title('原图频域图')
plt.imshow(fre_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(224)
plt.title("巴特沃斯高通滤波图像的频域图")
plt.imshow(fre_butterworth_img, cmap=plt.cm.gray)
plt.axis("off")
plt.show()
 
#3.指数高通滤波器
my_img = src.copy()
exp_filter = exp_high_filter(my_img, D0=40, rank=2)  # 生成指数高通滤波器
exp_img = filter_use(my_img, exp_filter)  # 将滤波器应用到图像，生成指数高通滤波图像
fre_exp_img = DFT_show(exp_img)  # 指数高通滤波图像的频域图像
plt.figure(dpi=300)
plt.rcParams['font.sans-serif'] = ['SimHei']#用来正常显示中文标签
plt.subplot(221)
plt.title('原图')
plt.imshow(my_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(222)
plt.title("指数高通滤波图像")
plt.imshow(exp_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(223)
plt.title('原图频域图')
plt.imshow(fre_img, cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(224)
plt.title("指数高通滤波图像的频域图")
plt.imshow(fre_exp_img, cmap=plt.cm.gray)
plt.axis("off")
plt.show()