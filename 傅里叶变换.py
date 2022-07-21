import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
 
if __name__ == "__main__":
    image = cv2.imread("da.jpg", 0)
    #对image进行快速傅里叶变换，输出为复数形式
    fft = np.fft.fft2(image)
    # np.fft.fftshift将零频率分量移动到频谱中心
    fft_shift = np.fft.fftshift(fft)
    # 取绝对值，将复数形式变为实数
    # 取绝对值目的为了将数据变化到较小的范围
    s1 = np.log(np.abs(fft))
    s2 = np.log(np.abs(fft_shift))
    # np.angle能够直接根据复数的虚部和实部求出角度，默认的角度是弧度
    phase_f = np.angle(fft)
    phase_f_shift = np.angle(fft_shift)
    # 进行傅里叶逆变换
    ifft_shift = np.fft.ifftshift(fft_shift)
    ifft = np.fft.ifft2(ifft_shift)
    #出来的是复数，无法显示
    image_back = np.abs(ifft)
    # 显示
    plt.subplot(231), plt.imshow(image, 'gray'), plt.title("原图"), plt.xticks([]), plt.yticks([])
    plt.subplot(232), plt.imshow(s1, 'gray'), plt.title("幅度谱中心化以前"), plt.xticks([]), plt.yticks([])
    plt.subplot(233), plt.imshow(s2, 'gray'), plt.title("幅度谱中心化以后"), plt.xticks([]), plt.yticks([])
    plt.subplot(234), plt.imshow(phase_f, 'gray'), plt.title("相位谱中心化以前"), plt.xticks([]), plt.yticks([])
    plt.subplot(235), plt.imshow(phase_f_shift, 'gray'), plt.title("相位谱中心化以后"), plt.xticks([]), plt.yticks([])
    plt.subplot(236), plt.imshow(image_back, 'gray'), plt.title("傅里叶逆变换结果"), plt.xticks([]), plt.yticks([])
    plt.show()
