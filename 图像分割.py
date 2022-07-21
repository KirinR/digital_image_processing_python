from turtle import width
import cv2
import numpy as np
import matplotlib.pyplot as plt

#按区域生长来分割，选定三个种子，分割三个种子所在的区域
img = cv2.imread("1.jpg")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def get_binary_img(img):
    # gray img to bin image
    bin_img = np.zeros(shape=(img.shape), dtype=np.uint8)
    h = img.shape[0]
    w = img.shape[1]
    for i in range(h):
        for j in range(w):
            bin_img[i][j] = 255 if img[i][j] > 127 else 0
    return bin_img


# 调用
bin_img = get_binary_img(gray_img)
out_img = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
hei = bin_img.shape[0]
wid = bin_img.shape[1]
# 选择初始3个种子点
seeds = [(176, 255), (229, 405), (347, 165)]
for seed in seeds:
    x = seed[0]
    y = seed[1]
    out_img[y][x] = 255
# 8 邻域
directs = [(-1, -1), (0, -1), (1, -1), (1, 0),
            (1, 1), (0, 1), (-1, 1), (-1, 0)]
visited = np.zeros(shape=(bin_img.shape), dtype=np.uint8)
while len(seeds):
    seed = seeds.pop(0)
    x = seed[0]
    y = seed[1]
    # visit point (x,y)
    visited[y][x] = 1
    for direct in directs:
        cur_x = x + direct[0]
        cur_y = y + direct[1]
        # 非法
        if cur_x <0 or cur_y<0 or cur_x >= wid or cur_y >=hei :
            continue
        # 没有访问过且属于同一目标
        if (not visited[cur_y][cur_x]) and (bin_img[cur_y][cur_x]==bin_img[y][x]) :
            out_img[cur_y][cur_x] = 255
            visited[cur_y][cur_x] = 1
            seeds.append((cur_x,cur_y))

bake_img = img.copy()
h = bake_img.shape[0]
w = bake_img.shape[1]
for i in range(h):
    for j in range(w):
        if out_img[i][j] != 255:
            bake_img[i][j][0] = 0
            bake_img[i][j][1] = 0
            bake_img[i][j][2] = 0
cv2.imshow('分割结果',bake_img)
cv2.waitKey(0)

