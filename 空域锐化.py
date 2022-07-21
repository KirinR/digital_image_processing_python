import cv2
import numpy as np
#三个一阶算子是从底层实现的，而拉普拉斯算子通过调库实现

def robert(img):
    h,w=img.shape[:2]
    r=[[-1,-1],[1,1]]
    for i in range(h):
        for j in range(w):
            if (j+2<w) and (i+2<=h):
                process_img=img[i:i+2,j:j+2]
                list_robert=r*process_img
                img[i,j]=abs(list_robert.sum())

    return img

img=cv2.imread("da.jpg",cv2.IMREAD_GRAYSCALE)
cv2.imshow('orl_img',img)
img=robert(img)
cv2.imshow('robert',img)
cv2.waitKey(0)


def sobel(img):
    h,w=img.shape
    new_img=np.zeros([h,w])
    x_img=np.zeros(img.shape)
    y_img=np.zeros(img.shape)
    sobel_x=np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    sobel_y=np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
    for i in range(h-2):
        for j in range(w-2):
            x_img[i+1,j+1]=abs(np.sum(img[i:i+3,j:j+3]*sobel_x))
            y_img[i+1,j+1]=abs(np.sum(img[i:i+3,j:j+3]*sobel_y))
            new_img[i+1,j+1]=np.sqrt(np.square(x_img[i+1,j+1])+np.square(y_img[i+1,j+1]))

    return np.uint8(new_img)

img=cv2.imread("da.jpg",cv2.IMREAD_GRAYSCALE)
img=sobel(img)
cv2.imshow('sobel',img)
cv2.waitKey(0)


def prewitt(img):
    h,w=img.shape
    new_img=np.zeros([h,w])
    x_img=np.zeros(img.shape)
    y_img=np.zeros(img.shape)
    prewitt_x=np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    prewitt_y=np.array([[-1,-1,-1],[0,0,0],[1,1,1]])
    for i in range(h-2):
        for j in range(w-2):
            x_img[i+1,j+1]=abs(np.sum(img[i:i+3,j:j+3]*prewitt_x))
            y_img[i+1,j+1]=abs(np.sum(img[i:i+3,j:j+3]*prewitt_y))
            new_img[i+1,j+1]=np.sqrt(np.square(x_img[i+1,j+1])+np.square(y_img[i+1,j+1]))

    return np.uint8(new_img)

img=cv2.imread("da.jpg",cv2.IMREAD_GRAYSCALE)
img=prewitt(img)
cv2.imshow('prewitt',img)
cv2.waitKey(0)



img=cv2.imread("da.jpg",cv2.IMREAD_GRAYSCALE)
img_lap=cv2.Laplacian(src=img,ddepth=-1,ksize=3)
cv2.imshow('Laplacian',img_lap)
cv2.waitKey(0)
