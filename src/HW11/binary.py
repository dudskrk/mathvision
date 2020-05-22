import numpy as np
import cv2

# read img and Otsu thresholding
src = cv2.imread('hw11_sample.png', 0)
w, h = src.shape[:2]
thresh, src_otsu = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
src_adaptive = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

cv2.imshow('src otsu : '+ str(thresh), src_otsu)
cv2.imshow('src adaptive', src_adaptive)

# initializing for pseudo inverse
A = np.zeros((w * h, 6), np.float64)
Y = np.zeros((w * h, 1), np.float64)
for i in range(h):
    for j in range(w):
        A[i * w + j][0] = i * i
        A[i * w + j][1] = j * j
        A[i * w + j][2] = i * j
        A[i * w + j][3] = i
        A[i * w + j][4] = j
        A[i * w + j][5] = 1
        Y[i * w + j][0] = src[j][i]
#pinvA = np.linalg.pinv(A)
pinvA = np.linalg.inv(A.transpose().dot(A)).dot(A.transpose())
X = pinvA.dot(Y)
a, b, c, d, e, f = X[:, 0]

# estimate background surface with LS
LS_surface = np.zeros((w, h), np.float64)
for i in range(h):
    for j in range(w):
        LS_surface[j][i] = a * i * i + b * j * j + c * i * j + d * i + e * j + f

# it's not correct using raw LS_surface for binarization
LS_surface_u8 = np.uint8(LS_surface)
cv2.imshow('src', src)
cv2.imshow('LS_surface_u8', LS_surface_u8)
cv2.waitKey(0)

# Give range to approximate value for binarization
LS_range = np.full((w, h), 255, np.float64)
cv2.imshow('src - LS_surface_u8', abs(src - LS_surface_u8))
for i in range(h):
    for j in range(w):
        if (abs(src[j][i] - LS_surface[j][i]) <= 10):
            LS_range[j][i] = 0

LS_thresh = np.uint8(LS_range)
cv2.imshow('LS_thresh', LS_thresh)
cv2.waitKey(0)
