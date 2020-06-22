import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import sympy as sym
from sympy import lambdify, hessian, ordered, Matrix
from mpl_toolkits import mplot3d
import cv2

def sin(arr, A, B, C, D):
    r = []
    for i in range(arr.shape[0]):
        r.append(A * sym.sin(B * arr[i, 0] + C) + D - arr[i, 1])
    return Matrix(r)

def np_sin(x, A, B, C, D):
    r = np.zeros((800, 1), dtype=float)
    for i in range(x.shape[0]):
        r[i] = (A * np.sin(B * x[i] + C) + D)
    return r

def Jacobian(v_str, f_list):
    vars = sym.symbols(v_str)
    f = sym.sympify(f_list)
    J = sym.zeros(len(f), len(vars))
    for i, fi in enumerate(f):
        for j, s in enumerate(vars):
            J[i, j] = sym.diff(fi, s)
    return J

def draw_sine(A, B, C, D, count, manual=False, step=False):
    global frame, frame_copy
    frame_clone = frame.copy()
    x_list = np.arange(0, 800, 1)
    y_list = np_sin(x_list, A, B, C, D)

    if step == True:
        for i in range(x_list.shape[0] - 1):
            cv2.line(frame, (x_list[i], y_list[i]), (x_list[i + 1], y_list[i + 1]), (0, count * 1.5, 255 - count * 1.5), 1, cv2.LINE_AA)
        frame += frame_copy
        cv2.imshow('frame', frame)
    else:
        for i in range(x_list.shape[0] - 1):
            cv2.line(frame_clone, (x_list[i], y_list[i]), (x_list[i + 1], y_list[i + 1]), (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame_clone)
    if manual == True:
        cv2.waitKey(0)
    else:
        cv2.waitKey(10)

def arg_sort(arr):
    count = 0
    x = arr[:, 0]
    sort = arr[np.argsort(x)]
    for i in range(1, arr.shape[0] - 1):
        if ((sort[i - 1][1] - sort[i][1]) * (sort[i][1] - sort[i + 1][1]) <= 0):
            count += 1
    #cv2.waitKey(0)
    return count

def fit_sine(gamma=0):
    global points, frame
    arr = np.array(points)
    arr_y = arr[:, 1]

    A, B, C, D = sym.symbols('A B C D')
    s = (A, B, C, D)
    r = sin(arr, A, B, C, D)
    j_r = Jacobian('A B C D', r)
    g_r = j_r.transpose() * r
    h_r = (j_r.transpose() * j_r)
    #j = lambdify(s, r, modules='numpy')
    #J_r = lambdify(s, j_r, modules='numpy')
    H_r = lambdify(s, h_r, modules='numpy')
    G_r = lambdify(s, g_r, modules='numpy')

    #xk_a = xk_A = (np.max(arr_y) - np.min(arr_y)) / 2
    xk_a = xk_A = np.std(arr_y)
    xk_b = xk_B = arg_sort(arr) * 0.0035
    xk_c = xk_C = 10
    xk_d = xk_D = np.mean(arr_y)
    count = 0

    draw_sine(xk_A, xk_B, xk_C, xk_D, 0)
    while True:
        H_r_np = H_r(xk_A, xk_B, xk_C, xk_D)
        H_r_inv = np.linalg.inv(H_r_np)
        xk1_A = xk_A - (H_r_inv.dot(G_r(xk_A, xk_B, xk_C, xk_D)))[0] + gamma * (xk_A - xk_a)
        xk1_B = xk_B - (H_r_inv.dot(G_r(xk_A, xk_B, xk_C, xk_D)))[1] + gamma * (xk_B - xk_b)
        xk1_C = xk_C - (H_r_inv.dot(G_r(xk_A, xk_B, xk_C, xk_D)))[2] + gamma * (xk_C - xk_c)
        xk1_D = xk_D - (H_r_inv.dot(G_r(xk_A, xk_B, xk_C, xk_D)))[3] + gamma * (xk_D - xk_d)
        draw_sine(xk1_A, xk1_B, xk1_C, xk1_D, count=count, step=True)
        if np.sqrt((xk1_A - xk_A) ** 2 + (xk1_B - xk_B) ** 2 + (xk1_C - xk_C) ** 2 + (xk1_D - xk_D) ** 2) < 0.001:
            break
        xk_a = xk_A; xk_b = xk_B; xk_c = xk_C; xk_d = xk_D;
        xk_A = float(xk1_A); xk_B = float(xk1_B); xk_C = float(xk1_C); xk_D = float(xk1_D)
        count += 1
        print(count)
    print('f(x)=', xk_A, 'sin(', xk_B, 'x+', xk_C, ')+', xk_D)
    draw_sine(xk1_A, xk1_B, xk1_C, xk1_D, count)

def refreshWindow():
    global points, frame
    points.clear()
    frame = np.zeros((600, 800, 3), np.uint8)
    cv2.imshow('frame', frame)

def draw_base(event, x, y, flags, param):
    global points, frame, frame_copy, data_ready

    # get points
    if event == cv2.EVENT_LBUTTONDOWN:
        if data_ready == False:
            print('(', x, y, ')')
            points.append((x, y))
            cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)

    # clear all, refresh window
    elif event == cv2.EVENT_RBUTTONDOWN:
        data_ready = False
        refreshWindow()

    # fitting sine
    elif event == cv2.EVENT_LBUTTONDBLCLK:
        data_ready = True
        frame_copy = frame.copy()
        fit_sine(gamma=0.9)
        #fit_sine()

data_ready = False
points = []
frame = np.zeros((600, 800, 3), np.uint8)
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', draw_base)

while True:
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)

    if (key == 27):
        break
cv2.destroyWindow('frame')