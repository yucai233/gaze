import torch
import model
import torch.nn as nn
import cv2
import face_recognition
import numpy as np
import copy
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 1
import random
import os
import time
import threading
import tkinter as tk
from PIL import Image, ImageTk, ImageDraw
from ttkthemes import ThemedTk

from screeninfo import get_monitors
import win32gui
import win32con
from tkinter import *

import time
import util
import myCalibration





def showImg(img):
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()
#
# def getRect(img):
#     arr = np.zeros((3, 4))
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # 首先拿到脸的定位
#     loc = face_recognition.face_locations(img)
#     if len(loc) == 0:   #说明是空列表
#         print("你的脸呢？")
#         return None
#     loc = loc[0]
#     arr[0][0] = loc[3]
#     arr[0][1] = loc[0]
#     arr[0][2] = loc[1]
#     arr[0][3] = loc[2]
#
#     # 拿到两只眼睛的定位
#     res = face_recognition.face_landmarks(img)
#     str_list = ["left_eye", "right_eye"]
#     for i, s in enumerate(str_list):
#         x_ave, y_ave = np.sum(res[0][s], axis=0) / 6
#
#         w = res[0][s][3][0] - res[0][s][0][0]
#         w *= 1.7
#         h = w
#         arr[i + 1][0] = int(x_ave - w / 2)
#         arr[i + 1][1] = int(y_ave - h / 2)
#         arr[i + 1][2] = int(x_ave + w / 2)
#         arr[i + 1][3] = int(y_ave + h / 2)
#
#     return arr.astype(np.int64)

def getPoint(img, net, device):
    output = util.Util().getPoint(img)
    if isinstance(output, type(None)):
        return None

    return net(torch.from_numpy(output.numpy()[np.newaxis, :]).to(device)).cpu().detach().numpy().squeeze()

#
# #首先导入模型
# face_cascade = cv2.CascadeClassifier('E:/Anoconda/envs/py37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier('E:/Anoconda/envs/py37/Lib/site-packages/cv2/data/haarcascade_eye.xml')
#
# def getRect(img):
#     img = copy.copy(img)
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     eye_rect = eye_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
#     face_rect = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
#     if isinstance(eye_rect, tuple) or isinstance(face_rect, tuple):
#         print("无效")
#         return None
#
#     arr = np.zeros((3, 4))
#     rect = np.vstack((face_rect, eye_rect))
#     if rect.shape[0] < 3:
#         return None
#
#
#     rect[:, 2] += rect[:, 0]
#     rect[:, 3] += rect[:, 1]
#     return rect[0:3, :]



# def getPoint(img, net, device):
#     rects = getRect(img)
#     if isinstance(rects, type(None)):
#         print("未检测到人脸")
#         return None
#
#     # 拿到分割图片
#     img_list = []
#     for rect in rects:
#         subimg = img[rect[1]:rect[3], rect[0]:rect[2]]  # .astype(np.float64)
#         img_list.append(subimg)
#         # showImg(subimg)
#
#     #图片预处理
#     face_img = cv2.resize(img_list[0], (224, 224))
#     face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
#     face_img = face_img / 255
#     face_img = face_img.transpose(2, 0, 1)[np.newaxis, :]
#
#     leftEye_img = cv2.resize(img_list[1], (112, 112))
#     leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
#     leftEye_img = leftEye_img / 255
#     leftEye_img = leftEye_img.transpose(2, 0, 1)[np.newaxis, :]
#
#     rightEye_img = cv2.resize(img_list[2], (112, 112))
#     rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
#     rightEye_img = cv2.flip(rightEye_img, 1)
#     rightEye_img = rightEye_img / 255
#     rightEye_img = rightEye_img.transpose(2, 0, 1)[np.newaxis, :]
#
#     #拿到模型输入的rects
#     #使用间接方法拿到用于输入的rects
#     rects = rects.astype(np.float32)
#     rects[:, 0] = rects[:, 0] / img.shape[1]
#     rects[:, 2] = rects[:, 2] / img.shape[1]
#     rects[:, 1] = rects[:, 1] / img.shape[0]
#     rects[:, 3] = rects[:, 3] / img.shape[0]
#     rects = rects.flatten().reshape(1, -1)
#     # print("输入模型的rects为：\n", rects)
#
#
#     gazes = net(torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(device),
#                 torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(device),
#                 torch.from_numpy(face_img).type(torch.FloatTensor).to(device),
#                 torch.from_numpy(rects).type(torch.FloatTensor).to(device))
#
#     return gazes.cpu().detach().numpy().squeeze()


def calibration(cap, net, device, n):
    # n = 20 # 控制应该输入多少数据
    truth_arr = np.zeros((n, 2))
    pred_arr = np.zeros((n, 2))
    # 拿到屏幕的大小
    w, h = pyautogui.size()

    i = 0
    while i < n:
        # 首先随机挑选鼠标的位置
        x = random.randint(0, w)
        y = random.randint(0, h)

        # 将鼠标移动到指定位置
        pyautogui.moveTo(x, y)

        # 等待用户拍照
        temp = input("按下Enter拍照:")
        ret, img = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
        # showImg(img)

        # 拿到照片之后，送给函数，拿到预测值
        output = getPoint(img, net, device)

        if isinstance(output, type(None)) or ret == False:
            continue

        # 将位置放到数组中
        truth_arr[i, 0] = x
        truth_arr[i, 1] = y
        pred_arr[i, 0] = output[0]
        pred_arr[i, 1] = output[1]

        i += 1
        print("进度 【{}/{}】".format(i, n))

    # 完事之后将两个数组存起来
    np.save(r"c:/Users/jchao/Desktop/truth_arr", truth_arr)
    np.save(r"c:/Users/jchao/Desktop/pred_arr", pred_arr)

def moveToTarget(c, circle, x_begin, y_begin, x_end, y_end, t, n):
    x = (x_end - x_begin) / n
    y = (y_end - y_begin) / n
    for i in range(n):
        c.move(circle, x, y)  # move方法，把circle往（x，y）的方向移动
        c.update()  # 刷新屏幕
        time.sleep(t)  # 设置间隔时间，每0.03秒执行一次循环

def timeOfOneFrame(cap, net, device):
    #判断拿到一帧图像的时间是多少
    for i in range(10):
        ret, frame = cap.read()
        getPoint(frame, net, device)

    before = time.time()
    for i in range(100):
        ret, frame = cap.read()
        getPoint(frame, net, device)
        print("{}/100".format(i))
    after = time.time()
    print("一帧图像平均用时：", (after - before) / 100)


def func(cap, net, device):
    screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

    # 创建透明窗口
    root = Tk()
    root.overrideredirect(True)
    root.wm_attributes('-transparentcolor', 'black')
    root.attributes('-topmost', True)
    root.geometry(f"{screen_width}x{screen_height}+0+0")
    # 设置窗口为无焦点状态
    win32gui.SetWindowLong(root.winfo_id(), win32con.GWL_EXSTYLE,
                           win32gui.GetWindowLong(root.winfo_id(), win32con.GWL_EXSTYLE) | win32con.WS_EX_NOACTIVATE)

    screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

    # print(screen_width, screen_height)

    c = Canvas(root, width=screen_width, height=screen_height, highlightthickness=0, bg='black')#绘制一个画布
    c.pack()


    # # 在画布上显示图像
    begin1, begin2 = 200, 200
    circle = c.create_oval(begin1, begin2, 65, 62, fill='red')#绘制一个圆

    while True:
        # target1 = random.randint(0, screen_width)
        # target2 = random.randint(0, screen_height)
        ret, frame = cap.read()
        output = getPoint(frame, net, device)
        if ret == False or isinstance(output, type(None)):
            continue
        target1, target2 = output
        # target1 = 55 * target1 + 1085
        # target2 = -55 * target2 + 148
        target1 += 960
        print(target1, target2)
        moveToTarget(c, circle, begin1, begin2, target1, target2, 0.01, 5)
        begin1, begin2  = target1, target2

    # c.mainloop()


def buildMode():
    # 模型准备
    # global device, net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = myCalibration.CalibrationModel(cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\59.png"))
    # net = model.model()
    net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
    state_dict = torch.load(r".\checkPoint1\Iter_1401_feiwu-Net.pt")
    # state_dict = torch.load(r".\checkPoint\Iter_2_AFF-Net.pt")
    net.load_state_dict(state_dict)
    net = net.module
    net.to(device)
    net.eval()  # 在模型test的时候加入
    print("Model building done...")
    return net, device

#
# if __name__ == "__main__":
#     net, device = buildMode()
#
#     # 打开摄像头
#     cap = cv2.VideoCapture(0)
#     # 设置分辨率
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#     # 检查摄像头是否成功打开
#     if not cap.isOpened():
#         print("无法打开摄像头")
#         exit()
#
#     # calibration(cap, net, device, 50)
#     func(cap, net, device)
#
#
#
#     cap.release()


if __name__ == "__main__":
    net, device = buildMode()
    img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\59.png")
    print(getPoint(img, net, device))




