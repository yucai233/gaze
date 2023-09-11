import torch
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

import frameToPoint


#
# def calibration(cap, net, device, n):
#     # n = 20 # 控制应该输入多少数据
#     truth_arr = np.zeros((n, 2))
#     pred_arr = np.zeros((n, 2))
#     # 拿到屏幕的大小
#     w, h = pyautogui.size()
#
#     i = 0
#     while i < n:
#         # 首先随机挑选鼠标的位置
#         x = random.randint(0, w)
#         y = random.randint(0, h)
#
#         # 将鼠标移动到指定位置
#         pyautogui.moveTo(x, y)
#
#         # 等待用户拍照
#         temp = input("按下Enter拍照:")
#         ret, img = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
#         # showImg(img)
#
#         # 拿到照片之后，送给函数，拿到预测值
#         output = getPoint(img, net, device)
#
#         if isinstance(output, type(None)) or ret == False:
#             continue
#
#         # 将位置放到数组中
#         truth_arr[i, 0] = x
#         truth_arr[i, 1] = y
#         pred_arr[i, 0] = output[0]
#         pred_arr[i, 1] = output[1]
#
#         i += 1
#         print("进度 【{}/{}】".format(i, n))
#
#     # 完事之后将两个数组存起来
#     np.save(r"c:/Users/jchao/Desktop/truth_arr", truth_arr)
#     np.save(r"c:/Users/jchao/Desktop/pred_arr", pred_arr)

def moveToTarget(c, circle, x_begin, y_begin, x_end, y_end, t, n):
    x = (x_end - x_begin) / n
    y = (y_end - y_begin) / n
    for i in range(n):
        c.move(circle, x, y)  # move方法，把circle往（x，y）的方向移动
        c.update()  # 刷新屏幕
        time.sleep(t)  # 设置间隔时间，每0.03秒执行一次循环

# def timeOfOneFrame(cap, net, device):
#     #判断拿到一帧图像的时间是多少
#     for i in range(10):
#         ret, frame = cap.read()
#         getPoint(frame, net, device)
#
#     before = time.time()
#     for i in range(100):
#         ret, frame = cap.read()
#         getPoint(frame, net, device)
#         print("{}/100".format(i))
#     after = time.time()
#     print("一帧图像平均用时：", (after - before) / 100)


def func(cap, util):
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
        output = util.frameToPoint(frame)
        if ret == False or isinstance(output, type(None)):
            continue
        target1, target2 = output
        # target1 = 55 * target1 + 1085
        # target2 = -55 * target2 + 148
        # target1 += 960
        print(target1, target2)
        moveToTarget(c, circle, begin1, begin2, target1, target2, 0.001, 5)
        begin1, begin2  = target1, target2

    # c.mainloop()


# def buildMode():
#     # 模型准备
#     # global device, net
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print("Model building")
#     net = myCalibration.CalibrationModel(cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\59.png"))
#     # net = model.model()
#     net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
#     state_dict = torch.load(r".\checkPoint1\Iter_1401_feiwu-Net.pt")
#     # state_dict = torch.load(r".\checkPoint\Iter_2_AFF-Net.pt")
#     net.load_state_dict(state_dict)
#     net = net.module
#     net.to(device)
#     net.eval()  # 在模型test的时候加入
#     print("Model building done...")
#     return net, device


if __name__ == "__main__":
    # net, device = buildMode()
    # util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\gagachao", r"../checkpoint2/Iter_120_zuhefeiwuceshi-Net.pt")
    util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint/Iter_20_jie2.pt")
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # calibration(cap, net, device, 50)
    func(cap, util)



    cap.release()




