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




def showImg(img):
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()

def getRect(img):
    arr = np.zeros((3, 4))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 首先拿到脸的定位
    loc = face_recognition.face_locations(img)
    if len(loc) == 0:   #说明是空列表
        print("你的脸呢？")
        return None
    loc = loc[0]
    arr[0][0] = loc[3]
    arr[0][1] = loc[0]
    arr[0][2] = loc[1]
    arr[0][3] = loc[2]

    # 拿到两只眼睛的定位
    res = face_recognition.face_landmarks(img)
    str_list = ["left_eye", "right_eye"]
    for i, s in enumerate(str_list):
        x_ave, y_ave = np.sum(res[0][s], axis=0) / 6

        w = res[0][s][3][0] - res[0][s][0][0]
        w *= 1.7
        h = w
        arr[i + 1][0] = int(x_ave - w / 2)
        arr[i + 1][1] = int(y_ave - h / 2)
        arr[i + 1][2] = int(x_ave + w / 2)
        arr[i + 1][3] = int(y_ave + h / 2)

    return arr.astype(np.int64)

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



def getPoint(img, net):
    rects = getRect(img)
    if isinstance(rects, type(None)):
        print("未检测到人脸")
        return None

    # 拿到分割图片
    img_list = []
    for rect in rects:
        subimg = img[rect[1]:rect[3], rect[0]:rect[2]]  # .astype(np.float64)
        img_list.append(subimg)
        # showImg(subimg)

    #图片预处理
    face_img = cv2.resize(img_list[0], (224, 224))
    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    face_img = face_img / 255
    face_img = face_img.transpose(2, 0, 1)[np.newaxis, :]

    leftEye_img = cv2.resize(img_list[1], (112, 112))
    leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
    leftEye_img = leftEye_img / 255
    leftEye_img = leftEye_img.transpose(2, 0, 1)[np.newaxis, :]

    rightEye_img = cv2.resize(img_list[2], (112, 112))
    rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
    rightEye_img = cv2.flip(rightEye_img, 1)
    rightEye_img = rightEye_img / 255
    rightEye_img = rightEye_img.transpose(2, 0, 1)[np.newaxis, :]

    #拿到模型输入的rects
    #使用间接方法拿到用于输入的rects
    rects = rects.astype(np.float32)
    rects[:, 0] = rects[:, 0] / img.shape[1]
    rects[:, 2] = rects[:, 2] / img.shape[1]
    rects[:, 1] = rects[:, 1] / img.shape[0]
    rects[:, 3] = rects[:, 3] / img.shape[0]
    rects = rects.flatten().reshape(1, -1)
    # print("输入模型的rects为：\n", rects)


    gazes = net(torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(device),
                torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(device),
                torch.from_numpy(face_img).type(torch.FloatTensor).to(device),
                torch.from_numpy(rects).type(torch.FloatTensor).to(device))

    return gazes.cpu().detach().numpy().squeeze()


def calibration(n):
    cap = cv2.VideoCapture(0)
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
        output = getPoint(img, net)

        if isinstance(output, type(None)) or ret == False:
            continue

        # 将位置放到数组中
        truth_arr[i, 0] = x
        truth_arr[i, 1] = y
        pred_arr[i, 0] = output[0]
        pred_arr[i, 1] = output[1]

        i += 1
        print("进度 【{}/{}】".format(i, n))

    cap.release()  # 释放
    # 完事之后将两个数组存起来
    np.save(r"c:/Users/jchao/Desktop/truth_arr", truth_arr)
    np.save(r"c:/Users/jchao/Desktop/pred_arr", pred_arr)


###################################################################################################

def getDpi():
    root = tk.Tk()
    dpi = root.winfo_fpixels('1i')
    root.destroy()
    return dpi


def start_tracking():
    global f
    f = True
    thread = threading.Thread(target=start)
    thread.start()


def start():
    #首先拿到dpi
    dpi = getDpi()
    while f:
        time.sleep(0.1)
        # 读取当前帧
        ret, frame = cap.read()
        # cv2.imwrite(r"c:\Users\jchao\Desktop\frame.jpg", frame)
        # assert False

        # 检查当前帧是否成功读取
        if not ret:
            print("无法获取当前帧")
            exit()

        print("开始处理")
        # TODO 处理图像
        output = getPoint(frame, net)
        if isinstance(output, type(None)):
            print("未识别到有效信息")
            continue
        output_1 = output[0]
        output_2 = output[1]

        x = 1411.9683811444247 + 127.28654564849158 * output_1
        y = 314.0647885583127 - 123.97994830816558 * output_2
        smooth_move_circle(x, y)


def end_tracking():
    global f
    f = False
    print("stop")

# 循环显示摄像头内容
# def show_frame():
#     ret, frame = cap.read()
#     if ret:
#         # 缩小图像
#         scale_percent = 60  # 设置缩放比例
#         width = int(frame.shape[1] * scale_percent / 100)
#         height = int(frame.shape[0] * scale_percent / 100)
#         dim = (width, height)
#         resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
#
#         image = Image.fromarray(resized_frame)
#         image = ImageTk.PhotoImage(image)
#         label.config(image=image)
#         label.image = image
#     window.after(10, show_frame)


def mouseTrackingGaze():
    # 打开摄像头
    global cap
    cap = cv2.VideoCapture(0)

    # 设置分辨率
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    # # 创建窗口
    # window = ThemedTk(theme="Plastik")
    # window.title("摄像头追踪")
    # window.geometry("800x500")
    # # 设置水平和垂直方向的内边距
    # window['padx'] = 20
    # window['pady'] = 10
    #
    # # 创建开始追踪按钮的回调函数
    # f = True
    #
    #
    # # 创建按钮
    # start_button = tk.Button(window, text="开始追踪", command=start_tracking)
    # start_button.pack(side=tk.BOTTOM)
    # end_button = tk.Button(window, text="停止追踪", command=end_tracking)
    # end_button.pack(side=tk.BOTTOM)
    #
    # # 创建显示摄像头内容的标签
    # label = tk.Label(window)
    # label.pack()
    #
    # show_frame()
    #
    # # 启动窗口消息循环
    # window.mainloop()

    # 释放摄像头资源
    cap.release()


def mainn():
    #模型准备
    global device, net
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = model.model()
    net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
    state_dict = torch.load(r".\checkPoint\Iter_10_AFF-Net.pt")
    net.load_state_dict(state_dict)
    net = net.module
    net.to(device)
    net.eval()  # 在模型test的时候加入
    print("Model building done...")

    # for a in range(1, 17):
    #     print(a)
    #     img = cv2.imread(r"C:/Users/jchao/Desktop/bisai/picture/{}.jpg".format(a))
    #     print(getPoint(img, net))

    # calibration(60)


    mouseTrackingGaze()
    start()


# # 定义平滑移动函数
# def smooth_move_circle(target_x, target_y, step=1):
#     x = canvas.coords(image_item)[0]
#     y = canvas.coords(image_item)[1]
#     print(target_x, x)
#     print(target_y, y)
#     if x < target_x:
#         x += step
#     elif x > target_x:
#         x -= step
#
#     if y < target_y:
#         y += step
#     elif y > target_y:
#         y -= step
#
#     canvas.coords(image_item, x, y)
#     if x > target_x + 2 or y > target_y + 2 or x < target_x - 2 or y < target_y - 2:
#         window0.after(1, lambda: smooth_move_circle(target_x, target_y, step))
#
# # 定义移动函数
# def move_circle():
#     print("dskfnskd")
#     ret, frame = cap.read()
#     # target_x = random.randint(0, 1920)
#     # target_y = random.randint(0, 1200)
#     tmp = getPoint(frame, net)
#     if not isinstance(tmp, type(None)):
#         print("看到了")
#         [target_x, target_y] = tmp
#         target_x = 1412 + 127 * target_x
#         target_y = 314 - 124 * target_y
#
#         smooth_move_circle(target_x, target_y)
#
#         print(target_x, target_y)
#     window0.after(100, move_circle)  # 每1秒随机移动一次

import asyncio

# 定义平滑移动函数
async def smooth_move_circle(target_x, target_y, step=1):
    x = canvas.coords(image_item)[0]
    y = canvas.coords(image_item)[1]

    while x > target_x + 2 or y > target_y + 2 or x < target_x - 2 or y < target_y - 2:
        await asyncio.sleep(0.001)  # 等待一小段时间，以允许事件循环处理其他任务
        if x < target_x:
            x += step
        elif x > target_x:
            x -= step
        if y < target_y:
            y += step
        elif y > target_y:
            y -= step
        canvas.coords(image_item, x, y)

# 定义移动函数
async def move_circle():
    print("dskfnskd")
    ret, frame = cap.read()
    # target_x = random.randint(0, 1920)
    # target_y = random.randint(0, 1200)
    tmp = getPoint(frame, net)
    if not isinstance(tmp, type(None)):
        print("看到了")
        [target_x, target_y] = tmp
        target_x = 1412 + 127 * target_x
        target_y = 314 - 124 * target_y

        await smooth_move_circle(target_x, target_y)

        print(target_x, target_y)

async def main():
    while True:
        await move_circle()  # 等待 move_circle() 函数完成

    # 在这里添加其他需要执行的代码

    # await asyncio.sleep(1)  # 每1秒执行一次 move_circle()


# # 启动随机移动函数
# move_circle()

#模型准备
global device, net
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Model building")
net = model.model()
net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
state_dict = torch.load(r".\checkPoint\Iter_10_AFF-Net.pt")
net.load_state_dict(state_dict)
net = net.module
net.to(device)
net.eval()  # 在模型test的时候加入
print("Model building done...")

#获取屏幕大小
# 打开摄像头
cap = cv2.VideoCapture(0)

# 设置分辨率
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()

screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

# 创建透明窗口
window0 = tk.Tk()
window0.overrideredirect(True)
window0.wm_attributes('-transparentcolor', 'black')
window0.attributes('-topmost', True)
window0.geometry(f"{screen_width}x{screen_height}+0+0")

# 设置窗口为无焦点状态
win32gui.SetWindowLong(window0.winfo_id(), win32con.GWL_EXSTYLE,
                       win32gui.GetWindowLong(window0.winfo_id(), win32con.GWL_EXSTYLE) | win32con.WS_EX_NOACTIVATE)

# 创建画布
canvas = tk.Canvas(window0, width=screen_width, height=screen_height, highlightthickness=0, bg='black')
canvas.pack()

# 创建PIL图像
diameter = 100
shadow_size = 5
image_size = diameter + shadow_size * 2
image = Image.new('RGBA', (image_size, image_size), (0, 0, 0, 0))
draw = ImageDraw.Draw(image)

gradient_radius = diameter / 2
gradient_center = (image_size // 2, image_size // 2)

for i in range(shadow_size):
    alpha = int((1 - i / shadow_size) * 255)
    shadow_color = (128, 128, 128, alpha)
    shadow_radius = gradient_radius + i
    shadow_bbox = (
        gradient_center[0] - shadow_radius,
        gradient_center[1] - shadow_radius,
        gradient_center[0] + shadow_radius,
        gradient_center[1] + shadow_radius
    )
    draw.ellipse(shadow_bbox, outline=shadow_color)

circle_radius = diameter // 2
circle_bbox = (
    gradient_center[0] - circle_radius,
    gradient_center[1] - circle_radius,
    gradient_center[0] + circle_radius,
    gradient_center[1] + circle_radius
)
draw.ellipse(circle_bbox, outline='white')

# 将PIL图像转换为Tkinter图像
tk_image = ImageTk.PhotoImage(image)

# 在画布上显示图像
image_item = canvas.create_image(200, 200, image=tk_image)

loop = asyncio.get_event_loop()
def eventLoop(loop):
    asyncio.set_event_loop(loop)
    loop.run_until_complete(main())
thread = threading.Thread(target=lambda : eventLoop(loop))
thread.start()

window0.mainloop()
# while True:
# time.sleep(0.1)
# move_circle()
# 运行窗口主循环
print("水电费即可萨芬看电视剧")

cap.release()


