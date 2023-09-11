import timeit

import cv2
import pyautogui
import math
# pyautogui.FAILSAFE = False
# pyautogui.PAUSE = 1

from screeninfo import get_monitors
import win32gui
import win32con
from tkinter import *
import time

import frameToPoint

from mcpi.minecraft import Minecraft
import pydirectinput
pydirectinput.FAILSAFE = False

import copy
from multiprocessing import Process, Pipe
import threading


#可视化PoG类
class PoGVisualable():
    def __init__(self):
        screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

        # 创建透明窗口
        root = Tk()
        root.overrideredirect(True)
        root.wm_attributes('-transparentcolor', 'black')
        root.attributes('-topmost', True)
        root.geometry(f"{screen_width}x{screen_height}+0+0")
        # 设置窗口为无焦点状态
        win32gui.SetWindowLong(root.winfo_id(), win32con.GWL_EXSTYLE,
                               win32gui.GetWindowLong(root.winfo_id(),
                                                      win32con.GWL_EXSTYLE) | win32con.WS_EX_NOACTIVATE)

        screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height

        # 绘制一个画布
        self.c = Canvas(root, width=screen_width, height=screen_height, highlightthickness=0, bg='black')
        self.c.pack()

        # # 在画布上显示图像
        self.begin1, self.begin2 = 200, 200
        self.circle = self.c.create_oval(self.begin1, self.begin2, 65, 62, fill='red')  # 绘制一个圆


    def moveToTarget(self, x_end, y_end, t, n):
        x = (x_end - self.begin1) / n
        y = (y_end - self.begin2) / n
        for i in range(n):
            self.c.move(self.circle, x, y)  # move方法，把circle往（x，y）的方向移动
            self.c.update()  # 刷新屏幕
            time.sleep(t)  # 设置间隔时间，每0.03秒执行一次循环
        self.begin1, self.begin2 = x_end, y_end

#控制MC类
class MCControl():
    def __init__(self):
        # 连接到Minecraft游戏
        self.mc = Minecraft.create()
        # 获取当前玩家的水平角度信息
        self.rotation = self.mc.player.getRotation()
        #控制前进后退的嘴巴的阈值
        self.earThreshold = 0.3

    def moveVA(self, target1, target2, ear):
        #移动玩家水平视角
        biasX, biasY = target1 - 960, target2 - 540
        print(biasX, biasY)
        if -100 < biasX < 100:
            return
        else:
            self.rotation = (self.rotation + 360 + 45 * (biasX / 1920)) % 360
        # else:
        #     rotation = (rotation + 10) % 360
        self.mc.player.setRotation(self.rotation)

        # # 移动玩家水平视角
        # if target1 > 910 and target1 < 1010:
        #     return
        # elif target1 < 910:
        #     for i in range(20):
        #         self.rotation = (self.rotation + 360 - 1) % 360
        #         self.mc.player.setRotation(self.rotation)
        #         time.sleep(0.001)
        # else:
        #     for i in range(20):
        #         self.rotation = (self.rotation + 1) % 360
        #         self.mc.player.setRotation(self.rotation)
        #         time.sleep(0.001)

        # 根据嘴巴控制移动与停止
        if ear >= self.earThreshold:
            pydirectinput.keyDown("w")
        else:
            pydirectinput.keyUp("w")

#定义子进程（只用来提取图片和进行人脸识别相关的内容）
def childP(child_pipe):
    print("子进程开始运行")

    #首先进行准备工作
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    #准备工作完毕，等待主进程发送开始工作的指示
    child_pipe.recv()

    #开始工作
    while True:
        #首先拿到一帧图片
        ret, frame = cap.read()
        if ret == False:
            print("图片读取失败")
            continue
        #对图片进行人脸识别相关操作
        result = frameToPoint.getRectOfDlib(frame)
        if isinstance(result, type(None)):
            print("你的脸呢？")
            continue
        #可视化出来
        img = frame.copy()
        cv2.putText(img, "MOUTH_EAR: {:.2f}".format(result[1]), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow("img", img)
        # cv2.imshow("frame", frame)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break

        #构建返回值
        result = list(result)
        result.append(frame)
        #返回给主进程
        child_pipe.send(result)
        #等待主进程拿走数据
        child_pipe.recv()

    cv2.destroyAllWindows()
    cap.release()
    child_pipe.close()


#定义主进程
if __name__ == "__main__":
    #首先分离子进程出去
    child_pipe, parent_pipe = Pipe(True)
    child_process = Process(target=childP, args=(child_pipe,))
    child_process.start()
    print("子进程创建完毕")

    #开始做准备工作
    util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./Iter_10_jie2.pt")
    pogv = PoGVisualable()
    # mcc = MCControl()
    print("准备工作完成")

    #向子进程发送信号，可以开始工作了
    parent_pipe.send(int(1))
    while (True):
        #首先等待子进程的数据
        data_list = parent_pipe.recv()
        #拿到数据之后，对子进程发信号，让它继续工作
        parent_pipe.send(int(1))

        #对子进程发送过来的数据进行处理
        rects, ear, frame = data_list
        output = util.frameToPoint(frame, rects)
        if isinstance(output, type(None)):
            print("你的脸呢")
            continue

        target1, target2 = output
        #移动小球
        pogv.moveToTarget(target1, target2, 0.001, 5)
        #移动视角
        # mcc.moveVA(target1, target2, ear)
        # time.sleep(1)




