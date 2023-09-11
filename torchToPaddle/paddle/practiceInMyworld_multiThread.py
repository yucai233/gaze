import cv2
import time

import pyautogui
from screeninfo import get_monitors
import win32gui
import win32con
from tkinter import *

import pydirectinput
pydirectinput.FAILSAFE = False

from multiprocessing import Process, Pipe
import threading

from mcpi.minecraft import Minecraft

import frameToPoint



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
        self.begin1, self.begin2 = 960, 540  #being1和begin2分别表示圆心的x和y
        self.radius = 80
        #根据圆心和半径绘制圆
        self.circle = self.c.create_oval(self.begin1 - self.radius,
                                         self.begin2 - self.radius,
                                         self.begin1 + self.radius,
                                         self.begin2 + self.radius,
                                         fill='red')  # 绘制一个圆

    def moveToTarget(self, x_end, y_end, t, n):
        #判断在x和y方向各自应该移动多少
        x = (x_end - self.begin1) / n
        y = (y_end - self.begin2) / n
        #开始移动
        for i in range(n):
            self.c.move(self.circle, x, y)  # move方法，把circle往水平竖直方向分别移动x和y个单位
            self.c.update()  # 刷新屏幕
            time.sleep(t)  # 设置间隔时间，每0.03秒执行一次循环

        #更新圆心坐标
        self.begin1, self.begin2 = x_end, y_end


# #调试小球
# if __name__ == "__main__":
#     rp = PoGVisualable()
#
#     while True:
#         x_mouse, y_mouse = pyautogui.position()
#         rp.moveToTarget(x_mouse, y_mouse, 0.001, 5)
#
#     # rp.moveToTarget(200, 200, 0.001, 5)
#     # pyautogui.moveTo(200, 200)
#     # while True:
#     #     rp.c.update()



#控制MC类
class MCControl():
    def __init__(self):
        # 连接到Minecraft游戏
        self.mc = Minecraft.create()
        # 获取当前玩家的水平角度信息
        self.rotation = self.mc.player.getRotation()
        #控制前进后退的嘴巴的阈值
        self.earThreshold = 0.2

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

#定义用于线程控制的全局变量
event = threading.Event()
mylock = threading.RLock()
parameter = []

#定义子线程
#子线程的作用主要是和主进程打交道，
def childThread(child_pipe):
    print("子线程开始运行")
    global event, mylock, parameter
    while True:
        #首先监视主进程发送过来的请求
        child_pipe.recv()
        #当拿到主进程的请求之后，首先确定数据是否已经就绪
        event.wait()
        #数据就绪后，拿到锁，开始访问数据，并发送给主进程
        mylock.acquire()
        child_pipe.send(parameter)
        #重置数据就绪信号
        event.clear()
        mylock.release()




#定义子进程（只用来提取图片和进行人脸识别相关的内容）
#主线程需要做两件事：
#第一件：显示视屏
#第二件，将处理好的参数放到全局变量parameter中供子线程随时取用
def childP(child_pipe):
    CAP_NAME = " "
    CAP_W_SIZE = 200
    print("子进程开始运行")

    global event, mylock, parameter
    event.clear()
    #将子线程分离出去
    childT = threading.Thread(target=childThread, args=(child_pipe,))
    childT.start()

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

    #对摄像头窗口进行设置
    ret, frame = cap.read()
    cv2.imshow(CAP_NAME, frame)
    hwnd = win32gui.FindWindow(None, CAP_NAME)
    # 将窗口置顶
    win32gui.SetWindowPos(hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, win32con.SWP_NOMOVE | win32con.SWP_NOSIZE)
    # 去除选项
    style = win32gui.GetWindowLong(hwnd, win32con.GWL_STYLE)
    style &= ~win32con.WS_SYSMENU
    # style &= ~win32con.WS_CAPTION
    win32gui.SetWindowLong(hwnd, win32con.GWL_STYLE, style)
    cv2.waitKey(1) & 0xff

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
        h, w, _ = img.shape
        img = cv2.resize(img, (CAP_W_SIZE, int(CAP_W_SIZE / w * h)))
        cv2.putText(img, "EAR: {:.2f}".format(result[1]), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(CAP_NAME, img)
        key = cv2.waitKey(1) & 0xff
        if key == ord("q"):
            break

        #构建返回值
        #申请锁
        mylock.acquire()
        parameter = list(result)
        parameter.append(frame)
        event.set()
        mylock.release()


    cv2.destroyAllWindows()
    cap.release()
    child_pipe.close()


#定义主进程
if __name__ == "__main__":
    #首先分离子进程出去
    child_pipe, parent_pipe = Pipe(True)
    child_process = Process(target=childP, args=(child_pipe,))
    child_process.start()
    print("子线程创建完毕")

    #开始做准备工作
    util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagachao", r"./checkpoint_3_layer/Iter_4_comb.pdparams")
    pogv = PoGVisualable()
    # mcc = MCControl()
    print("准备工作完成")

    while (True):
        #向子进程发送数据请求
        parent_pipe.send(int(1))
        #首先等待子进程的数据
        data_list = parent_pipe.recv()

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




