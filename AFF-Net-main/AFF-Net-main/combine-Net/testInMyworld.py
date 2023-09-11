import cv2
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 1

from screeninfo import get_monitors
import win32gui
import win32con
from tkinter import *
import time

import frameToPoint

from mcpi.minecraft import Minecraft



def moveToTarget(c, circle, x_begin, y_begin, x_end, y_end, t, n):
    x = (x_end - x_begin) / n
    y = (y_end - y_begin) / n
    for i in range(n):
        c.move(circle, x, y)  # move方法，把circle往（x，y）的方向移动
        c.update()  # 刷新屏幕
        time.sleep(t)  # 设置间隔时间，每0.03秒执行一次循环



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

    # 连接到Minecraft游戏
    mc = Minecraft.create()
    #获取当前玩家的水平角度信息
    rotation = mc.player.getRotation()

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

        #移动玩家水平视角
        if target1 > 910 and target1 < 1010:
            continue
        elif target1 < 910:
            rotation = (rotation + 360 - 10) % 360
        else:
            rotation = (rotation + 360) % 360
        mc.player.setRotation(rotation)





if __name__ == "__main__":
    # 连接到Minecraft游戏
    mc = Minecraft.create()
    #获取当前玩家的水平角度信息
    rotation = mc.player.getRotation()

    while True:
        x = 1100
        # #首先使用pyautogui获取当前鼠标的位置
        # x, y = pyautogui.position()
        #然后根据当前鼠标的位置，判断视角的转变
        if x > 910 and x < 1010:
            continue
        elif x < 910:
            rotation = (rotation + 360 - 10) % 360
        else:
            rotation = (rotation + 10) % 360
        mc.player.setRotation(rotation)
        time.sleep(1)
        print("haha")



