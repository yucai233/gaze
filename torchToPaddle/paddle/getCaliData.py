import sys
import time
import win32gui
import win32con
from screeninfo import get_monitors
from tkinter import *
import pyautogui
import cv2



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
        self.radius = 20
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


#调试小球
if __name__ == "__main__":
    rp = PoGVisualable()

    while True:
        x_mouse, y_mouse = pyautogui.position()
        rp.moveToTarget(x_mouse, y_mouse, 0.001, 5)

    # rp.moveToTarget(200, 200, 0.001, 5)
    # pyautogui.moveTo(200, 200)
    # while True:
    #     rp.c.update()