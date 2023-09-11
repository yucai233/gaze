import cv2
import numpy as np
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.1
import random
import os




def calibration(cap, n, path):
    # n 控制应该输入多少数据
    # truth_arr = np.zeros((n, 2))
    truth_arr = np.zeros((n + 15, 2))
    # 拿到屏幕的大小
    w, h = pyautogui.size()


    #首先将随机坐标存放到数组中
    randint_list = []
    for i in range(n):
        randint_list.append((random.randint(0, w), random.randint(0, h)))

    #然后是9个坐标
    for i in np.linspace(0, w, 5):
        for j in np.linspace(0, h, 3):
            if i == 1920:
                i -= 20
            if j == 1080:
                j -= 20
            randint_list.append((i, j))
    # #最后是一个坐标
    # randint_list.append((w / 2, 0))


    #前n张照片用随机的坐标
    for i, (x, y) in enumerate(randint_list):
        # 将鼠标移动到指定位置
        pyautogui.moveTo(x, y)

        # 等待用户拍照
        temp = input("按下Enter拍照:")
        ret, img = cap.read()  # cao.read()返回两个值，第一个存储一个bool值，表示拍摄成功与否。第二个是当前截取的图片帧。
        # showImg(img)

        if ret == False:
            print("保存失败")
            break

        # 将位置放到数组中
        truth_arr[i, 0] = x
        truth_arr[i, 1] = y
        #将照片保存
        cv2.imwrite(os.path.join(path, "{}.png".format(i)), img)

        print("进度 【{}/{}】".format(i + 1, n))


    # 完事之后将两个数组存起来

    np.save(os.path.join(path, "truth"), truth_arr)


if __name__ == "__main__":
    # 打开摄像头
    cap = cv2.VideoCapture(0)
    # 设置分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
    # 检查摄像头是否成功打开
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    calibration(cap, 0, r"C:\Users\jchao\Desktop\ceshi")

    cap.release()
