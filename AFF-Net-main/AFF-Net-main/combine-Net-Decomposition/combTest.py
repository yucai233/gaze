import time

import combModel
import torch
from torch import nn
import numpy as np
import cv2
import face_recognition
import pyautogui
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 1
from scipy.stats import norm


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

def getPar(img):
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

    return leftEye_img, rightEye_img, face_img, rects

def getDiff(img1, img2, net, device):
    leftEye1, rightEye1, face1, rects1 = getPar(img1)
    leftEye2, rightEye2, face2, rects2 = getPar(img2)
    gazes = net(torch.from_numpy(leftEye1).type(torch.FloatTensor).to(device),
                torch.from_numpy(rightEye1).type(torch.FloatTensor).to(device),
                torch.from_numpy(face1).type(torch.FloatTensor).to(device),
                torch.from_numpy(rects1).type(torch.FloatTensor).to(device),
                torch.from_numpy(leftEye2).type(torch.FloatTensor).to(device),
                torch.from_numpy(rightEye2).type(torch.FloatTensor).to(device),
                torch.from_numpy(face2).type(torch.FloatTensor).to(device),
                torch.from_numpy(rects2).type(torch.FloatTensor).to(device)
                )

    return gazes.cpu().detach().numpy().squeeze()

def buileModel(modelPath, type):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = combModel.model()

    net = nn.DataParallel(net)  # 用到了显卡，所以显卡中必须有一份net的模型
    net = net.module

    state_dict = torch.load(modelPath)
    net.load_state_dict(state_dict)
    net.to(device)
    if type == "eval":
        net.eval()
    elif type == "train":
        net.train()
    else:
        print("error")
        return None

    print("Model building done")

    return net, device

# def getImgForCalibration(n = 15):
#     imgs_list = []
#     for i in range(n):
#         imgs_list.append(cv2.imread(r"C:\Users\jchao\Desktop\c_15\{}.png".format(i)))
#     imgs_pos = np.load(r"C:\Users\jchao\Desktop\c_15\truth_arr.npy")
#     imgs_pos[imgs_pos < 0] = 0
#
#     return imgs_list, imgs_pos

def getImgForCalibration(n = 9):
    imgs_list = []
    for i in range(50, 59):
        imgs_list.append(cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(i)))
    imgs_pos = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\truth_arr.npy")[-10:-1, :]
    imgs_pos[imgs_pos < 0] = 0

    return imgs_list, imgs_pos


def getK(imgs_list, imgs_pos, net, device):
    kx, ky = 0, 0
    divided = 0

    #拿到任意两张图片的组合
    n = len(imgs_list)
    for i in range(n):
        for j in range(i + 1, n):
            diff = getDiff(imgs_list[i], imgs_list[j], net, device)
            # print("diff----------------------------------------")
            # print(diff)
            kx += (imgs_pos[i][0] - imgs_pos[j][0]) / diff[0]
            ky += (imgs_pos[i][1] - imgs_pos[j][1]) / diff[1]
            divided += 1

    kx /= divided
    ky /= divided
    print("kx:{} ky:{}".format(kx, ky))
    return kx, ky

def getOneGaze(img1, img2, net, device, kx, ky, gaze_img2_x, gaze_img2_y):
    diff = getDiff(img1, img2, net, device)
    gaze_img1_x = gaze_img2_x + kx * diff[0]
    gaze_img1_y = gaze_img2_y + ky * diff[1]

    return gaze_img1_x, gaze_img1_y, abs(diff[0]) + abs(diff[1])

# def getBestGaze(img, imgs_list, imgs_pos, net, device, kx, ky):
#     gaze_x, gaze_y = 0, 0
#     minDiff = 100000
#     n = len(imgs_list)
#
#     for i in range(n):
#         x, y, diff = getOneGaze(img, imgs_list[i], net, device, kx, ky, imgs_pos[i][0], imgs_pos[i][1])
#         if diff < minDiff:
#             #如果发现有更合适的，就更新参数
#             gaze_x, gaze_y = x, y
#             minDiff = diff
#
#     return gaze_x, gaze_y


def getGaze(img, imgs_list, imgs_pos, net, device, kx, ky):
    gaze_x, gaze_y = 0, 0
    n = len(imgs_list)
    w_sum = 0

    for i in range(n):
        x, y, diff = getOneGaze(img, imgs_list[i], net, device, kx, ky, imgs_pos[i][0], imgs_pos[i][1])
        w = 1 / diff
        # w = norm.pdf(diff, loc=0, scale=0.1)
        # print(diff, w)
        gaze_x += x * w
        gaze_y += y * w
        w_sum += w

    gaze_x /= w_sum
    gaze_y /= w_sum

    return gaze_x, gaze_y


if __name__ == "__main__":
    net, device = buileModel("../checkpoint2/Iter_39_zuhefeiwu-Net.pt", "eval")
    arr = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\truth_arr.npy")
    arr[arr < 0] = 0

    #首先拿到用于校正的n张图片
    imgs_list, imgs_pos = getImgForCalibration()
    print(len(imgs_list))
    print(imgs_pos)
    #计算kx和ky
    # kx = 1920 / 34.5353
    # ky = 1080 / 19.426
    kx, ky = getK(imgs_list, imgs_pos, net, device)

    for i in range(59):
    # while True:
    #     i = int(input("->"))
    #     j = int(input("->"))
        img1 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(i))
        # img2 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(j))

        print("pred->", getGaze(img1, imgs_list, imgs_pos, net, device, kx, ky))
        print("real->", arr[i])

        # diff = getDiff(img1, img2, net, device)
        # print("diff:", diff)
        # print("gaze_diff", kx * diff[0], ky * diff[1])
        # print("img1 pred", getOneGaze(img1, img2, net, device, kx, ky, arr[j][0], arr[j][1]))
        # print("img1_real", arr[i][0], arr[i][1])
        # print("img2_real", arr[j][0], arr[j][1])


# if __name__ == "__main__":
#     net, device = buileModel("../checkpoint2/Iter_39_zuhefeiwu-Net.pt", "eval")
#     img1 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(1))
#     img2 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(2))
#     print(getDiff(img1, img2, net, device))