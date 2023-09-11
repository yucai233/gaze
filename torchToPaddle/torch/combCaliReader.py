import os

import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import random
import copy
import pandas as pd
import face_recognition
import dlib
from imutils import face_utils

detector = dlib.get_frontal_face_detector()
# 使用模型构建特征提取器
predictor = dlib.shape_predictor('E:/Anoconda/envs/py37/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')


def getRectOfRec(img):
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


def getRectOfDlib(img):
    arr = np.zeros((3, 4))
    # 将图片转化为灰度图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 首先拿到人脸
    dets = detector(img, 0)
    #判断是否检测到人脸
    if len(dets) == 0:
        print("你的脸呢？")
        return None
    arr[0][0] = dets[0].left()
    arr[0][1] = dets[0].top()
    arr[0][2] = dets[0].right()
    arr[0][3] = dets[0].bottom()

    # 然后拿到人的眼睛
    dlib_shape = predictor(img, dets[0])
    str_list = ["left_eye", "right_eye"]
    res = face_utils.shape_to_np(dlib_shape)
    for i, s in enumerate(str_list):
        b, e = face_utils.FACIAL_LANDMARKS_IDXS[s]
        x_ave, y_ave = np.sum(res[b:e, :], axis=0) / 6

        w = res[b:e, :][3][0] - res[b:e, :][0][0]
        w *= 1.7
        h = w
        arr[i + 1][0] = int(x_ave - w / 2)
        arr[i + 1][1] = int(y_ave - h / 2)
        arr[i + 1][2] = int(x_ave + w / 2)
        arr[i + 1][3] = int(y_ave + h / 2)

    return arr.astype(np.int64)

def getPar(img, label):
    rects = getRectOfDlib(img)
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
    face_img = face_img.transpose(2, 0, 1)

    leftEye_img = cv2.resize(img_list[1], (112, 112))
    leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
    leftEye_img = leftEye_img / 255
    leftEye_img = leftEye_img.transpose(2, 0, 1)

    rightEye_img = cv2.resize(img_list[2], (112, 112))
    rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
    rightEye_img = cv2.flip(rightEye_img, 1)
    rightEye_img = rightEye_img / 255
    rightEye_img = rightEye_img.transpose(2, 0, 1)

    #拿到模型输入的rects
    #使用间接方法拿到用于输入的rects
    rects = rects.astype(np.float32)
    rects[:, 0] = rects[:, 0] / img.shape[1]
    rects[:, 2] = rects[:, 2] / img.shape[1]
    rects[:, 1] = rects[:, 1] / img.shape[0]
    rects[:, 3] = rects[:, 3] / img.shape[0]
    rects = rects.flatten().reshape(1, -1).squeeze()
    # print("输入模型的rects为：\n", rects)

    return leftEye_img, rightEye_img, face_img, rects, label



class loader(Dataset):

    def __init__(self, cali_data_path, truth_arr_name):
        super(loader, self).__init__()

        self.pair = []  #用来挑选哪两张图片的
        self.eles = []  #用来存放每一张图片的所有元素

        #首先把label拿到手
        labels = np.load(os.path.join(cali_data_path, truth_arr_name))
        width_mm = 345.353
        width_pixel = 1920.0
        height_mm = 194.26
        height_pixel = 1080
        labels[:, 0] = (labels[:, 0] * width_mm / width_pixel - width_mm / 2) / 10
        labels[:, 1] = -1 * labels[:, 1] * height_mm / height_pixel / 10

        #判断一共有多少图片
        n = len(os.listdir(cali_data_path)) - 1
        #将图片全部导入并做处理
        for i in range(n):
            img = cv2.imread(os.path.join(cali_data_path, "{}.png".format(i)))
            # self.eles.append(getPar(img, labels[i]))
            self.eles.append(getPar(img, labels[i]))

        #拿到图片位置信息
        for i in range(n):
            for j in range(i, n):
                self.pair.append((i, j))


        print("初始化完成~~~~~~")



    def __len__(self):
        return len(self.pair)


    def __getitem__(self, idx):
        idx1, idx2 = self.pair[idx]
        leftEye_img1, rightEye_img1, face_img1, inputRects1, label1 = self.eles[idx1]
        leftEye_img2, rightEye_img2, face_img2, inputRects2, label2 = self.eles[idx2]

        label = label1 - label2
        return {"faceImg1": torch.from_numpy(face_img1).type(torch.FloatTensor),
                "leftEyeImg1": torch.from_numpy(leftEye_img1).type(torch.FloatTensor),
                "rightEyeImg1": torch.from_numpy(rightEye_img1).type(torch.FloatTensor),
                "rects1": torch.from_numpy(inputRects1).type(torch.FloatTensor),
                "faceImg2": torch.from_numpy(face_img2).type(torch.FloatTensor),
                "leftEyeImg2": torch.from_numpy(leftEye_img2).type(torch.FloatTensor),
                "rightEyeImg2": torch.from_numpy(rightEye_img2).type(torch.FloatTensor),
                "rects2": torch.from_numpy(inputRects2).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor)}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, "truth.npy")
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load



if __name__ == "__main__":
    l = loader(r"C:\Users\jchao\Desktop\gagajie", "truth.npy")
    print(len(l))
    # print(l.__getitem__(37090))


    # img = cv2.imread(r"C:\Users\jchao\Desktop\gagajie\0.png")
    # leftEye_img, rightEye_img, face_img, rects, _ = getPar(img, None)
    # print(leftEye_img.shape)
    # print(rightEye_img.shape)
    # print(face_img.shape)
    # print(rects.shape)
    #
    # leftEye_img, rightEye_img, face_img, rects, _ = getInput(img, None)
    # print(leftEye_img.shape)
    # print(rightEye_img.shape)
    # print(face_img.shape)
    # print(rects.shape)