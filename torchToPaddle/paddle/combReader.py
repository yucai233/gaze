import os.path

import numpy as np
import cv2
import random
import copy
import pandas as pd

import paddle


def getRects(pos_arr):
    x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, x6, y6 = pos_arr
    arr = np.zeros((3, 4))
    # 计算左眼
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    w = (x2 - x1) * 1.7
    arr[0][0] = x_center - w / 2
    arr[0][1] = y_center - w / 2
    arr[0][2] = x_center + w / 2
    arr[0][3] = y_center + w / 2

    # 计算右眼
    x_center = (x3 + x4) / 2
    y_center = (y3 + y4) / 2
    w = (x4 - x3) * 1.7
    arr[1][0] = x_center - w / 2
    arr[1][1] = y_center - w / 2
    arr[1][2] = x_center + w / 2
    arr[1][3] = y_center + w / 2

    # 计算脸
    x_center = (x1 + x2 + x3 + x4 + x5 + x6) / 6
    y_center = (y1 + y2 + y3 + y4 + y5 + y6) / 6
    w *= 1 / 0.3
    #     w *= 4
    arr[2][0] = x_center - w / 2
    arr[2][1] = y_center - w / 2
    arr[2][2] = x_center + w / 2
    arr[2][3] = y_center + w / 2

    #如果碰到有负数，将它变成0
    arr[arr < 0] = 0
    return arr

def getInputRects(rects, shape):
    rects = copy.copy(rects)
    #使用间接方法拿到用于输入的rects
    ratio_w = 1000 / shape[1]
    ratio_h = 1000 / shape[0]
    rects[:, 0] = rects[:, 0] * ratio_w
    rects[:, 2] = rects[:, 2] * ratio_w
    rects[:, 1] = rects[:, 1] * ratio_h
    rects[:, 3] = rects[:, 3] * ratio_h

    return rects


class loader(paddle.io.Dataset):

    def __init__(self, data_path, screenSize_path):
        super(loader, self).__init__()
        self.data_path = data_path   #数据集的路径
        self.screenSize_path = screenSize_path  #存放每个对象的电脑尺寸

        self.subs_num = []           #定义

        self.imgs_path = []
        self.args = np.empty((0, 14), float)
        self.screensSize = np.loadtxt(screenSize_path, delimiter=' ')

        #直接找每个对象底下的txt文件就可以了
        for i in range(15):
            if i < 10:
                df = pd.read_table(os.path.join(self.data_path, r"p0{}/p0{}.txt".format(i, i)), header=None, sep=' ')
                # 拿到该对象下的所有照片
                imgs_temp = df.iloc[:, 0].to_numpy().tolist()
                self.imgs_path += [os.path.join(os.path.join(self.data_path, "p0{}".format(i)), img_path) for img_path in imgs_temp]
            else:
                df = pd.read_table(os.path.join(self.data_path, r"p{}/p{}.txt".format(i, i)), header=None, sep=' ')
                # 拿到该对象下的所有照片
                imgs_temp = df.iloc[:, 0].to_numpy().tolist()
                self.imgs_path += [os.path.join(os.path.join(self.data_path, "p{}".format(i)), img_path) for img_path in imgs_temp]
            #拿到该对象照片对应的label和landmarks
            subjectArgs = df.iloc[:, 1:15].to_numpy().astype(np.float64)
            height_mm, height_pixel, width_mm, width_pixel = self.screensSize[i]

            subjectArgs[:, 0] = (subjectArgs[:, 0] * width_mm / width_pixel - width_mm / 2) / 10
            subjectArgs[:, 1] = -1 * subjectArgs[:, 1] * height_mm / height_pixel / 10

            self.args = np.vstack((self.args, subjectArgs))
            self.subs_num.append(len(df))        #每一个对象有多少个元素

        #对subs_num进行简单处理
        s = 0
        for i in range(len(self.subs_num)):
            s += self.subs_num[i]
            self.subs_num[i] = s




    def __len__(self):
        return len(self.imgs_path)

    def getInput(self, idx):
        img = cv2.imread(self.imgs_path[idx])
        img = np.array(img)
        rects = getRects(self.args[idx][2:])
        inputRects = getInputRects(rects, img.shape)
        inputRects = inputRects.reshape(1, -1).squeeze() / 1000
        rects = rects.astype(np.int64)

        leftEye_img = img[rects[0][1]:rects[0][3], rects[0][0]:rects[0][2]]
        rightEye_img = img[rects[1][1]:rects[1][3], rects[1][0]:rects[1][2]]
        face_img = img[rects[2][1]:rects[2][3], rects[2][0]:rects[2][2]]

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img / 255
        face_img = face_img.transpose(2, 0, 1)

        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img, 1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        label = self.args[idx][:2]

        return face_img, leftEye_img, rightEye_img, inputRects, label

    def getInterval(self, idx):
        mi = 0
        ma = 0
        for i in range(len(self.subs_num)):
            ma = self.subs_num[i]
            if idx < ma:
                break
            else:
                mi = ma

        if mi == ma:
            return None
        else:
            return mi, ma

    def __getitem__(self, idx):
        # print(idx)
        face_img1, leftEye_img1, rightEye_img1, inputRects1, label1 = self.getInput(idx)

        #拿到另一张照片的索引
        mi, ma = self.getInterval(idx)
        idx_comp = random.randint(mi, ma - 1)
        face_img2, leftEye_img2, rightEye_img2, inputRects2, label2 = self.getInput(idx_comp)

        label = label1 - label2
        return {"faceImg1": paddle.to_tensor(face_img1, dtype="float32"),
                "leftEyeImg1": paddle.to_tensor(leftEye_img1, dtype="float32"),
                "rightEyeImg1": paddle.to_tensor(rightEye_img1, dtype="float32"),
                "rects1": paddle.to_tensor(inputRects1, dtype="float32"),
                "faceImg2": paddle.to_tensor(face_img2, dtype="float32"),
                "leftEyeImg2": paddle.to_tensor(leftEye_img2, dtype="float32"),
                "rightEyeImg2": paddle.to_tensor(rightEye_img2, dtype="float32"),
                "rects2": paddle.to_tensor(inputRects2, dtype="float32"),
                "label": paddle.to_tensor(label, dtype="float32")}


def txtload(data_path, screenSize_path, batch_size, shuffle=False, num_workers=0):
    dataset = loader(data_path, screenSize_path)
    print("[Read Data]: MPIIFaceGaze Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    load = paddle.io.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load



if __name__ == "__main__":
    l = loader(r"D:/MPIIFaceGaze", r"c:/Users/jchao/Desktop/screenSize.txt")
    print(l.__getitem__(30000))