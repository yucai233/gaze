import copy
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torch
import pandas as pd
import torch.nn as nn
import model
import random

def getInterval(subs_num, idx):
    """
    通过传入的idx拿到对应对象的区间(左闭右开)
    :param subs_num:
    :param idx:
    :return:
    """
    subs_num = copy.copy(subs_num)
    mi = 0
    ma = 0
    for i in range(len(subs_num)):
        ma = subs_num[i]
        if idx < ma:
            break
        else:
            mi = ma

    if mi == ma:
        return None
    else:
        return mi, ma

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

def getX(img, net, device, rects):
    # 拿到分割图片
    rects = rects.astype(np.int64)
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

    return gazes.detach().squeeze()


def buildMode():
    # 模型准备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = model.model()
    net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
    state_dict = torch.load(r"..\checkPoint\Iter_2_AFF-Net.pt")
    net.load_state_dict(state_dict)
    net = net.module
    net.to(device)
    net.eval()  # 在模型test的时候加入
    print("Model building done...")
    return net, device


class loader(Dataset):
    def __init__(self, data_path, data_type):
        super(loader, self).__init__()
        self.data_path = data_path   #数据集的路径
        self.data_type = data_type   #判断是训练还是测试
        self.net, self.device = buildMode()

        self.imgs_path = []
        self.args = np.empty((0, 14), float)
        self.screensSize = np.loadtxt("C:/Users/jchao/Desktop/screenSize.txt", delimiter=' ')
        self.subs_num = []

        #直接找每个对象底下的txt文件就可以了
        for i in range(15):
            if i < 10:
                df = pd.read_table(r"D:/MPIIFaceGaze/p0{}/p0{}.txt".format(i, i), header=None, sep=' ')
                # 拿到该对象下的所有照片
                imgs_temp = df.iloc[:, 0].to_numpy().tolist()
                self.imgs_path += ["D:/MPIIFaceGaze/p0{}/".format(i) + img_path for img_path in imgs_temp]
            else:
                df = pd.read_table(r"D:/MPIIFaceGaze/p{}/p{}.txt".format(i, i), header=None, sep=' ')
                # 拿到该对象下的所有照片
                imgs_temp = df.iloc[:, 0].to_numpy().tolist()
                self.imgs_path += ["D:/MPIIFaceGaze/p{}/".format(i) + img_path for img_path in imgs_temp]

            #拿到该对象照片对应的label和landmarks
            subjectArgs = df.iloc[:, 1:15].to_numpy().astype(np.float64)
            height_mm, height_pixel, width_mm, width_pixel = self.screensSize[i]

            subjectArgs[:, 0] = (subjectArgs[:, 0] * width_mm / width_pixel - width_mm / 2) / 10
            subjectArgs[:, 1] = -1 * subjectArgs[:, 1] * height_mm / height_pixel / 10

            self.args = np.vstack((self.args, subjectArgs))

            self.subs_num.append(len(df))
        #对subs_num进行简单处理
        s = 0
        for i in range(len(self.subs_num)):
            s += self.subs_num[i]
            self.subs_num[i] = s

    def __getitem__(self, idx):
        img = cv2.imread(self.imgs_path[idx])
        rects = getRects(self.args[idx][2:])
        x1 = getX(img, self.net, self.device, rects)

        #拿到另外一张做对比的照片
        mi, ma = getInterval(self.subs_num, idx)
        idx_comp = random.randint(mi, ma - 1)  #random.randint是左闭右闭的
        img_comp = cv2.imread(self.imgs_path[idx_comp])
        rects_comp = getRects(self.args[idx_comp][2:])
        x2 = getX(img_comp, self.net, self.device, rects_comp)

        #拿到两个label
        label = self.args[idx][:2]
        label_comp = self.args[idx_comp][:2]

        label_ret = label - label_comp

        #返回参数
        return {
            "x1": x1.type(torch.FloatTensor),
            "x2": x2.type(torch.FloatTensor),
            "label": torch.from_numpy(label_ret).type(torch.FloatTensor)
        }

    def __len__(self):
        return len(self.imgs_path)



def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load





if __name__ == "__main__":
    subs_num = [2, 3, 5, 4]
    ret = getInterval(subs_num, 15)
    if isinstance(ret, type(None)):
        print("错误")
    else:
        print(ret)
