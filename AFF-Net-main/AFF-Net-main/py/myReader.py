import numpy as np
import cv2
import os

import pandas
from torch.utils.data import Dataset, DataLoader
import torch
import json
import random
import copy
import pandas as pd


# randomly move the bounding box around
def aug_line(line, width, height):
    bbox = np.array(line[2:5])
    bias = round(30 * random.uniform(-1, 1))
    bias = max(np.max(-bbox[0, [0, 2]]), bias)
    bias = max(np.max(-2*bbox[1:, [0, 2]]+0.5), bias)
    
    line[2][0] += int(round(bias))
    line[2][1] += int(round(bias))
    line[2][2] += int(round(bias))
    line[2][3] += int(round(bias))

    line[3][0] += int(round(0.5*bias))
    line[3][1] += int(round(0.5*bias))
    line[3][2] += int(round(0.5*bias))
    line[3][3] += int(round(0.5*bias))

    line[4][0] += int(round(0.5*bias))
    line[4][1] += int(round(0.5*bias))
    line[4][2] += int(round(0.5*bias))
    line[4][3] += int(round(0.5*bias))

    line[5][2] = line[2][2]/width
    line[5][3] = line[2][0]/height

    line[5][6] = line[3][2]/width
    line[5][7] = line[3][0]/height

    line[5][10] = line[4][2]/width
    line[5][11] = line[4][0]/height
    return line


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


class loader(Dataset):

    def __init__(self, data_path, data_type):
        self.data_path = data_path   #数据集的路径
        self.data_type = data_type   #判断是训练还是测试

        self.imgs_path = []
        self.args = np.empty((0, 14), float)
        self.screensSize = np.loadtxt("C:/Users/jchao/Desktop/screenSize.txt", delimiter=' ')

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


        # subjects = os.listdir(data_path)[:-1]
        # subjects.sort()
        #
        # for subject in subjects:
        #     #得到每一个对象所在的文件夹
        #     subject_path = os.path.join(data_path, subject)
        #
        #     # 首先将所有的照片路径加载到imgs_path中
        #
        #     days_dir = os.listdir(subject_path)[1:-1]
        #     for day in days_dir:
        #         day_path = os.path.join(subject_path, day)
        #         for img_name in os.listdir(day_path):
        #             img_path = os.path.join(day_path, img_name)
        #             self.imgs_path.append(img_path)


        # print(self.imgs_path[:100])
        # print(self.args.shape)
        # print(self.args)



    def __len__(self):
        return len(self.imgs_path)
#
    def __getitem__(self, idx):
        # print(idx)
        #   0        1     2     3     4     5     6
        # subject, name, face, left, right, rect, 8pts

        img = cv2.imread(self.imgs_path[idx])

        # cv2.imshow("face", img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()
        # print(self.args[idx][2:])

        img = np.array(img)

        rects = getRects(self.args[idx][2:])
        inputRects = getInputRects(rects, img.shape)
        inputRects = inputRects.reshape(1, -1).squeeze() / 1000
        # print(inputRects.shape)

        rects = rects.astype(np.int64)
        # print(rects)

        # print(inputRects)


        leftEye_img = img[rects[0][1]:rects[0][3], rects[0][0]:rects[0][2]]
        rightEye_img = img[rects[1][1]:rects[1][3], rects[1][0]:rects[1][2]]
        face_img = img[rects[2][1]:rects[2][3], rects[2][0]:rects[2][2]]

        # cv2.imshow("left", leftEye_img)
        # cv2.imshow("right", rightEye_img)
        # cv2.imshow("face", face_img)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        # print(face_img)

        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img/255
        face_img = face_img.transpose(2, 0, 1)


        leftEye_img = cv2.resize(leftEye_img, (112, 112))
        leftEye_img = cv2.cvtColor(leftEye_img, cv2.COLOR_BGR2RGB)
        leftEye_img = leftEye_img / 255
        leftEye_img = leftEye_img.transpose(2, 0, 1)

        rightEye_img = cv2.resize(rightEye_img, (112, 112))
        rightEye_img = cv2.cvtColor(rightEye_img, cv2.COLOR_BGR2RGB)
        rightEye_img = cv2.flip(rightEye_img,1)
        rightEye_img = rightEye_img / 255
        rightEye_img = rightEye_img.transpose(2, 0, 1)

        label = self.args[idx][:2]
        # print(label)

        return {"faceImg": torch.from_numpy(face_img).type(torch.FloatTensor),
                "leftEyeImg": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
                "rightEyeImg": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
                "rects": torch.from_numpy(inputRects).type(torch.FloatTensor),
                "label": torch.from_numpy(label).type(torch.FloatTensor)}
                # "exlabel": torch.from_numpy(np.array(line[6])).type(torch.FloatTensor), "frame": line}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

#
# if __name__ == "__main__":
#     path = "/home/byw/Dataset/GazeCapture/"
#     type = "train"
#
#     loader = txtload(path, type, batch_size=2)
#     for i, (data) in enumerate(loader):
#         #print(data['frame'][0][0] + ' ' + data['frame'][1][0])
#         '''print(data['faceImg'][0].shape)
#                                 print(torch.mean(data['faceImg'][0]))
#                                 print(torch.mean(data['leftEyeImg'][0]))
#                                 print(torch.mean(data['rightEyeImg'][0]))
#                                 print(data['rects'][0])
#                                 print(data['exlabel'][0])'''
#         break

#
if __name__ == "__main__":
    l = loader("D:\\MPIIFaceGaze", 'train')
    # for i in range(len(l)):
    #     l.__getitem__(i)
    #     print(i)
    l.__getitem__(5993)