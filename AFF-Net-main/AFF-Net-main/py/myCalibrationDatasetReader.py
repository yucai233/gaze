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
from util import Util


def showImg(img):
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()


class loader(Dataset):

    def __init__(self, data_path, data_type):
        self.data_path = data_path   #数据集的路径
        self.data_type = data_type   #判断是训练还是测试

        self.ut = Util()
        # self.imgs = np.zeros(60)
        self.imgs = []
        self.labels = np.load(os.path.join(self.data_path, "truth_arr.npy"))
        # self.imgs = np.load(r"c:/Users/jchao/Desktop/imgsGetPoint.npy")

        imgs_path = os.path.join(self.data_path, "jie")
        i = 0
        for img_name in os.listdir(imgs_path):
            img_path = os.path.join(imgs_path, img_name)
            # self.imgs[i] = self.ut.getPoint(cv2.imread(img_path))
            # self.imgs.append(self.ut.getPoint(cv2.imread(img_path)))
            self.imgs.append(cv2.imread(img_path))

        # np.save(r"c:/Users/jchao/Desktop/imgsGetPoint", self.imgs)

    def __len__(self):
        return len(self.imgs)
#
    def __getitem__(self, idx):
        # print(self.labels[len(self.imgs) - 1])
        # return {"img": self.imgs[idx].type(torch.FloatTensor),
        #         "label": torch.from_numpy(self.labels[idx] - self.labels[len(self.imgs) - 1]).type(torch.FloatTensor)}
        return {"img": self.ut.getPoint(self.imgs[idx]).type(torch.FloatTensor),
                "label": torch.from_numpy(self.labels[idx] - self.labels[len(self.imgs) - 1]).type(torch.FloatTensor)}


def txtload(path, type, batch_size, shuffle=False, num_workers=0):
    dataset = loader(path, type)
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load



if __name__ == "__main__":
    l = loader(r"C:\Users\jchao\Desktop\calibrationDataset", 'train')
    ret = l.__getitem__(len(l) - 1)
    print(ret["img"])
    print(ret["label"])
