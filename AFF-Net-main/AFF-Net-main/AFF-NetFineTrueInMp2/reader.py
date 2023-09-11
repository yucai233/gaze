import torch
from torch import nn
import numpy as np
import cv2
import face_recognition
from torch.utils.data import Dataset, DataLoader



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

    return leftEye_img, rightEye_img, face_img, rects


class loader(Dataset):
    def __init__(self):
        super(loader, self).__init__()
        #读到所有照片的label，并转化成以cm为单位
        self.labels = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\jie\truth_arr.npy")
        width_mm = 345.353
        width_pixel = 1920.0
        height_mm = 194.26
        height_pixel = 1080
        self.labels[:, 0] = (self.labels[:, 0] * width_mm / width_pixel - width_mm / 2) / 10
        self.labels[:, 1] = -1 * self.labels[:, 1] * height_mm / height_pixel / 10

        #读到所有照片的组件
        self.imgCompent = []
        for i in range(60):
            img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(i))
            self.imgCompent.append(getPar(img))



            # ########
            # leftEye_img, rightEye_img, face_img, rects = self.imgCompent[i]
            # print(leftEye_img.shape)
            # print(rightEye_img.shape)
            # print(face_img.shape)
            # print(rects.shape)
            # break

    def __len__(self):
        return len(self.imgCompent)

    def __getitem__(self, idx):
        leftEye_img, rightEye_img, face_img, rects = self.imgCompent[idx]
        label = self.labels[idx]

        return {
            "leftEye_img": torch.from_numpy(leftEye_img).type(torch.FloatTensor),
            "rightEye_img": torch.from_numpy(rightEye_img).type(torch.FloatTensor),
            "face_img": torch.from_numpy(face_img).type(torch.FloatTensor),
            "rects": torch.from_numpy(rects).type(torch.FloatTensor),
            "label": torch.from_numpy(label).type(torch.FloatTensor)
        }


def txtload(type, batch_size, shuffle=False, num_workers=0):
    dataset = loader()
    print("[Read Data]: GazeCapture Dataset")
    print("[Read Data]: Total num: {:d}".format(len(dataset)))
    print("[Read Data]: Dataset type: {:s}".format(type))
    load = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return load

if __name__ == "__main__":
    l = loader()



