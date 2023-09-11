import torch
import torch.nn as nn
import model
import numpy as np
import face_recognition
import cv2

def buildMode():
    # 模型准备
    # global device, net
    device = torch.device("cpu")
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = model.model()
    net = nn.DataParallel(net)  # 当数据足够大的时候，开多个GPU
    state_dict = torch.load(r".\checkPoint\Iter_2_AFF-Net.pt")
    net.load_state_dict(state_dict)
    net = net.module
    net.to(device)
    net.eval()  # 在模型test的时候加入
    print("Model building done...")
    return net, device


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

class Util:
    # def __init__(self):
    #     self.net, self.device = buildMode()

    net, device = buildMode()

    @classmethod
    def getPoint(self, img):
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

        # 图片预处理
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

        # 拿到模型输入的rects
        # 使用间接方法拿到用于输入的rects
        rects = rects.astype(np.float32)
        rects[:, 0] = rects[:, 0] / img.shape[1]
        rects[:, 2] = rects[:, 2] / img.shape[1]
        rects[:, 1] = rects[:, 1] / img.shape[0]
        rects[:, 3] = rects[:, 3] / img.shape[0]
        rects = rects.flatten().reshape(1, -1)
        # print("输入模型的rects为：\n", rects)

        gazes = self.net(torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(self.device),
                    torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(self.device),
                    torch.from_numpy(face_img).type(torch.FloatTensor).to(self.device),
                    torch.from_numpy(rects).type(torch.FloatTensor).to(self.device))

        return gazes.detach().squeeze()



if __name__ == "__main__":
    img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\0.png")
    ret = Util().getPoint(img)
    print(ret.shape)