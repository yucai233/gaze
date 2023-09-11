import cv2
import torch
import torch.nn as nn
import os
import numpy as np
import face_recognition
import copy
import time
import timeit
import dlib
from imutils import face_utils
from scipy.spatial.distance import euclidean

import onnxruntime



face_cascade = cv2.CascadeClassifier('E:/Anoconda/envs/py37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
# 使用模型构建特征提取器
predictor = dlib.shape_predictor('E:/Anoconda/envs/py37/Lib/site-packages/face_recognition_models/models/shape_predictor_68_face_landmarks.dat')

def mouth_ear(mouth):
    a = euclidean(mouth[1], mouth[7])
    b = euclidean(mouth[2], mouth[6])
    c = euclidean(mouth[3], mouth[5])
    d = euclidean(mouth[0], mouth[4])
    return (a + b + c) / (3 * d)
#     return (a + b + c) / 3    #这里必须要除以d，因为d是嘴巴张大过程中相对不变的量，可以排除距离的变化导致的变化


# def getRectOfCV(img):
#     arr = np.empty((0, 4), float)
#     #首先通过cv2中的函数拿到人脸图像
#     #将图片转化为灰度图像
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     ret_list = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
#     for x, y, w, h in ret_list:
#         #裁剪人脸
#         gray_face_img = gray_img[y:y+h, x:x+w]
#         #然后通过face_recogonition拿到两只眼睛的坐标
#         res = face_recognition.face_landmarks(gray_face_img)
#         #如果没有识别到人脸，那么res就为空列表
#         if len(res) == 0:
#             print("这次没检测到")
#             continue
#         #cv2检测到的不算，只有当recognition检测到了人脸，才将人脸放进数组中
#         arr = np.vstack((arr, np.array([x, y, x + w, y + h])))
#         str_list = ["left_eye", "right_eye"]
#         for i, s in enumerate(str_list):
#             x_ave, y_ave = np.sum(res[0][s], axis=0) / 6
#
#             w = res[0][s][3][0] - res[0][s][0][0]
#             w *= 1.7
#             h = w
#             arr = np.vstack((arr, np.array([x + int(x_ave - w / 2),
#                                             y + int(y_ave - h / 2),
#                                             x + int(x_ave + w / 2),
#                                             y + int(y_ave + h / 2)])))
#         print(arr.shape)
#         return arr.astype(np.int64)
#     print("你的脸呢？", ret_list)
#     print(ret_list)
#
#     return None

def getRectOfRec_CV(img):
    arr = np.empty((0, 4), float)
    #首先通过cv2中的函数拿到人脸图像
    #将图片转化为灰度图像
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face_detect_ret = face_cascade.detectMultiScale(gray_img, scaleFactor=1.2, minNeighbors=5)
    #判断是否是人脸
    if isinstance(face_detect_ret, tuple):
        print("你的脸呢？")
        return None
    x, y, w, h = face_detect_ret[0]
    arr = np.vstack((arr, np.array([x, y, x + w, y + h])))
    #裁剪人脸
    gray_face_img = gray_img[y:y+h, x:x+w]
#     face_img = img[y:y+h, x:x+w]
#     showImg(gray_face_img)
    #然后通过face_recogonition拿到两只眼睛的坐标
    res = face_recognition.face_landmarks(gray_face_img)
    #如果没有识别到人脸，那么res就为空列表
    if len(res) == 0:
        print("你的脸呢？")
        return None
    str_list = ["left_eye", "right_eye"]
    for i, s in enumerate(str_list):
        x_ave, y_ave = np.sum(res[0][s], axis=0) / 6

        w = res[0][s][3][0] - res[0][s][0][0]
        w *= 1.7
        h = w
        arr = np.vstack((arr, np.array([x + int(x_ave - w / 2),
                                        y + int(y_ave - h / 2),
                                        x + int(x_ave + w / 2),
                                        y + int(y_ave + h / 2)])))
    return arr.astype(np.int64)

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
    #shape转化成numpy
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


    #拿到关于嘴的闭合参数
    b, e = face_utils.FACIAL_LANDMARKS_IDXS['inner_mouth']
    ear = mouth_ear(res[b:e, :])

    return arr.astype(np.int64), ear


def getRectOfDlib_CV(img):
    arr = np.zeros((3, 4))
    # 将图片转化为灰度图片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 首先拿到人脸(使用cv库)
    face_detect_ret = face_cascade.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5)
    # 判断是否是人脸
    if isinstance(face_detect_ret, tuple):
        print("你的脸呢？")
        return None
    x, y, w, h = face_detect_ret[0]
    #     print("脸", x, y, w, h)
    arr[0][0] = x
    arr[0][1] = y
    arr[0][2] = x + w
    arr[0][3] = y + h

    det = dlib.rectangle(x, y, x + w, y + h)
    # 然后拿到人的眼睛
    dlib_shape = predictor(img, det)
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

def getPar(img, rects):
    #rects保证不是None类型

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

#定义第一组网络（北航的检测头）
class AGN(nn.Module):
    def __init__(self, input_size, channels):
        super(AGN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, channels * 2),
            nn.LeakyReLU()
        )

    def forward(self, x, G, factor):
        style = self.fc(factor)
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)

        N, C, H, W = x.shape
        x = x.view(N * G, -1)
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        x = (x - mean) / (var + 1e-8).sqrt()
        x = x.view([N, C, H, W])

        x = x * (style[:, 0, :, :, :] + 1.) + style[:, 1, :, :, :]
        return x


class SELayer(nn.Module):
    def __init__(self, channel_num, compress_rate):
        super(SELayer, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Linear(channel_num, (channel_num) // compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_num) // compress_rate, channel_num, bias=True),
            nn.Sigmoid()
        )

    def forward(self, feature):
        batch_size, num_channels, H, W = feature.size()
        squeeze_tensor = self.gap(feature)
        squeeze_tensor = squeeze_tensor.view(squeeze_tensor.size(0), -1)
        fc_out = self.se(squeeze_tensor)
        output_tensor = torch.mul(feature, fc_out.view(batch_size, num_channels, 1, 1))
        return output_tensor


class EyeImageModel(nn.Module):
    def __init__(self):
        super(EyeImageModel, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=1)
        self.features1_1 = nn.Sequential(
            nn.Conv2d(3, 24, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(3, 24),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.features1_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            SELayer(48, 16),
            nn.Conv2d(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.features1_3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.features2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.features2_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.features2_3 = nn.ReLU(inplace=True)

        self.AGN1_1 = AGN(128, 48)
        self.AGN1_2 = AGN(128, 64)
        self.AGN2_1 = AGN(128, 128)
        self.AGN2_2 = AGN(128, 64)

    def forward(self, x, factor):
        x1 = self.features1_3(self.AGN1_2(self.features1_2(self.AGN1_1(self.features1_1(x), 6, factor)), 8, factor))
        x2 = self.features2_3(self.AGN2_2(self.features2_2(self.AGN2_1(self.features2_1(x1), 16, factor)), 8, factor))

        return torch.cat((x1, x2), 1)


class FaceImageModel(nn.Module):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 48, kernel_size=5, stride=2, padding=0),
            nn.GroupNorm(6, 48),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 96, kernel_size=5, stride=1, padding=0),
            nn.GroupNorm(12, 96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(96, 128, kernel_size=5, stride=1, padding=2),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(16, 192),
            nn.ReLU(inplace=True),
            SELayer(192, 16),
            nn.Conv2d(192, 128, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),
            SELayer(128, 16),
            nn.Conv2d(128, 64, kernel_size=3, stride=2, padding=0),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            SELayer(64, 16),
        )
        self.fc = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class beihang(nn.Module):

    def __init__(self):
        super(beihang, self).__init__()
        self.eyeModel = EyeImageModel()
        self.eyesMerge_1 = nn.Sequential(
            SELayer(256, 16),
            nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.eyesMerge_AGN = AGN(128, 64)
        self.eyesMerge_2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SELayer(64, 16)
        )
        self.faceModel = FaceImageModel()
        # Joining both eyes
        self.eyesFC = nn.Sequential(
            nn.Linear(5 * 5 * 64, 128),
            nn.LeakyReLU(inplace=True),
        )

        self.rects_fc = nn.Sequential(
            nn.Linear(12, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 96),
            nn.LeakyReLU(inplace=True),
            nn.Linear(96, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, eyesLeft, eyesRight, faces, rects):
        xFace = self.faceModel(faces)
        xRect = self.rects_fc(rects)
        # print(xFace.shape)
        # print(xRect.shape)
        factor = torch.cat((xFace, xRect), 1)

        xEyeL = self.eyeModel(eyesLeft, factor)
        xEyeR = self.eyeModel(eyesRight, factor)

        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesMerge_2(self.eyesMerge_AGN(self.eyesMerge_1(xEyes), 8, factor))
        xEyes = xEyes.view(xEyes.size(0), -1)
        xEyes = self.eyesFC(xEyes)

        # Cat all
        x = torch.cat((xEyes, xFace, xRect), 1)

        return x


#定义第二组网络（全连接层）
class mlp(nn.Module):
    def __init__(self):
        super(mlp, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 2)
        )

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        x = self.fc(x)
        return x


if __name__ == "__main__":
    #首先构建网络并导入网络参数
    device = torch.device("cpu")
    # self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 首先需要有两个已经微调过的网络
    net1 = beihang()
    net1 = nn.DataParallel(net1)
    net1 = net1.module
    net2 = mlp()
    net2 = nn.DataParallel(net2)
    net2 = net2.module
    # 为两个网络导入参数
    state_dict = torch.load("./Iter_10_jie2.pt")

    d1 = net1.state_dict()
    keys_list1 = d1.keys()
    for key in keys_list1:
        d1[key] = state_dict[key]
    d2 = net2.state_dict()
    keys_list2 = d2.keys()
    for key in keys_list2:
        d2[key] = state_dict[key]

    net1.load_state_dict(d1)
    net2.load_state_dict(d2)
    net1.to(device)
    net2.to(device)
    net1.eval()
    net2.eval()
    print("网络参数导入完毕")

    #然后拿到部署网络
    ss1 = onnxruntime.InferenceSession(r"c:/Users/jchao/Desktop/net1.onnx")
    ss2 = onnxruntime.InferenceSession(r"c:/Users/jchao/Desktop/net2.onnx")
    print("onnx模型加载完成")


    #加载并处理图片
    img_parame = []

    for i in range(1, 6):
        #加载图片
        img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
        #拿到图片参数并保存
        rects, _ = getRectOfDlib(img)
        img_parame.append(getPar(img, rects))


    #对两张图片对比结果，判断部署是否成功
    for i in range(len(img_parame)):
        for j in range(i + 1, len(img_parame)):
            print("{}-{}".format(i, j))

            leftEye_img1, rightEye_img1, face_img1, inputRects1 = img_parame[i]
            leftEye_img2, rightEye_img2, face_img2, inputRects2 = img_parame[j]

            #判断直接使用网络得到的结果
            print("直接使用网络得到的结果：")
            print(net2(
                net1(
                    torch.from_numpy(leftEye_img1).type(torch.FloatTensor).to(device),
                    torch.from_numpy(rightEye_img1).type(torch.FloatTensor).to(device),
                    torch.from_numpy(face_img1).type(torch.FloatTensor).to(device),
                    torch.from_numpy(inputRects1).type(torch.FloatTensor).to(device)
                ),
                net1(
                    torch.from_numpy(leftEye_img2).type(torch.FloatTensor).to(device),
                    torch.from_numpy(rightEye_img2).type(torch.FloatTensor).to(device),
                    torch.from_numpy(face_img2).type(torch.FloatTensor).to(device),
                    torch.from_numpy(inputRects2).type(torch.FloatTensor).to(device)
                )
            ).cpu().detach().numpy().squeeze())


            in11 = {"leftEye_img": leftEye_img1.astype(np.float32),
                    "rightEye_img": rightEye_img1.astype(np.float32),
                    "face_img": face_img1.astype(np.float32),
                    "inputRects": inputRects1.astype(np.float32)}

            in12 = {"leftEye_img": leftEye_img2.astype(np.float32),
                    "rightEye_img": rightEye_img2.astype(np.float32),
                    "face_img": face_img2.astype(np.float32),
                    "inputRects": inputRects2.astype(np.float32)}

            in21 = ss1.run(['ret'], in11)[0]
            in22 = ss1.run(['ret'], in12)[0]

            print("使用部署网络得到的结果：")
            print(ss2.run(['ret'], {'x1': in21, 'x2': in22})[0])
            print()










    # #首先对net1进行部署(部署好以后输入是numpy，输出也是numpy)
    # img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\1.png")
    # #拿到组件，输入网络处理
    # rects, _ = getRectOfDlib(img)
    # leftEye_img, rightEye_img, face_img, inputRects = getPar(img, rects)
    # #处理成可以直接输入网络的形式
    # face_img = torch.from_numpy(face_img).type(torch.FloatTensor).to(device)
    # leftEye_img = torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(device)
    # rightEye_img = torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(device)
    # inputRects = torch.from_numpy(inputRects).type(torch.FloatTensor).to(device)
    #
    # with torch.no_grad():
    #     torch.onnx.export(
    #         net1,
    #         (leftEye_img, rightEye_img, face_img, inputRects),
    #         r"C:\Users\jchao\Desktop\net1.onnx",
    #         input_names=['leftEye_img', 'rightEye_img', 'face_img', 'inputRects'],
    #         output_names=['ret']
    #     )
    # print("net1部署完毕")
    #
    # #然后对net2进行部署
    # x = net1(leftEye_img, rightEye_img, face_img, inputRects)
    # with torch.no_grad():
    #     torch.onnx.export(
    #         net2,
    #         (x, x),
    #         r"C:\Users\jchao\Desktop\net2.onnx",
    #         input_names=['x1', 'x2'],
    #         output_names=['ret']
    #     )
    # print("net2部署完毕")
