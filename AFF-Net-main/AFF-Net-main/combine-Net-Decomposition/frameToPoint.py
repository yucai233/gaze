import cv2
import torch
import torch.nn as nn
import os
import numpy as np
import face_recognition
import copy
import time
import timeit



face_cascade = cv2.CascadeClassifier('E:/Anoconda/envs/py37/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

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

def getRectOfCV(img):
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

def getPar(img, choice):
    if choice == "rec":
        rects = getRectOfRec(img)
    elif choice == "cv":
        rects = getRectOfCV(img)
        # rects = getRectOfRec(img)
    else:
        print("getPar error")
        assert(False)

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


class FrameToPoint():
    def __init__(self, cali_data_path, dis_data_path, model_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #首先需要有两个已经微调过的网络
        self.net1 = beihang()
        self.net1 = nn.DataParallel(self.net1)
        self.net1 = self.net1.module
        self.net2 = mlp()
        self.net2 = nn.DataParallel(self.net2)
        self.net2 = self.net2.module
        #为两个网络导入参数
        # state_dict = torch.load("../checkpoint2/Iter_59_zuhefeiwu-Net.pt")
        state_dict = torch.load(model_path)
        # print(state_dict.keys())
        # print(self.net1.state_dict().keys())
        # print(self.net2.state_dict().keys())


        # d1 = {k:v for k, v in state_dict.items() if k in self.net1.state_dict().keys()}
        # d2 = {k:v for k, v in state_dict.items() if k in self.net2.state_dict().keys()}
        d1 = self.net1.state_dict()
        keys_list1 = d1.keys()
        for key in keys_list1:
            d1[key] = state_dict[key]
        d2 = self.net2.state_dict()
        keys_list2 = d2.keys()
        for key in keys_list2:
            d2[key] = state_dict[key]

        self.net1.load_state_dict(d1)
        self.net2.load_state_dict(d2)
        self.net1.to(self.device)
        self.net2.to(self.device)
        self.net1.eval()
        self.net2.eval()
        print("网络参数导入完毕")

        #有处理好的微调照片
        self.prepossed_imgs = []
        self.imgsComponent = []
        self.labels = np.load(os.path.join(cali_data_path, "truth.npy"))
        #判断照片个数
        n = self.labels.shape[0]
        #首先导入所有微调照片
        for i in range(n):
            #拿到每一张图片的路径
            img_path = os.path.join(cali_data_path, "{}.png".format(i))
            img = cv2.imread(img_path)
            #拿到组件，输入网络处理
            leftEye_img, rightEye_img, face_img, inputRects = getPar(img, "rec")
            # face_img, leftEye_img, rightEye_img, inputRects, label = getInput(img, self.labels[i])
            self.imgsComponent.append((face_img, leftEye_img, rightEye_img, inputRects, self.labels[i]))
            #处理成可以直接输入网络的形式
            face_img = torch.from_numpy(face_img).type(torch.FloatTensor).to(self.device)
            leftEye_img = torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(self.device)
            rightEye_img = torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(self.device)
            inputRects = torch.from_numpy(inputRects).type(torch.FloatTensor).to(self.device)

            # print(leftEye_img.shape)
            # print(rightEye_img.shape)
            # print(face_img.shape)
            # print(inputRects.shape)

            #输入网络
            self.prepossed_imgs.append(self.net1(leftEye_img, rightEye_img, face_img, inputRects))
        print("微调图片处理完毕")


        #拿到和屏幕有关的系数
        self.kx, self.ky = 0, 0
        divided = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = self.net2(self.prepossed_imgs[i], self.prepossed_imgs[j]).cpu().detach().numpy().squeeze()
                # print("diff----------------------------------------")
                # print(diff)
                self.kx += (self.labels[i][0] - self.labels[j][0]) / diff[0]
                self.ky += (self.labels[i][1] - self.labels[j][1]) / diff[1]
                divided += 1
        self.kx /= divided
        self.ky /= divided
        # w_cm = 34.5353
        # w_pixl = 1920
        # h_cm = 19.426
        # h_pixl = 1080
        # self.kx = w_pixl / w_cm
        # self.ky = h_pixl / h_cm
        print("k初始化完毕->", self.kx, self.ky)


        #拿到视线分解之后和subject有关的因素
        self.x_offset, self.y_offset = 0, 0
        dis_labels = np.load(os.path.join(dis_data_path, "truth.npy"))
        #对于所有的校正图片，拿到网络的预测值
        for i in range(dis_labels.shape[0]):
            #拿到图片路径
            img_path = os.path.join(dis_data_path, "{}.png".format(i))
            #拿到图片
            img = cv2.imread(img_path)
            #拿到网络预测值
            x_net, y_net = self.getGaze(img, "rec")
            #拿到照片的真实坐标
            x_truth, y_truth = dis_labels[i]
            #累加
            self.x_offset += x_truth - x_net
            self.y_offset += y_truth - y_net
        self.x_offset /= n
        self.y_offset /= n
        print("视线分解处理完毕-》", self.x_offset, self.y_offset)


    def getGaze(self, frame, choice):
        #首先对frame做处理
        # 拿到组件，输入网络处理
        # face_img, leftEye_img, rightEye_img, inputRects, _ = getInput(frame, None)
        ret = getPar(frame, choice)
        if isinstance(ret, type(None)):
            return None
        leftEye_img, rightEye_img, face_img, inputRects = ret
        # 处理成可以直接输入网络的形式
        face_img = torch.from_numpy(face_img).type(torch.FloatTensor).to(self.device)
        leftEye_img = torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(self.device)
        rightEye_img = torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(self.device)
        inputRects = torch.from_numpy(inputRects).type(torch.FloatTensor).to(self.device)
        # 输入网络
        x1 = self.net1(leftEye_img, rightEye_img, face_img, inputRects)

        gaze1_x, gaze1_y, w_sum = 0, 0, 0
        #两两之间求一次差值
        for i, x2 in enumerate(self.prepossed_imgs):
            gaze2_x, gaze2_y = self.labels[i][0], self.labels[i][1]
            diff = self.net2(x1, x2).cpu().detach().numpy().squeeze()

            tmp_x = gaze2_x + self.kx * diff[0]
            tmp_y = gaze2_y + self.ky * diff[1]
            w = 1 / (abs(diff[0]) + abs(diff[1]))

            gaze1_x += w * tmp_x
            gaze1_y += w * tmp_y
            w_sum += w

        gaze1_x /= w_sum
        gaze1_y /= w_sum

        return gaze1_x, gaze1_y

    def frameToPoint(self, frame):
        ret = self.getGaze(frame, "rec")
        if isinstance(ret, type(None)):
            return None

        x_net, y_net = ret
        x_truth, y_truth = x_net + self.x_offset, y_net + self.y_offset
        return x_truth, y_truth





#
# if __name__ == "__main__":
#     img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\1.png")
#     timem = timeit.timeit(lambda : getRect(img), number=10)
#     print(timem / 10)




if __name__ == "__main__":
    # img1 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(1))
    # face_img, leftEye_img, rightEye_img, inputRects, _ = getInput(img1, None)
    # # 处理成可以直接输入网络的形式
    # face_img = torch.from_numpy(face_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # leftEye_img = torch.from_numpy(leftEye_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # rightEye_img = torch.from_numpy(rightEye_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # inputRects = torch.from_numpy(inputRects).type(torch.FloatTensor).to(util.device)
    # x1 = util.net1(leftEye_img, rightEye_img, face_img, inputRects)
    #
    # img2 = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\{}.png".format(2))
    # face_img, leftEye_img, rightEye_img, inputRects, _ = getInput(img2, None)
    # # 处理成可以直接输入网络的形式
    # face_img = torch.from_numpy(face_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # leftEye_img = torch.from_numpy(leftEye_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # rightEye_img = torch.from_numpy(rightEye_img[np.newaxis, :]).type(torch.FloatTensor).to(util.device)
    # inputRects = torch.from_numpy(inputRects).type(torch.FloatTensor).to(util.device)
    # x2 = util.net1(leftEye_img, rightEye_img, face_img, inputRects)
    #
    # # leftEye1, rightEye1, face1, rects1 = getPar(img1)
    # # leftEye2, rightEye2, face2, rects2 = getPar(img2)
    # # x1 = util.net1(torch.from_numpy(leftEye1).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(rightEye1).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(face1).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(rects1).type(torch.FloatTensor).to(util.device))
    # #
    # # x2 = util.net1(torch.from_numpy(leftEye2).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(rightEye2).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(face2).type(torch.FloatTensor).to(util.device),
    # #             torch.from_numpy(rects2).type(torch.FloatTensor).to(util.device))
    #
    #
    # print(util.net2(x1, x2).cpu().detach().numpy())






    # print(util.prepossed_imgs[0].shape)
    util = FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint/Iter_20_jie2.pt")
    # util = FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"../checkpoint2/Iter_20_zuhefeiwuceshi-Net.pt")
    arr = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\truth_arr.npy")
    arr[arr < 0] = 0
    #拿到所有的图片进行测试
    offset = 0
    n = 0
    for i in range(59):
        frame = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
        ret = util.frameToPoint(frame)
        if isinstance(ret, type(None)):
            continue
        pred_x, pred_y = ret
        truth_x, truth_y = arr[i]
        pred_x = pred_x if pred_x > 0 else 0
        pred_y = pred_y if pred_y > 0 else 0
        offset += abs(pred_x - truth_x) + abs(pred_y - truth_y)
        # print(i)
        print("pred-> ", pred_x, pred_y)
        print("real-> ", truth_x, truth_y)
        n += 1
    print("组合： ", offset / n)
    print("n定于", n)

