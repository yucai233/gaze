import torch
import model
import torch.nn as nn
import cv2
import face_recognition
import numpy as np
import copy

def showImg(img):
    cv2.imshow("", img)
    cv2.waitKey()
    cv2.destroyAllWindows()



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




def buildModel():
    print("model buinding")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = model.model()
    net = nn.DataParallel(net)
    net = net.module
    state_dict = torch.load("../checkpoint/Iter_20_AFF_finetrue-Net.pt")
    net.load_state_dict(state_dict)
    net.to(device)
    net.eval()
    print("model building done")
    return net, device


# if __name__ == "__main__":
#     net, device = buildModel()
#
#     #拿到图片，拿到预测值和真实值
#     pred_arr = np.empty((0, 2), float)
#     for i in range(59):
#         img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
#         leftEye_img, rightEye_img, face_img, rects = getPar(img)
#         ret = net(torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(device),
#                   torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(device),
#                   torch.from_numpy(face_img).type(torch.FloatTensor).to(device),
#                   torch.from_numpy(rects).type(torch.FloatTensor).to(device))
#         pred_arr = np.vstack((pred_arr, ret.cpu().detach().numpy()))
#
#     #完事之后将pred_arr存起来
#     np.save(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\pred_arr", pred_arr)
#     print("over")


if __name__ == "__main__":
    net, device = buildModel()
    truth_arr = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\truth_arr.npy")
    truth_arr[truth_arr < 0] = 0
    offset = 0
    w_cm = 34.5353
    w_pixl = 1920
    h_cm = 19.426
    h_pixl = 1080
    for i in range(50):
        img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
        leftEye_img, rightEye_img, face_img, rects = getPar(img)
        ret = net(torch.from_numpy(leftEye_img).type(torch.FloatTensor).to(device),
                  torch.from_numpy(rightEye_img).type(torch.FloatTensor).to(device),
                  torch.from_numpy(face_img).type(torch.FloatTensor).to(device),
                  torch.from_numpy(rects).type(torch.FloatTensor).to(device)).cpu().detach().numpy()[0]
        # pred_x, pred_y = 55.632639313389745 * ret[0] + 959.9648403177628, -55.82785868896932 * ret[1] + -2.301096345553219
        pred_x = (ret[0] + w_cm / 2) / w_cm * w_pixl
        pred_y = -ret[1] / h_cm * h_pixl
        pred_x = pred_x if pred_x > 0 else 0
        pred_y = pred_y if pred_y > 0 else 0
        truth_x, truth_y = truth_arr[i]

        offset += abs(pred_x - truth_x) + abs(pred_y - truth_y)
        # print(i)

        print("pred-> ", pred_x,
              pred_y)
        print("real->", truth_x, truth_y)

    print("北航：", offset / 50)