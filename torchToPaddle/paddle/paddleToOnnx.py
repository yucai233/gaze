import cv2
import numpy as np
import dlib
from imutils import face_utils
from scipy.spatial.distance import euclidean
import paddle
import onnxruntime



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
    rects = rects.flatten().reshape((1, -1))
    # print("输入模型的rects为：\n", rects)

    return leftEye_img, rightEye_img, face_img, rects

#定义第一组网络（北航的检测头）
class AGN(paddle.nn.Layer):
    def __init__(self, input_size, channels):
        super(AGN, self).__init__()
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(input_size, channels * 2),
            paddle.nn.LeakyReLU()
        )

    def forward(self, x, G, factor):
        style = self.fc(factor)
        shape = [-1, 2, x.shape[1], 1, 1]
        style = style.reshape(shape)

        N, C, H, W = x.shape
        x = x.reshape((N * G, -1))
        mean = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True)
        x = (x - mean) / (var + 1e-8).sqrt()
        x = x.reshape([N, C, H, W])

        x = x * (style[:, 0, :, :, :] + 1.) + style[:, 1, :, :, :]
        return x


class SELayer(paddle.nn.Layer):
    def __init__(self, channel_num, compress_rate):
        super(SELayer, self).__init__()
        self.gap = paddle.nn.AdaptiveAvgPool2D(1)
        self.se = paddle.nn.Sequential(
            paddle.nn.Linear(channel_num, (channel_num) // compress_rate, bias_attr=True),
            paddle.nn.ReLU(),
            paddle.nn.Linear((channel_num) // compress_rate, channel_num, bias_attr=True),
            paddle.nn.Sigmoid()
        )

    def forward(self, feature):
        batch_size, num_channels, H, W = feature.shape
        squeeze_tensor = self.gap(feature)
        squeeze_tensor = squeeze_tensor.reshape((squeeze_tensor.shape[0], -1))
        fc_out = self.se(squeeze_tensor)
        output_tensor = paddle.multiply(feature, fc_out.reshape((batch_size, num_channels, 1, 1)))
        return output_tensor


class EyeImageModel(paddle.nn.Layer):
    def __init__(self):
        super(EyeImageModel, self).__init__()
        self.maxpool = paddle.nn.MaxPool2D(kernel_size=3, stride=1)
        self.features1_1 = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 24, kernel_size=5, stride=2, padding=0),
            paddle.nn.GroupNorm(3, 24),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(24, 48, kernel_size=5, stride=1, padding=0),
        )
        self.features1_2 = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            SELayer(48, 16),
            paddle.nn.Conv2D(48, 64, kernel_size=5, stride=1, padding=1),
        )
        self.features1_3 = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
        )
        self.features2_1 = paddle.nn.Conv2D(64, 128, kernel_size=3, stride=1, padding=1)

        self.features2_2 = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            SELayer(128, 16),
            paddle.nn.Conv2D(128, 64, kernel_size=3, stride=1, padding=1),
        )
        self.features2_3 = paddle.nn.ReLU()

        self.AGN1_1 = AGN(128, 48)
        self.AGN1_2 = AGN(128, 64)
        self.AGN2_1 = AGN(128, 128)
        self.AGN2_2 = AGN(128, 64)

    def forward(self, x, factor):
        x1 = self.features1_3(self.AGN1_2(self.features1_2(self.AGN1_1(self.features1_1(x), 6, factor)), 8, factor))
        x2 = self.features2_3(self.AGN2_2(self.features2_2(self.AGN2_1(self.features2_1(x1), 16, factor)), 8, factor))

        return paddle.concat((x1, x2), 1)


class FaceImageModel(paddle.nn.Layer):

    def __init__(self):
        super(FaceImageModel, self).__init__()
        self.conv = paddle.nn.Sequential(
            paddle.nn.Conv2D(3, 48, kernel_size=5, stride=2, padding=0),
            paddle.nn.GroupNorm(6, 48),
            paddle.nn.ReLU(),
            paddle.nn.Conv2D(48, 96, kernel_size=5, stride=1, padding=0),
            paddle.nn.GroupNorm(12, 96),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(96, 128, kernel_size=5, stride=1, padding=2),
            paddle.nn.GroupNorm(16, 128),
            paddle.nn.ReLU(),
            paddle.nn.MaxPool2D(kernel_size=3, stride=2),
            paddle.nn.Conv2D(128, 192, kernel_size=3, stride=1, padding=1),
            paddle.nn.GroupNorm(16, 192),
            paddle.nn.ReLU(),
            SELayer(192, 16),
            paddle.nn.Conv2D(192, 128, kernel_size=3, stride=2, padding=0),
            paddle.nn.GroupNorm(16, 128),
            paddle.nn.ReLU(),
            SELayer(128, 16),
            paddle.nn.Conv2D(128, 64, kernel_size=3, stride=2, padding=0),
            paddle.nn.GroupNorm(8, 64),
            paddle.nn.ReLU(),
            SELayer(64, 16),
        )
        self.fc = paddle.nn.Sequential(
            paddle.nn.Linear(5 * 5 * 64, 128),
            paddle.nn.LeakyReLU(),
            paddle.nn.Linear(128, 64),
            paddle.nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.reshape((x.shape[0], -1))
        x = self.fc(x)
        return x


class beihang(paddle.nn.Layer):

    def __init__(self):
        super(beihang, self).__init__()
        self.eyeModel = EyeImageModel()
        self.eyesMerge_1 = paddle.nn.Sequential(
            SELayer(256, 16),
            paddle.nn.Conv2D(256, 64, kernel_size=3, stride=2, padding=1),
        )
        self.eyesMerge_AGN = AGN(128, 64)
        self.eyesMerge_2 = paddle.nn.Sequential(
            paddle.nn.ReLU(),
            SELayer(64, 16)
        )
        self.faceModel = FaceImageModel()
        # Joining both eyes
        self.eyesFC = paddle.nn.Sequential(
            paddle.nn.Linear(5 * 5 * 64, 128),
            paddle.nn.LeakyReLU(),
        )

        self.rects_fc = paddle.nn.Sequential(
            paddle.nn.Linear(12, 64),
            paddle.nn.LeakyReLU(),
            paddle.nn.Linear(64, 96),
            paddle.nn.LeakyReLU(),
            paddle.nn.Linear(96, 128),
            paddle.nn.LeakyReLU(),
            paddle.nn.Linear(128, 64),
            paddle.nn.LeakyReLU(),
        )

    def forward(self, eyesLeft, eyesRight, faces, rects):
        xFace = self.faceModel(faces)
        xRect = self.rects_fc(rects)
        # print(xFace.shape)
        # print(xRect.shape)
        factor = paddle.concat((xFace, xRect), 1)

        xEyeL = self.eyeModel(eyesLeft, factor)
        xEyeR = self.eyeModel(eyesRight, factor)

        # Cat and FC
        xEyes = paddle.concat((xEyeL, xEyeR), 1)
        xEyes = self.eyesMerge_2(self.eyesMerge_AGN(self.eyesMerge_1(xEyes), 8, factor))
        xEyes = xEyes.reshape((xEyes.shape[0], -1))
        xEyes = self.eyesFC(xEyes)

        # Cat all
        x = paddle.concat((xEyes, xFace, xRect), 1)

        return x


#定义第二组网络（全连接层）
class mlp(paddle.nn.Layer):
    def __init__(self):
        super(mlp, self).__init__()

        # #四层
        # self.diff = paddle.nn.Sequential(
        #     paddle.nn.Linear(512, 128),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Dropout(),
        #     paddle.nn.Linear(128, 32),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Dropout(),
        #     paddle.nn.Linear(32, 8),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Dropout(),
        #     paddle.nn.Linear(8, 2)
        # )

        # 三层
        self.diff = paddle.nn.Sequential(
            paddle.nn.Linear(512, 128),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(),
            paddle.nn.Linear(128, 32),
            paddle.nn.ReLU(),
            paddle.nn.Dropout(),
            paddle.nn.Linear(32, 2)
        )

        # #两层
        # self.diff = paddle.nn.Sequential(
        #     paddle.nn.Linear(512, 64),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Dropout(),
        #     paddle.nn.Linear(64, 2)
        # )

    def forward(self, x1, x2):
        x = paddle.concat((x1, x2), 1)
        x = self.diff(x)
        return x


if __name__ == "__main__":
    paddle.device.set_device("cpu")
    model_path = "./checkpoint_3_layer/Iter_4_comb.pdparams"

    # 首先需要有两个已经微调过的网络
    net1 = beihang()
    net2 = mlp()

    # 导入两个网络共同的参数
    state_dict = paddle.load(model_path)
    # 为两个网络分别导入参数
    d1 = net1.state_dict()
    keys_list1 = d1.keys()
    for key in keys_list1:
        d1[key] = state_dict[key]
    d2 = net2.state_dict()
    keys_list2 = d2.keys()
    for key in keys_list2:
        d2[key] = state_dict[key]
    net1.set_state_dict(d1)
    net2.set_state_dict(d2)
    # 将模型设置为训练
    net1.eval()
    net2.eval()
    print("网络参数导入完毕")

    # 然后拿到部署网络
    ss1 = onnxruntime.InferenceSession(r"c:/Users/jchao/Desktop/net1_paddle.onnx")
    ss2 = onnxruntime.InferenceSession(r"c:/Users/jchao/Desktop/net2_paddle.onnx")
    print("onnx模型加载完成")

    # 加载并处理图片
    img_parame = []

    for i in range(1, 6):
        # 加载图片
        img = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
        # 拿到图片参数并保存
        rects, _ = getRectOfDlib(img)
        img_parame.append(getPar(img, rects))

    # 对两张图片对比结果，判断部署是否成功
    for i in range(len(img_parame)):
        for j in range(i + 1, len(img_parame)):
            print("{}-{}".format(i, j))

            leftEye_img1, rightEye_img1, face_img1, inputRects1 = img_parame[i]
            leftEye_img2, rightEye_img2, face_img2, inputRects2 = img_parame[j]

            # 判断直接使用网络得到的结果
            print("直接使用网络得到的结果：")
            print(net2(
                net1(
                    paddle.to_tensor(leftEye_img1, dtype="float32"),
                    paddle.to_tensor(rightEye_img1, dtype="float32"),
                    paddle.to_tensor(face_img1, dtype="float32"),
                    paddle.to_tensor(inputRects1, dtype="float32")
                ),
                net1(
                    paddle.to_tensor(leftEye_img2, dtype="float32"),
                    paddle.to_tensor(rightEye_img2, dtype="float32"),
                    paddle.to_tensor(face_img2, dtype="float32"),
                    paddle.to_tensor(inputRects2, dtype="float32")
                )
            ).numpy())

            in11 = {"leftEye_img": leftEye_img1.astype(np.float32),
                    "rightEye_img": rightEye_img1.astype(np.float32),
                    "face_img": face_img1.astype(np.float32),
                    "inputRects": inputRects1.astype(np.float32)}

            in12 = {"leftEye_img": leftEye_img2.astype(np.float32),
                    "rightEye_img": rightEye_img2.astype(np.float32),
                    "face_img": face_img2.astype(np.float32),
                    "inputRects": inputRects2.astype(np.float32)}

            in21 = ss1.run(None, in11)[0]
            in22 = ss1.run(None, in12)[0]

            print("使用部署网络得到的结果：")
            print(ss2.run(None, {'x1': in21, 'x2': in22})[0])
            print()


    # #尝试将net1转onnx
    # #拿到每一张图片的路径
    # #拿到net1的输入形状
    # img_path = r"c:/Users/jchao/Desktop/calibrationDataset/jie/0.png"
    # img = cv2.imread(img_path)
    # #拿到组件，输入网络处理
    # rects, _ = getRectOfDlib(img)
    # leftEye_img, rightEye_img, face_img, inputRects = getPar(img, rects)
    #
    # x_l = paddle.static.InputSpec.from_numpy(leftEye_img.astype(np.float32), "leftEye_img")
    # x_r = paddle.static.InputSpec.from_numpy(rightEye_img.astype(np.float32), "rightEye_img")
    # x_f = paddle.static.InputSpec.from_numpy(face_img.astype(np.float32), "face_img")
    # x_i = paddle.static.InputSpec.from_numpy(inputRects.astype(np.float32), "inputRects")
    # #开始转化
    # paddle.onnx.export(net1, r"c:/Users/jchao/Desktop/net1_paddle", input_spec=[x_l, x_r, x_f, x_i], opset_version=11)
    # print("net1转化完毕")
    #
    #
    # #开始处理net2
    # face_img = paddle.to_tensor(face_img, dtype="float32")
    # leftEye_img = paddle.to_tensor(leftEye_img, dtype="float32")
    # rightEye_img = paddle.to_tensor(rightEye_img, dtype="float32")
    # inputRects = paddle.to_tensor(inputRects, dtype="float32")
    # x1 = paddle.static.InputSpec.from_tensor(net1(leftEye_img, rightEye_img, face_img, inputRects), 'x1')
    # x2 = paddle.static.InputSpec.from_tensor(net1(leftEye_img, rightEye_img, face_img, inputRects), 'x2')
    # paddle.onnx.export(net2, r"c:/Users/jchao/Desktop/net2_paddle", input_spec=[x1, x2], opset_version=11)
    # print("net2转化完毕")



# class FrameToPoint():
#     def __init__(self, cali_data_path, model_path):
#         paddle.device.set_device("cpu")
#
#         #首先需要有两个已经微调过的网络
#         self.net1 = beihang()
#         self.net2 = mlp()
#         # self.net1 = paddle.DataParallel(self.net1)
#         # self.net2 = paddle.DataParallel(self.net2)
#
#         #导入两个网络共同的参数
#         state_dict = paddle.load(model_path)
#         #为两个网络分别导入参数
#         d1 = self.net1.state_dict()
#         keys_list1 = d1.keys()
#         for key in keys_list1:
#             d1[key] = state_dict[key]
#         d2 = self.net2.state_dict()
#         keys_list2 = d2.keys()
#         for key in keys_list2:
#             d2[key] = state_dict[key]
#         self.net1.set_state_dict(d1)
#         self.net2.set_state_dict(d2)
#         #将模型设置为训练
#         self.net1.eval()
#         self.net2.eval()
#         print("网络参数导入完毕")
#
#         #有处理好的微调照片
#         self.prepossed_imgs = []
#         self.imgsComponent = []
#         self.labels = np.load(os.path.join(cali_data_path, "truth.npy"))
#         #判断照片个数
#         n = self.labels.shape[0]
#         #首先导入所有微调照片
#         for i in range(n):
#             #拿到每一张图片的路径
#             img_path = os.path.join(cali_data_path, "{}.png".format(i))
#             img = cv2.imread(img_path)
#             #拿到组件，输入网络处理
#             rects, _ = getRectOfDlib(img)
#             leftEye_img, rightEye_img, face_img, inputRects = getPar(img, rects)
#             # face_img, leftEye_img, rightEye_img, inputRects, label = getInput(img, self.labels[i])
#             self.imgsComponent.append((face_img, leftEye_img, rightEye_img, inputRects, self.labels[i]))
#             #处理成可以直接输入网络的形式
#             face_img = paddle.to_tensor(face_img, dtype="float32")
#             leftEye_img = paddle.to_tensor(leftEye_img, dtype="float32")
#             rightEye_img = paddle.to_tensor(rightEye_img, dtype="float32")
#             inputRects = paddle.to_tensor(inputRects, dtype="float32")
#
#             #输入网络
#             self.prepossed_imgs.append(self.net1(leftEye_img, rightEye_img, face_img, inputRects))
#         print("微调图片处理完毕")
#
#
#         # 拿到和屏幕有关的系数
#         self.kx, self.ky = 0, 0
#         divided = 0
#         for i in range(n):
#             for j in range(i + 1, n):
#                 diff = self.net2(self.prepossed_imgs[i], self.prepossed_imgs[j]).cpu().detach().numpy().squeeze()
#                 # if abs(diff[0]) + abs(diff[1]) < 18.5:
#                 self.kx += (self.labels[i][0] - self.labels[j][0]) / diff[0]
#                 self.ky += (self.labels[i][1] - self.labels[j][1]) / diff[1]
#                 divided += 1
#         self.kx /= divided
#         self.ky /= divided
#
#         # w_cm = 34.5353
#         # w_pixl = 1920
#         # h_cm = 19.426
#         # h_pixl = 1080
#         # self.kx = w_pixl / w_cm
#         # self.ky = h_pixl / h_cm
#
#         print("k初始化完毕->", self.kx, self.ky)
#
#
#     #做平均
#     def frameToPoint(self, frame, rects):
#         ret = getPar(frame, rects)
#         if isinstance(ret, type(None)):
#             return None
#         leftEye_img, rightEye_img, face_img, inputRects = ret
#         # 处理成可以直接输入网络的形式
#         face_img = paddle.to_tensor(face_img, dtype="float32")
#         leftEye_img = paddle.to_tensor(leftEye_img, dtype="float32")
#         rightEye_img = paddle.to_tensor(rightEye_img, dtype="float32")
#         inputRects = paddle.to_tensor(inputRects, dtype="float32")
#         # 输入网络
#         x1 = self.net1(leftEye_img, rightEye_img, face_img, inputRects)
#
#         gaze1_x, gaze1_y, w_sum = 0, 0, 0
#         #两两之间求一次差值
#         for i, x2 in enumerate(self.prepossed_imgs):
#             gaze2_x, gaze2_y = self.labels[i][0], self.labels[i][1]
#             diff = self.net2(x1, x2).cpu().detach().numpy().squeeze()
#             # print(diff)
#
#             tmp_x = gaze2_x + self.kx * diff[0]
#             tmp_y = gaze2_y + self.ky * diff[1]
#             w = 1 / (abs(diff[0]) + abs(diff[1]))
#
#             gaze1_x += w * tmp_x
#             gaze1_y += w * tmp_y
#             w_sum += w
#
#         gaze1_x /= w_sum
#         gaze1_y /= w_sum
#
#         return gaze1_x, gaze1_y
#
#
#
# class FrameToPoint_fit():
#     def __init__(self, cali_data_path, model_path):
#         paddle.device.set_device("gpu")
#
#         #首先需要有两个已经微调过的网络
#         self.net1 = beihang()
#         self.net2 = mlp()
#         # self.net1 = paddle.DataParallel(self.net1)
#         # self.net2 = paddle.DataParallel(self.net2)
#
#         #导入两个网络共同的参数
#         state_dict = paddle.load(model_path)
#         #为两个网络分别导入参数
#         d1 = self.net1.state_dict()
#         keys_list1 = d1.keys()
#         for key in keys_list1:
#             d1[key] = state_dict[key]
#         d2 = self.net2.state_dict()
#         keys_list2 = d2.keys()
#         for key in keys_list2:
#             d2[key] = state_dict[key]
#         self.net1.set_state_dict(d1)
#         self.net2.set_state_dict(d2)
#         #将模型设置为训练
#         self.net1.eval()
#         self.net2.eval()
#         print("网络参数导入完毕")
#
#         #有处理好的微调照片
#         self.prepossed_imgs = []
#         self.imgsComponent = []
#         self.labels = np.load(os.path.join(cali_data_path, "truth.npy"))
#         #判断照片个数
#         n = self.labels.shape[0]
#         #首先导入所有微调照片
#         for i in range(n):
#             #拿到每一张图片的路径
#             img_path = os.path.join(cali_data_path, "{}.png".format(i))
#             img = cv2.imread(img_path)
#             #拿到组件，输入网络处理
#             rects, _ = getRectOfDlib(img)
#             leftEye_img, rightEye_img, face_img, inputRects = getPar(img, rects)
#             # face_img, leftEye_img, rightEye_img, inputRects, label = getInput(img, self.labels[i])
#             self.imgsComponent.append((face_img, leftEye_img, rightEye_img, inputRects, self.labels[i]))
#             #处理成可以直接输入网络的形式
#             face_img = paddle.to_tensor(face_img, dtype="float32")
#             leftEye_img = paddle.to_tensor(leftEye_img, dtype="float32")
#             rightEye_img = paddle.to_tensor(rightEye_img, dtype="float32")
#             inputRects = paddle.to_tensor(inputRects, dtype="float32")
#
#             #输入网络
#             self.prepossed_imgs.append(self.net1(leftEye_img, rightEye_img, face_img, inputRects))
#         print("微调图片处理完毕")
#
#
#         #考虑从cm到pixl的转化或者是其他的客观因素
#         #导入随机照片集并进行处理
#         temp_path = r"C:\Users\jchao\Desktop\calibrationDataset\gagarandomjie"   #存放数据集
#         temp_preprossed_img = []  #存放处理好的数据
#         temp_labels = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\gagarandomjie\truth.npy")
#         for i in range(15):  #随机照片集一共有15张
#             temp_img_path = os.path.join(temp_path, "{}.png".format(i))
#             temp_img = cv2.imread(temp_img_path)
#             rects, _ = getRectOfDlib(temp_img)
#             leftEye_img, rightEye_img, face_img, inputRects = getPar(temp_img, rects)
#             temp_preprossed_img.append(self.net1(
#                 paddle.to_tensor(leftEye_img, dtype="float32"),
#                 paddle.to_tensor(rightEye_img, dtype="float32"),
#                 paddle.to_tensor(face_img, dtype="float32"),
#                 paddle.to_tensor(inputRects, dtype="float32")
#             ))
#         #将网络预测出来的结果转化成实际的结果
#         diff_net_all = np.empty((0, 2), dtype=float)
#         diff_label_all = np.empty((0, 2), dtype=float)
#         for i in range(n):
#             for j in range(i + 1, n):
#                 diff_net = self.net2(temp_preprossed_img[i], temp_preprossed_img[j]).cpu().detach().numpy().squeeze()
#                 diff_label = temp_labels[i] - temp_labels[j]
#                 diff_net_all = np.vstack((diff_net_all, diff_net))
#                 diff_label_all = np.vstack((diff_label_all, diff_label))
#         # #拿到数据以后通过线性模型进行预测
#         # np.save(r"c:/Users/jchao/Desktop/diff_net_all.npy", diff_net_all)
#         # np.save(r"c:/Users/jchao/Desktop/diff_label_all.npy", diff_label_all)
#         # print("保存完毕")
#
#         #构建线性模型
#         self.lr1 = LinearRegression()
#         self.lr2 = LinearRegression()
#         self.lr1.fit(diff_net_all[:, 0].reshape(-1, 1), diff_label_all[:, 0].reshape(-1, 1))
#         self.lr2.fit(diff_net_all[:, 1].reshape(-1, 1), diff_label_all[:, 1].reshape(-1, 1))
#
#
#     #做平均
#     def frameToPoint(self, frame, rects):
#         ret = getPar(frame, rects)
#         if isinstance(ret, type(None)):
#             return None
#         leftEye_img, rightEye_img, face_img, inputRects = ret
#         # 处理成可以直接输入网络的形式
#         face_img = paddle.to_tensor(face_img, dtype="float32")
#         leftEye_img = paddle.to_tensor(leftEye_img, dtype="float32")
#         rightEye_img = paddle.to_tensor(rightEye_img, dtype="float32")
#         inputRects = paddle.to_tensor(inputRects, dtype="float32")
#         # 输入网络
#         x1 = self.net1(leftEye_img, rightEye_img, face_img, inputRects)
#
#         gaze1_x, gaze1_y, w_sum = 0, 0, 0
#         #两两之间求一次差值
#         for i, x2 in enumerate(self.prepossed_imgs):
#             gaze2_x, gaze2_y = self.labels[i][0], self.labels[i][1]
#             diff = self.net2(x1, x2).cpu().detach().numpy().squeeze()
#             # print(diff)
#
#             tmp_x = gaze2_x + self.lr1.predict(np.array([diff[0]]).reshape(1, -1)).squeeze()
#             tmp_y = gaze2_y + self.lr2.predict(np.array([diff[1]]).reshape(1, -1)).squeeze()
#             w = 1 / (abs(diff[0]) + abs(diff[1]))
#
#             gaze1_x += w * tmp_x
#             gaze1_y += w * tmp_y
#             w_sum += w
#
#         gaze1_x /= w_sum
#         gaze1_y /= w_sum
#
#         return gaze1_x, gaze1_y
#
#
# # if __name__ == "__main__":
# #     util = FrameToPoint_fit(r"C:\Users\jchao\Desktop\calibrationDataset\gagarandomjie",
# #                         r"./checkpoint/Iter_25_jie.pdparams")
#
#
#
# if __name__ == "__main__":
#     # util = FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./jicao_gaijin/Iter_6_gaijin.pdparams")
#     util = FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint_4_layer/Iter_11_comb.pdparams")
#     # util = FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint_3_layer/Iter_4_comb.pdparams")
#
#
#     arr = np.load(r"C:\Users\jchao\Desktop\calibrationDataset\jie2/truth_arr.npy")
#     arr[arr < 0] = 0
#     #拿到所有的图片进行测试
#     offset = 0
#     n = 0
#     start = time.time()
#     for i in range(59):
#         print(i)
#         frame = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(i))
#         parameter, ear = getRectOfDlib(frame)
#         if isinstance(parameter, type(None)):
#             print("你的脸呢？")
#             continue
#         ret = util.frameToPoint(frame, parameter)
#         if isinstance(ret, type(None)):
#             continue
#         pred_x, pred_y = ret
#         truth_x, truth_y = arr[i]
#         pred_x = pred_x if pred_x > 0 else 0
#         pred_y = pred_y if pred_y > 0 else 0
#         offset += abs(pred_x - truth_x) + abs(pred_y - truth_y)
#         # print(i)
#         print("pred-> ", pred_x, pred_y)
#         print("real-> ", truth_x, truth_y)
#         print("ear:", ear)
#         n += 1
#     print("组合： ", offset / n)
#     print("n定于", n)
#
#     end = time.time()
#     print("时间：", end - start)

