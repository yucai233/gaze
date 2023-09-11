import paddle

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



class model(paddle.nn.Layer):

    def __init__(self):
        super(model, self).__init__()
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
        # Gaze Regression
        # # 两层
        # self.diff = paddle.nn.Sequential(
        #     paddle.nn.Linear(512, 64),
        #     paddle.nn.ReLU(),
        #     paddle.nn.Dropout(),
        #     paddle.nn.Linear(64, 2)
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

    def getX(self, eyesLeft, eyesRight, faces, rects):
        xFace = self.faceModel(faces)
        xRect = self.rects_fc(rects)
        factor = paddle.concat((xFace, xRect), 1)

        xEyeL = self.eyeModel(eyesLeft, factor)
        xEyeR = self.eyeModel(eyesRight, factor)

        # Cat and FC
        xEyes = paddle.concat((xEyeL, xEyeR), 1)
        xEyes = self.eyesMerge_2(self.eyesMerge_AGN(self.eyesMerge_1(xEyes), 8, factor))
        xEyes = xEyes.reshape((xEyes.shape[0], -1))
        xEyes = self.eyesFC(xEyes)

        x = paddle.concat((xEyes, xFace, xRect), 1)

        return x

    def forward(self,
                eyesLeft1, eyesRight1, faces1, rects1,
                eyesLeft2, eyesRight2, faces2, rects2):
        x1 = self.getX(eyesLeft1, eyesRight1, faces1, rects1)
        x2 = self.getX(eyesLeft2, eyesRight2, faces2, rects2)

        x = paddle.concat((x1, x2), 1)
        # x.stop_gradient = True   #冻结前面的若干层，只训练最后一层
        x = self.diff(x)
        return x





if __name__ == '__main__':
    paddle.device.set_device("cpu")
    m = model()
    sd1 = paddle.load("./Iter_1_testPaddle.pdparams")
    sd2 = paddle.load("./Iter_6_comb.pdparams")
    m.set_state_dict(sd1)
    print(m.state_dict()['eyesFC.0.bias'])
    m.set_state_dict(sd2)
    print(m.state_dict()['eyesFC.0.bias'])