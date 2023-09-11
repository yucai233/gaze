import torch
import torch.nn as nn
import paddle

class AGN(nn.Module):
    def __init__(self, input_size, channels):
        super(AGN, self).__init__()
        self.fc = nn.Sequential(
                    nn.Linear(input_size, channels*2),
                    nn.LeakyReLU()
                    )

    def forward(self, x, G, factor):
        style = self.fc(factor) 
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)  
        
        N, C, H, W = x.shape
        x = x.view(N*G, -1)
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
            nn.Linear(channel_num, (channel_num)//compress_rate, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear((channel_num)//compress_rate, channel_num, bias=True),
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



class model(nn.Module):

    def __init__(self):
        super(model, self).__init__()
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
        # Gaze Regression
        self.fc = nn.Sequential(
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(64, 2)
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

    def getX(self, eyesLeft, eyesRight, faces, rects):
        xFace = self.faceModel(faces)
        xRect = self.rects_fc(rects)
        factor = torch.cat((xFace, xRect), 1)

        xEyeL = self.eyeModel(eyesLeft, factor)
        xEyeR = self.eyeModel(eyesRight, factor)

        # Cat and FC
        xEyes = torch.cat((xEyeL, xEyeR), 1)
        xEyes = self.eyesMerge_2(self.eyesMerge_AGN(self.eyesMerge_1(xEyes), 8, factor))
        xEyes = xEyes.view(xEyes.size(0), -1)
        xEyes = self.eyesFC(xEyes)

        x = torch.cat((xEyes, xFace, xRect), 1)

        return x

    def forward(self,
                eyesLeft1, eyesRight1, faces1, rects1,
                eyesLeft2, eyesRight2, faces2, rects2):
        x1 = self.getX(eyesLeft1, eyesRight1, faces1, rects1)
        x2 = self.getX(eyesLeft2, eyesRight2, faces2, rects2)

        x = torch.cat((x1, x2), 1)
        x = self.fc(x)
        return x


def checkpoint_torch2paddle(checkpoint_torch, checkpoint_paddle):
    #首先读取checkpoint_torch
    dict_torch = torch.load(checkpoint_torch)
    #读取一个替身用的paddle_toch
    dict_paddle = paddle.load(r"C:\Users\jchao\Desktop\torchToPaddle\paddle\net.pdparams")

    #第一步，将dict_paddle中的所有key对应的values值转变为dict_torch
    for key in dict_paddle.keys():
        dict_paddle[key] = paddle.to_tensor(dict_torch[key].cpu().detach().numpy())

    #第二步，将dict_paddle中的所有线性层的weight全部转置
    #通过pytorch的模型排查线性层
    m = model()
    for modul in m.named_modules():
        if isinstance(modul[1], torch.nn.modules.linear.Linear):
            name = modul[0] + ".weight"
            dict_paddle[name] = dict_paddle[name].T
    paddle.save(dict_paddle, checkpoint_paddle)



if __name__ == '__main__':
    checkpoint_torch2paddle("./jicao.pt", "../paddle/jicao.pdparams")

