import numpy as np
import torch
import torch.nn as nn
from util import Util
import cv2




class CalibrationModel(nn.Module):
    def __init__(self, img):
        super(CalibrationModel, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.calibrationImg = Util().getPoint(img).numpy()[np.newaxis, :]
        # self.device = torch.device("cpu")
        # self.calibrationImg = torch.from_numpy(Util().getPoint(img).numpy()[np.newaxis, :].repeat(batch_size, 0)).to(self.device)

        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.LeakyReLU(inplace=True),
            # nn.Dropout(0.3),
            nn.Linear(128, 2))

    def forward(self, x):
        # print(self.calibrationImg.shape)
        # print(x.shape)
        ci = torch.from_numpy(self.calibrationImg.repeat(x.shape[0], 0)).to(self.device)
        x = torch.cat((ci, x), 1)
        # print(x.shape)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    img1 = cv2.imread(r"C:\Users\jchao\Desktop\bisai\picture\3.jpg")
    img2 = cv2.imread(r"C:\Users\jchao\Desktop\bisai\picture\4.jpg")

    m = CalibrationModel(img1)
    u = Util()
    print(m(torch.from_numpy(u.getPoint(img2).numpy()[np.newaxis, :])))