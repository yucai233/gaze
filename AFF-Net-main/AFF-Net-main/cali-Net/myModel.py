#在这里构建模型
import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

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
