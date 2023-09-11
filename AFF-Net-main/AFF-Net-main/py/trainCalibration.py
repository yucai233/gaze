import myCalibration
import myCalibrationDatasetReader
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
import copy
import yaml
import math
import time


if __name__ == "__main__":
    data_path = r"C:\Users\jchao\Desktop\calibrationDataset"
    model_name = "feiwu-Net"
    save_path = r"C:\Users\jchao\Desktop\AFF-Net-main\AFF-Net-main\checkpoint1"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 16
    all_epoch = 1600
    save_step = 200
    decay_steps = [500, 1000, 1500]
    cur_decay_index = 0


    #数据准备
    print("Read data")
    dataset = myCalibrationDatasetReader.txtload(data_path, "train", batch_size, shuffle=True,
                             num_workers=0)
    length = len(dataset)

    #模型准备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = myCalibration.CalibrationModel(cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie\59.png"))
    net.train()
    net = nn.DataParallel(net)
    net.to(device)
    print("Model building done")



    #损失函数和优化器
    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    base_lr = 0.01
    optimizer = torch.optim.Adam(net.parameters(), base_lr,
                                weight_decay=0.0005)
    print("optimizer building done")



    print("Traning")

    for epoch in range(1, all_epoch + 1):
        for i, (data) in enumerate(dataset):
            #判断是否需要衰减
            if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
                cur_decay_index += 1
                base_lr /= 10
                for param_group in optimizer.param_groups:
                    param_group["lr"] = base_lr

            #拿到数据
            data["img"] = data["img"].to(device)
            label = data["label"].to(device)
            # print(data["leftEyeImg"].shape)
            # print(data["rightEyeImg"].shape)
            # print(data["faceImg"].shape)
            # print(data["rects"].shape)

            #参数优化
            # print("进入网络之前：", data["img"])
            gaze = net(data["img"])
            loss = loss_op(gaze, label)
            # loss = loss_op(gaze, label)*4   #损失加权？没看懂
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #日志输出
            log = f"[{epoch}/{all_epoch}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr}\n"
            print(log)

        if epoch % save_step == 0:
            torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch + 1}_{model_name}.pt"))

