import os
import torch
from torch import nn
import reader
import model


if __name__ == "__main__":
    model_name = "AFF_finetrue-Net"
    save_path = r"C:\Users\jchao\Desktop\AFF-Net-main\AFF-Net-main\checkpoint"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 16
    all_epoch = 20
    save_step = 5
    base_lr = 0.001


    #数据准备
    print("Read data")
    dataset = reader.txtload("train", batch_size, shuffle=True, num_workers=0)
    length = len(dataset)

    #模型准备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = model.model()
    net = nn.DataParallel(net)
    net = net.module
    state_dict = torch.load("../checkpoint/Iter_20_AFF_finetrue-Net.pt")
    net.load_state_dict(state_dict)
    net.to(device)
    net.train()
    print("Model building done")



    #损失函数和优化器
    print("optimizer building")
    loss_op = nn.SmoothL1Loss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), base_lr,
                                weight_decay=0.0005)
    print("optimizer building done")



    print("Traning")

    for epoch in range(1, all_epoch + 1):
        if epoch == 10:
            base_lr /= 10
            for param_group in optimizer.param_groups:
                param_group["lr"] = base_lr
        for i, (data) in enumerate(dataset):
            #拿到数据
            data["face_img"] = data["face_img"].to(device)
            data["leftEye_img"] = data["leftEye_img"].to(device)
            data["rightEye_img"] = data["rightEye_img"].to(device)
            data["rects"] = data["rects"].to(device)
            label = data["label"].to(device)

            gaze = net(data["leftEye_img"], data["rightEye_img"], data["face_img"], data["rects"])

            loss = loss_op(gaze, label) * 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #日志输出
            log = f"[{epoch}/{all_epoch}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr}\n"
            print(log)

        if epoch % save_step == 0:
            torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))

