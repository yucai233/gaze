import os
import torch
from torch import nn
import combModel
import combReader
import combCaliReader


if __name__ == "__main__":
    data_path = r"C:\Users\jchao\Desktop\calibrationDataset\gagachao"
    model_name = "chao"
    save_path = r"C:\Users\jchao\Desktop\AFF-Net-main\AFF-Net-main\combine-Net\checkpoint"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 16
    all_epoch = 10
    save_step = 10
    decay_steps = [i for i in range(1, 11)]
    cur_decay_index = 0
    base_lr = 0.0001


    #数据准备
    print("Read data")
    dataset = combCaliReader.txtload(data_path, "train", batch_size, shuffle=True,
                             num_workers=0)
    length = len(dataset)

    #模型准备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Model building")
    net = combModel.model()
    net = nn.DataParallel(net)
    net = net.module
    # state_dict = torch.load("../checkpoint/Iter_2_AFF-Net.pt")
    # state_dict2 = torch.load("../checkpoint1/Iter_3_feiwu-Net.pt")
    # state_dict.pop("module.fc.2.weight")
    # state_dict.pop("module.fc.2.bias")
    # state_dict["module.fc.3.weight"] = state_dict2['fc.3.weight']
    # state_dict["module.fc.3.bias"] = state_dict2['fc.3.bias']
    # state_dict["module.fc.0.weight"] = state_dict2['fc.0.weight']
    # state_dict["module.fc.0.bias"] = state_dict2['fc.0.bias']
    # state_dict = torch.load("./checkpoint/jicao.pt")
    state_dict = torch.load("./checkpoint/Iter_30_chao.pt")
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
        # #调整学习率
        # if epoch == 10:
        #     base_lr /= 10
        #     for param_group in optimizer.param_groups:
        #         param_group["lr"] = base_lr
        for i, (data) in enumerate(dataset):
            # #判断是否需要衰减
            # if cur_decay_index < len(decay_steps) and epoch == decay_steps[cur_decay_index]:
            #     cur_decay_index += 1
            #     base_lr /= 10
            #     for param_group in optimizer.param_groups:
            #         param_group["lr"] = base_lr

            #拿到数据
            data["faceImg1"] = data["faceImg1"].to(device)
            data["leftEyeImg1"] = data["leftEyeImg1"].to(device)
            data["rightEyeImg1"] = data["rightEyeImg1"].to(device)
            data["rects1"] = data["rects1"].to(device)
            data["faceImg2"] = data["faceImg2"].to(device)
            data["leftEyeImg2"] = data["leftEyeImg2"].to(device)
            data["rightEyeImg2"] = data["rightEyeImg2"].to(device)
            data["rects2"] = data["rects2"].to(device)

            label = data["label"].to(device)

            diff = net(data["leftEyeImg1"], data["rightEyeImg1"], data["faceImg1"], data["rects1"],
                       data["leftEyeImg2"], data["rightEyeImg2"], data["faceImg2"], data["rects2"])

            loss = loss_op(diff, label) * 4
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            #日志输出
            log = f"[{epoch}/{all_epoch}]: [{i}/{length}] loss:{loss:.5f} lr:{base_lr}\n"
            print(log)

        if epoch % save_step == 0:
            torch.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pt"))

