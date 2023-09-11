import os
import combModel
import combCaliReader
import combReader

import paddle


if __name__ == "__main__":
    data_path = r"C:\Users\jchao\Desktop\calibrationDataset\gagawei"
    model_name = "layer_3_fineturn"
    save_path = r"./"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    batch_size = 16
    all_epoch = 5
    save_step = 1
    decay_steps = [i for i in range(1, 11)]
    cur_decay_index = 0
    base_lr = 0.0001


    #数据准备
    print("Read data")
    dataset = combCaliReader.txtload(data_path, "train", batch_size, shuffle=True, num_workers=0)
    # dataset = combReader.txtload(r"D:/MPIIFaceGaze", r"c:/Users/jchao/Desktop/screenSize.txt", 16, shuffle=True, num_workers=0)
    length = len(dataset)

    #模型准备
    paddle.device.set_device("gpu")
    print("Model building")
    net = combModel.model()
    state_dict = paddle.load("./checkpoint_3_layer/Iter_4_comb.pdparams")
    net.set_state_dict(state_dict)
    net.train()
    print("Model building done")



    #损失函数和优化器
    print("optimizer building")
    loss_op = paddle.nn.SmoothL1Loss()
    optimizer = paddle.optimizer.Adam(parameters=net.parameters(), learning_rate=base_lr, weight_decay=0.0005)
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
            label = data["label"]

            diff = net(data["leftEyeImg1"], data["rightEyeImg1"], data["faceImg1"], data["rects1"],
                       data["leftEyeImg2"], data["rightEyeImg2"], data["faceImg2"], data["rects2"])

            loss = loss_op(diff, label)
            # loss = loss_op(diff, label) * 4
            optimizer.clear_grad()
            loss.backward()
            optimizer.step()

            #日志输出
            log = f"[{epoch}/{all_epoch}]: [{i}/{length}] loss:{loss.numpy()[0]:.5f} lr:{base_lr}\n"
            print(log)

        if epoch % save_step == 0:
            paddle.save(net.state_dict(), os.path.join(save_path, f"Iter_{epoch}_{model_name}.pdparams"))

