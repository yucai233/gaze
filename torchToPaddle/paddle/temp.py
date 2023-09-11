# import paddle
import torch

# #首先打开两个文件
# dict_of_torch = torch.load(r"./Iter_10_jie2.pt")
# dict_of_paddle = paddle.load(r"./net.pdparams")
#
# #将dict_of_paddle中的内容修改为和dict_of_torch的一样
# for key in dict_of_paddle.keys():
#     dict_of_paddle[key] = paddle.to_tensor(dict_of_torch[key].cpu().detach().numpy())
#
# paddle.save(dict_of_paddle, r"./net.pdparams")


class model(torch.nn.Module):
    def __init__(self):
        super(model, self).__init__()
        self.a = 1

m = model()
print(m.a)