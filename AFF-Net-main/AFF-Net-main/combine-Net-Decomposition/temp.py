import frameToPoint
import timeit
import cv2

class test():
    def __init__(self):
        self.b = 123
        self.a = self.func()

    def func(self):
        return self.b

# if __name__ == "__main__":
#     util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint/Iter_20_jie2.pt")
#     # frame = cv2.imread(r"C:\Users\jchao\Desktop\ceshi\0.png")
#     frame = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(2))
#
#     print("总共耗时：", timeit.timeit(lambda: util.frameToPoint(frame), number=100))
#     print("人脸识别与眼睛识别耗时：", timeit.timeit(lambda : frameToPoint.getPar(frame, "cv"), number=100))

if __name__ == "__main__":
    t = test()
    print(t.a)