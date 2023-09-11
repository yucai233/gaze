# import frameToPoint
# import timeit
# import cv2
#
# if __name__ == "__main__":
#     util = frameToPoint.FrameToPoint(r"C:\Users\jchao\Desktop\calibrationDataset\gagajie2", r"./checkpoint/Iter_20_jie2.pt")
#     # frame = cv2.imread(r"C:\Users\jchao\Desktop\ceshi\0.png")
#     frame = cv2.imread(r"C:\Users\jchao\Desktop\calibrationDataset\jie2\{}.png".format(2))
#
#     print("总共耗时：", timeit.timeit(lambda: util.frameToPoint(frame), number=100))
#     print("人脸识别与眼睛识别耗时：", timeit.timeit(lambda : frameToPoint.getPar(frame, "cv"), number=100))


#
# #测试多进程
# from multiprocessing import Process, Pipe
# import time
# import numpy as np
#
# def sonP(pipe):
#     out_pipe, in_pipe = pipe
#     #子进程主要用来接受数据，所以关闭in_pipe管道
#     in_pipe.close()
#
#     #循环从out_pipe中读取数据，并显示出来
#     while True:
#         try:
#             msg = out_pipe.recv()
#             print(msg)
#             time.sleep(1)
#         except EOFError:
#             break
#
#     out_pipe.close()
#     print("子进程结束，成功退出")
#
# if __name__ == "__main__":
#     ret = Pipe(True)
#     son_process = Process(target=sonP, args=(ret,))
#     son_process.start()
#
#     out_pipe, in_pipe = ret
#     out_pipe.close()
#     for i in range(10):
#         in_pipe.send([1, 2, 3])
#         # time.sleep(1)
#
#     # while True:
#     #     msg = input("->")
#     #     in_pipe.send(msg)
#     in_pipe.close()
#     print("发送完毕，管道关闭成功")
#     son_process.join()
#     print("主线程执行完毕")


from multiprocessing import Process, Pipe
import time


def childP(child_pipe):
    #子进程首先等待主进程发送开始的信号
    print("子---》子进程启动成功，现在开始等待主进程发送开始信号")
    child_pipe.recv()
    print("子---》接受主进程开始信号成功，现在开始工作")

    #拿到开始信号后，子进程开始处理图片，并将处理结果返回给主进程
    while True:
        time.sleep(20)
        a = 2
        print("子---》向主进程发送数据")
        child_pipe.send(a)  # 继续等待主进程发送信号
        #等待主进程拿走数据
        print("子---》等待主进程拿走数据")
        child_pipe.recv()
        print("子---》主进程已经拿走数据，现在继续工作, 子进程处理需要20s")


if __name__ == "__main__":
    child_pipe, parent_pipe = Pipe(True)
    #创建子进程
    child_process = Process(target=childP, args=(child_pipe,))
    #运行子进程
    child_process.start()

    #首先向子进程发送信号，表示可以开始工作了
    parent_pipe.send(int(1))
    print("主----》主进程发送开始信号成功, 现在开始等待接受子进程发送过来的数据")
    while True:
        msg = parent_pipe.recv()
        print("主----》拿到子进程发送过来的数据，想子进程发信号，表示可以继续发送了")
        parent_pipe.send(int(1))
        #接下来是处理子进程发送过来的数据
        print("主----》开始处理数据，处理时间：10s")
        time.sleep(10)
        print("主----》处理完毕，这是子进程发送过来的数据", msg)
        print()

