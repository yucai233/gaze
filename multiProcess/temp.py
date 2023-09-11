#测试多线程问题
import threading
import time
import random

#
# def func(i, event):
#     print("线程{}启动成功".format(i))
#     event.wait()
#     print("线程{}执行完毕".format(i))
#
#
# if __name__ == "__main__":
#     event = threading.Event()
#
#     thread_list = []
#     for i in range(4):
#         thread_list.append(threading.Thread(target=func, args=(i, event)))
#
#     #开始启动子线程
#     for t in thread_list:
#         t.start()
#
#     time.sleep(5)
#     event.set()
#
#     for t in thread_list:
#         t.join()
#     print("程序结束")




#
# #红绿灯的例子
# def car(i, event):
#     #首先为汽车获取一个随机到达路口的时间
#     t = random.randint(1, 11)
#     print("汽车{}开始出发，还有{}秒到达路口".format(i, t))
#     time.sleep(t)
#     print("汽车{}到达路口".format(i))
#     event.wait()
#     print("汽车{}穿过红绿灯".format(i))
#
# if __name__ == "__main__":
#     #首先拿到事件
#     event = threading.Event()
#     #设置当前为红灯
#     event.clear()
#     #创建5辆汽车
#     car_list = []
#     for i in range(5):
#         car_list.append(threading.Thread(target=car, args=(i, event)))
#
#     #将5辆汽车开动
#     for car in car_list:
#         car.start()
#
#     #开始管控红绿灯
#     #首先是2秒红灯，然后是2秒绿灯，以此类推
#     while threading.activeCount() > 1:
#         #只要还有车子没有通过红绿灯，那么就继续循环
#         print("当前是红灯")
#         event.clear()
#         time.sleep(2)
#
#         print("当前是绿灯")
#         event.set()
#         time.sleep(2)

#
# def func(name):
#     for i in range(10):
#         time.sleep(1)
#         print(name, i)
#
# if __name__ == "__main__":
#     t1 = threading.Thread(target=func, args=("Threading 1 ->",))
#     t2 = threading.Thread(target=func, args=("Threading 2 ->",))
#
#     t1.start()
#     t2.start()
#
#     t1.join()
#     t2.join()




#模拟真实场景
import threading
import time


event = threading.Event()
mylock = threading.RLock()
parameter = []

#定义子线程
def childThread():
    global event, mylock, parameter

    while True:
        #首先等待主进程发送请求信号
        print("开始等待主进程发送请求信号：")
        time.sleep(1)
        #然后等待数据就绪
        print("接受到主进程信号，开始等待数据就绪")
        event.wait()
        print("数据就绪，清空就绪标志")
        event.clear() #等待成功后清理掉标志，代表最新鲜的数据已经拿走了
        #开始访问共享内存中的数据，但是需要先加锁
        print("拿到线程锁")
        mylock.acquire()
        print("数据内容为：", parameter)
        print("释放线程锁")
        mylock.release()

if __name__ == "__main__":
    event.clear()

    child_thread = threading.Thread(target=childThread)
    child_thread.start()

    i = 0
    while True:
        #不断生产数据
        time.sleep(2)
        print("生产数据成功，当前数据为", i)
        #申请锁
        mylock.acquire()
        parameter = [i]
        #释放锁
        mylock.release()
        #更新就绪信号
        event.set()

        i += 1

    child_thread.join()


#
# #在主进程中定义的全局变量，在子进程中会有相同的一份吗
# import multiprocessing
#
# #首先在主进程中定义全局变量
# a = 1
# b = 2
# c = []
#
# def childP():
#     print("子进程启动成功")
#     global a, b, c
#     print("子进程->", a)
#     print("子进程->", b)
#     print("子进程->", c)
#
#     #修改变量的值
#     a = 100
#     b = 200
#     c.append(1)
#     print("子进程->", a)
#     print("子进程->", b)
#     print("子进程->", c)
#     time.sleep(10)
#
# if __name__ == "__main__":
#     #首先分离子进程
#     child_process = multiprocessing.Process(target=childP)
#     child_process.start()
#
#     time.sleep(5)
#     print("主进程->", a)
#     print("主进程->", b)
#     print("主进程->", c)
#
#     child_process.join()
#     print("程序结束")



