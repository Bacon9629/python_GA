import matplotlib.pyplot as plt
import numpy as np
from my_airplane import Airplane
from test_control_object import Test
from control_object import ControlObject


class PID:
    def __init__(self, control_object: ControlObject, p, i, d, target_function, sample_time=0.01, start_time=0.0, end_time=5.0):
        """F_function(time): 定義要追隨的波型，參數有一個 time，為系統執行時間"""
        self.control_object = control_object
        self.p = p
        self.i = i
        self.d = d
        self.target_function = target_function
        self.sample_time = sample_time
        self.start_time = start_time
        self.end_time = end_time

        # self.last_error = 0
        self.last_error = control_object.get_value()
        self.i_error = 0

        self.time_list = np.arange(start_time, end_time, sample_time)
        self.count = 1

        self.targets = np.zeros(self.time_list.shape, dtype=float)
        self.my_values = self.targets.copy()
        # self.last_value = 0

    def clear_and_init(self, p=None, i=None, d=None):
        if p is not None:
            self.p = p
        if i is not None:
            self.i = i
        if d is not None:
            self.d = d
        self.control_object.clear_and_init()
        self.last_error = self.control_object.get_value()
        self.i_error = 0

        self.time_list = np.arange(self.start_time, self.end_time, self.sample_time)
        self.count = 1

        self.targets = np.zeros(self.time_list.shape, dtype=float)
        self.my_values = self.targets.copy()

    def next(self):
        """
        回傳：
        1. 要追蹤的目標值
        2. 我的現在值
        3. 現在系統模擬時間
        """
        now_time = self.time_list[self.count]
        delta_time = self.sample_time
        now_target = self.target_function(now_time)  # 系統目標值
        now_error = now_target - self.control_object.get_value()  # 現在我的值跟系統的差距
        # print(now_error)
        # delta_error = now_error - self.last_error
        p_adjust = now_error * self.p
        self.i_error += now_error * delta_time
        i_adjust = self.i_error * self.i
        if delta_time != 0:
            d_adjust = (now_error / delta_time) * self.d
        else:
            d_adjust = 0

        p_adjust, self.i_error, i_adjust, d_adjust = self.__check_float_infinite(p_adjust, self.i_error, i_adjust, d_adjust)

        # self.last_value += (p_adjust + i_adjust + d_adjust)
        new_height = self.control_object.next(p_adjust + i_adjust + d_adjust)

        self.targets[self.count] = now_target
        self.my_values[self.count] = new_height

        self.count += 1

        return now_target, new_height, self.time_list[self.count-1]

    def __check_float_infinite(self, *args):
        if float('inf') in args or -float('inf') in args:
            _temp = np.array(args)
            positive_idx = _temp == float("inf")
            negative_idx = _temp == -float("inf")
            _temp[positive_idx] = 1e300
            _temp[negative_idx] = -1e300
            return _temp
        else:
            return args

    def simulate(self):
        _t = 0.0
        while _t < self.time_list[-1]:
            # self.next()
            a, b, _t = self.next()
            # print(a)

        return self.targets, self.my_values, self.time_list

    def get_loss(self):
        # loss_list = np.zeros(self.time_list.shape, dtype=np.float)
        # loss = ((self.targets - self.my_values) ** 2).sum()
        loss = np.sqrt(((self.targets - self.my_values) ** 2).sum())
        return loss

if "__main__" == __name__:
    def F(x):

        # return 0
        # x = x/2 - x//2
        # if x <= 0.5:
        #     return 0
        # else:
        #     return 5

        # if x < 0.5:
        #     return 0
        # elif x < 1:
        #     return 5
        # elif x < 1.5:
        #     return 0
        # else:
        #     return 5

        return -(x*x*x - 2 * x*x)
        # return (x - 0.2) ** 2
        # return 400/(x**2 + x)
        # return 400*np.exp(-2*x)/(x**2 + x)
        # return 2

    # pid_list = [100, 300, 200]
    pid_list = [14.79994563, -94.37529224, 3.76736659]

    print("start")
    sample_time = 0.01
    # pid = PID(Airplane(sample_time), 1, 0, 0, F, sample_time, 0.2, 2)
    pid = PID(Airplane(sample_time), *pid_list, F, sample_time, 0.2, 2)

    # pid = PID(Test(), *pid_list, F, sample_time, 0.2, 5)

    targets_y, my_values_y, t = pid.simulate()
    loss_list = pid.get_loss()
    print(loss_list)

    print("end")

    plt.plot(t, my_values_y, color="blue")
    plt.plot(t, targets_y, color="red")
    plt.show()

    # print(pid.F(1))





