# import matplotlib.pyplot as plt
import numpy as np
from PID.control_object import ControlObject


class Airplane(ControlObject):
    def __init__(self, sample_time):
        super().__init__()
        self.sample_time = sample_time

        self.height = 0  # 我的高度，就是我的回傳值
        # self.last_power = 0
        self.last_acceleration = 0  # 加速度

        self.delay_sample_time = 0  # 延遲系統要delay幾個sample time
        self.power_queue = [0 for i in range(self.delay_sample_time)]

    def clear_and_init(self):
        self.height = 0  # 我的高度，就是我的回傳值
        self.last_acceleration = 0  # 加速度
        self.power_queue = [0 for i in range(self.delay_sample_time)]

    def acceleration_power_function(self, power):
        # if power > 100:
        #     power = 100
        # elif power < 100:
        #     power = -100

        return power + self.last_acceleration * 0.5

    def __height_update(self, acceleration):
        self.height += self.sample_time * acceleration * acceleration
        self.last_acceleration = acceleration

        # self.height += self.get_noise()

    def get_value(self):
        """取得現在控制項的值"""
        return self.height

    def get_noise(self):
        """有側風吹過來，就是雜訊"""
        return (np.random.random() - 0.5) * self.height * 0.1

    def next(self, power):
        """更新控制項的值並回傳"""
        self.__height_update(self.acceleration_power_function(power))

        # 延遲系統
        # self.__height_update(self.acceleration_power_function(self.power_queue.pop(0)))
        # self.power_queue.append(power)

        return self.get_value()


