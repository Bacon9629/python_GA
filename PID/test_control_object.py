from PID.control_object import ControlObject


class Test(ControlObject):
    def __init__(self):
        super().__init__()
        self.last_value = 0

    def clear_and_init(self):
        """清空and初始化"""
        self.last_value = 0
        pass

    def get_value(self):
        """取得現在要控制的值"""
        return self.last_value

    def next(self, x):
        """更新現在要控制的值，輸入是某種參數，可依輸入值影響輸出"""
        # a = (3*x*x + 2*x + 4) + self.last_value
        a = x/(x*x + 5*x + 2)
        # a = (x) + self.last_value
        if a == float("inf"):
            a = 1e300
        elif a == -float("inf"):
            a = -1e300
        self.last_value = a
        return self.get_value()

