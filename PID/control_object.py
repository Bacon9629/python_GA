class ControlObject:
    def __init__(self):
        # self.value = 0
        pass

    def clear_and_init(self):
        """清空and初始化"""
        pass

    def get_value(self):
        """取得現在要控制的值"""
        pass

    def next(self, input_val):
        """更新現在要控制的值，輸入是某種參數，可依輸入值影響輸出"""
        return self.get_value()
