import time
import numpy as np
import matplotlib.pyplot as plt


class PID:
    def __init__(self,Kp,Ki,Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.sample_time = 0.01
        self.Ki_error = 0

        self.last_time = time.time()
        self.last_error = 0

    def adjust(self,target,feedback):
        now = time.time()
        error = target - feedback
        delta_error = error - self.last_error
        delta_time = now - self.last_time

        if(delta_time >= self.sample_time):
            Kp_error = self.Kp * error
            self.Ki_error += error * delta_time
            self.Kd_error = delta_error / delta_time
            self.last_time = now
            self.last_error = error

            return Kp_error + self.Ki * self.Ki_error + self.Kd * self.Kd_error


PID = PID(0.6,0.13,0.001)
# PID = PID(0.6, 0, 0)
feedback = 0
target = 20
feedback_list = []
target_list = []

for x in range(100):
    while True:
        result = PID.adjust(target,feedback)
        if result is not None:
            break
    feedback += result
    feedback_list.append(feedback)
    target_list.append(target)

print('max: ',max(feedback_list))

plt.plot(target_list)
plt.plot(feedback_list)
plt.show()


