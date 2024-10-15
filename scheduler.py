import math
from math import cos

def transformation(value):
    return (math.cos(value ** 4) / 2) - 0.5

class Scheduler:
    def __init__(self, first_finite):
        self.first_finite = first_finite
        self.norm_factor = math.pi ** 0.25
    def sample(self, percentage_done):
        # cosine only valid for anything between 0.1 < x < 0.6
        value = self.norm_factor * (1 - percentage_done)
        return transformation(value) * self.first_finite


scheduler = Scheduler(100)
max_count = 100
for i in range(max_count):
    print(i, scheduler.sample(i/max_count))