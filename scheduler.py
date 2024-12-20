import math
from math import cos

def transformation(value):
    return (math.cos(value ** 6) / 2) - 0.5

class Scheduler:
    def __init__(self, first_finite):
        self.first_finite = first_finite
        self.norm_factor = math.pi ** (1/6)
    def sample(self, percentage_done):
        # cosine only valid for anything between 0.1 < x < 0.6
        value = self.norm_factor * (1 - percentage_done)
        return transformation(value) * self.first_finite

if __name__ == "__main__":
    thingy = Scheduler(100)
    count = 1000
    for i in range(count):
        print(i, thingy.sample(i/count))