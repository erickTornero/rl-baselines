"""
    Cummulate a metric to get mean value, etc
"""
class Cummulator:
    def __init__(self):
        self.value = 0
        self.counter = 0

    def step(self, value):
        self.value += value
        self.counter += 1

    def get(self):
        return self.value
    
    def reset(self):
        self.value = 0
        self.counter = 0

    def mean(self):
        return self.value/self.counter