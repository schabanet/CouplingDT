from .Utils import DataItem
from scipy.stats import expon
import numpy as np

class StreamBase:
    def __init__(self, arrivalRate = 1):
        self.arrivalNumber = 0
        self.arrivalRate = arrivalRate

    def generateTime(self):
        return expon.rvs(scale = self.arrivalRate)

    def generateDataItem(self):
        self.arrivalNumber += 1
        return DataItem(self.arrivalNumber-1)


