import pickle
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class PlotComparison():
    def __init__(self, filenames=list()):
        self.filenames  = filenames
    
    def appendfile(self, names):
        self.filenames.append(names)
    
    def show(self, step=100, max_n_steps = -1):
        files = list()
        for fn in self.filenames:
            files.append(open(fn, 'rb'))
        
        rw_lists = list()
        min_len = 10e6
        for f in files:
            rw_lists.append(pickle.load(f))
            min_len = min(min_len, len(rw_lists[-1]))
        
        if max_n_steps != -1:
            min_len = min(max_n_steps, min_len)
        print('Steps to Print> ', min_len)

        xvals = np.arange(1, min_len * step, step)

        for rw_list in rw_lists:
            plt.plot(xvals, np.asarray(rw_list[:min_len]))
        plt.legend(self.filenames)
        plt.show()
    
plothandler =   PlotComparison(['XS001-rw.pckl','XS002-rw.pckl'])
plothandler.show(step=100, max_n_steps=200)