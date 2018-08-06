from __future__ import print_function
import os
from collections import OrderedDict
import numpy as np

import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt

class Logger():
    def __init__(self, save_weight_path):
        self.log_dir = save_weight_path
        self.iter_log = []

    def iterLogUpdate(self, loss):
        """
        iteration log: [iter][loss]
        """
        self.iter_log.append(loss)

    def draw_loss_curve(self):
        fig = plt.figure()
        ax  = fig.add_subplot(111)
        ax.plot(range(len(self.iter_log)), self.iter_log)
        ax.set_title('Loss Curve')
        plt.tight_layout()
        fig.savefig(os.path.join(self.log_dir, 'epoch_summary.pdf'))
        plt.close(fig)

            


