import matplotlib.pyplot as plt
import math

class SubPlots:
    def _figure(self):
        def figure():
            self.index += 1
            return plt.subplot(self.n_rows, self.n_cols,self.index-1)
        return figure
    
    @staticmethod
    def _do_nothing():
        pass

    def __init__(self, n_plots: int, max_cols: int, 
                 subplotting: bool = True, new_figure: bool = True):
        if new_figure:
            if subplotting:
                self.n_cols = min(max_cols, n_plots)
                self.n_rows = math.ceil(n_plots/self.n_cols)
                self.index = 1
                self.figure = self._figure()
            else:
                self.figure = plt.figure
        else:
            self.figure = SubPlots._do_nothing