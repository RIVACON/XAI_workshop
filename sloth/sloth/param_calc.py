# This file contains different classes handling default parameters


class HistogramPlot():
    @staticmethod
    def alpha(n_graphs):
        return 1.0/n_graphs
    @staticmethod
    def bins(n_datapoints):
        if n_datapoints > 2500:
            return int(n_datapoints/500)
        elif n_datapoints > 500:
            return int(n_datapoints/100)
        elif n_datapoints > 50:
            return int(n_datapoints/10)
        else:
            return n_datapoints
    # @staticmethod
    # def get_hist_params(n_graphs, n_datapoints):
    #     return {'alpha': HistogramPlot.alpha(n_graphs),
    #             'bins': HistogramPlot.bins(n_datapoints)}