import abc
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sloth.features import DataType
from sloth.validation_task import ValidationTask
from sloth.explainers.base_explainer import BaseExplainer, BaseExplanation
import sloth.result_cache as result_cache
from sloth import param_calc
from sloth.explainers.utils import UtilsClass

from itertools import cycle
from matplotlib import cm
from copy import deepcopy
from matplotlib.ticker import StrMethodFormatter

logger = logging.getLogger('sloth')

class  BaseGlobalExplanation(BaseExplanation):
    def __init__(self, task: ValidationTask, explainer_hash: str, method: str):
        """
        Explanation object generated by Explainer with plot functionality and opportunity to add results.
        """
        super().__init__(task, explainer_hash)
        self._method = method
        self.results = {}
        
    def add_result(self, feature_name: str, x: np.ndarray, y: np.ndarray):
        """
        Adds given data to results.
        """
        self.results[feature_name] = {'x': x, 'y': y}

    def plot(self,
             **kwargs):
        """_summary_

        Args:
            n_subplot_cols (int, optional): number of columns for subplots
                                            default: rounded root of number of features
                                            limited by 3
            include_rug (bool, optional): if True, draw and return seaborn.rug_plot at the bottom
                                            default: True
            include_hist (bool, optional): if True, draw and return matplotlib.hist
                                            default: True
            features (str, optional): _description_. Defaults to None.
            label (str, optional): #TODO wofür
            hist_params (dict, optional): _description_. Defaults to None.
            #TODO rug_params

        Raises:
            ValueError: _description_
            NotImplementedError: _description_
        """

        #hist_para = global_params.HistogramPlot.get_hist_params(1, len(self._task.data))
        parameter = UtilsClass.set_config('plot')
        #parameter.update(hist_para)
        parameter.update(kwargs)
        for k, v in parameter.items():
            setattr(self, k, v)

        if self.alpha == 'auto':
            self.alpha = param_calc.HistogramPlot.alpha(1)
        if self.bins == 'auto':
            self.bins = param_calc.HistogramPlot.bins(len(self._task.data))
        self.hist_param.update({'alpha': self.alpha, 'bins': self.bins})

        num_plot = 1
        output = next(iter(self._task.output_features.values()))
        if self.features is None:
            self.features = [k for k in self.results.keys()]
        elif isinstance(self.features, str):
            if self.features not in self.results.keys():
                raise ValueError('Feature %s not found in results.' % self.features)
            self.features = [self.features]
        if self.label is None:
            self.label = self.features

        # Determine optimal grid for subplots and color settings
        cycler = cycle(cm.tab10.colors)
        hist_color = next(cycler)
        rug_color = next(cycler)
        line_color = next(cycler)
        line_label = None
        rug_label = None
        if self.n_subplot_cols is None:
            if self.include_hist:
                self.n_subplot_cols = min(int(len((self.features))**(1/2)),3)
            else:
                self.n_subplot_cols = -1
        n_plots = None
        if self.n_subplot_cols > 0:
            line_label = 'prediction'
            rug_label = 'quantity'
            if len(self.features) < self.n_subplot_cols:
                self.n_subplot_cols = len(self.features)
            n_plots = [(len(self.features)) // (self.n_subplot_cols), self.n_subplot_cols]
            if len(self.features) % self.n_subplot_cols > 0:
                n_plots[0] += 1

        fig = plt.figure()
        for f in self.features:
            fff = self._task.input_features[f]
            if fff.data_type != DataType.ORDINAL.value:
                raise NotImplementedError('Plotting up to now only implemented for ordinal features.')
            explanation = self.results[f]
            projection, projected_values = explanation['x'], explanation['y']
            if self.n_subplot_cols > 0:
                ax1 = plt.subplot(n_plots[0], n_plots[1], num_plot)
            else:
                ax1 = plt.subplot()
                line_label = f
                line_color = next(cycler)
                rug_color = deepcopy(line_color)
            num_plot += 1

            ax1.ticklabel_format(useOffset=True)
            plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.5f}'))
            ax1.plot(projection, projected_values, '-', linewidth=3.0, label=line_label, c=line_color)
            plt.title(f, loc='left', y=1.0, pad=-13)
            plt.xlabel('feature value')

            rug_axis = ax1
            ax2 = None

            if self.include_hist:
                f_ = self._task.input_features[f]
                ax2 = ax1.twinx()
                rug_axis = ax2
                ax1.set_zorder(ax2.get_zorder() + 1)
                ax1.set_frame_on(False)
                if f_.inverse_transform is not None:
                    ax2.hist(f_.inverse_transform(self._task.data[:, f_.column]),
                             density=True, label='frequency', color=hist_color, **self.hist_param)
                else:
                    ax2.hist(self._task.data[:, f_.column], density=True, label='frequency', color=hist_color, **self.hist_param)
            if self.include_rug:
                f_ = self._task.input_features[f]
                if f_.inverse_transform is not None:
                    sns.rugplot(f_.inverse_transform(self._task.data[:, f_.column]), lw=1,
                                ax=rug_axis, alpha=0.3, color=rug_color, label=rug_label, height= 0.05)
                else:
                    sns.rugplot(data=self._task.data[:, f_.column], lw=1, alpha=0.3,
                                ax=rug_axis, color=rug_color, label=rug_label, height= 0.05)

            print()
        # general layout settings
        handles, labels = ax1.get_legend_handles_labels()
        if ax2 is not None:
            # dummy axes for y labels
            ax_right = fig.add_subplot(1, 1, 1)
            ax_right.set_xticks([])
            ax_right.set_yticks([])
            [ax_right.spines[side].set_visible(False) for side in ('left', 'top', 'right', 'bottom')]
            ax_right.patch.set_visible(False)
            ax_right.yaxis.set_label_position('right')
            ax_right.set_ylabel('frequency', labelpad=30, fontsize='large')
            # shared legend
            handles2, labels2 = ax2.get_legend_handles_labels()
            for item, label in zip(handles2, labels2):
                handles.append(item)
                labels.append(label)
        fig.legend(handles=handles, labels=labels, ncol=3)
        plt.suptitle(self._method, fontsize='xx-large')
        plt.subplots_adjust(wspace=0.4, hspace=0.4)
        fig.supylabel(output.name, fontsize='large')

        if self.show:
            plt.show()



class BaseGlobalExplainer(BaseExplainer):
    def __init__(self, task: ValidationTask,  **kwargs):
        super().__init__(task, **kwargs)

    @abc.abstractmethod
    def _explain(self)->BaseExplanation:
        pass

    def explain(self)->BaseExplanation:
        if result_cache.has_result(self._hashkey):
            logger.info('Using cached result for hashkey  %s', self._hashkey)
            return result_cache.get_result(self._hashkey)
        else:
            result = self._explain()
            result_cache.add_result(self._hashkey, result)
            return result