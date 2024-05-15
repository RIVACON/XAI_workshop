import numpy as np
import logging
try:
    import hiplot as hip
    has_hiplot = True
except ImportError:
    has_hiplot = False

from sloth.features import DataType
from sloth.validation_task import ValidationTask
from sloth.explainers.datapoints.prototypes_criticisms import MMD
from sloth.explainers.datapoints.extreme_points import ExtremePoints, ExtremePredictions, DataPoints

logger = logging.getLogger('sloth')

class Examples(DataPoints):

    def __init__(self,
                 task: ValidationTask,
                 extreme_prediction=True,
                 coordinate_extreme_points=True,
                 mmd=True,
                 **kwargs,
                 ):


        super().__init__(task,
                         extreme_prediction=extreme_prediction,
                         coordinate_extreme_points=coordinate_extreme_points,
                         mmd=mmd,
                         **kwargs)
        self.points = []
        # self.compute_points()

    def compute_points(self):
        if self.coordinate_extreme_points:
            logging.debug('Start computing extreme points.')
            extreme_points = ExtremePoints(self.task)
            self.points = extreme_points._get_coordinate_extreme_points()
            logging.debug('Finished computing extreme points.')
        if self.extreme_prediction:
            if self.task.problemtype != 'regression':
                raise TypeError("Points of extreme predictions can only be analyzed in the case of regression. "
                                "Set 'extreme_prediction' to 'false' and start process again to continue.")
            elif self.task.output_dim() > 1:
                raise NotImplementedError("Computing examples from extreme prediction values"
                                          "only implemented for tasks with one output."
                                          "Set 'extreme_prediction' to 'false' and start process again to continue.")
            else:
                logging.debug('Start computing points of extreme predictions.')
                extreme_predictions = ExtremePredictions(self.task)
                q = min(extreme_predictions.quantile,
                        extreme_predictions.max_n_points / self.task.data.shape[0])
                tmp = np.where(self.task.y_pred<=np.quantile(self.task.y_pred, q))[0]
                self.points.extend(tmp.tolist())
                tmp = np.where(self.task.y_pred>=np.quantile(self.task.y_pred, 1.0-q))[0]
                self.points.extend(tmp.tolist())
                logging.debug('Finished computing points of extreme predictions.')
        if self.mmd:
            logging.debug('Start computing MMD prototypes and criticisms.')
            mmd = MMD(self.task)
            tmp = mmd.compute_prototypes_and_criticisms()
            logging.debug('Finished computing MMD prototypes and criticisms.')
            self.points.extend(tmp['prototypes'])
            self.points.extend(tmp['criticisms'])
        logging.debug('Remove redundant points.')
        points_before = len(self.points)
        self.points = np.array(list(set(self.points)))
        logging.debug('Removed ' + str(points_before-self.points.shape[0]) + ' points, ' \
                      + str(self.points.shape[0]) + ' points remaining.')

    def add_example(self, point: int, name: str=None):
        """Adds given data to points and thus helps to analyze own examples or special points."""
        self.points = np.concatenate([self.points, [point]])

    def hiplot(self):
        if not has_hiplot:
            raise Exception('HiPlot not installed. To use this method first install hiplot, see https://facebookresearch.github.io/hiplot/index.html')
        experiments = []
        for index in self.points:
            tmp={}
            #index = self.points[i]
            y = self.task.y_pred[index]
            for f in self.task.output_features.values():
                try:
                    tmp[f.name] = y[f.column]
                except:
                    tmp[f.name] = y
            for f in self.task.input_features.values():
                if f.data_type != DataType.ONE_HOT_ENCODED.value:
                    tmp[f.name] = self.task.data[index, f.column]
                else:
                    for i in range(len(f.column)):
                        if self.task.data[index, f.column[i]] == 1:
                            tmp[f.name] = f.category_names[i]
            experiments.append(tmp)
        
        order_by =  [f.name  for f in self.task.output_features.values()]        
        exp = hip.Experiment.from_iterable(experiments)
        exp.display_data(hip.Displays.TABLE).update({
                # In the table, order rows by default
                'order_by': [order_by],
                #'order': ['test loss']
        })#exp.display_data(hip.Displays.PARALLEL_PLOT)
        exp.display_data(hip.Displays.PARALLEL_PLOT).update({
                #'order': ['stdev test loss', 'train loss', 'test loss'], 

        })
        exp.display()

if __name__ == '__main__':
    import sloth
    # validation_task1 = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=3, f=0)
    validation_task2 = sloth.datasets.test_sets.simple_regression_ordinal_discrete_ohe(n_samples=1_000, x=2, f=0)
    ex = Examples(validation_task2)
    ex.compute_points()
    # ex.hiplot()
    print(ex.points)

