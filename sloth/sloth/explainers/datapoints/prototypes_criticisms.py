import logging
from sklearn.metrics.pairwise import pairwise_kernels

try:
    import hiplot as hip
    has_hiplot = True
except ImportError:
    has_hiplot = False
    
from sloth.validation_task import ValidationTask
from sloth.explainers.datapoints.extreme_points import DataPoints


logger = logging.getLogger('sloth')

class MMD(DataPoints):
    def __init__(self,
                 task: ValidationTask,
                 n_prototypes=5,
                 n_criticisms=5,
                 metric='rbf',
                 witness_penalty=1.0,
                 max_n_points=10000,
                 pw_kernel_kwargs={},
                 **kwargs):
        """
        Class to compute prototypes and criticisms using the maximum mean discrepancy (MMD) as a measure of discrepancy between two distributions.

        This method calculates prototypes and criticisms for a set of given datapoints. Prototypes are the typical representatives of the datapoints, while criticisms are  datapoints that are not well represented by the data. Here, a simple greedy algorithm using the squared maximum mean discrepancy is used. A simple penalty is applied within the computation of criticisms to avoid  (see e.g. C. Molnar, Interpretable Machine Learning).
        In more detail, we estimate the density of the prototypes and the data using  kernel density estimators and compare the discrepancy between the two with the maximum mean discrepancy (MMD) that can be calculated by:

        .. math::

           MMD^2 = \\frac{1}{m^2}\\sum_{i=1}^m \\sum_{j=1}^m k(z_i,z_j) - \\frac{2}{mn}\\sum_{i=1}^m \\sum_{j=1}^n k(z_i,x_j) + \\frac{1}{n^2}\\sum_{i=1}^n\\sum_{j=1}^n k(x_i,x_j)

        The prototypes are then computed by a greedy search, looking for the next prototype by simply computing which next datapoint reduces the current MMD most. The criticisms are computed by looking for the datapoints that increase the MMD most, which is measured by the witness function $w$

        .. math::
            w(x):=\\frac{1}{n}\sum_{i=1}^n k(x,x_i) - \\frac{1}{m}\sum_{i=1}^m k(x,z_i),

        where $x$ is a candidate point, $z_i$ are the prototypes and $x_i$ are the datapoints. To avoid too close criticisms, a penalty is applied to the witness function.

        Args:
            _prototypes (int): Number of prototypes.
            n_criticisms (int): Number of criticisms.
            metric (str or callable, optional): The metric to use when calculating kernel between instances in a feature array.
                If metric is a string, it must be one of the metrics in sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS.
                If metric is precomputed, X is assumed to be a kernel matrix. Alternatively, if metric is a callable function,
                it is called on each pair of instances (rows) and the resulting value recorded.
                The callable should take two arrays from X as input and return a value indicating the distance between them.
                Currently, sklearn provides the following strings: ‘additive_chi2’, ‘chi2’, ‘linear’, ‘poly’, ‘polynomial’, ‘rbf’,
                                                    ‘laplacian’, ‘sigmoid’, ‘cosine’
            witness_penalty (float): Penalty parameter to include some penalty to avoid too close criticisms.
            max_n_points (int): Maximum number of points to use for the computation of prototypes and criticisms. If the dataset has more points, we randomly choose max_n_points points from the dataset used for the calculations.
            **kwds: optional keyword parameters that are passed directly to the kernel function.
        """
        super().__init__(validation_task=task,
                         n_prototypes=n_prototypes,
                         n_criticisms=n_criticisms,
                         metric=metric,
                         witness_penalty=witness_penalty,
                         max_n_points=max_n_points,
                         pw_kernel_kwargs=pw_kernel_kwargs,
                         **kwargs)

    def compute_prototypes_and_criticisms(self):
        """This methods computes for given datapoints prototypes and criticisms.

        Args:
            task (ValidationTask): The validation task for which prototypes and criticisms are computed.

        Raises:
            Exception: If sklearn is not installed

        Returns:
           dict: Dictionary with list of prototypes (under the key 'prototypes') and list of criticisms (under the key 'criticisms')
        """
        if self.n_prototypes >= self.task.data.shape[0]:
            raise Exception('Number of prototypes must be less then number of datapoints.')
        if self.max_n_points < self.task.data.shape[0]:
            raise Exception \
                ('Number of datapoints must be less then max_n_points. Either increase max_n_points or reduce number of datapoints.')

        X = self.task.data[:, self.task.get_input_cols_ordinal()]
        kernel_matrix = pairwise_kernels(X, metric=self.metric, **self.pw_kernel_kwargs)

        prototypes = self.compute_prototype(kernel_matrix)
        criticisms = self.compute_critcism(prototypes, kernel_matrix)

        logger.info('Computed '+ str(len(prototypes)) + ' prototypes and ' + str(len(criticisms)) + ' criticisms.')
        return {'prototypes': prototypes, 'criticisms': criticisms}

    def compute_prototype(self, kernel_matrix):
        prototypes = []
        n = float(kernel_matrix.shape[0])
        # To compute prototypes we minimize the Maximum Mean Discrepancy (MMD), .. math::
        #     MMD^2 = \frac{1}{m^2}\sum_{i=1}^m \sum_{j=1}^m k(z_i,z_j) - \frac{2}{mn}\sum_{i=1}^m \sum_{j=1}^n k(z_i,x_j) + \frac{1}{n^2}\sum_{i=1}^n\sum_{j=1}^n k(x_i,x_j)
        #     We are doing this using a greedy search, looking for the next prototype by simply computing which next datapoint reduces the current MMD most.
        #     For this we compute simply
        #     the impact on the MMD if a new point x is used as a prototype. The impact is computed by .. math::
        #     \frac{2}{(m+1)^2}(k(x,x) + \sum{i=1}^mk(z_i,x)) -  \frac{2}{(m+1)n}\sum_{j=1}^n k(x,x_j)for i in range(n_prototypes):
        # max_impact  = 1.0e8
        for i in range(self.n_prototypes):
            m = float(len(prototypes))
            max_impact = 1.0e8
            new_prototype = None
            for candidate in range(kernel_matrix.shape[0]):
                if candidate not in prototypes:
                    impact = kernel_matrix[candidate][prototypes].sum() / ((m + 1) ** 2) - kernel_matrix[candidate,
                                                                                           :].sum() / ((m + 1) * n)
                    if impact < max_impact:
                        new_prototype = candidate
                        max_impact = impact
            if new_prototype is None:
                raise Exception('Cannot find a new prototype.')
            prototypes.append(new_prototype)
        return prototypes

    def compute_critcism(self, prototypes, kernel_matrix):
        m = float(len(prototypes))
        n = float(kernel_matrix.shape[0])
        criticisms = []
        for i in range(self.n_criticisms):
            # m = float(len(overall))
            m_criticisms = float(len(criticisms))
            max_witness = -1.0e8
            new_criticism = None
            for candidate in range(kernel_matrix.shape[0]):
                if candidate not in criticisms:
                    witness = kernel_matrix[candidate].sum() / n - kernel_matrix[candidate][prototypes].sum() / m
                    if m_criticisms > 0:
                        regularizer = kernel_matrix[candidate][criticisms].max()
                    else:
                        regularizer = 0
                    cost = abs(witness) - self.witness_penalty * regularizer
                    if cost > max_witness:
                        max_witness = cost
                        new_criticism = candidate
            if new_criticism is None:
                raise Exception('Cannot find a new criticism.')
            criticisms.append(new_criticism)
        return criticisms

if __name__=='__main__':
    from sloth.datasets.test_sets import simple_regression_ordinal_discrete_ohe as test_task
    task = test_task(n_samples=1_000, x=2, f=0)
    test = MMD(task)
    test.compute_prototypes_and_criticisms()

