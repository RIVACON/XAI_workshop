from sloth.explainers.utils import UtilsClass
import sloth.datasets as datasets
from sloth.validation_task import ValidationTask
from sloth.features import *

from sloth.explainers.global_explainers.ale_plot import ALE
from sloth.explainers.global_explainers.cohort_shapley_variance import CohortShapleyVariance
from sloth.explainers.global_explainers.m_plots import MarginalPlots
from sloth.explainers.global_explainers.partial_dependence_plot import *

# from sloth.explainers.local_explainers.anchors import *
from sloth.explainers.local_explainers.anchors.anchors import Anchors
from sloth.explainers.local_explainers.anchors.anchors_alibi import AnchorsAlibi
from sloth.explainers.local_explainers.anchors.anchors_exp import AnchorsExp
from sloth.explainers.local_explainers.cohort_shapley import CohortShapleyValues
from sloth.explainers.local_explainers.ice import ICE
from sloth.explainers.local_explainers.lime_file import Lime
from sloth.explainers.local_explainers.shapley import ShapExplainer

from sloth.explainers.datapoints.examples import Examples, MMD

from sloth.explainers.base_explainer import BaseExplainer
#Parameter._name_to_object = {cls.__name__: cls for cls in Parameter.__subclasses__()}
