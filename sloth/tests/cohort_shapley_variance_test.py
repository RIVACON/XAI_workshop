from unittest import main, TestCase
import numpy as np
import sloth

class CohortShapleyVarianceTest(TestCase):
    def test_simple_regression(self):
        """Test cohort shapley variance feature importance for simple examples (regression test).
        """
        # simple regression problem with ordinal features
        validation_task = sloth.datasets.test_sets.simple_regression_ordinal(n_samples=1_000, x=0, f=0)
        cohort_shap_variance = sloth.explainers.global_explainers.CohortShapleyVariance(validation_task, similarity_bins=10)
        values, feature_names = cohort_shap_variance.explain()
        self.assertAlmostEqual(values[0], 0.05347124, places=5)
        self.assertAlmostEqual(values[1], 0.05451855, places=5)
        self.assertAlmostEqual(values[2], 0.0041261, places=5)
        # simple regression problem with ordinal features and a third ordinal discrete feature
        validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete(n_samples=1_000, x=0, f=0)
        cohort_shap_variance = sloth.explainers.global_explainers.CohortShapleyVariance(validation_task, similarity_bins=10)
        values, feature_names = cohort_shap_variance.explain()
        self.assertAlmostEqual(values[0], 0.05517951, places=5)
        self.assertAlmostEqual(values[1], 0.05520965, places=5)
        self.assertAlmostEqual(values[2], 0.00104226, places=5)
        # simple regression problem with ordinal features, one ordinal discrete feature and one one hot encoded feature categorical feature
        validation_task = sloth.datasets.test_sets.simple_regression_ordinal_discrete_ohe(n_samples=1_000, x=0, f=0)
        cohort_shap_variance = sloth.explainers.global_explainers.CohortShapleyVariance(validation_task, similarity_bins=10)
        values, feature_names = cohort_shap_variance.explain()
        self.assertAlmostEqual(values[0], 0.05258371, places=5)
        self.assertAlmostEqual(values[1], 0.05312617, places=5)
        self.assertAlmostEqual(values[2], 0.00272663, places=5)
        self.assertAlmostEqual(values[3], 0.00403138, places=5)

        
        
if __name__ == '__main__':
    main()
