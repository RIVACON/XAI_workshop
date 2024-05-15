from unittest import main, TestCase
import numpy as np
import sloth

class AnchorTest(TestCase):
    def test_simple_classification(self):
        """Test anchors for simple classification problem with ordinal features.
        """
        validation_task = sloth.datasets.test_sets.simple_classification_ordinal(n_samples=1_000, x=0, f=0)
        anchor_params = sloth.AnchorParameter(threshold=0.8, 
                                      percentiles=[k for k in range(5,95,2)],
                                      alibi_params={'beam_size': 20,})
        anchors = sloth.Anchors(validation_task, anchor_params)
        p = np.where(validation_task.y_pred==True)[0][0]
        self.assertEqual(p, 11)
        a = anchors.compute(x=[p])
        self.assertEqual(a['anchor'][0],'x_1 > 0.55 AND x_2 > 0.52')

if __name__ == '__main__':
    main()