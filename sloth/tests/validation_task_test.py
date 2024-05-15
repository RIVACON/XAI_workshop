from unittest import main, TestCase
import numpy as np
import sloth

class TestValidationTask(TestCase):
    
    def test_data_ohe_as_ordinal(self):
        """Test method get_data_ohe_as_ordinal
        """
        data = np.empty((20,5))
        data[:,0] = 5
        data[:,4] = 8
        #one hot encoded features
        data[:10,1] = 1
        data[10:,1] = 0
        data[:10,2] = 0
        data[10:,2] = 1
        data[:,3] = 0
        task = sloth.ValidationTask([sloth.OrdinalFeatureDescription('f0', 0,),
                                     sloth.OneHotEncodedFeatureDescription('f1', columns=[1,2,3]),
                                    sloth.OrdinalFeatureDescription('f2', 4)],
                                    [sloth.OrdinalFeatureDescription('y', 0)],
                                    data, problemtype='regression', 
                                    predict= lambda x: x[:,0])
        data_, names = task.get_data_ohe_as_ordinal()
        self.assertEqual(data_.shape, (20, 3))
        self.assertEqual(names, ['f0', 'f1', 'f2'])
        self.assertTrue(np.all(data_[:,0]==5))
        self.assertTrue(np.all(data_[:,2]==8))
        self.assertTrue(np.all(data_[:10,1]==0))
        self.assertTrue(np.all(data_[10:,1]==1))
        
if __name__ == '__main__':
    main()