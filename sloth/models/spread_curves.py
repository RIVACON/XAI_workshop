import sys
sys.path.insert(0, '../../RiVaPy/')
import datetime as dt
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import  OrdinalEncoder
sys.path.append('..')
from sloth.features import OrdinalFeatureDescription, DiscreteOrdinalFeatureDescription
from sloth.validation_task import ValidationTask

import rivapy
from rivapy.tools.enums import Rating, Sector, Country, Currency, SecuritizationLevel, ESGRating
from rivapy.sample_data.market_data.spread_curves import SpreadCurveSampler

ref_date = dt.datetime(2023,1,1)



n_samples = 1000
categories = {'rating' : [r.value for r in Rating if r != Rating.NONE],
    'sec_lvl': ['SENIOR_SECURED', 'SENIOR_UNSECURED', 'SUBORDINATED'],
     'sector': [r.value for r in Sector if r!= Sector.UNDEFINED],
     'currency' : [Currency.USD.value, Currency.EUR.value, Currency.GBP.value, Currency.JPY.value],
     'country': [x.value for x in Country],
    'esg_rating': [r.value for r in ESGRating],
    }

_categories = list(categories.keys())
spread_curve_categories = [categories[k] for k in _categories]
oe_prep = OrdinalEncoder(categories = spread_curve_categories)


class OrdinalEncodedModel:
    @staticmethod
    def sample_data(n_samples=1000, seed = 42):
        np.random.seed(seed)
        result = np.array([np.random.choice(categories[v], size=n_samples) for v in _categories])
        #result['T'] = np.random.uniform(low=0.5, high=15.0, size=n_samples)
        #result.transpose().shape
        spread_curve_data = pd.DataFrame(np.transpose(result), columns = _categories)
        oe_prep.fit(spread_curve_data)
        X_oe = np.concatenate([oe_prep.transform(spread_curve_data), 
                        np.random.randint(30,15*365,size=(spread_curve_data.shape[0],1))], axis=1)
        return X_oe
    
    def __init__(self, seed = 42, 
                 n_sector_factors=4, 
                 n_country_factors=3):
        np.random.seed(seed)
        sector_weights = np.random.uniform(0.0, 5.0, size=(len(Sector), n_sector_factors))
        sector_weights = sector_weights**5
        row_sums = sector_weights.sum(axis=1)
        sector_weights = sector_weights / row_sums[:, np.newaxis]

        country_weights = np.random.uniform(0.0, 5.0, size=(len(Country), n_country_factors))
        row_sums = country_weights.sum(axis=1)
        country_weights = country_weights / row_sums[:, np.newaxis]
        self.curve_sampler = SpreadCurveSampler(sector_weights = sector_weights, 
                                   country_weights = country_weights)
        self.curve_sampler.sample(ref_date)

    def predict(self, x):
        cache = dict()
        x_ = oe_prep.inverse_transform(x[:,:-1])
        y = np.empty(x.shape[0])
        curve = None
        for i in range(x.shape[0]):
            curve = self.curve_sampler._get_curve(rating = x_[i,0], 
                                            country=x_[i,4],
                                            esg_rating=x_[i,5],
                                            sector=x_[i,2], 
                                            currency=x_[i,3],
                                            securitization_level=x_[i,1])
            y[i] = curve.value_rate(ref_date, ref_date+dt.timedelta(days= x[i,-1]))
        return 100.0*y
        
_model =  OrdinalEncodedModel()

def get_validation_task(n_samples=1000):
    input_features = [DiscreteOrdinalFeatureDescription('rating', column=0),
                      DiscreteOrdinalFeatureDescription('country',  column=4),
                      DiscreteOrdinalFeatureDescription('ESG rating',  column=5),
                      DiscreteOrdinalFeatureDescription('sector',  column=2),
                      DiscreteOrdinalFeatureDescription('currency', column=3),
                      DiscreteOrdinalFeatureDescription('securitization level', column=1),
                      DiscreteOrdinalFeatureDescription('days to maturity', column=6),]
    output_features = [OrdinalFeatureDescription('spread', column=0)]
    x = OrdinalEncodedModel.sample_data(n_samples)
    return ValidationTask(input_features, output_features, x, _model.predict)
