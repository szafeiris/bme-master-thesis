from src.main.configurator import configurator as conf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from ITMO_FS.filters.multivariate import MRMR

import numpy as np

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class FeatureSelectionAlgorithm(BaseEstimator, TransformerMixin):
    def __init__(self, estimator=None):
        self.__estimator = estimator
    
    def fit(self, X, y=None, **kwargs):
        self.__estimator.fit(X, y)

    def transform(self, X, y=None, **kwargs):
        self.__estimator.transform(X, y)

    def get_params(self, deep=True):
        return self.__estimator.get_params()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.__estimator, parameter, value)
        return self

class mRMR(FeatureSelectionAlgorithm):
    def __init__(self, k=10, estimator=None):
        super().__init__(estimator)
        self.k = k
        self._isFitted_ = False

    def fit(self, X: np.array, y: np.array, **kwargs):
        features = [i for i in range(X.shape[1])]
        selectedFeatures = []
        freeFeatures = [i for i in features if i not in selectedFeatures]

        while len(selectedFeatures) != self.k:
            mrmr = MRMR(np.asarray(selectedFeatures), np.asarray(freeFeatures), X, y)
            mxPos = np.argmax(mrmr)
            log.debug(f"mrmr {mrmr}, mxPos {mxPos}")

            selectedFeatures.append(freeFeatures[mxPos])
            freeFeatures = [i for i in features if i not in selectedFeatures]

        featureNames = None
        self._selectedFeaturesNames = None
        if 'featureNames' in kwargs:
           featureNames = kwargs['featureNames']

        if featureNames is not None:
            selectedFeaturesNames = [featureNames[s] for s in selectedFeatures]
            self._selectedFeaturesNames = selectedFeaturesNames
        
        self._selectedFeatures = selectedFeatures
        self._isFitted_ = True        
    
        return self

    def transform(self, X, y=None, **kwargs):
        if not self._isFitted_:
            raise NotFittedError()
        
        return X[:, self._selectedFeatures]

    def get_params(self, deep=True):
        return {
            'selected_features_': self._selectedFeatures,
            'k': self.k,
            'selected_feature_names_': self._selectedFeaturesNames if self._selectedFeaturesNames else '',
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self