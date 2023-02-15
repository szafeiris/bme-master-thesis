from src.main.configurator import configurator as conf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from sklearn.svm import SVC

from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter, spearman_corr, pearson_corr
from ITMO_FS.filters.multivariate import MultivariateFilter

import numpy as np
import abc

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class FeatureSelectionAlgorithm(BaseEstimator, TransformerMixin):
    @abc.abstractclassmethod    
    def fit(self, X, y=None, **kwargs):
        pass

    @abc.abstractclassmethod
    def transform(self, X, y=None, **kwargs):
        pass

    def get_params(self, deep=True):
        return self.__dict__

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

## ITMO  filter methods
ITMO_UV_METHODS = {
    'PEARSON': pearson_corr,
    'SPEARMAN': spearman_corr
}

ITMO_MV_METHODS = ['MIM', 'MRMR', 'JMI', 'CIFE', 'MIFS', 'CMIM', 'ICAP', 'DCSF', 'CFR', 'MRI', 'IWFS']

class ItmoFsAlgorithm(FeatureSelectionAlgorithm):
    def __init__(self, methodName=None, nFeatures=100):
        if nFeatures <= 0:
            raise ValueError('`nFeatures` must be greater than zero')
        
        self._cleaning()
        self._methodName = methodName
        self.nFeatures = nFeatures

    def fit(self, X, y=None, **kwargs):
        self._cleaning()

        self._method.fit(X, y)
        self.selectedFeatures = self._method.selected_features
        self._isFitted = True

        if 'featureNames' in kwargs:
            self.selectedFeaturesNames = [kwargs['featureNames'][i] for i in self.selectedFeatures]
        return self

    def transform(self, X, y=None, **kwargs):
        if not self._isFitted:
            raise NotFittedError('`algorithm` must be fitted first')

        return self._method.transform(X)

    def get_params(self, deep=True):
        return {
            'method': self._methodName,
            'nFeatures': self.nFeatures,
            'selectedFeatures': self.selectedFeatures if hasattr(self, 'selectedFeatures') else [],
            'selectedFeaturesNames': self.selectedFeaturesNames if hasattr(self, 'selectedFeaturesNames') else [],
        }

    def set_params(self, **parameters):
        self._cleaning()
        if 'nFeatures' in parameters:
            self.nFeatures = parameters['nFeatures']
        
        if 'methodName' in parameters:
            self._methodName = parameters['methodName']
        
        del self._method
        return self
    
    def _cleaning(self):
        self._isFitted = False

        if hasattr(self, 'selectedFeatures'):
            del self.selectedFeatures

        if hasattr(self, 'selectedFeaturesNames'):
            del self.selectedFeaturesNames
    
    def __name__(self):
        return f"{self._methodName}#{self.nFeatures}_"
    
    def __dict__(self):
        return self.get_params()

class MultivariateIFsAlgorithm(ItmoFsAlgorithm):
    def __init__(self, methodName=None, nFeatures=100):
        super().__init__(methodName, nFeatures)

        if not methodName in ITMO_MV_METHODS:
            raise KeyError(f'method `{methodName}` is not in ITMO_MV_METHODS')
        
        self._method = MultivariateFilter(methodName, nFeatures)

    def set_params(self, **parameters):
        super().set_params(**parameters)

        if not self._methodName in ITMO_MV_METHODS:
            raise KeyError(f'method `{self._methodName}` is not in ITMO_MV_METHODS')

        self._method = MultivariateFilter(self._methodName, self.nFeatures)
        return self

class UnivariateIFsAlgorithm(ItmoFsAlgorithm):
    def __init__(self, methodName=None, nFeatures=100):
        super().__init__(methodName, nFeatures)
        
        self._methodName, self.nFeatures = methodName, nFeatures 
        self.__createMethodFromName()

    def set_params(self, **parameters):
        super().set_params(**parameters)
        self.__createMethodFromName()
        return self
    
    def __createMethodFromName(self):
        try:
            self._method = UnivariateFilter(ITMO_UV_METHODS[self._methodName], select_k_best(self.nFeatures))
        except KeyError:
            raise KeyError(f'method `{self._methodName}` is not in ITMO_UV_METHODS')


ALGORITHMS = {
    'FS_METHODS': {

    },
    'MODELS': {
        'svm': {
            'model': SVC(),
            'modelParams': {
                'kernel': 'rbf'
            }
        },

    }
}

## Fill ALGORITHMS dictionary
# ITMO Univariate methods
for method in ITMO_UV_METHODS:
    ALGORITHMS['FS_METHODS'][method.lower()] = {
        'method': UnivariateIFsAlgorithm(method),
        'methodParams': {
            'nFeatures': 50
        }
    }

# ITMO Multivariate methods
for method in ITMO_MV_METHODS:
    ALGORITHMS['FS_METHODS'][method.lower()] = {
        'method': MultivariateIFsAlgorithm(method),
        'methodParams': {
            'nFeatures': 50
        }
    }


def decodeMethod(methodName: str, params={}, applyParams=True):
    if not methodName in ALGORITHMS['FS_METHODS'].keys():
        raise ValueError(f'{methodName} does not exist in ALGORITHMS')
    
    methodParams = {
        **ALGORITHMS['FS_METHODS'][methodName]['methodParams'],
        **params
    }
    method = ALGORITHMS['FS_METHODS'][methodName]['method']

    if applyParams:
        method.set_params(**methodParams)
    
    return (method, methodParams)


def decodeModel(modelName: str, params={}, applyParams=True):
    if not modelName in ALGORITHMS['MODELS']:
        raise ValueError(f'{modelName} does not exist in ALGORITHMS')
    
    modelParams = {
        **ALGORITHMS['MODELS'][modelName]['modelParams'],
        **params
    }
    model = ALGORITHMS['MODELS'][modelName]['model']
    
    if applyParams:
        model.set_params(**modelParams)

    return (model, modelParams)
