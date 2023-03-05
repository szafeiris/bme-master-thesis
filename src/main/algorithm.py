from src.main.configurator import configurator as conf

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter, spearman_corr, pearson_corr
from ITMO_FS.filters.multivariate import MultivariateFilter

from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF

from boruta.boruta_py import BorutaPy

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
        self.selectedFeatures = [int(a) for a in self._method.selected_features]
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

class BorutaFsAlgorithm(FeatureSelectionAlgorithm):
    def __init__(self, estimator=None, n_estimators=1000, perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=None, verbose=0):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
        self._method = BorutaPy(self.estimator, self.n_estimators, self.perc, self.alpha, self.two_step, self.max_iter, self.random_state, self.verbose)
        self._cleaning()

    def fit(self, X, y=None, **kwargs):
        self._cleaning()
        self._method.fit(X, y)
        self.selectedFeatures = self._method.support_
        self.selectedWeakFeatures = self._method.support_weak_
        self.featureRanking = self._method.ranking_
        return self

    def transform(self, X, y=None, **kwargs):
        return self._method.transform(X)

    def get_params(self, deep=True):
        return {
            # 'estimator': self.estimator.get_params() if deep else type(self.estimator),
            'n_estimators': self.n_estimators,
            'perc': self.perc,
            'alpha': self.alpha,
            'two_step': self.two_step,
            'max_iter': self.max_iter,
            'selectedFeatures': [int(a) for a in np.arange(self.selectedFeatures.shape[0])[self.selectedFeatures]] if hasattr(self, 'selectedFeatures') else [],
            'selectedWeakFeatures': [int(a) for a in np.arange(self.selectedWeakFeatures.shape[0])[self.selectedWeakFeatures]] if hasattr(self, 'selectedWeakFeatures') else [],
            'featureRanking': [int(a) for a in self.featureRanking] if hasattr(self, 'featureRanking') else [],
        }

    def set_params(self, **parameters):
        self._cleaning()
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        del self._method
        self._method = BorutaPy(self.estimator, self.n_estimators, self.perc, self.alpha, self.two_step, self.max_iter, self.random_state, self.verbose)
        return self
    
    def _cleaning(self):
        if hasattr(self, 'selectedFeatures'):
            del self.selectedFeatures
        
        if hasattr(self, 'selectedWeakFeatures'):
            del self.selectedWeakFeatures
        
        if hasattr(self, 'featureRanking'):
            del self.featureRanking
    
    def __name__(self):
        return f"boruta_"
    
    def __dict__(self):
        return self.get_params()

ALGORITHMS = {
    'FS_METHODS': {
        'relieff': {
            'method': ReliefF(),
            'methodParams': {
                'n_neighbors': 100,
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        'surf': {
            'method': SURF(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        'surfstar': {
            'method': SURFstar(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        'multisurf': {
            'method': MultiSURF(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        'multisurfstar': {
            'method': MultiSURFstar(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        'turf': {
            'method': TuRF(core_algorithm='relieff'),
            'methodParams': {
                'core_algorithm': 'relieff',
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1
            }
        },
        
        'boruta': {
            'method': BorutaFsAlgorithm(estimator=RandomForestClassifier(class_weight='balanced', max_depth=5)),
            'methodParams': {
                'estimator': RandomForestClassifier(class_weight='balanced', max_depth=5),
                'n_estimators': 'auto',
                'perc': 50,
                'alpha': 0.05,
                'two_step': True,
                'max_iter': 20,
                'random_state': 42,
                'verbose': 2,
            }
        },

    },
    'MODELS': {
        'svm': {
            'model': SVC(),
            'modelParams': {
                'kernel': 'linear'
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
