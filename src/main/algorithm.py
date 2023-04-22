from sklearn.linear_model import Lasso
from src.main.configurator import log

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import NotFittedError

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import *
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter, spearman_corr, pearson_corr
from ITMO_FS.filters.multivariate import MultivariateFilter

from skrebate import ReliefF, SURF, SURFstar, MultiSURF, MultiSURFstar, TuRF

from boruta.boruta_py import BorutaPy

import numpy as np
import abc

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

    def __getstate__(self):
        return self.get_params().copy()

    def __setstate__(self, params):
        self.set_params(**params)
        return self

## ITMO  filter methods
ITMO_UV_METHODS = {
    'PEARSON': pearson_corr,
    'SPEARMAN': spearman_corr
}

ITMO_MV_METHODS = ['MIFS', 'JMI', 'CMIM', 'MRMR']
# ITMO_MV_METHODS = ['MIM', 'MRMR', 'JMI', 'CIFE', 'MIFS', 'CMIM', 'ICAP', 'DCSF', 'CFR', 'MRI', 'IWFS']

class ItmoFsAlgorithm(FeatureSelectionAlgorithm):
    def __init__(self, methodName=None, nFeatures=100, **kwargs):
        if nFeatures <= 0:
            raise ValueError('`nFeatures` must be greater than zero')
        
        self._cleaning()
        self.nFeatures = nFeatures
        self.method = methodName
        if methodName is None and 'method' in kwargs:
            self.method = kwargs['method']
           
        try:
            self.selectedFeatures = kwargs['selectedFeatures']
        except:
            pass

        try:
            self.method = kwargs['method']
        except:
            pass
        
        try:
            self.nFeatures = kwargs['nFeatures']
        except:
            pass

        try:
            self.selectedFeaturesNames = kwargs['selectedFeaturesNames']
        except:
            pass

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
            'method': self.method,
            'nFeatures': self.nFeatures,
            'selectedFeatures': self.selectedFeatures if hasattr(self, 'selectedFeatures') else [],
            'selectedFeaturesNames': self.selectedFeaturesNames if hasattr(self, 'selectedFeaturesNames') else [],
        }

    def set_params(self, **parameters):
        self._cleaning()
        if 'nFeatures' in parameters:
            self.nFeatures = parameters['nFeatures']
        
        if 'method' in parameters:
            self.method = parameters['method']
        
        return self
    
    def _cleaning(self):
        self._isFitted = False

        if hasattr(self, 'selectedFeatures'):
            del self.selectedFeatures

        if hasattr(self, 'selectedFeaturesNames'):
            del self.selectedFeaturesNames
    
    def __name__(self):
        return f"{self.method}#{self.nFeatures}_"
    
    def __dict__(self):
        return self.get_params()

class MultivariateIFsAlgorithm(ItmoFsAlgorithm):
    def __init__(self, methodName='MRMR', nFeatures=100, beta=None, **kwargs):
        super().__init__(methodName, nFeatures, **kwargs)
        self.beta = beta

        if not methodName in ITMO_MV_METHODS:
            raise KeyError(f'method `{methodName}` is not in ITMO_MV_METHODS')
        
        if self.beta is None:
            self._method = MultivariateFilter(methodName, nFeatures)
        else:
            self._method = MultivariateFilter(methodName, nFeatures, self.beta)

    def set_params(self, **parameters):
        super().set_params(**parameters)

        if not self.method in ITMO_MV_METHODS:
            raise KeyError(f'method `{self.method}` is not in ITMO_MV_METHODS')

        self.beta = parameters['beta'] if 'beta' in parameters else None
        if self.beta is None:
            self._method = MultivariateFilter(self.method, self.nFeatures)
        else:
            self._method = MultivariateFilter(self.method, self.nFeatures, self.beta)
            
        return self

class UnivariateIFsAlgorithm(ItmoFsAlgorithm):
    def __init__(self, methodName=None, nFeatures=100, **kwargs):
        super().__init__(methodName, nFeatures, **kwargs)
        self.__createMethodFromName()

    def set_params(self, **parameters):
        super().set_params(**parameters)
        self.__createMethodFromName()
        return self
    
    def __createMethodFromName(self):
        try:
            self._method = UnivariateFilter(ITMO_UV_METHODS[self.method], select_k_best(self.nFeatures))
        except KeyError:
            raise KeyError(f'method `{self.method}` is not in ITMO_UV_METHODS')

class BorutaFsAlgorithm(FeatureSelectionAlgorithm):
    def __init__(self, estimator=None, n_estimators=1000, perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=None, verbose=0, **kwargs):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.perc = perc
        self.alpha = alpha
        self.two_step = two_step
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose

        try:
            self.selectedFeatures = kwargs['selectedFeatures']
            self.selectedWeakFeatures = kwargs['selectedWeakFeatures']
            self.featureRanking = kwargs['featureRanking']
        except:
            pass
        
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
            'selectedFeatures': np.arange(self.selectedFeatures.shape[0])[self.selectedFeatures] if hasattr(self, 'selectedFeatures') else [],
            'selectedWeakFeatures': np.arange(self.selectedWeakFeatures.shape[0])[self.selectedWeakFeatures] if hasattr(self, 'selectedWeakFeatures') else [],
            'featureRanking': self.featureRanking if hasattr(self, 'featureRanking') else [],
        }

    def set_params(self, **parameters):
        self._cleaning()
        
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        
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


class LassoFsAlgorithm(FeatureSelectionAlgorithm):
    def __init__(self, alpha=0.14, **kwargs) -> None:
        self.__lasso = Lasso(alpha=alpha, fit_intercept=False, copy_X=True, random_state=42)
        self.alpha = alpha

        try:
            self.selectedFeatures = kwargs['selectedFeatures']
        except:
            pass

    def fit(self, X, y=None, **kwargs):
        self.__lasso.fit(X, y)
        importance = abs(self.__lasso.coef_)
        self.selectedFeatures = np.arange(importance.shape[0])[importance > 0]
        return self

    def transform(self, X, y=None, **kwargs):
        X_ret = np.copy(X)
        importance = abs(self.__lasso.coef_)
        X_ret = X_ret[:, importance > 0]
        return X_ret

    def get_params(self, deep=True):
        return {
            'selectedFeatures': self.selectedFeatures  if hasattr(self, 'selectedFeatures') else np.array([]),
            'alpha': self.alpha
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)

        self.__lasso = Lasso(alpha=self.alpha, fit_intercept=False, copy_X=True, random_state=42)
        return self
    
    def __name__(self):
        return f"lasso_"
    
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
                'n_jobs': -1,
                'verbose': True
            }
        },
        'surf': {
            'method': SURF(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1,
                'verbose': True
            }
        },
        'surfstar': {
            'method': SURFstar(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1,
                'verbose': True
            }
        },
        'multisurf': {
            'method': MultiSURF(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1,
                'verbose': True
            }
        },
        'multisurfstar': {
            'method': MultiSURFstar(),
            'methodParams': {
                'n_features_to_select': 2,
                'discrete_threshold': 10,
                'n_jobs': -1,
                'verbose': True
            }
        },
        'boruta': {
            'method': BorutaFsAlgorithm(estimator=RandomForestClassifier(class_weight='balanced', max_depth=5)),
            'methodParams': {
                'estimator': RandomForestClassifier(class_weight='balanced', max_depth=5, random_state=42),
                'n_estimators': 'auto',
                'perc': 50,
                'alpha': 0.05,
                'two_step': True,
                'max_iter': 20,
                'random_state': 42,
                'verbose': 1,
            }
        },
        'lasso': {
            'method': LassoFsAlgorithm(),
            'methodParams': {
            }
        },

    },
    'MODELS': {
        'svm-linear': {
            'model': SVC(),
            'modelParams': {
                'kernel': 'linear'
            }
        },
        'svm-rbf': {
            'model': SVC(),
            'modelParams': {
                'kernel': 'rbf',
            }
        },
        'rf': {
            'model': RandomForestClassifier(),
            'modelParams': {
                'max_depth': 5,
                'class_weight':'balanced',
                'random_state': 42
            }
        },
        'gnb': {
            'model': GaussianNB(),
            'modelParams': {}
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'modelParams': {
                'n_neighbors': 5
            }
        },
        'xgb': {
            'model': xgb.XGBClassifier(),
            'modelParams': {}
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
