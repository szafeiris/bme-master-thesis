from sklearn.base import BaseEstimator

class FeatureSelectionAlgorithm(BaseEstimator):
    def __init__(self, estimator=None):
        self.__estimator = estimator
    
    def fit(self, X, y, **kargs):
        self.__estimator.fit(X, y)

    def fit_transform(self, X, y, **kargs):
        self.__estimator.fit_transform(X, y)

    def transform(self, X, y, **kargs):
        self.__estimator.transform(X, y)

    def get_params(self, deep=True):
        return self.__estimator.get_params()

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self.__estimator, parameter, value)
        return self

class MRMR(FeatureSelectionAlgorithm):
    def __init__(self, k, estimator=None):
        super().__init__(estimator)
        self._k = k

    def fit(self, X, y, **kargs):
        self.__estimator.fit(X, y)

    def fit_transform(self, X, y, **kargs):
        self.__estimator.fit_transform(X, y)

    def transform(self, X, y, **kargs):
        self.__estimator.transform(X, y)

    def get_params(self, deep=True):
        return {'_best_features': self._best_features, 'k': self._k}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self