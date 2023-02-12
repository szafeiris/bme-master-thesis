from src.main.configurator import configurator as conf
from src.main.data import *
from src.main.algorithm import *

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

import json

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class EvaluationResult:
    def __init__(self) -> None:
        self._name = ''    
        self._result = dict()
    
    def to_dict(self):
        out = dict()
        out['name'] = self._name
        out['result'] = self._result
        
        return dict
    
    def __str__(self) -> str:
        return f'name: {self._name}, results: {self._result}'


class EvaluationSession:
    def __init__(self) -> None:
        self._pipeline = Pipeline([])
    
    def __enter__(self):
        self._pipeline.steps = []

        return self

    def __exit__(self, type, value, traceback):
        del self._pipeline

        return
    
    def evaluate(self, X, y, **kwargs):
        self._patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        self._radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None
        
        # Get feature selection method 
        if not 'method' in kwargs:
            raise ValueError('`method` must be declared')
        self._featureSelectionSettings = kwargs['methodParams'] if 'methodParams' in kwargs else {}       
        self.method = decodeMethod(kwargs['method'], self._featureSelectionSettings)

        del kwargs['method']
        if 'methodParams' in kwargs:
            del kwargs['methodParams']

        # Get model
        if not 'model' in kwargs:
            raise ValueError('`model` must be declared')
        self._modelParams = kwargs['modelParams'] if 'modelParams' in kwargs else {}
        self.model = decodeModel(kwargs['model'], self._modelParams)

        del kwargs['model']
        if 'modelParams' in kwargs:
            del kwargs['modelParams']
        
        self._isStratifiedCV = kwargs['stratifiedCV'] if 'stratifiedCV' in kwargs else True
        self._folds = kwargs['folds'] if 'folds' in kwargs else 10
        self._splitTest = kwargs['splitTest'] if 'splitTest' in kwargs else True
        self._enableCV = kwargs['enableCV'] if 'enableCV' in kwargs else True
        self._randomState = kwargs['randomState'] if 'randomState' in kwargs else 42
        self._testSize = kwargs['testSize'] if 'testSize' in kwargs else 1/3

        log.debug('Evaluation session starts.')
        log.debug(f"Execute {self.method[0].__class__.__name__} and {self.model[0].__class__.__name__}")

        evaluationResults = []
        if self._enableCV:
            self.cv = StratifiedKFold(n_splits=self._folds) if self._isStratifiedCV else KFold(n_splits=self._folds)
            for i, (train_index, test_index) in enumerate(self.cv.split(X, y)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                pipeline = self.__getPipeline()
                pipeline.fit(X_train, y_train)
                
                params = pipeline.get_params()
                trainScore = pipeline.score(X_train, y_train)
                testScore = pipeline.score(X_test, y_test)
                log.debug(f'Fold: {i}')
                log.debug(f'params: {params}')
                log.debug(f'trainScore: {trainScore}')
                log.debug(f'testScore: {testScore}')

                er = EvaluationResult()
                er._name = f"{self.method[0].__class__.__name__}_{self.model[0].__class__.__name__}_cv"
                er._result = {
                    'fold': i,
                    'params': params,
                    'trainScore': trainScore,
                    'testScore': testScore,
                }
                evaluationResults.append(er)

        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self._testSize, random_state=self._randomState)

            pipeline = self.__getPipeline()
            pipeline.fit(X_train, y_train)

            params = pipeline.get_params()
            trainScore = pipeline.score(X_train, y_train)
            testScore = pipeline.score(X_test, y_test)
            log.debug(f'params: {params}')
            log.debug(f'trainScore: {trainScore}')
            log.debug(f'testScore: {testScore}')

            er = EvaluationResult()
            er._name = f"{self.method[0].__class__.__name__}_{self.model[0].__class__.__name__}"
            er._result = {
                'params': params,
                'trainScore': trainScore,
                'testScore': testScore,
            }

            evaluationResults.append(er)
        
        return evaluationResults
    
    def __getPipeline(self):
        self._pipeline.steps = [
            ('fs', self.method[0]),
            ('model', self.model[0])
        ]

        return self._pipeline

class Evaluator:

    def evaluate(self, X, y, **kwargs) -> EvaluationResult:
        patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None

        experimentData = {
            'method': 'mRMR',
            'methodParams': {},
            'model': 'svm',
            'modelParams': {},

            'enableCV': True,
            'stratifiedCV': True,
            'folds': 10,

        }

        if patientIds:
            experimentData['patientIds'] = patientIds

        if radiomicFeaturesNames:
            experimentData['radiomicFeaturesNames'] = radiomicFeaturesNames

        with EvaluationSession() as sess:
            evaluationResult = sess.evaluate(X, y, **experimentData)
        
        return evaluationResult
    