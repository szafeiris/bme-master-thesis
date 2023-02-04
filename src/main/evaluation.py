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
        self._result = {}
    
    def to_dict(self):
        out = dict()
        out['name'] = self._name
        out['result'] = self._result
        
        return dict
    
    def __str__(self) -> str:
        return json.dumps(self.to_dict())


class EvaluationSession:
    def __init__(self, **kwargs) -> None:
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
        
        self._featureSelectionMethod = kwargs['method'] if 'method' in kwargs else None
        self._featureSelectionSettings = kwargs['methodParams'] if 'methodParams' in kwargs else {}
        self._model = kwargs['model'] if 'model' in kwargs else None
        self._modelParams = kwargs['modelParams'] if 'modelParams' in kwargs else {}
        
        self._isStratifiedCV = kwargs['stratifiedCV'] if 'stratifiedCV' in kwargs else True
        self._folds = kwargs['folds'] if 'folds' in kwargs else 10

        self._cv = StratifiedKFold(n_splits=self._folds) if self._isStratifiedCV else KFold(n_splits=self._folds)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        log.debug('Evaluation session starts.')
        log.debug(f"Execute {mRMR().getName()}")
        self._pipeline.steps =[
                ('fs', mRMR()),
                ('model', SVC(kernel='linear'))
            ]
        
        self._pipeline.fit(X_train, y_train)
        log.debug(self._pipeline.get_params())
        log.debug(self._pipeline.score(X_test, y_test))

class Evaluator:

    def evaluate(self, X, y, **kwargs) -> EvaluationResult:
        patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None

        experimentData = {
            'method': 'mRMR',
            'methodParams': {},
            'model': 'SVM',
            'modelParams': {},

            'stratifiedCV': False
        }

        if patientIds:
            experimentData['patientIds'] = patientIds

        if radiomicFeaturesNames:
            experimentData['radiomicFeaturesNames'] = radiomicFeaturesNames

        with EvaluationSession() as sess:
            evaluationResult = sess.evaluate(X, y, **experimentData)
        
        return evaluationResult
    