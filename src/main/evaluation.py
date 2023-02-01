from src.main.configurator import configurator as conf
from src.main.data import *
from src.main.algorithm import *

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class EvaluationResult:
    def __init__(self, result=None):
        if result:
            self._result = result
    
    def setResult(self, result):
        self._result = result
        return self
    
    def getResult(self):
        return self._result


class EvaluationSession:
    def __init__(self, **kwargs) -> None:
        self._folds = kwargs['folds'] if 'folds' in kwargs else 10
    
    def evaluate(self):
        log.info('Evaluation starts.')
