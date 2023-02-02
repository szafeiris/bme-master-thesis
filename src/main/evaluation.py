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
    def __init__(self) -> None:
        self.name = ''    
        self.result = {}

class EvaluationContext:
    def __init__(self, steps=[]) -> None:
        self.__pipeline = Pipeline(steps)

    def setSteps(self, steps):
        self.__pipeline.steps = steps
    
    def resetPipeline(self):
        self.__pipeline.steps = []
    
    def getPipelineParams(self):
        return self.__pipeline.get_params(deep=True)
    
    def setPipelineParams(self, **parameters):
        return self.__pipeline.set_params(**parameters)

    def fit(self, X, y, **kwargs):
        return self.__pipeline.fit(X, y, **kwargs)
    
    def predict(self, X, **kwargs):
        return self.__pipeline.predict(X, **kwargs)

    def score(self, X, y, **kwargs):
        return self.__pipeline.score(X, y, **kwargs)


class EvaluationSession:
    def __init__(self, **kwargs) -> None:
        self.__evaluationContext = EvaluationContext()
        self._folds = kwargs['folds'] if 'folds' in kwargs else 10
    
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        pass
    
    def evaluate(self, X, y, **kwargs):
        log.info('Evaluation starts.')
        log.debug(f"Execute {mRMR().getName()}")
        self.__evaluationContext.setSteps([
                ('fs', mRMR()),
                ('model', SVC())
            ])
        self.__evaluationContext.fit(X, y)
        log.debug(self.__evaluationContext.getPipelineParams())
        log.debug(self.__evaluationContext.score(X, y))

class Evaluator:
    def evaluate(self, X, y, **kwargs):
        with EvaluationSession() as sess:
            sess.evaluate(X, y, **kwargs)
    