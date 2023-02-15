from src.main.configurator import configurator as conf
from src.main.data import *
from src.main.algorithm import *

from sklearn.metrics import *
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline

import progressbar


## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class EvaluationResult:
    def __init__(self) -> None:
        self.name = ''
        self.method = {}
        self.model = {}
        self.results = {
            'trainIndexes': [],
            'testIndexes': [],
            'trainPredictions': [],
            'testPredictions': [],
        }
        self.pipelineParams = {}
        self.fold = None
    
    def __str__(self) -> str:
        sufix = f'fold: {self.fold}' if self.fold else '' 
        return sufix + f'name: {self.name}, method: {self.method}, model: {self.model}'
    
    def calculateMetrics(self, labels) -> dict:
        metrics = {
            'trainMetrics': {
                'accuracy': accuracy_score(labels[self.results['trainIndexes']], self.results['trainPredictions']),
	            'balanced_accuracy': balanced_accuracy_score(labels[self.results['trainIndexes']], self.results['trainPredictions']),
	            'f1': f1_score(labels[self.results['trainIndexes']], self.results['trainPredictions']),
	            'precision': precision_score(labels[self.results['trainIndexes']], self.results['trainPredictions']), # Sensitivity
	            'recall': recall_score(labels[self.results['trainIndexes']], self.results['trainPredictions']),       # Specificity
	            'roc_auc': roc_auc_score(labels[self.results['trainIndexes']], self.results['trainPredictions']),
            },
            'testMetrics': {
                'accuracy': accuracy_score(labels[self.results['testIndexes']], self.results['testPredictions']),
                'balanced_accuracy': balanced_accuracy_score(labels[self.results['testIndexes']], self.results['testPredictions']),
                'f1': f1_score(labels[self.results['testIndexes']], self.results['testPredictions']),
                'precision': precision_score(labels[self.results['testIndexes']], self.results['testPredictions']), # Sensitivity
                'recall': recall_score(labels[self.results['testIndexes']], self.results['testPredictions']),       # Specificity
                'roc_auc': roc_auc_score(labels[self.results['testIndexes']], self.results['testPredictions']),
                
                'confusion_matrix': confusion_matrix(labels[self.results['testIndexes']], self.results['testPredictions']),
            }
        }

        # FPR, TPR, thresholds = roc_curve(labels[self.results['testIndexes']], self.results['testPredictions'], pos_label=1)
        # metrics['testMetrics']['FPR'] = FPR
        # metrics['testMetrics']['TPR'] = TPR
        # metrics['testMetrics']['thresholds'] = thresholds

        return metrics


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
        self._extraSteps = kwargs['extraSteps'] if 'extraSteps' in kwargs else None
        
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

        self._methodName = self.method[0].__name__()
        self._modelName = self.model[0].__class__.__name__

        log.info('Evaluation session starts.')
        log.info(f"Execute {self._methodName} and {self._modelName}")

        evaluationResults = []
        if self._enableCV:
            self.cv = StratifiedKFold(n_splits=self._folds) if self._isStratifiedCV else KFold(n_splits=self._folds)

            widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='_'),  progressbar.FormatLabel(' Fold %(value)d out of %(max)d - '), progressbar.AdaptiveETA()]
            bar = progressbar.ProgressBar(maxval = self._folds, widgets=widgets).start()

            for i, (train_index, test_index) in enumerate(self.cv.split(X, y)):
                self._train_index, self._test_index = train_index, test_index
                self._X_train, self._X_test = X[train_index], X[test_index]
                self._y_train, self._y_test = y[train_index], y[test_index]

                pipeline = self.__getPipeline()
                if self._radiomicFeaturesNames:
                    pipeline.fit(self._X_train, self._y_train, fs__featureNames=self._radiomicFeaturesNames)
                else:
                    pipeline.fit(self._X_train, self._y_train)
                evaluationResult = self._createEvaluationResult(pipeline, fold=i)
                evaluationResults.append(evaluationResult)
                bar.update(i)
            
            bar.finish()
        else:
            self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(X, y, test_size=self._testSize, random_state=self._randomState)           
            self._train_index, self._test_index = np.where(X == self._X_train), np.where(X == self._X_test)

            pipeline = self.__getPipeline()
            if self._radiomicFeaturesNames:
                pipeline.fit(self._X_train, self._y_train, fs__featureNames=self._radiomicFeaturesNames)
            else:
                pipeline.fit(self._X_train, self._y_train)
            evaluationResult = self._createEvaluationResult(pipeline)
            evaluationResults.append(evaluationResult)

        log.info('Evaluation session teminated.')
        return evaluationResults
    
    def __getPipeline(self):
        pipelineSteps = []

        if self._extraSteps:
            for eStep in self._extraSteps:
                pipelineSteps.append(eStep)
        
        pipelineSteps.append(('fs', self.method[0]))
        pipelineSteps.append(('model', self.model[0]))
        self._pipeline.steps = pipelineSteps
        return self._pipeline
    
    def _createEvaluationResult(self, pipeline: Pipeline, fold=None) -> EvaluationResult:
        evaluationResult = EvaluationResult()
        evaluationResult.name = f'{self._methodName}_{self._modelName}'
        evaluationResult.method = self.method[0].get_params()
        evaluationResult.model = self.model[0].get_params()
        evaluationResult.results = {
            'trainIndexes': np.asarray([i for i in self._train_index]),
            'testIndexes': np.asarray([i for i in self._test_index]),
            'trainPredictions': np.asarray([i for i in pipeline.predict(self._X_train)]),
            'testPredictions': np.asarray([i for i in pipeline.predict(self._X_test)]),
        }

        evaluationResult.pipelineParams = {**pipeline.get_params()}
        evaluationResult.fold = fold

        return evaluationResult

class Evaluator:
    def evaluate(self, X, y, **kwargs) -> EvaluationResult:
        patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None

        experimentData = {
            'method': 'spearman',
            'methodParams': {
                'nFeatures': 1100
            },
            'model': 'svm',
            'modelParams': {
                'kernel': 'rbf'
            },

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
    