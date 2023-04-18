from sklearn.preprocessing import StandardScaler
from .configurator import configurator as conf, log
from .data import *
from .algorithm import *
from .notification import send_to_telegram

from sklearn.metrics import *
from sklearn.model_selection import *
from sklearn.pipeline import Pipeline

import numpy as np
import progressbar
import json

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
        sufix = f'fold: {self.fold}, ' if self.fold else '' 
        return sufix + f'name: {self.name}, method: {self.method}, model: {self.model}, results: {self.results}, metrics: {self.metrics}'
    
    def dict(self):
        dictionary = {
            "name": self.name,
            "method": self.method,
            "model": self.model,
            "results": {
                'trainIndexes': [int(a) for a in self.results['trainIndexes']],
                'testIndexes': [int(a) for a in self.results['testIndexes']],
                'trainPredictions': [int(a) for a in self.results['trainPredictions']],
                'testPredictions': [int(a) for a in self.results['testPredictions']],
            },
            "metrics": self.metrics,
        }

        if not self.fold is None:
            return {
                f"{self.fold}": dictionary
            }

        return dictionary
    
    def calculateMetrics(self, labels) -> dict:
        metrics = {
            'trainMetrics': {
                'accuracy': float(accuracy_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])),
	            'balanced_accuracy': float(balanced_accuracy_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])),
	            'f1': float(f1_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])),
	            'precision': float(precision_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])), # Sensitivity
	            'recall': float(recall_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])),       # Specificity
	            'roc_auc': float(roc_auc_score(labels[self.results['trainIndexes']], self.results['trainPredictions'])),
                'confusion_matrix': [int(a) for a in confusion_matrix(labels[self.results['trainIndexes']], self.results['trainPredictions']).ravel()],
            },
            'testMetrics': {
                'accuracy': float(accuracy_score(labels[self.results['testIndexes']], self.results['testPredictions'])),
                'balanced_accuracy': float(balanced_accuracy_score(labels[self.results['testIndexes']], self.results['testPredictions'])),
                'f1': float(f1_score(labels[self.results['testIndexes']], self.results['testPredictions'])),
                'precision': float(precision_score(labels[self.results['testIndexes']], self.results['testPredictions'])), # Sensitivity
                'recall': float(recall_score(labels[self.results['testIndexes']], self.results['testPredictions'])),       # Specificity
                'roc_auc': float(roc_auc_score(labels[self.results['testIndexes']], self.results['testPredictions'])),
                'confusion_matrix': [int(a) for a in confusion_matrix(labels[self.results['testIndexes']], self.results['testPredictions']).ravel()],
            }
        }

        # FPR, TPR, thresholds = roc_curve(labels[self.results['testIndexes']], self.results['testPredictions'], pos_label=1)
        # metrics['testMetrics']['FPR'] = FPR
        # metrics['testMetrics']['TPR'] = TPR
        # metrics['testMetrics']['thresholds'] = thresholds
        
        self.metrics = metrics
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
        
        self._folds = kwargs['crossValidationNFolds'] if 'crossValidationNFolds' in kwargs else 10
        self._crossValidation = kwargs['crossValidation'] if 'crossValidation' in kwargs else None
        self._crossValidationShuffle = kwargs['crossValidationShuffle'] if 'crossValidationShuffle' in kwargs else False
        self._randomState = kwargs['randomState'] if 'randomState' in kwargs else 42
        self._testSize = kwargs['testSize'] if 'testSize' in kwargs else 1/3

        self._methodName = self.method[0].__name__() if hasattr(self.method[0], '__name__') else self.method[0].__class__.__name__
        self._modelName = self.model[0].__class__.__name__
        if self._modelName == 'SVC':
            kernel = self.model[0].get_params()['kernel']
            degree = self.model[0].get_params()['degree'] if kernel == 'poly' else ''
            self._modelName = f"{self._modelName}#{kernel}#{degree}"

        log.info('Evaluation session starts.')
        log.info(f"Execute {self._methodName} and {self._modelName}")

        evaluationResults = []
        if not self._crossValidation is None:
            self._crossValidation.n_splits = self._folds
            self._crossValidation.random_state = self._randomState
            self._crossValidation.shuffle = self._crossValidationShuffle

            widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='_'),  progressbar.FormatLabel(' Fold %(value)d out of %(max)d - '), progressbar.AdaptiveETA()]
            bar = progressbar.ProgressBar(maxval = self._folds, widgets=widgets).start()

            for i, (train_index, test_index) in enumerate(self._crossValidation.split(X, y)):
                self._train_index, self._test_index = train_index, test_index
                self._X_train, self._X_test = X[train_index], X[test_index]
                self._y_train, self._y_test = y[train_index], y[test_index]

                evaluationResults.append(
                    self.__executePipeline(fold = i + 1)
                )

                bar.update(i)
            
            bar.finish()
        else:
            shuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=self._testSize, random_state=self._randomState)
            shuffleSplit.get_n_splits(X, y)
            self._train_index, self._test_index = next(shuffleSplit.split(X, y)) 
            self._X_train, self._X_test = X[self._train_index], X[self._test_index] 
            self._y_train, self._y_test = y[self._train_index], y[self._test_index]

            evaluationResults.append(
                self.__executePipeline()
            )

        log.info('Evaluation session teminated.')
        return evaluationResults
    
    def __executePipeline(self, fold: int = None) -> EvaluationResult:
        pipeline = self.__getPipeline()
        if isinstance(self.method[0], TuRF):
            pipeline.fit(self._X_train, self._y_train, fs__headers=self._radiomicFeaturesNames)
        elif self._radiomicFeaturesNames and not isinstance(self.method[0], ReliefF):
            pipeline.fit(self._X_train, self._y_train, fs__featureNames=self._radiomicFeaturesNames)
        else:
            pipeline.fit(self._X_train, self._y_train)
        return self._createEvaluationResult(pipeline, fold=fold) if fold else self._createEvaluationResult(pipeline)
        
    
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
        saveResults = kwargs['saveResults'] if 'saveResults' in kwargs else False
        saveSufix = kwargs['saveSufix'] if 'saveSufix' in kwargs else ''
        
        if not 'experimentData' in kwargs:
            raise AttributeError('`experimentData` is missing')
        experimentData = kwargs['experimentData']

        if patientIds:
            experimentData['patientIds'] = patientIds

        if radiomicFeaturesNames:
            experimentData['radiomicFeaturesNames'] = radiomicFeaturesNames

        with EvaluationSession() as sess:
            evaluationResults = sess.evaluate(X, y, **experimentData)

            if saveResults:
                evaluationResultsDictionary = []
                for evaluationResult in evaluationResults:
                    evaluationResult.calculateMetrics(y)
                    evaluationResultDictionary = evaluationResult.dict()
                    evaluationResultsDictionary.append(evaluationResultDictionary)

                filename = f"{evaluationResultsDictionary[0]['1']['name']}_CV" if len(evaluationResultsDictionary) > 1 else f"{evaluationResultsDictionary[0]['name']}"
                filename = f"{saveSufix}{filename}.json"
                json.dump(
                    evaluationResultsDictionary,
                    open(os.path.join(conf.RESULTS_DIR, filename), 'w'),
                    indent = '\t',
                    sort_keys = True
                )

        return evaluationResults

class CrossCombinationEvaluator(Evaluator):
    def evaluate(self, X, y, **kwargs) -> EvaluationResult:
        featureNumbers = kwargs['featuresNo'] if 'featuresNo' in kwargs else [int(X.shape[1]/2)]

        for method, model in self.getCrossCombinations():
            log.debug((method, model))
            experimentData = {
                'method': method,
                'model': model,            
                # 'crossValidation': StratifiedKFold(),
                'crossValidationNFolds': 10,
                'testSize': 1/3,
                # 'testSize': 0.35,
            }

            if 'boruta' == method:
                args = { 
                    **kwargs,
                    'experimentData': experimentData,
                    'saveSufix': 'cross_combination_'
                }
                super().evaluate(X, y, **args)
                continue
            
            for featureNo in featureNumbers:
                if 'urf' in method or 'relieff' in method:
                    experimentData['methodParams'] = {
                        'n_features_to_select': featureNo
                    }
                else:
                    experimentData['methodParams'] = {
                        'nFeatures': featureNo
                    }
                args = { 
                    **kwargs,
                    'experimentData': experimentData,
                    'saveSufix': f'cross_combination_feature_{featureNo}_'
                }
                log.debug(f'for {featureNo} features...')
                send_to_telegram(f'Running {method}/{model}/{featureNo}')
                try:
                    super().evaluate(X, y, **args)
                except Exception as e:
                    log.exception(e)
                    send_to_telegram('Exception occured: ' + str(e))

    
    def getCrossCombinations(self) -> list:
        combinations = []
        for method in list(ALGORITHMS['FS_METHODS'].keys()):
            for model in list(ALGORITHMS['MODELS'].keys()):
                combinations.append((method, model))
        return combinations


class GridSearchNestedCVEvaluation:
    def __init__(self, **kwargs) -> None:
        self.patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        self.radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None
        
        featureStart = kwargs['featureStart'] if 'featureStart' in kwargs else None
        featureStop = kwargs['featureStop'] if 'featureStop' in kwargs else None
        featureStep = kwargs['featureStep'] if 'featureStep' in kwargs else None
        self.featureNumbers = [int(a) for a in np.arange(start=featureStart, step=featureStep, stop=featureStop)]
        # if self.featureNumbers[-1] != featureStop:
        #     self.featureNumbers.append(featureStop)

        self.combinations = []
        for method in list(ALGORITHMS['FS_METHODS'].keys()):
            for model in list(ALGORITHMS['MODELS'].keys()):
                self.combinations.append((method, model))

    def evaluateAll(self, X, y, yStrat):
        data = {}
        
        # for combination in [('lasso', 'svm-linear')]:
        for combination in [('mrmr', 'svm-linear')]:
        # for combination in [('mim', 'svm-linear')]:
        # for combination in [('pearson', 'svm-linear'), ('spearman', 'svm-linear')]:
        # for combination in self.combinations:
            try:
                results = self.evaluateSingle(X.copy(), y, yStrat, combination[0], combination[1])
                data[f'{combination[0]}_{combination[1]}'] = results
                json.dump(results, open(f'{conf.RESULTS_DIR}/{combination[0]}_{combination[1]}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
            except Exception as ex:
                log.exception(ex)
                log.error(f'Error during evaluation of {combination[0]}_{combination[1]}: {str(type(ex).__name__)} {str(ex.args)}')

        return data

    def evaluateSingle(self, X, y, yStrat, methodName, modelName):
        scoring = {
            'auc': 'roc_auc',
            'accuracy': make_scorer(accuracy_score),
            'balanced_accuracy': make_scorer(balanced_accuracy_score),
            'f1': make_scorer(f1_score),
            'precision': make_scorer(precision_score),
            'recall': make_scorer(recall_score),
            'roc_auc': make_scorer(roc_auc_score),
            'cohen_kappa': make_scorer(cohen_kappa_score),
        }
        results = {}
        
        log.info(f'Executing {methodName}/{modelName}.')
        send_to_telegram(f'Executing {methodName}/{modelName}.')

        # Outter loop  (5-fold stratified cross validation)
        stratifiedKFoldCV = StratifiedKFold(n_splits=5, shuffle=False)
        for i, (train_index, test_index) in enumerate(stratifiedKFoldCV.split(X, yStrat)):
            X_train_outter, X_test_outter = X[train_index], X[test_index]
            y_train_outter, y_test_outter = y[train_index], y[test_index]
            data = {}

            log.info(f'Fold: {i + 1}')
            # Evaluate ITMO methods with multiple features
            if ('urf' in methodName) or ('relieff' == methodName):
                param_grid = {
                    'feature_selector__n_features_to_select': self.featureNumbers,
                }
            else:
                param_grid = {
                    'feature_selector__nFeatures': self.featureNumbers,
                }

            pipeline = Pipeline([
                ('standard_scaler', StandardScaler()),
                ('feature_selector', decodeMethod(methodName)[0]),
                ('classifier', decodeModel(modelName)[0])
            ])
   
            if 'boruta' in methodName: 
                pipeline.fit(X_train_outter, y_train_outter)
                data['best_method_params'] = pipeline.steps[1][1].get_params()

                predictions = pipeline.predict(X_test_outter)

                data['test_predictions'] = predictions
                data['test_labels'] = y_test_outter
                data['test_labels_strat'] = yStrat[test_index]
                data['classification_report'] = classification_report(y_test_outter, predictions)
                log.debug(data['classification_report'])
     
                results[f'{i + 1}'] = data.copy()

                continue

            if 'lasso' in methodName:
                search = GridSearchCV(pipeline,
                                      {
                                        'feature_selector__alpha': np.arange(0.01, 1, 0.01),
                                        'feature_selector__random_state': [42],
                                        'feature_selector__fit_intercept': [False],
                                        'feature_selector__copy_X': [True],
                                      },
                                      cv=3,
                                      scoring="roc_auc",
                                      verbose=1,
                                      n_jobs=-1,
                                      refit=True)
                
                search.fit(X_train_outter, y_train_outter)
                best_params = search.best_estimator_.named_steps['feature_selector'].get_params()
                pipeline.steps[1][1].set_params(**best_params)

                pipeline.fit(X_train_outter, y_train_outter)
                data['best_method_params'] = pipeline.steps[1][1].get_params()

                predictions = pipeline.predict(X_test_outter)

                data['test_predictions'] = predictions
                data['test_labels'] = y_test_outter
                data['test_labels_strat'] = yStrat[test_index]
                data['classification_report'] = classification_report(y_test_outter, predictions)
                log.debug(data['classification_report'])
     
                results[f'{i + 1}'] = data.copy()

                continue

            grid = GridSearchCV(
                    pipeline,
                    param_grid,
                    scoring=scoring,
                    cv=5,
                    refit="auc",
                    verbose=0,
                    n_jobs=-1,
                    return_train_score=True
                )
   
            # Fit the model using grid search 
            grid.fit(X_train_outter, y_train_outter) 
 
            bestParams = grid.best_params_            
            log.debug(bestParams)
            data['bestParameters'] = bestParams

            cv_results = grid.cv_results_
            data['cross_validation_results'] = cv_results
            data['best_method_params'] = grid.best_estimator_.get_params()['steps'][1][1].get_params()
            
            grid_predictions = grid.predict(X_test_outter)
            data['test_predictions'] = grid_predictions
            data['test_labels'] = y_test_outter
            data['test_labels_strat'] = yStrat[test_index]
            data['classification_report'] = classification_report(y_test_outter, grid_predictions)
            log.debug(data['classification_report'])

            results[f'{i + 1}'] = data.copy()
        
        return results




