from typing import Dict, List
from .algorithm import ALGORITHMS, decodeMethod, decodeModel
from .utils.notification import send_to_telegram
from .utils import PATHS, getLogger, CustomJSONEncoder, printTime, getTime

from sklearn.metrics import make_scorer, accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, cohen_kappa_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np
import json

class EvaluationCombination:
    def __init__(self, method: str, model: str) -> None:
        self.method = method
        self.model = model
        
    def __str__(self) -> str:
        return f'{self._method}/{self._model}'
    
    def toDict(self) -> Dict[str, str]:
        return { 'method': self.method, 'model': self.model, }
    
    def toKey(self) -> str:
        return f'{self._method}_{self._model}'
    
    @property
    def method(self):
        """The feature selection method to use in evaluation."""
        return self._method

    @method.setter
    def method(self, newMethod):
        if newMethod in ALGORITHMS['FS_METHODS']:
            self._method = newMethod
        else:
            raise ValueError(f'Feature selection method `{newMethod}` is not implemented yet')
    
    @property
    def model(self):
        """The classification model to use in evaluation."""
        return self._model

    @model.setter
    def model(self, newModel):
        if newModel in ALGORITHMS['MODELS']:
            self._model = newModel
        else:
            raise ValueError(f'Classifier `{newModel}` is not implemented yet')
    
def getAllEvaluationCombinations() -> List[EvaluationCombination]:
    evaluationCombination: List[EvaluationCombination] = []
    for method in list(ALGORITHMS['FS_METHODS']):
        for model in list(ALGORITHMS['MODELS']):
            evaluationCombination.append(EvaluationCombination(method, model))
    return evaluationCombination

class GridSearchNestedCVEvaluation:
    def __init__(self, **kwargs) -> None:
        self.patientIds = kwargs['patientIds'] if 'patientIds' in kwargs else None
        self.radiomicFeaturesNames = kwargs['radiomicFeaturesNames'] if 'radiomicFeaturesNames' in kwargs else None
        self.train_idx = kwargs['train_idx'] if 'train_idx' in kwargs else None
        self.test_idx = kwargs['test_idx'] if 'test_idx' in kwargs else None
        
        featureStart = kwargs['featureStart'] if 'featureStart' in kwargs else 4
        featureStop = kwargs['featureStop'] if 'featureStop' in kwargs else 100
        featureStep = kwargs['featureStep'] if 'featureStep' in kwargs else 5
        
        self.featureNumbers = [int(a) for a in np.arange(start=featureStart, step=featureStep, stop=featureStop)]
        self.thresholds = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95,]
        self._logger = getLogger()
        

    def evaluateAll(self, X, y, dataset='', skipEvaluated: bool = True):
        results = {}
        startTime = getTime()
        self._logger.info(f"Evaluation of `{dataset}` started @ {printTime(startTime)}")
        send_to_telegram(f"Evaluation of `{dataset}` started @ {printTime(startTime)}")
        # for combination in [('mifs', 'rf')]:
        #for combination in [('pearson', 'svm-linear'), ('pearson', 'svm-rbf'), ('pearson', 'rf'), ('pearson', 'knn'), ('pearson', 'gnb'), ('pearson', 'xgb'), ('spearman', 'svm-linear'), ('spearman', 'svm-rbf'), ('spearman', 'rf'), ('spearman', 'knn'), ('spearman', 'gnb'), ('spearman', 'xgb')]:
        # for combination in [('boruta', 'svm-linear'), ('lasso', 'svm-linear'), ('relieff', 'svm-linear'), ('mifs', 'svm-linear')]:
        # for combination in [('pearson', 'svm-linear')]:
        
        for evaluationcombination in getAllEvaluationCombinations():
            try:
                combinationResultsPath = PATHS.getResultsForCombinationDir(dataset, evaluationcombination.method, evaluationcombination.model)
                if combinationResultsPath.exists() and skipEvaluated:
                    self._logger.info(f'Skipped evaluation of {str(evaluationcombination)} for `{dataset}`.')
                    results[f'{evaluationcombination.toKey()}'] = json.load(combinationResultsPath.open())
                    continue
                
                result = self.evaluate(X.copy(), y.copy(), evaluationcombination, dataset)
                results[f'{evaluationcombination.toKey()}'] = result
                json.dump(result, combinationResultsPath.open('w'), cls=CustomJSONEncoder, sort_keys=True, indent=1)
            except Exception as ex:
                self._logger.error(f'Error during evaluation of {evaluationcombination.toKey()}: {str(type(ex).__name__)} {str(ex.args)}')
                self._logger.exception(ex)
                send_to_telegram(f'Error during evaluation of {evaluationcombination.toKey()}: {str(type(ex).__name__)} {str(ex.args)} [{str(ex)}]')
        
        endTime = getTime()
        self._logger.info(f"Evaluation of `{dataset}` ended @ {printTime(endTime)} [Elapsed time: {printTime(endTime - startTime)}]")
        send_to_telegram(f"Evaluation of `{dataset}` ended @ {printTime(endTime)} [Elapsed time: {printTime(endTime - startTime)}]")
        return results

    def evaluate(self, X, y, evaluationCombination: EvaluationCombination, dataset=''):
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
        
        self._logger.info(f'Executing {str(evaluationCombination)} for `{dataset}`.')
        send_to_telegram(f'Executing {str(evaluationCombination)} for `{dataset}`.')
        
        X_train, X_test = X[self.train_idx], X[self.test_idx]
        y_train, y_test = y[self.train_idx], y[self.test_idx]
        data = {}

        if ('urf' in evaluationCombination.method) or ('relieff' == evaluationCombination.method):
            param_grid = {
                'feature_selector__n_features_to_select': self.featureNumbers,
            }
        elif evaluationCombination.method == 'pearson' or evaluationCombination.method == 'spearman' or evaluationCombination.method == 'kendall':
            param_grid = {
                'feature_selector__threshold': self.thresholds,
            }
        else:
            param_grid = {
                'feature_selector__nFeatures': self.featureNumbers,
            }
            
            if 'mifs' in evaluationCombination.method:
                param_grid = {
                    **param_grid,
                    'feature_selector__beta': [1],
                }

        pipeline = Pipeline([
            ('standard_scaler', StandardScaler()),
            ('feature_selector', decodeMethod(evaluationCombination.method)),
            ('classifier', decodeModel(evaluationCombination.model))
        ])
   
        if 'boruta' in evaluationCombination.method: 
            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
            
            data['best_method_params'] = pipeline.steps[1][1].get_params()
            data['test_predictions'] = predictions
            data['classification_report'] = classification_report(y_test, predictions)
            TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
            data['confusion_matrix'] = { 'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP) }
            self._logger.debug(data['classification_report'])
            
            return data.copy()

        if 'lasso' in evaluationCombination.method:
            search = GridSearchCV(pipeline,
                {
                    'feature_selector__alpha': np.arange(0.01, 0.5, 0.01),
                    'feature_selector__random_state': [42],
                    'feature_selector__fit_intercept': [False],
                    'feature_selector__copy_X': [True],
                },
                cv=3,
                scoring=scoring,
                verbose=0,
                n_jobs=-1,
                refit="auc")
                
            search.fit(X_train, y_train)
            best_params = search.best_estimator_.named_steps['feature_selector'].get_params()
            pipeline.steps[1][1].set_params(**best_params)

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)
                        
            data['best_method_params'] = pipeline.steps[1][1].get_params()
            data['test_predictions'] = predictions
            data['classification_report'] = classification_report(y_test, predictions)
            TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
            data['confusion_matrix'] = { 'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP) }
            self._logger.debug(data['classification_report'])
     
            return data.copy()

        grid = GridSearchCV(
                pipeline,
                param_grid,
                scoring=scoring,
                cv=3,
                refit="auc",
                verbose=0,
                n_jobs=-1,
                return_train_score=True
            )
   
        # Fit the model using grid search 
        grid.fit(X_train, y_train) 
 
        bestParams = grid.best_params_            
        data['bestParameters'] = bestParams

        cv_results = grid.cv_results_
        data['cross_validation_results'] = cv_results
        data['best_method_params'] = grid.best_estimator_.get_params()['steps'][1][1].get_params()
            
        grid_predictions = grid.predict(X_test)
        data['test_predictions'] = grid_predictions
        data['classification_report'] = classification_report(y_test, grid_predictions)
        TN, FP, FN, TP = confusion_matrix(y_test, grid_predictions).ravel()
        data['confusion_matrix'] = { 'TN': int(TN), 'FP': int(FP), 'FN': int(FN), 'TP': int(TP) }
        
        if ('urf' in evaluationCombination.method) or ('relieff' == evaluationCombination.method):
            data['selected_features'] = grid.best_estimator_.get_params()['steps'][1][1].top_features_[:grid.best_estimator_.get_params()['steps'][1][1].n_features_to_select]
        
        return data.copy()

# class HybridFsEvaluator:
#     def __init__(self, train_index, test_index) -> None:
#         self.scoring = {
#             'accuracy': accuracy_score,
#             'balanced_accuracy': balanced_accuracy_score,
#             'f1': f1_score,
#             'precision': precision_score,
#             'recall': recall_score,
#             'roc_auc': roc_auc_score,
#             'cohen_kappa': cohen_kappa_score,
#         }
        
#         self.train_index = train_index
#         self.test_index = test_index
    
#     def evaluateOptimals(self, X, y, yStrat, sufix=''):                       
#         method1Names = ['pearson', 'spearman']
#         optimalThresholds = [0.7, 0.7]
        
#         if sufix == '_norm':
#             optimalMethod2 = 'multisurf'
#             optimalMethod2FeatureNo = 13
#             optimalModel = 'svm-linear'
#         elif sufix == '_n4':
#             optimalMethod2 = 'relieff'
#             optimalMethod2FeatureNo = 68
#             optimalModel = 'knn'
#         elif sufix == '_n4_norm':
#             optimalThresholds = [0.85, 0.85]
#             optimalMethod2 = 'mrmr'
#             optimalMethod2FeatureNo = 23
#             optimalModel = 'rf'
#         elif sufix == '_fat':
#             optimalMethod2 = 'mrmr'
#             optimalMethod2FeatureNo = 23
#             optimalModel = 'rf'
#         elif sufix == '_muscle':
#             optimalThresholds = [0.75, 0.75]
#             optimalMethod2 = 'multisurf'
#             optimalMethod2FeatureNo = 18
#             optimalModel = 'svm-rbf'
#         else: # original
#             optimalThresholds = [0.85, 0.85]
#             optimalMethod2 = 'cmim'
#             optimalMethod2FeatureNo = 73
#             optimalModel = 'xgb'
        
#         for combo in zip(method1Names, optimalThresholds):
#             res = self.evaluateSingle(X, y, yStrat, combo[0], combo[1], optimalMethod2, optimalMethod2FeatureNo, optimalModel, sufix='')
#             json.dump(res, open(f'{conf.RESULTS_DIR}/hybrid_optimals_{combo[0]}{sufix}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
#             break
    
#     def evaluateOptimalsGsCV(self, X, y, yStrat, sufix=''):                       
#         method1Names = ['pearson', 'spearman']
#         optimalThresholds = [0.7, 0.7]
#         optimalMethod2FeatureNo = 100
        
#         if sufix == '_norm':
#             optimalMethod2 = 'multisurf'
#             optimalMethod2FeatureNo = 143
#             optimalModel = 'svm-linear'
#         elif sufix == '_n4':
#             optimalMethod2 = 'relieff'
#             optimalMethod2FeatureNo = 97
#             optimalModel = 'knn'
#         elif sufix == '_n4_norm':
#             optimalThresholds = [0.85, 0.85]
#             optimalMethod2 = 'mrmr'
#             optimalMethod2FeatureNo = 252
#             optimalModel = 'rf'
#         elif sufix == '_fat':
#             optimalMethod2 = 'mrmr'
#             optimalModel = 'rf'
#             optimalMethod2FeatureNo = 148
#         elif sufix == '_muscle':
#             optimalThresholds = [0.75, 0.75]
#             optimalMethod2 = 'multisurf'
#             optimalMethod2FeatureNo = 178
#             optimalModel = 'svm-rbf'
#         else: # original
#             optimalThresholds = [0.85, 0.85]
#             optimalMethod2 = 'cmim'
#             optimalMethod2FeatureNo = 175
#             optimalModel = 'xgb'
            
#         # optimalThresholds = [0.95, 0.95]
        
#         for combo in zip(method1Names, optimalThresholds):
#             log.info(f'Started: hybrid_optimals_{combo[0]}{sufix}')
#             send_to_telegram(f'Started: hybrid_optimals_{combo[0]}{sufix}')
            
#             res = self.evaluateSingleWithGSCV(X, y, yStrat, combo[0], combo[1], optimalMethod2, optimalMethod2FeatureNo, optimalModel, sufix='')
#             json.dump(res, open(f'{conf.RESULTS_DIR}/hybrid_optimals_cv_{combo[0]}{sufix}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
            
#             log.info(f'Ended: hybrid_optimals_{combo[0]}{sufix}')
#             send_to_telegram(f'Ended: hybrid_optimals_{combo[0]}{sufix}')
            
#             break
            
#     def evaluateSingle(self, X, y, yStrat, evaluationCombination.method1, featureNumber1, evaluationCombination.method2, featureNumber2, evaluationCombination.model, sufix=''):
#         if sufix != '':
#             sufix = f'{sufix[1:].replace("_", "-")}'
        
#         log.info(f'Executing {evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}.')
#         send_to_telegram(f'Executing {evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}.')
        
#         X_train, X_test = X[self.train_index], X[self.test_index]
#         y_train, y_test = y[self.train_index], y[self.test_index]
                    
#         pipeline = Pipeline([
#             ('standard_scaler', StandardScaler()),
#             ('feature_selector_1', decodeMethod(evaluationCombination.method1)),
#             ('feature_selector_2', decodeMethod(evaluationCombination.method2)),
#             ('classifier', decodeModel(evaluationCombination.model))
#         ])

#         if ('urf' in evaluationCombination.method1) or ('relieff' == evaluationCombination.method1):
#             pipeline.named_steps['feature_selector_1'].set_params(n_features_to_select=featureNumber1)
#         elif evaluationCombination.method1 == 'pearson' or evaluationCombination.method1 == 'spearman' or evaluationCombination.method1 == 'kendall':
#             pipeline.named_steps['feature_selector_1'].set_params(threshold=featureNumber1)
#         else:
#             pipeline.named_steps['feature_selector_1'].set_params(nFeatures=featureNumber1)

#         if ('urf' in evaluationCombination.method2) or ('relieff' == evaluationCombination.method2):
#             pipeline.named_steps['feature_selector_2'].set_params(n_features_to_select=featureNumber2)
#         elif evaluationCombination.method2 == 'pearson' or evaluationCombination.method2 == 'spearman' or evaluationCombination.method1 == 'kendall':
#             pipeline.named_steps['feature_selector_2'].set_params(threshold=featureNumber2)
#         else:
#             pipeline.named_steps['feature_selector_2'].set_params(nFeatures=featureNumber2)
        
#         # Fit the model and get results
#         pipeline.fit(X_train, y_train)
#         predictions = pipeline.predict(X_test)
#         TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
#         data = {
#             'name': f"{evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}",
#             'params': [featureNumber1, featureNumber2],
#             'TN': int(TN),
#             'FP': int(FP),
#             'FN': int(FN), 
#             'TP': int(TP),
#         }
        
#         for score in self.scoring.keys():
#             data = {
#                 **data,
#                 score: float(self.scoring[score](y_test, predictions))
#             }
#             log.debug(f" > Model {data['name']} {score} (test set): {data[score]}.")
        
#         return data.copy()
    
#     def evaluateSingleWithGSCV(self, X, y, yStrat, evaluationCombination.method1, featureNumber1, evaluationCombination.method2, featureStop, evaluationCombination.model, sufix=''):
#         if sufix != '':
#             sufix = f'{sufix[1:].replace("_", "-")}'
        
#         log.info(f'Executing {evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}.')
#         send_to_telegram(f'Executing {evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}.') 
        
#         X_train, X_test = X[self.train_index], X[self.test_index]
#         y_train, y_test = y[self.train_index], y[self.test_index]
                    
#         pipeline = Pipeline([
#             ('standard_scaler', StandardScaler()),
#             ('feature_selector_1', decodeMethod(evaluationCombination.method1)),
#             ('feature_selector_2', decodeMethod(evaluationCombination.method2)),
#             ('classifier', decodeModel(evaluationCombination.model))
#         ])

#         param_grid = {}
#         # if ('urf' in evaluationCombination.method1) or ('relieff' == evaluationCombination.method1):
#         #     pipeline.named_steps['feature_selector_1'].set_params(n_features_to_select=featureNumber1)
#         # elif evaluationCombination.method1 == 'pearson' or evaluationCombination.method1 == 'spearman' or evaluationCombination.method1 == 'kendall':
#         #     pipeline.named_steps['feature_selector_1'].set_params(threshold=featureNumber1)
#         # else:
#         #     pipeline.named_steps['feature_selector_1'].set_params(nFeatures=featureNumber1)
        
#         if ('urf' in evaluationCombination.method1) or ('relieff' == evaluationCombination.method1):
#             param_grid = {
#                 'feature_selector_1__n_features_to_select': [int(a) for a in np.arange(start=3, step=5, stop=featureStop)]
#             }
#         elif evaluationCombination.method1 == 'pearson' or evaluationCombination.method1 == 'spearman' or evaluationCombination.method1 == 'kendall':
#             param_grid = {
#                 'feature_selector_1__threshold': [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
#             }
#         else:
#             param_grid = {
#                 'feature_selector_1__nFeatures': [int(a) for a in np.arange(start=3, step=5, stop=featureStop)]
#             }

#         if ('urf' in evaluationCombination.method2) or ('relieff' == evaluationCombination.method2):
#             param_grid = {
#                 **param_grid,
#                 'feature_selector_2__n_features_to_select': [int(a) for a in np.arange(start=3, step=5, stop=featureStop)]
#             }
#         elif evaluationCombination.method2 == 'boruta' or evaluationCombination.method2 == 'lasso':
#             param_grid = {
#                 **param_grid,
#             }
#         elif evaluationCombination.method2 == 'pearson' or evaluationCombination.method2 == 'spearman' or evaluationCombination.method1 == 'kendall':
#             param_grid = {
#                 **param_grid,
#                 'feature_selector_2__threshold': [int(a) for a in np.arange(start=3, step=5, stop=featureStop)]
#             }
#         else:
#             param_grid = {
#                 **param_grid,
#                 'feature_selector_2__nFeatures': [int(a) for a in np.arange(start=3, step=5, stop=featureStop)]
#             }
        
        
#         grid = GridSearchCV(
#             pipeline,
#             param_grid,
#             scoring = {
#                 'auc': 'roc_auc',
#                 'accuracy': make_scorer(accuracy_score),
#                 'balanced_accuracy': make_scorer(balanced_accuracy_score),
#                 'f1': make_scorer(f1_score),
#                 'precision': make_scorer(precision_score),
#                 'recall': make_scorer(recall_score),
#                 'roc_auc': make_scorer(roc_auc_score),
#                 'cohen_kappa': make_scorer(cohen_kappa_score),
#             },
#             # cv=[(self.train_index, self.test_index)],
#             cv=3,
#             refit="balanced_accuracy",
#             verbose=0,
#             n_jobs=-1,
#             return_train_score=True
#         )
   
#         # Fit the model using grid search 
#         # grid.fit(X_train, y_train)
#         grid.fit(X, y)
#         predictions = grid.predict(X_test)
#         TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
#         data = {
#             'name': f"{evaluationCombination.method1}/{evaluationCombination.method2}/{evaluationCombination.model}{sufix}",
#             'params': [featureNumber1, grid.best_params_[list(grid.best_params_.keys())[0]]],
#             'TN': int(TN),
#             'FP': int(FP),
#             'FN': int(FN), 
#             'TP': int(TP),
#         }
        
#         for score in self.scoring.keys():
#             data = {
#                 **data,
#                 score: float(self.scoring[score](y_test, predictions))
#             }
#             log.debug(f" > Model {data['name']} {score} (test set): {data[score]}.")
        
#         return data.copy()

# class FusionFsEvaluator:
#     def __init__(self) -> None:
#         pass
    
#     def evaluate(self, X, y, yStrat, sufix=''):        
#         if sufix != '':
#             sufix = f'{sufix[1:].replace("_", "-")}'
        
#         log.info(f'Executing fusion{sufix}.')
#         send_to_telegram(f'Executing fusion{sufix}.')
        
#         X_train, X_test = X[train_index], X[test_index]
#         y_train, y_test = y[train_index], y[test_index]
        
        
#         evaluationResults = {}
#         for classifier in ALGORITHMS['MODELS']:
#             log.info(f'Fusion{sufix} - {classifier}.')
#             send_to_telegram(f'Fusion{sufix} - {classifier}.')
            
#             if sufix == '_norm':
#                 estimators = [
#                     ('svm-linear', decodeModel('svm-linear')),
#                     ('gnb', decodeModel('gnb')),
#                 ]
#             elif sufix == '_n4':
#                 estimators = [
#                     ('knn', decodeModel('knn')),
#                     ('xgb', decodeModel('xgb')),
#                 ]
#             elif sufix == '_n4_norm':
#                 estimators = [
#                     ('rf', decodeModel('rf')),
#                     ('xgb', decodeModel('xgb')),
#                 ]
#             elif sufix == '_fat':
#                 estimators = [
#                     ('rf', decodeModel('rf')),
#                     ('svm-rbf', decodeModel('svm-rbf')),
#                 ]
#             elif sufix == '_muscle':
#                 estimators = [
#                     ('svm-rbf', decodeModel('svm-rbf')),
#                     ('rf', decodeModel('rf')),
#                 ]
#             else: # original
#                 estimators = [
#                     ('svm-linear', decodeModel('svm-linear')),
#                     ('xgb', decodeModel('xgb')),
#                 ]
            
#             stackingClassifier = StackingClassifier(
#                 estimators=estimators,
#                 final_estimator=decodeModel(classifier),
#                 cv=5, verbose=0, n_jobs=-1
#             )  

#             method = 'pearson'
#             threshold = 0.70
            
#             pipeline = Pipeline([
#                 ('standard_scaler', StandardScaler()),
#                 ('feature_selector', decodeMethod(method)),
#                 ('classifier', stackingClassifier)
#             ])
#             pipeline.named_steps['feature_selector'].set_params(threshold=threshold)
            
#             pipeline.fit(X_train, y_train)
#             predictions = pipeline.predict(X_test)
#             TN, FP, FN, TP = confusion_matrix(y_test, predictions).ravel()
        
#             data = {
#                 'accuracy_score': float(accuracy_score(y_test, predictions)),
#                 'balanced_accuracy_score': float(balanced_accuracy_score(y_test, predictions)),
#                 'f1_score': float(f1_score(y_test, predictions)),
#                 'precision_score': float(precision_score(y_test, predictions)),
#                 'recall_score': float(recall_score(y_test, predictions)),
#                 'roc_auc_score': float(roc_auc_score(y_test, predictions)),
#                 'cohen_kappa_score': float(cohen_kappa_score(y_test, predictions)),
#                 'TN': int(TN),
#                 'FP': int(FP),
#                 'FN': int(FN), 
#                 'TP': int(TP),
#             }
            
#             log.debug(f'Fusion{sufix} - {classifier}. Balanced accuracy: {data["balanced_accuracy_score"]}.')
            
#             evaluationResults = {
#                 **evaluationResults,
#                 classifier: data.copy()
#             }
        
#         evaluationResults = {
#             'method': method,
#             'threshold': threshold,
#             **evaluationResults
#         }
        
#         log.info(f'End of fusion{sufix}.')
#         send_to_telegram(f'End of fusion{sufix}.')

#         return evaluationResults
        