import json
from pathlib import Path
from glob import glob as g
import numpy as np
import pandas as pd

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, accuracy_score


def mergeResults(resultsBase='./**'):
    """
    Merge all .json files with results into one

    Args:
        resultsBase (str, optional): The path that all folders containing the performance results, for a specific dataset, in .json files, reside. Defaults to './**'.
    """
    results = {}
    for path in g(resultsBase):
        path = Path(path)
        if not path.is_dir():
            continue
        
        evaluationResults = [evaluationResult for evaluationResult in path.glob("*.json") if 'evaluation' not in evaluationResult.absolute().name]
        print(path.name, len(evaluationResults))
        datasetResults = {}
        for evaluationResult in evaluationResults:
            datasetResults = {
                **datasetResults,
                str(evaluationResult.name.removesuffix('.json')): json.load(open(evaluationResult.absolute(), 'r'))
            }
        
        results = {
            **results,
            path.name: datasetResults,
        }
    
    return results

def extractMetadata(resultName):
    metadata = resultName.split('_')
    
    featureSelectionMethodName = metadata[0]
    classifierName = metadata[1]
    datasetName = 'original'
    isNormalized = 'norm' in resultName
    
    if len(metadata) == 4 or (len(metadata) == 3 and not isNormalized):
        datasetName = metadata[2]
        
    return featureSelectionMethodName, classifierName, datasetName, isNormalized
    
if __name__ == "__main__":
    mergedResultsPath = Path('./data/results/mergedResults.json')
    if mergedResultsPath.exists():
        mergedResults = json.load(open(mergedResultsPath.absolute(), 'r'))
    else:
        mergedResults = mergeResults('./data/results/**')
        json.dump(mergedResults, open(mergedResultsPath.absolute(), 'w'), indent=1)
    
    
    for dataset in mergedResults.keys():
        print(dataset, len(mergedResults[dataset].keys()))
        # print(mergedResults[dataset].keys())
        
        data = []
        for key in mergedResults[dataset].keys():
            featureSelectionMethodName, classifierName, datasetName, isNormalized = extractMetadata(key)
            evaluationResultsDict = {
                'featureSelectionMethodName': featureSelectionMethodName,
                'classifierName': classifierName,
                'datasetName': datasetName,
                'isNormalized': isNormalized,
            }
            # print(key)
            # print(featureSelectionMethodName, classifierName, datasetName, isNormalized)
            
            # print(mergedResults[dataset][key].keys())
            # print(mergedResults[dataset][key]['best_method_params'])
            
            # Get number of selected features
            if 'surf' in key or 'relief' in key:
                selectedFeaturesNo = mergedResults[dataset][key]['best_method_params']['n_features_to_select']
            else:
                selectedFeatures = mergedResults[dataset][key]['best_method_params']['selectedFeatures']
                selectedFeaturesNo = len(selectedFeatures)
            
            # print(f'{featureSelectionMethodName} and {classifierName} selected {selectedFeaturesNo} features.')
            evaluationResultsDict = {
                **evaluationResultsDict,
                'selectedFeaturesNo': selectedFeaturesNo,
            }
            
            yTrue = np.array(mergedResults[dataset][key]['test_labels'])
            yPred = np.array(mergedResults[dataset][key]['test_predictions'])
            
            # print(yPred.shape, yTrue.shape)
            
            TN, FP, FN, TP = confusion_matrix(yTrue, yPred).ravel()
            evaluationResultsDict = {
                **evaluationResultsDict,   
                'accuracy_score': accuracy_score(yTrue, yPred),
                'balanced_accuracy_score': balanced_accuracy_score(yTrue, yPred),
                'f1_score': f1_score(yTrue, yPred),
                'precision_score': precision_score(yTrue, yPred),
                'recall_score': recall_score(yTrue, yPred),
                'roc_auc_score': roc_auc_score(yTrue, yPred),
                'cohen_kappa_score': cohen_kappa_score(yTrue, yPred),
                'TN': TN,
                'FP': FP,
                'FN': FN, 
                'TP': TP,
            }
            data.append(evaluationResultsDict)
            
        df = pd.DataFrame(data)
        df.to_csv(f'data/results/{dataset}_results.csv', index=False)
