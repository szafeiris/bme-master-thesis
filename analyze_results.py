from pathlib import Path
import pandas as pd

def printMaxScore(scoreName, dataset):
    bestScore = dataset.loc[dataset[scoreName].idxmax()]
    print(f"{'='*11} Max {scoreName} {'='*11}\n{bestScore}\n")
    

n4DatasetPath = Path('n4-norm_results.csv')
n4Dataset = pd.read_csv(n4DatasetPath.absolute())

origDatasetPath = Path('original_results.csv')
origDataset = pd.read_csv(origDatasetPath.absolute())

merged = pd.concat([origDataset, n4Dataset], ignore_index=True)
merged.sort_values(by=['balanced_accuracy_score', 'roc_auc_score', 'datasetName', 'featureSelectionMethodName', 'classifierName'])

printMaxScore('roc_auc_score', merged)


