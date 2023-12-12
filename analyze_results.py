from BMEMasterThesis.utils import log, Datasets
from BMEMasterThesis.utils.config import PATHS
from BMEMasterThesis.services import PicaiDataService

import pandas as pd

def printMaxScore(scoreName: str, dataset: pd.DataFrame):
    bestScore = dataset.loc[dataset[scoreName].idxmax()]
    print(f"{'='*11} Max {scoreName} {'='*11}\n{bestScore}\n")
    


dataService = PicaiDataService()

originalResultScores = dataService.getScores(Datasets.ORIGINAL)
originalNormalizedResultScores = dataService.getScores(Datasets.ORIGINAL_NORMALIZED)
n4ResultScores = dataService.getScores(Datasets.N4)
n4NormalizedResultScores = dataService.getScores(Datasets.N4_NORMALIZED)
fatNormalizedResultScores = dataService.getScores(Datasets.FAT_NORMALIZED)
muscleNormalizedResultScores = dataService.getScores(Datasets.MUSCLE_NORMALIZED)

resultScores = pd.concat([originalResultScores, originalNormalizedResultScores, n4ResultScores, n4NormalizedResultScores, fatNormalizedResultScores, muscleNormalizedResultScores], ignore_index=True)
resultScores.sort_values(by=['Balanced Accuracy', 'ROC AUC', 'Dataset', 'Feature Selection Method', 'Classification Algorithm'])

