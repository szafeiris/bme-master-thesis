from BMEMasterThesis.utils import log, Datasets
from BMEMasterThesis.utils.config import PATHS
from BMEMasterThesis.services import PicaiDataService

import matplotlib.pyplot as plt
from pathlib import Path
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

print(resultScores.groupby(by='Dataset')['Dataset'].aggregate(['count']))

def generateBoxplotForScore(scoreName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
    resultScoresBoxplot = resultScore.boxplot(
        column=[
            scoreName,
        ],
        by= 'Dataset',
        rot=0, figsize=(10, 10))
    resultScoresBoxplot.plot()
    plt.title(f'Distribution of {scoreName}')
    plt.suptitle('')
    plt.xlabel('Dataset')
    plt.ylabel(scoreName)
    plt.grid(axis='x')
    
    if not (path is None):
        if isinstance(path, str):
            path = Path(path)
        plt.savefig(path)
    
    if showPlot:
        plt.show()

# generateBoxplotForScore('Balanced Accuracy', resultScores, True, path='Balanced Accuracy Boxplot.jpg')
# generateBoxplotForScore('ROC AUC', resultScores, path='ROC AUC Boxplot.jpg')
# generateBoxplotForScore('Cohen Kappa', resultScores, path='Cohen Kappa Boxplot.jpg')


# printMaxScore('Balanced Accuracy', originalResultScores)
# printMaxScore('Balanced Accuracy', originalNormalizedResultScores)
# printMaxScore('Balanced Accuracy', n4ResultScores)
# printMaxScore('Balanced Accuracy', n4NormalizedResultScores)
# printMaxScore('Balanced Accuracy', fatNormalizedResultScores)
# printMaxScore('Balanced Accuracy', muscleNormalizedResultScores)


# resultScores.groupby(['Dataset','Classification Algorithm'])['Balanced Accuracy'].max().unstack().plot.bar(rot= 0, figsize= (10, 5))
# plt.xlabel('Dataset')
# plt.ylabel('Balanced Accuracy')
# plt.title('Highest Balanced Accuracy achieved per Classification Algorithm')
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
# plt.tight_layout()
# # plt.savefig('Highest Balanced Accuracy achieved per Classification Algorithm.jpg')
# plt.show()


# print(resultScores.groupby(['Dataset','Classification Algorithm'])['Balanced Accuracy'].max())


# resultScores.groupby(['Dataset','Feature Selection Method'])['Balanced Accuracy'].max().unstack().plot.bar(rot= 0, figsize= (15, 7))
# plt.xlabel('Dataset')
# plt.ylabel('Balanced Accuracy')
# plt.title('Highest Balanced Accuracy Achieved per Feature Selection Method')
# plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
# plt.tight_layout()
# # plt.savefig('Highest Balanced Accuracy achieved per Feature Selection Method.jpg')
# plt.show()


def plotScoreByClassifier(classifierName, resultScore):
    classfier = resultScore.loc[resultScore['Classification Algorithm'] == classifierName]
    classifierRes = classfier.groupby(['Dataset','Feature Selection Method'])['Balanced Accuracy'].max().unstack()
    ax = classifierRes.plot.bar(figsize=(10, 5), rot=0)
    plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0)
    plt.title(classifierName)
    plt.xlabel('Dataset')
    plt.ylabel('Balanced Accuracy')
    ax.set(axisbelow=True)
    plt.grid(axis='y')
          
    plt.savefig(f'0_{classifierName}_per_plot.jpeg', bbox_inches='tight', dpi=300)

def plotScoreByFSMethod(featureSelectionMethodName, resultScore):
    fsMethod = resultScore.loc[resultScore['Feature Selection Method'] == featureSelectionMethodName]
    fsMethodRes = fsMethod.groupby(['Dataset','Classification Algorithm'])['Balanced Accuracy'].max().unstack()
    ax = fsMethodRes.plot.bar(figsize=(10, 5), rot=0)
    plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0)
    plt.title(featureSelectionMethodName)
    plt.xlabel('Dataset')
    plt.ylabel('Balanced Accuracy')
    ax.set(axisbelow=True)
    plt.grid(axis='y')
    
    plt.savefig(f'1_{featureSelectionMethodName.replace("*", " Star")}_per_plot.jpeg', bbox_inches='tight', dpi=300)
    
# classifiers = resultScores['Classification Algorithm'].unique().tolist()
# methods = resultScores['Feature Selection Method'].unique().tolist()
# for c in classifiers:
#     plotScoreByClassifier(c, resultScores)

# for m in methods:
#     plotScoreByFSMethod(m, resultScores)


topResultScores = resultScores.loc[resultScores['Balanced Accuracy'] >= 0.60].copy()
topResultScores = topResultScores.sort_values(by=['Balanced Accuracy',  'Dataset'], ascending=False)

# print(topResultScores)

## Get unique for each column
# print(topResultScores['Feature Selection Method'].unique().tolist())
# print(topResultScores['Classification Algorithm'].unique().tolist())


## Show optimal evaluation combination per dataset
idxs = topResultScores.groupby('Dataset')['Balanced Accuracy'].transform(max) == topResultScores['Balanced Accuracy']
df = topResultScores[idxs].reset_index()
df = df[['Dataset', 'Feature Selection Method', 'Classification Algorithm', 'Balanced Accuracy', 'Optimal Feature Number', 'Optimal Threshold']]
# print(df)

