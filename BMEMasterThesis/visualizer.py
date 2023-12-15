from .utils.utils import simplifyFeatureSelectionMethodName, simplifyClassificationAlgorithmName
from .utils import log, Datasets, PATHS, getLogger
from .services import DataService, PicaiDataService

import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, List
import pandas as pd
import numpy as np
import json
import os

class Visualizer:
    def __init__(self, dataService: DataService) -> None:
        self._logger = getLogger(self.__class__.__name__)
        self._dataService = dataService
    
    def visualizeDataset(self, dataset: str) -> None:
        pass
    
    def visualizeDatasets(self, datasets: List[str], saveName: str | Path) -> None:
        pass
    
    def visualizeAllDatasets(self, saveName: str | Path) -> None:
        self.visualizeDatasets(Datasets._ALL_DATASETS, saveName)
    
    def getMaxScore(scoreName: str, resultScores: pd.DataFrame):
        return resultScores.loc[resultScores[scoreName].idxmax()]
    
    def generateBoxplotForScore(scoreName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
        resultScoresBoxplot = resultScore.boxplot(
            column = [scoreName,], 
            by = 'Dataset', 
            rot = 0, 
            figsize=(10, 10))
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
        plt.close()
    
    def generatePerClassifierPlotForScore(scoreName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
        resultScore.groupby(['Dataset','Classification Algorithm'])[scoreName].max().unstack().plot.bar(rot= 0, figsize= (10, 5))
        plt.xlabel('Dataset')
        plt.ylabel(scoreName)
        plt.title(f'Highest {scoreName} achieved per Classification Algorithm')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        
        if not (path is None):
            if isinstance(path, str):
                path = Path(path)
            plt.savefig(path)

        if showPlot:
            plt.show()
        plt.close()
    
    def generatePerMethodPlotForScore(scoreName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
        resultScore.groupby(['Dataset','Feature Selection Method'])[scoreName].max().unstack().plot.bar(rot= 0, figsize= (15, 7))
        plt.xlabel('Dataset')
        plt.ylabel(scoreName)
        plt.title(f'Highest {scoreName} Achieved per Feature Selection Method')
        plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0)
        plt.tight_layout()
        
        if not (path is None):
            if isinstance(path, str):
                path = Path(path)
            plt.savefig(path)

        if showPlot:
            plt.show()
        plt.close()
        
    def generateScorePlotByClassifier(scoreName: str, classifierName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
        classfier = resultScore.loc[resultScore['Classification Algorithm'] == classifierName]
        classifierRes = classfier.groupby(['Dataset','Feature Selection Method'])[scoreName].max().unstack()
        ax = classifierRes.plot.bar(figsize=(10, 5), rot=0)
        plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0)
        plt.title(classifierName)
        plt.xlabel('Dataset')
        plt.ylabel(scoreName)
        ax.set(axisbelow=True)
        plt.grid(axis='y')

        if not (path is None):
            if isinstance(path, str):
                path = Path(path)
            plt.savefig(path.joinpath(f'{classifierName}.jpeg'))

        if showPlot:
            plt.show()
        plt.close()

    def generateScorePlotByFSMethod(scoreName: str, featureSelectionMethodName: str, resultScore: pd.DataFrame, showPlot: bool = False, path: str | Path = None):
        fsMethod = resultScore.loc[resultScore['Feature Selection Method'] == featureSelectionMethodName]
        fsMethodRes = fsMethod.groupby(['Dataset','Classification Algorithm'])[scoreName].max().unstack()
        ax = fsMethodRes.plot.bar(figsize=(10, 5), rot=0)
        plt.legend(bbox_to_anchor=(1.01, 1.01), loc='upper left', borderaxespad=0)
        plt.title(featureSelectionMethodName)
        plt.xlabel('Dataset')
        plt.ylabel(scoreName)
        ax.set(axisbelow=True)
        plt.grid(axis='y')

        if not (path is None):
            if isinstance(path, str):
                path = Path(path)
            plt.savefig(path.joinpath(f'{featureSelectionMethodName.replace("*", " Star")}.jpeg'))

        if showPlot:
            plt.show()
        plt.close()
    
    def getNBestCombinationsForDataset(n: int, dataset: str, resultScores: pd.DataFrame, scoreName: str = 'Balanced Accuracy') -> pd.DataFrame:
        bestCombinationIdxs = resultScores['Dataset'] == Datasets.prettifyDataset(dataset)
        bestCombinations = resultScores[bestCombinationIdxs].sort_values(by=[scoreName], ascending=False).reset_index()
        return bestCombinations[:n]

    def getBestCombinationForDataset(dataset: str, resultScores: pd.DataFrame, scoreName: str = 'Balanced Accuracy')  -> pd.DataFrame:
        return Visualizer.getNBestCombinationsForDataset(1, dataset, resultScores, scoreName)
    
    def getBestCombinationsForDatasets(datasets: List[str], resultScores: pd.DataFrame, scoreName: str = 'Balanced Accuracy') -> List[pd.DataFrame]:
        return [Visualizer.getBestCombinationForDataset(dataset, resultScores, scoreName) for dataset in datasets]

    def getBestCombinationsForAllDatasets(resultScores: pd.DataFrame, scoreName: str = 'Balanced Accuracy') -> List[pd.DataFrame]:
        return Visualizer.getBestCombinationsForDatasets(Datasets._ALL_DATASETS, resultScores, scoreName)
    
class PicaiVisualizer(Visualizer):
    def __init__(self, dataService: PicaiDataService = PicaiDataService()) -> None:
        super().__init__(dataService)
        self._dataService = dataService
    
    def visualizeDataset(self, dataset: str) -> None:
        analysisDir = PATHS.getAnalysisDir(dataset)
        datasetResultScores = self._dataService.getScores(dataset)
        radiomicFeaturesNames = self._dataService.getRadiomicFeatureNames()
        
        maxBalancedAccuracy = Visualizer.getMaxScore('Balanced Accuracy', datasetResultScores)
        maxBalancedAccuracy.to_excel(analysisDir.joinpath('Max Balanced Accuracy.xlsx'))
        
        Visualizer.generateBoxplotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy Boxplot.jpg'))
        Visualizer.generateBoxplotForScore('ROC AUC', datasetResultScores, path=analysisDir.joinpath('ROC AUC Boxplot.jpg'))
        Visualizer.generateBoxplotForScore('Cohen Kappa', datasetResultScores, path=analysisDir.joinpath('Cohen Kappa Boxplot.jpg'))
        
        Visualizer.generatePerClassifierPlotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy per Classifier.jpg'))
        Visualizer.generatePerMethodPlotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy per Feature Selection Method.jpg'))
        
        classifiers = datasetResultScores['Classification Algorithm'].unique().tolist()
        methods = datasetResultScores['Feature Selection Method'].unique().tolist()
        for c in classifiers:
            Visualizer.generateScorePlotByClassifier('Balanced Accuracy', c, datasetResultScores, path=analysisDir.joinpath('plots').joinpath('classifiers'))
        for m in methods:
            Visualizer.generateScorePlotByFSMethod('Balanced Accuracy', m, datasetResultScores, path=analysisDir.joinpath('plots').joinpath('methods'))
        
        bestCombination = Visualizer.getBestCombinationForDataset(dataset, datasetResultScores)
        bestCombination.to_excel(analysisDir.joinpath('Top Combination.xlsx'), index = False)
        
        for n in [5, 10, 20]:
            topNCombinations = Visualizer.getNBestCombinationsForDataset(n, dataset, datasetResultScores)
            topNCombinations.to_excel(analysisDir.joinpath(f'Top {n} Combination.xlsx'), index = False)
        
        bestMethod = simplifyFeatureSelectionMethodName(bestCombination['Feature Selection Method'].get(0))
        bestModel = simplifyClassificationAlgorithmName(bestCombination['Classification Algorithm'].get(0))
        selectedRadiomicFeaturesNames = DataService.getNamedSelectedFeaturesForCombination(dataset, bestMethod, bestModel, radiomicFeaturesNames)
        json.dump(selectedRadiomicFeaturesNames, analysisDir.joinpath(f'Selected Features for Optimal Configuration.json').open('w'), indent= 1)
        
        selectedRadiomicFeaturesStatistics = PicaiDataService.calculateFeatureStatistics(selectedRadiomicFeaturesNames)
        selectedRadiomicFeaturesStatistics.to_excel(analysisDir.joinpath(f'Statistics for Selected Features of Optimal Configuration.xlsx'))
    
    def visualizeDatasets(self, datasets: List[str], saveName: str | Path) -> None:
        if isinstance(saveName, Path):
            saveName = str(saveName)
                    
        analysisDir = PATHS.getAnalysisDir(saveName)
        
        datasetResultScores = pd.concat([self._dataService.getScores(dataset) for dataset in datasets], ignore_index= True)
        datasetResultScores.sort_values(by=['Balanced Accuracy', 'ROC AUC', 'Dataset', 'Feature Selection Method', 'Classification Algorithm'])
        
        Visualizer.generateBoxplotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy Boxplot.jpg'))
        Visualizer.generateBoxplotForScore('ROC AUC', datasetResultScores, path=analysisDir.joinpath('ROC AUC Boxplot.jpg'))
        Visualizer.generateBoxplotForScore('Cohen Kappa', datasetResultScores, path=analysisDir.joinpath('Cohen Kappa Boxplot.jpg'))
        
        Visualizer.generatePerClassifierPlotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy per Classifier.jpg'))
        Visualizer.generatePerMethodPlotForScore('Balanced Accuracy', datasetResultScores, path=analysisDir.joinpath('Balanced Accuracy per Feature Selection Method.jpg'))
        
        classifiers = datasetResultScores['Classification Algorithm'].unique().tolist()
        methods = datasetResultScores['Feature Selection Method'].unique().tolist()
        for c in classifiers:
            Visualizer.generateScorePlotByClassifier('Balanced Accuracy', c, datasetResultScores, path=analysisDir.joinpath('plots').joinpath('classifiers'))
        for m in methods:
            Visualizer.generateScorePlotByFSMethod('Balanced Accuracy', m, datasetResultScores, path=analysisDir.joinpath('plots').joinpath('methods'))

        topResultScores = datasetResultScores.loc[datasetResultScores['Balanced Accuracy'] >= 0.60].copy()
        topResultScores = topResultScores.sort_values(by=['Balanced Accuracy',  'Dataset'], ascending=False)
        topResultScores.to_excel(analysisDir.joinpath(f'Scores with balanced accuracy over 0.60 (Top Scores).xlsx'), index= False)        
        
        json.dump({
            'Number of Combinations per Dataset': datasetResultScores.groupby(by='Dataset')['Dataset'].aggregate(['count']).to_dict()['count'],
            'Top Feature Selection Method Used': topResultScores['Feature Selection Method'].unique().tolist(),
            'Top Classification Algorithm Used': topResultScores['Classification Algorithm'].unique().tolist(),    
        }, analysisDir.joinpath(f'Selected Features for Optimal Configuration.json').open('w'), indent= 1)        
        
        # Optimal Evaluation Combination per Dataset
        idxs = datasetResultScores.groupby('Dataset')['Balanced Accuracy'].transform(max) == datasetResultScores['Balanced Accuracy']
        topResultScoresPerDataset = datasetResultScores[idxs].reset_index()
        # topResultScoresPerDataset = topResultScoresPerDataset[['Dataset', 'Feature Selection Method', 'Classification Algorithm', 'Balanced Accuracy', 'Optimal Feature Number', 'Optimal Threshold']]
        topResultScoresPerDataset.to_excel(analysisDir.joinpath(f'Best Combination per Dataset.xlsx'), index= False)

        
        