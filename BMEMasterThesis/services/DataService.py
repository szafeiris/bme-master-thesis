from BMEMasterThesis.services.utils import DataReader
from BMEMasterThesis.extractor import BasicRadiomicExtractor
from BMEMasterThesis.utils.config import PATHS

from typing import Dict, List, Tuple
from pathlib import Path
import pandas as pd
import json
import os

class DataService:
    def __init__(self, dataReader: DataReader = None, radiomicsExtractor: BasicRadiomicExtractor = None) -> None:
        self._dataReader = dataReader
        self._radiomicsExtractor = radiomicsExtractor
        
    def getMethodAndModelNamesFromFilename(filename: str | Path) -> Tuple[str, str]:
        if isinstance(filename, Path):
            filename = str(filename)
            
        combination = str(filename).split(os.sep)[-1].removesuffix('.json').split('_')
        return combination[0], combination[1]
            
    def getSelectedFeaturesForCombination(dataset: str, method: str, model: str) -> List[int]:
        combinationDir = PATHS.getResultsForCombinationDir(dataset, method, model)
        combinationResult = json.load(combinationDir.open())
        method, model = DataService.getMethodAndModelNamesFromFilename(combinationDir)

        if 'surf' in method or 'relief' in method:
            selectedFeatures = combinationResult['selected_features']
        elif ('pearson' in method or 'spearman' in method) and (not 'itmo' in method):
            selectedFeatures = combinationResult['best_method_params']['selectedFeatures']
        else:
            selectedFeatures = combinationResult['best_method_params']['selectedFeatures']

        return selectedFeatures
    
    def getNamedSelectedFeaturesForCombination(dataset: str, method: str, model: str, radiomicFeaturesNames: List[str]) -> List[str]:
        return [radiomicFeaturesNames[idx] for idx in DataService.getSelectedFeaturesForCombination(dataset, method, model)]    
    
    def calculateFeatureStatistics(featureNames: List[str], byType: bool = False) -> pd.DataFrame:   
        filters = set([featureName.split('_')[0] for featureName in featureNames])
        types = set([featureName.split('_')[1] for featureName in featureNames])
        
        if byType:
            radiomicFeatureData = [
                { 
                    'Type': t, 
                    'Number of Features': sum([1 for name in featureNames if t in name])
                } for t in types
            ]
        else:
            radiomicFeatureData = {
                filter: {
                    t: sum([1 for name in featureNames if (t in name) and (filter in name)]) for t in types
                } for filter in filters
            }
        
        return pd.DataFrame(radiomicFeatureData)
