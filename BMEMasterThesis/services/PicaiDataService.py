from .DataService import DataService
from .utils import DataReader, NiftyReader
from ..extractor import BasicRadiomicExtractor, MultiLabelRadiomicExtractor
from ..utils.config import PATHS, Datasets
from ..utils.log import getLogger
from ..utils.utils import prettifyClassificationAlgorithmName, prettifyFeatureSelectionMethodName

from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from numpy.typing import NDArray
from typing import List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np
import json
import sys
import os
 
class PicaiDataService(DataService):
    def __init__(self, dataReader: DataReader = NiftyReader(),
                       radiomicsExtractor: BasicRadiomicExtractor = MultiLabelRadiomicExtractor(PATHS.RADIOMICS_DIR)) -> None:
        super().__init__(dataReader, radiomicsExtractor)
        self._logger = getLogger(__name__)
    
    def generateBinWidth(self, dataset: str = None, bins: int = 32, normalizeScale: int = 100) -> Tuple[float, int]:
        self._logger.debug(f'Calculating bin width for dataset: {dataset}, using {bins} bins and normalization scale equal to {normalizeScale}')
        rangesFile = PATHS.getRangesFile(dataset)        
        if rangesFile.exists() and rangesFile.is_file():
            rangesData = json.load(rangesFile.open())
            return rangesData['binWidth'], rangesData['globalMin']
        
        with PATHS.PICAI_PATIENTS_ID_FILE.open() as patientIdsFile:
            ranges: List[float] = []
            rangeData = {}
            globalMin = sys.maxsize
            for patientId in patientIdsFile:
                patientId = patientId.strip()
                imageFile = PATHS.getImagePathByDatasetAndPatientId(dataset, patientId)
                maskFile = PATHS.getMaskPathByPatientId(patientId)
                    
                image = self._dataReader.read(imageFile)
                mask = self._dataReader.read(maskFile)
                
                rangeData = {
                    **rangeData,
                    patientId: {},
                }
                
                for maskValue in np.unique(mask)[1:]:
                    tempMask = np.zeros_like(mask)
                    tempMask[mask == maskValue] = 1
                    
                    tempImage = image * tempMask                    
                    
                    tempImageMax = np.max(tempImage)
                    tempImageMin = np.min(tempImage[tempImage > np.min(tempImage)])
                    if tempImageMin < globalMin:
                        globalMin = tempImageMin
                    
                    imageRange = tempImageMax - tempImageMin + 1
                    ranges.append(imageRange)
                       
                    rangeData[patientId] = {
                        **rangeData[patientId],
                        str(maskValue): {
                            'imageMax': tempImageMax,
                            'imageMin': tempImageMin,
                            'imageRange': imageRange,
                        }
                    }
            
            if (dataset in [ Datasets.FAT_NORMALIZED,
                             Datasets.MUSCLE_NORMALIZED,
                             Datasets.N4_NORMALIZED,
                             Datasets.ORIGINAL_NORMALIZED
                           ]):
                ranges = [ r * normalizeScale for r in ranges ]
            
            meanRanges = np.mean(ranges)
            binWidth = int(np.round(meanRanges / bins))
            
            self._logger.debug(f'Mean Ranges for dataset `{dataset}` is {meanRanges}')
            
            rangeData = {
                **rangeData,
                'binWidth': binWidth,
                'globalMin': globalMin,
                'binCount': bins,
                'normalizeScale': normalizeScale,
                'meanRanges': meanRanges,
            }
            
            json.dump(rangeData, rangesFile.open('w'), sort_keys= True, indent= 4)
            return binWidth, globalMin
                 
    def extractRadiomics(self, dataset: str, outputCsvFile: str | Path = None, keepDiagnosticsFeatures: bool = False, binWidth: int | None = None, shiftValue: float | int = None, isFixedBinWidth: bool = True, binCount: int = None, normalizeScale: int | None = None):
        csvData = {
            'Image': [],
            'Mask': [],
            'Patient ID': []
        }

        self._logger.info("Gathering image data")
        for mask in PATHS.PICAI_MASKS_DIR.glob('*'):
            patientCode = str(mask).split(os.sep)[-1].replace('.nii.gz', '')
            csvData['Patient ID'].append(patientCode)

            csvData['Mask'].append(str(PATHS.getMaskPathByPatientId(patientCode)))
            csvData['Image'].append(str(PATHS.getImagePathByDatasetAndPatientId(dataset, patientCode)))

        self._logger.info("Extracting radiomics features")
        if isFixedBinWidth:
            if (not binWidth is None) and (not shiftValue is None):
                radiomicFeaturesDataframe = self._radiomicsExtractor.extract(csvData,
                                                                             keepDiagnosticsFeatures=keepDiagnosticsFeatures,
                                                                             binWidth=binWidth,
                                                                             voxelArrayShift=shiftValue
                                                                            )
            else:
                raise ValueError(f'`binWidth` or `shiftValue` cannot be None')
        else:
            if not binCount is None:
                radiomicFeaturesDataframe = self._radiomicsExtractor.extract(csvData,
                                                                             keepDiagnosticsFeatures=keepDiagnosticsFeatures,
                                                                             binCount=binCount)
            else:
                raise ValueError(f'`binCount` cannot be None')
            
        if outputCsvFile is not None:
            self._logger.info('Saving radiomics file.')
            radiomicFeaturesDataframe.to_csv(outputCsvFile, index=False)

        return radiomicFeaturesDataframe
    
    def getMetadata(self):
        return pd.read_csv(PATHS.PICAI_METADATA_FILE)
    
    def getRadiomicFeatureNames(self):
        return json.load(PATHS.PICAI_RADIOMICS_NAMES_FILE.open())
    
    def getScores(self, dataset: str) -> pd.DataFrame:
        scoresCSVFile = PATHS.getScoresCsvFile(dataset)
        if scoresCSVFile.exists():
            return pd.read_csv(scoresCSVFile)
        return None
            
    def generateScoresFromResults(self, dataset: str, yTrue: NDArray, forceGenerate: bool = False):
        scoresCSVFile = PATHS.getScoresCsvFile(dataset)
        if scoresCSVFile.exists() and (not forceGenerate):
            return
        
        combinationResults = []
        for combinationResultFile in PATHS.getResultsDir(dataset).glob('*.json'):
            combinationResult = json.load(combinationResultFile.open())
            combination = str(combinationResultFile).split(os.sep)[-1].removesuffix('.json').split('_')
            method, model = combination[0], combination[1]            
            
            optimalThreshold = None
            optimalFeatureNumber = None
            if 'surf' in method or 'relief' in method:
                optimalFeatureNumber = combinationResult['best_method_params']['n_features_to_select']
            elif ('pearson' in method or 'spearman' in method) and (not 'itmo' in method):
                optimalThreshold = combinationResult['best_method_params']['threshold']
                optimalFeatureNumber = len(combinationResult['best_method_params']['selectedFeatures'])
            else:
                optimalFeatureNumber = len(combinationResult['best_method_params']['selectedFeatures'])
            
            yPred = np.array(combinationResult['test_predictions'])
            TP, FP = combinationResult['confusion_matrix']['TP'], combinationResult['confusion_matrix']['FP']
            TN, FN = combinationResult['confusion_matrix']['TN'], combinationResult['confusion_matrix']['FN']
                        
            combinationResultsDict = {
                'Feature Selection Method': prettifyFeatureSelectionMethodName(method),
                'Classification Algorithm': prettifyClassificationAlgorithmName(model),
                'Dataset': Datasets.prettifyDataset(dataset),
                'Optimal Feature Number': optimalFeatureNumber,
                'Optimal Threshold': optimalThreshold,
                'Balanced Accuracy': balanced_accuracy_score(yTrue, yPred),
                'Sensitivity': TP / (TP + FN),
                'Speficity': TN / (TN + FP),
                'Accuracy': accuracy_score(yTrue, yPred),
                'F1': f1_score(yTrue, yPred),
                'Precision': precision_score(yTrue, yPred),
                'Recall': recall_score(yTrue, yPred),
                'ROC AUC': roc_auc_score(yTrue, yPred),
                'Cohen Kappa': cohen_kappa_score(yTrue, yPred),              
                **combinationResult['confusion_matrix'],
            }
            
            combinationResults.append(combinationResultsDict)
        
        pd.DataFrame(combinationResults).to_csv(scoresCSVFile, index=False)
            
            
            
            
        
        
    