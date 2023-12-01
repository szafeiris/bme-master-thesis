from .DataService import DataService
from .utils import DataReader, NiftyReader
from ..extractor import BasicRadiomicExtractor, MultiLabelRadiomicExtractor
from ..utils.config import PATHS, Datasets
from ..utils.log import getLogger

from matplotlib import pyplot as plt
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
    