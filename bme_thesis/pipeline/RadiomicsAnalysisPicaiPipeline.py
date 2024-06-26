import os
from pathlib import Path
import pandas as pd
import numpy as np

from bme_thesis.pipeline.PicaiPipeline import PicaiPipeline
from bme_thesis.utils.dataset import Datasets
from bme_thesis.utils.json import JSON
from bme_thesis.utils.notification import sendNotification
from bme_thesis.utils.paths import Paths
from bme_thesis.utils.settings import bmeThesisSettings 
from bme_thesis.utils.time import TimeUtil


class RadiomicsAnalysisPicaiPipeline(PicaiPipeline):
    def run(self, **kwargs) -> None:
        dataset = self.ctx.dataset
        try:
            self.log.debug(f'Pipeline {self.__class__.__name__} for {dataset} dataset started')
            startTime = TimeUtil.now()
            
            self._unpackArgs(**kwargs)
            self.__step_1_extractRadiomics__()
            # self.__step_2_readData__()
            # self.__step_3_evaluate__()
            # self.__step_4_createScores__()
            # self.__step_5_visualize__()
            
            self.log.debug(f'Pipeline {self.__class__.__name__} for {dataset} dataset ended [Elapsed: {TimeUtil.format(TimeUtil.now() - startTime)}]')
        except KeyboardInterrupt:
            self.log.warning('Pipeline terminated by user')
        except Exception as e:
            self.log.error(f'An error occured during the execution of pipeline: {self.__class__.__name__}.')
            self.log.exception(e)
            sendNotification(f'An error occured during the execution of pipeline: {self.__class__.__name__} for dataset: {self.dataset} [{str(e)}]')

    def _unpackArgs(self, **kwargs):
        self.ctx.isFixedBinWidth = kwargs.get('isFixedBinWidth', True)
        self.ctx.binCount = kwargs['binCount'] if 'binCount' in kwargs and isinstance(kwargs['binCount'], int) and kwargs['binCount']  > 0 else 32
        self.ctx.normallizeScale = kwargs['normallizeScale'] if 'normallizeScale' in kwargs and isinstance(kwargs['normallizeScale'], int) and kwargs['normallizeScale']  > 0 else 100
    
    
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
            
            JSON.save(rangeData, rangesFile, sort_keys=True, indent=4)
            return binWidth, globalMin
        
    def extractRadiomics(self, dataset: str, outputCsvFile: str | Path = None, keepDiagnosticsFeatures: bool = False, binWidth: int | None = None, shiftValue: float | int = None, isFixedBinWidth: bool = True, binCount: int = None, normalizeScale: int | None = None):
        csvData = {
            'Image': [],
            'Mask': [],
            'Patient ID': []
        }

        self.log.info("Gathering image data")
        for mask in Paths.toPath(bmeThesisSettings.masks_path).glob('*'):
            patientCode = str(mask).split(os.sep)[-1].replace('.nii.gz', '')
            csvData['Patient ID'].append(patientCode)

            csvData['Mask'].append(str(Paths.getMaskPathByPatientId(patientCode)))
            csvData['Image'].append(str(Paths.getImagePathByDatasetAndPatientId(dataset, patientCode)))

        self.log.info("Extracting radiomics features")
        if isFixedBinWidth:
            if (not binWidth is None) and (not shiftValue is None):
                radiomicFeaturesDataframe = self.radiomicsExtractor.extract(csvData,
                                                                             keepDiagnosticsFeatures=keepDiagnosticsFeatures,
                                                                             binWidth=binWidth,
                                                                             voxelArrayShift=shiftValue
                                                                            )
            else:
                raise ValueError(f'`binWidth` or `shiftValue` cannot be None')
        else:
            if not binCount is None:
                radiomicFeaturesDataframe = self.radiomicsExtractor.extract(csvData,
                                                                             keepDiagnosticsFeatures=keepDiagnosticsFeatures,
                                                                             binCount=binCount
                                                                            )
            else:
                raise ValueError(f'`binCount` cannot be None')
            
        if outputCsvFile is not None:
            self.log.info('Saving radiomics file.')
            radiomicFeaturesDataframe.to_csv(outputCsvFile, index=False)

        return radiomicFeaturesDataframe
    
    def _step_1_extractRadiomics(self):
        self.ctx.radiomicsFile = Paths.getRadiomicFile(self.ctx.dataset)
        if self.ctx.radiomicsFile.exists() and self.ctx.radiomicsFile.is_file():
            self.ctx.radiomics = pd.read_csv(self.ctx.radiomicsFile)
            self.log.info(f'Radiomics for `{self.ctx.dataset}` are loaded successfully')
            return
        
        normalizeScale = self.ctx.normallizeScale if self.ctx.dataset in Datasets.NORMALIZED_DATASETS else None  
        if self.ctx.isFixedBinWidth:
            binWidth, globalMin = self.generateBinWidth(self.ctx.dataset, self.ctx.binCount, normalizeScale)
            normalizedGlobalMin = 0 if globalMin > 0 else -globalMin
            self.log.info(f'Bin width: {binWidth}, Global Minimum (Normalized): {globalMin} ({normalizedGlobalMin})')
            
            self.ctx.radiomics = self.extractRadiomics( self.ctx.dataset, 
                                                        self.ctx.radiomicsFile, 
                                                        binWidth=binWidth, 
                                                        shiftValue=normalizedGlobalMin, 
                                                        normalizeScale=normalizeScale
                                                       )
        else:
            self.ctx.radiomics = self.extractRadiomics( self.ctx.dataset,
                                                        self.ctx.radiomicsFile,
                                                        binCount=self.ctx.binCount,
                                                        normalizeScale=normalizeScale
                                                       )