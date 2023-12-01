import abc
from typing import Dict, List
from radiomics import featureextractor
import six
import pandas as pd
import SimpleITK as sitk
import numpy as np
import progressbar

from .utils import getLogger, configuration as conf

class BasicRadiomicExtractor:
    def __init__(self, paramFile: str) -> None:
        self.__paramFile__ = paramFile
        self._logger = getLogger(__name__)
    
    @abc.abstractclassmethod
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        raise NotImplementedError(f'Extraction is not implemented in class {__name__}')


class SingleLabelRadiomicExtractor(BasicRadiomicExtractor):
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)
        
        returnPandasDataframe = kwargs['returnPandasDataframe'] if 'returnPandasDataframe' in kwargs else False
        kwargs.pop('returnPandasDataframe', None)
        
        normalizeScale = kwargs['normalizeScale'] if 'normalizeScale' in kwargs else None
        kwargs.pop('normalizeScale', None)

        extractor = featureextractor.RadiomicsFeatureExtractor(conf.PYRADIOMICS_PARAMS_FILE, **kwargs)
        values = pd.DataFrame(csvData).values
        
        self._logger.info('Starting radiomics extraction')
        widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='.'),  progressbar.FormatLabel(' %(value)d/%(max)d '), '(', progressbar.Percentage(), ') - ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(maxval = values.shape[0], widgets=widgets).start()

        radiomics = []
        for i, data in enumerate(values):       
            result = extractor.execute(data[0], data[1])
            bar.update(i)

            radiomic = {
                'Patient_Id': data[2]              
            }
            for key, value in six.iteritems(result):
                if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                    radiomic[key] = value
            radiomics.append(radiomic)
        
        bar.finish()
        self._logger.info('Radiomics extraction finished')
        if returnPandasDataframe:
            return pd.DataFrame.from_dict(radiomics)
        
        return radiomics

class MultiLabelRadiomicExtractor(BasicRadiomicExtractor):
    
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)
        
        returnPandasDataframe = kwargs['returnPandasDataframe'] if 'returnPandasDataframe' in kwargs else True
        kwargs.pop('returnPandasDataframe', None)
        
        normalizeScale = kwargs['normalizeScale'] if 'normalizeScale' in kwargs else None
        kwargs.pop('normalizeScale', None)

        extractor = featureextractor.RadiomicsFeatureExtractor(conf.PYRADIOMICS_PARAMS_FILE, **kwargs)
        values = pd.DataFrame(csvData).values
        
        self._logger.debug(f'Settings: {extractor.settings}')
        self._logger.debug(f'Enabled Image Types: {extractor.enabledImagetypes}')
        self._logger.debug(f'Enabled Features: {extractor.enabledFeatures}')
        

        self._logger.info('Starting multi-label radiomics extraction')
        widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='.'),  progressbar.FormatLabel(' %(value)d/%(max)d '), '(', progressbar.Percentage(), ') - ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(maxval = values.shape[0], widgets=widgets).start()

        radiomics = []
        for i, data in enumerate(values):
            bar.update(i)

            # Load image            
            image = sitk.ReadImage(data[0])
            
            if not normalizeScale is None:
                imageNumpy = sitk.GetArrayFromImage(image)
                image = sitk.GetImageFromArray(imageNumpy * normalizeScale)
                self._logger.debug('normalize image...')
                     

            # Load mask and convert it to numpy array
            originalMask = sitk.ReadImage(data[1])
            originalNumpyMask = sitk.GetArrayFromImage(originalMask)
            originalNumpyMaskUnique = np.unique(originalNumpyMask[originalNumpyMask != 0])

            for label in originalNumpyMaskUnique:
                result = extractor.execute(image, originalMask, label=int(label))
                bar.update(i)
                
                radiomic = {
                    'Patient_Id': data[2],
                    'Label': label
                }
                for key, value in six.iteritems(result):
                    if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                        radiomic[key] = value
                radiomics.append(radiomic)
        
        bar.finish()
        self._logger.info('Radiomics extraction finished')
        if returnPandasDataframe:
            return pd.DataFrame.from_dict(radiomics)
        
        return radiomics
  