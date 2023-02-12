from . import configurator as conf
from radiomics import featureextractor
import six
import pandas as pd
import SimpleITK as sitk
import numpy as np
import progressbar

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class RadiomicExtractor:
    def __init__(self, paramFile=None):
        self.__paramFile__ = paramFile

    def extractFromCsv(self, csvData, **kwargs):
        extractor = featureextractor.RadiomicsFeatureExtractor(self.__paramFile__)
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        values = pd.DataFrame(csvData).values

        widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='.'),  progressbar.FormatLabel(' %(value)d/%(max)d '), '(', progressbar.Percentage(), ') - ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(maxval = values.shape[0], widgets=widgets).start()

        radiomics = []
        for i, data in enumerate(values):            
            radiomic = {}
            radiomic['Patient_Id'] = data[2]
            bar.update(i)
            
            result = extractor.execute(data[0], data[1])
            for key, value in six.iteritems(result):
                if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                    radiomic[key] = value
            radiomics.append(radiomic)
        
        bar.finish()
        return pd.DataFrame.from_dict(radiomics)

class MultiLabelRadiomicExtractor(RadiomicExtractor):
    def __init__(self, paramFile=None):
        self.__paramFile__ = paramFile

    def extractFromCsv(self, csvData, **kwargs):
        extractor = featureextractor.RadiomicsFeatureExtractor(self.__paramFile__)
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        values = pd.DataFrame(csvData).values

        widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='.'),  progressbar.FormatLabel(' %(value)d/%(max)d '), '(', progressbar.Percentage(), ') - ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(maxval = values.shape[0], widgets=widgets).start()

        radiomics = []
        for i, data in enumerate(values):            
            radiomic = {}
            radiomic['Patient_Id'] = data[2]
            bar.update(i)
            
            image = sitk.ReadImage(data[0])

            mask_ori = sitk.ReadImage(data[1])
            mask_np = sitk.GetArrayFromImage(mask_ori)
            
            mask_np[mask_np >= 2] = 1
            
            mask = sitk.GetImageFromArray(mask_np)
            mask.CopyInformation(mask_ori)
            mask.SetSpacing(mask_ori.GetSpacing())
            mask.SetOrigin(mask_ori.GetOrigin())
            mask.SetDirection(mask_ori.GetDirection())

            result = extractor.execute(image, mask)
            for key, value in six.iteritems(result):
                if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                    radiomic[key] = value
            radiomics.append(radiomic)
        
        bar.finish()
        return pd.DataFrame.from_dict(radiomics)
