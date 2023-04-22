from . import log
from radiomics import featureextractor
import six
import pandas as pd
import SimpleITK as sitk
import numpy as np
import progressbar

class RadiomicExtractor:
    def __init__(self, paramFile=None):
        self.__paramFile__ = paramFile

    def extractFromCsv(self, csvData, **kwargs):
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)
        extractor = featureextractor.RadiomicsFeatureExtractor(self.__paramFile_)
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
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)

        extractor = featureextractor.RadiomicsFeatureExtractor(self.__paramFile__, **kwargs)
        
        values = pd.DataFrame(csvData).values

        widgets=['[', progressbar.Timer(), '] ', progressbar.Bar(marker='.'),  progressbar.FormatLabel(' %(value)d/%(max)d '), '(', progressbar.Percentage(), ') - ', progressbar.AdaptiveETA()]
        bar = progressbar.ProgressBar(maxval = values.shape[0], widgets=widgets).start()

        radiomics = []
        for i, data in enumerate(values):
            bar.update(i)

            # Load image            
            image = sitk.ReadImage(data[0])

            # Load mask and convert it to numpy array
            originalMask = sitk.ReadImage(data[1])
            originalNumpyMask = sitk.GetArrayFromImage(originalMask)
            originalNumpyMaskUnique = np.unique(originalNumpyMask[originalNumpyMask != 0])

            for label in originalNumpyMaskUnique:
                radiomic = {}
                radiomic['Patient_Id'] = data[2]
                radiomic['Label'] = label

                result = extractor.execute(image, originalMask, label=int(label))
                for key, value in six.iteritems(result):
                    if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                        radiomic[key] = value
                radiomics.append(radiomic)
        
        bar.finish()
        return pd.DataFrame.from_dict(radiomics)
