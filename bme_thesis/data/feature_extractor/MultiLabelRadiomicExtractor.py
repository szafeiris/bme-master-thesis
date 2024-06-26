from radiomics import featureextractor
from typing import Dict, List
import SimpleITK as sitk
import pandas as pd
import numpy as np
import progressbar
import six

from bme_thesis.data.feature_extractor.RadiomicExtractor import RadiomicExtractor

class MultiLabelRadiomicExtractor(RadiomicExtractor):
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)
        
        returnPandasDataframe = kwargs['returnPandasDataframe'] if 'returnPandasDataframe' in kwargs else True
        kwargs.pop('returnPandasDataframe', None)
        
        normalizeScale = kwargs['normalizeScale'] if 'normalizeScale' in kwargs else None
        kwargs.pop('normalizeScale', None)

        extractor = featureextractor.RadiomicsFeatureExtractor(self.paramFile, **kwargs)
        values = pd.DataFrame(csvData).values
        
        self.log.debug(f'Settings: {extractor.settings}')
        self.log.debug(f'Enabled Image Types: {extractor.enabledImagetypes}')
        self.log.debug(f'Enabled Features: {extractor.enabledFeatures}')
        

        self.log.info('Starting multi-label radiomics extraction')
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
                self.log.debug('normalize image...')
                     

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
        self.log.info('Radiomics extraction finished')
        if returnPandasDataframe:
            return pd.DataFrame.from_dict(radiomics)
        
        return radiomics