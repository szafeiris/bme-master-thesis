from radiomics import featureextractor
from typing import Dict, List
import pandas as pd
import progressbar
import six

from bme_thesis.data.feature_extractor.RadiomicExtractor import RadiomicExtractor

class SingleLabelRadiomicExtractor(RadiomicExtractor):
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        keepDiagnosticsFeatures = kwargs['keepDiagnosticsFeatures'] if 'keepDiagnosticsFeatures' in kwargs else False
        kwargs.pop('keepDiagnosticsFeatures', None)
        
        returnPandasDataframe = kwargs['returnPandasDataframe'] if 'returnPandasDataframe' in kwargs else False
        kwargs.pop('returnPandasDataframe', None)
        
        normalizeScale = kwargs['normalizeScale'] if 'normalizeScale' in kwargs else None
        kwargs.pop('normalizeScale', None)

        extractor = featureextractor.RadiomicsFeatureExtractor(self.paramFile, **kwargs)
        values = pd.DataFrame(csvData).values
        
        self.log.info('Starting radiomics extraction')
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
        self.log.info('Radiomics extraction finished')
        if returnPandasDataframe:
            return pd.DataFrame.from_dict(radiomics)
        
        return radiomics