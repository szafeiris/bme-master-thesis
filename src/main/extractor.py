from . import configurator as conf
from radiomics import featureextractor
import six
import pandas as pd

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class RadiomicExtractor:
    def __init__(self, paramFile = conf.PYRADIOMICS_PARAMS_FILE):
        self.__paramFile__ = paramFile

    def extractFromCsv(self, csvData, keepDiagnosticsFeatures=False):
        extractor = featureextractor.RadiomicsFeatureExtractor(self.__paramFile__)

        radiomics = []
        for data in pd.DataFrame(csvData).values:            
            radiomic = {}
            radiomic['Patient_Id'] = data[2]
            log.debug(data)
            
            result = extractor.execute(data[0], data[1])
            for key, value in six.iteritems(result):
                if (not 'diagnostics_' in key) or keepDiagnosticsFeatures:
                    radiomic[key] = value
            radiomics.append(radiomic)
        
        return radiomics
