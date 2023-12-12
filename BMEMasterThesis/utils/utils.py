from ..algorithm import ALGORITHM_NAMES
from datetime import datetime as dt
import time

def getTime() -> float:
    return time.time()

def printTime(time: float) -> str:
    return dt.strftime(dt.utcfromtimestamp(time), '%d/%m/%Y %H:%M:%S')

def prettifyFeatureSelectionMethodName(featureSelectionMethodName: str) -> str:
    if featureSelectionMethodName in ALGORITHM_NAMES['FS_METHODS'].keys():
        return ALGORITHM_NAMES['FS_METHODS'][featureSelectionMethodName]

def simplifyFeatureSelectionMethodName(featureSelectionMethodName: str) -> str:
    if featureSelectionMethodName in ALGORITHM_NAMES['FS_METHODS'].values():
        return list(ALGORITHM_NAMES['FS_METHODS'].keys())[list(ALGORITHM_NAMES['FS_METHODS'].values()).index(featureSelectionMethodName)]

def prettifyClassificationAlgorithmName(classificationAlgorithmName: str) -> str:
    if classificationAlgorithmName in ALGORITHM_NAMES['MODELS'].keys():
        return ALGORITHM_NAMES['MODELS'][classificationAlgorithmName]

def simplifyClassificationAlgorithmName(classificationAlgorithmName: str) -> str:
    if classificationAlgorithmName in ALGORITHM_NAMES['MODELS'].values():
        return list(ALGORITHM_NAMES['MODELS'].keys())[list(ALGORITHM_NAMES['MODELS'].values()).index(classificationAlgorithmName)]