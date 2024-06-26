from bme_thesis.algorithms import ALGORITHM_NAMES


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