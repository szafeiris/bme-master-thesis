from src.main.configurator import configurator as conf
from src.main.data import *
from src.main.evaluation import *

import os

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

def runNsclcEvaluation():    
    dataService = NsclcRadiogenomicsDataService()

    # Convert folders to nifty
    if not os.path.exists(conf.NSCLC_NIFTI_IMAGES_DIR):
        dataService.convertToNifty(conf.NSCLC_IMAGES_DIR, conf.NSCLC_NIFTI_IMAGES_DIR)

    # Extract radiomics from images or read them
    if os.path.exists(conf.NSCLC_RADIOMICS_FILE):
        radiomicFeatures = dataService.readRadiomics(conf.NSCLC_RADIOMICS_FILE)
    else:
        radiomicFeatures = dataService.extractRadiomics(conf.NSCLC_NIFTI_IMAGES_DIR, conf.NSCLC_RADIOMICS_FILE)

    patientIds = radiomicFeatures.pop('Patient_Id').to_list()
    radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    X = radiomicFeatures.to_numpy()

    # Declare labels
    y = np.asarray([0, 0, 1, 1, 0, 0])

    evaluator = Evaluator()
    args = { 'patientIds': patientIds, 'radiomicFeaturesNames': radiomicFeaturesNames}
    evaluationResults = evaluator.evaluate(X, y, **args)
    for er in evaluationResults:
        log.debug(er)

def runPicaiEvaluation():    
    dataService = PicaiDataService()

    # Extract radiomics from images or read them
    if os.path.exists(conf.PICAI_RADIOMICS_FILE):
        radiomicFeatures = dataService.readRadiomics(conf.PICAI_RADIOMICS_FILE)
    else:
        radiomicFeatures = dataService.extractRadiomics(conf.PICAI_NIFTI_IMAGES_DIR, conf.PICAI_RADIOMICS_FILE)
    
    patientIds = radiomicFeatures.pop('Patient_Id').to_list()
    labels = radiomicFeatures.pop('Label')
    radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    
    X = radiomicFeatures.to_numpy()
    y = np.copy(labels)
    y[y == 2] = 0   # 0: ISUP = 2,
    y[y > 2] = 1    # 1: ISUP > 2
    
    evaluator = Evaluator()
    experimentData = {
            # 'method': 'boruta',
            # 'methodParams': {},
            'method': 'pearson',
            'methodParams': {
                'nFeatures': 2
                # 'n_features_to_select': 3
            },
            'model': 'svm',
            'modelParams': {
                'kernel': 'linear'
            },

            # 'crossValidation': StratifiedKFold(),
            'crossValidationNFolds': 3,
            # 'testSize': 1/3,
            'testSize': 0.35,
        }
    
    args = { 
        'patientIds': patientIds,
        'radiomicFeaturesNames': radiomicFeaturesNames,
        'experimentData': experimentData
    }
    evaluationResults = evaluator.evaluate(X, y, **args)
    
    evaluationResultsDictionary = []
    for evaluationResult in evaluationResults:
        evaluationResult.calculateMetrics(y)
        evaluationResultDictionary = evaluationResult.dict()
        evaluationResultsDictionary.append(evaluationResultDictionary)

    log.debug(evaluationResultsDictionary)
    filename = f"{evaluationResultsDictionary[0]['1']['name']}_CV.json" if len(evaluationResultsDictionary) > 1 else f"{evaluationResultsDictionary[0]['name']}.json"
    json.dump(
        evaluationResultsDictionary,
        open(os.path.join(conf.RESULTS_DIR, filename), 'w'),
        indent = '\t',
        sort_keys = True
    )
 
    log.debug('=====================================')
    log.debug(evaluationResultsDictionary)
    log.debug('=====================================')

from glob import glob as g
def computePicaiBinWidth():
    ranges = []

    if not os.path.exists('ranges.npy'):
        dataService = PicaiDataService()
        imagePaths = g(os.path.join(conf.PICAI_NIFTI_IMAGES_DIR, 'images/**'))
        for imagePath in imagePaths:
            print(imagePath)
            image = dataService.read(imagePath)
            ranges.append(int(np.max(image) - np.min(image)))
            del image

        ranges = np.asarray(ranges)
        np.save('ranges.npy', ranges)
    else:
        ranges = np.load('ranges.npy')

    meanRange = np.mean(ranges)

    binValues = []
    binWidthValues = []

    bins = 17
    binWidth = 1
    while bins > 16:
        bins = meanRange/binWidth
        if 16 <= bins <= 128:
            print(f'Bins: {int(bins)}, BinWidth: {binWidth}')
            binValues.append(int(bins))
            binWidthValues.append(binWidth)
        
        binWidth += 1

    binValues = np.asarray(list(binValues))
    binWidthValues = np.asarray(list(binWidthValues))

    print(binValues.shape)
    print(binWidthValues.shape)

    print(np.median(binValues))
    print(np.median(binWidthValues))

if __name__ == '__main__':
    # runNsclcEvaluation()
    # computePicaiBinWidth()
    runPicaiEvaluation()