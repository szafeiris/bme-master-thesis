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
    log.debug(np.unique(y, return_counts=True))
    y[y == 2] = 0   # 0: ISUP = 2,
    y[y > 2] = 1    # 1: ISUP > 2

    u, uc = np.unique(y, return_counts=True)
    log.debug(u)
    log.debug(uc)
    
    evaluator = Evaluator()
    args = { 'patientIds': patientIds, 'radiomicFeaturesNames': radiomicFeaturesNames}
    evaluationResults = evaluator.evaluate(X, y, **args)
    for er in evaluationResults:
        log.debug(er)

if __name__ == '__main__':
    # runNsclcEvaluation()
    runPicaiEvaluation()