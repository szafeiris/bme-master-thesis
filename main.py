from src.main.configurator import configurator as conf
from src.main.data import NsclcRadiogenomicsDataService
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
    dataService = PicaiRadiogenomicsDataService()

    # Extract radiomics from images or read them
    if os.path.exists(conf.PICAI_RADIOMICS_FILE):
        radiomicFeatures = dataService.readRadiomics(conf.PICAI_RADIOMICS_FILE)
    else:
        radiomicFeatures = dataService.extractRadiomics(conf.PICAI_NIFTI_IMAGES_DIR, conf.PICAI_RADIOMICS_FILE)
    
    patientIds = radiomicFeatures.pop('Patient_Id').to_list()
    radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    X = radiomicFeatures.to_numpy()

    # Declare labels
    y = np.random.choice([0, 1], 220, p=[0.5, 0.5])
    
    evaluator = Evaluator()
    args = { 'patientIds': patientIds, 'radiomicFeaturesNames': radiomicFeaturesNames}
    evaluationResults = evaluator.evaluate(X, y, **args)
    for er in evaluationResults:
        log.debug(er)

if __name__ == '__main__':
    # runNsclcEvaluation()
    runPicaiEvaluation()