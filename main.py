from src.main.configurator import configurator as conf, log
from src.main.data import *
from src.main.evaluation import *
from src.main.notification import send_to_telegram

import os
from glob import glob as g

def runPicaiEvaluation():    
    dataService = PicaiDataService()

    # Extract radiomics from images or read them
    if os.path.exists(conf.PICAI_RADIOMICS_FILE):
        radiomicFeatures = dataService.readRadiomics(conf.PICAI_RADIOMICS_FILE)
    else:
        binWidth = dataService.computeBinWidth(conf.PICAI_NIFTI_IMAGES_DIR)
        # log.info(f'Bin Width is {binWidth}.')
        radiomicFeatures = dataService.extractRadiomics(conf.PICAI_NIFTI_IMAGES_DIR, conf.PICAI_RADIOMICS_FILE, binWidth=binWidth)
    
    # Create labels for stratification
    picaiMetadata = dataService.getMetadata(conf.PICAI_METADATA_PATH)    
    jointDfs = pd.merge(picaiMetadata, radiomicFeatures, on='Patient_Id')
    
    conditions = [
        (jointDfs['Label'] == 2) & (jointDfs['Manufacturer'] == 'Philips Medical Systems'),
        (jointDfs['Label'] == 2) & (jointDfs['Manufacturer'] == 'SIEMENS'),
        (jointDfs['Label'] > 2)  & (jointDfs['Manufacturer'] == 'Philips Medical Systems'),
        (jointDfs['Label'] > 2)  & (jointDfs['Manufacturer'] == 'SIEMENS'),
    ]
    jointDfs['StratifiedLabels'] = np.select(conditions, [0, 1, 2, 3])
    yStrat = jointDfs['StratifiedLabels'].to_numpy()
    
    # jointDfs2 = jointDfs[['Label', 'MagneticFieldStrength', 'StratifiedLabels', 'Patient_Id', 'Manufacturer']]
    # jointDfs2.to_csv('joined_metadata.csv', index=False)
    # log.debug(np.unique(yStrat, return_counts=True))

    # Get features and original labels
    patientIds = radiomicFeatures.pop('Patient_Id').to_list()
    labels = radiomicFeatures.pop('Label')
    radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    
    X = radiomicFeatures.to_numpy()
    y = np.copy(labels)
    y[y == 2] = 0   # 0: ISUP = 2,
    y[y > 2] = 1    # 1: ISUP > 2

    # Configure evaluation
    args = {
        'patientIds': patientIds,
        'radiomicFeaturesNames': radiomicFeaturesNames,
        'featureStart': 5,
        'featureStep': 10,
        'featureStop': 225,
    }
    evaluator = GridSearchNestedCVEvaluation(**args)
    send_to_telegram("Evaluation started.")
    # evaluationResults = evaluator.evaluateAll(X, y, yStrat)
    evaluationResults = evaluator.evaluateSingle(X, y, yStrat, 'pearson', 'svm-linear')
    json.dump(evaluationResults, open(f'{conf.RESULTS_DIR}/evaluation.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
    send_to_telegram("Evaluation ended.")
    
if __name__ == '__main__':
    try:
        runPicaiEvaluation()
    except Exception as e:
        log.exception(e)
        send_to_telegram('Exception occured: ' + str(e))
    