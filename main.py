from src.main.configurator import configurator as conf, log
from src.main.data import *
from src.main.evaluation import *
from src.main.notification import send_to_telegram

import os
from glob import glob as g

def runPicaiEvaluation(sufix=''): 
    try:
        dataService = PicaiDataService()

        if sufix != '' and sufix[0] != '_':
            sufix = '_' + sufix

        # Extract radiomics from images or read them
        radiomicsFileName = f'{conf.PICAI_RADIOMICS_FILE}{sufix}.csv'
        if os.path.exists(radiomicsFileName):
            radiomicFeatures = dataService.readRadiomics(radiomicsFileName)
        else:
            log.info(f'Generating radiomic features for `images{sufix}`.')
            binWidth, shiftValue = dataService.computeBinWidth(conf.PICAI_NIFTI_IMAGES_DIR, sufix=sufix)
            log.info(f'Selected bin Width is {binWidth} (Shift: {shiftValue}).')
            radiomicFeatures = dataService.extractRadiomics(conf.PICAI_NIFTI_IMAGES_DIR, radiomicsFileName, binWidth=binWidth, shiftValue=shiftValue, sufix=sufix)

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
            'featureStart': 3,
            'featureStep': 5,
            'featureStop': 100,
        }
        evaluator = GridSearchNestedCVEvaluation(**args)
        evaluationResults = evaluator.evaluateAll(X, y, yStrat, sufix=sufix)
        json.dump(evaluationResults, open(f'{conf.RESULTS_DIR}/evaluation{sufix}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
        json.dump(args, open(f'{conf.RESULTS_DIR}/evaluation{sufix}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
    
    except Exception as e:
        log.exception(e)
        log.error('Exception occured: ' + str(e))
        send_to_telegram('Exception occured: ' + str(e))
        send_to_telegram(f"{'=' * 50}\n{str(e)}\n{'=' * 50}")


if __name__ == '__main__':
    from multiprocessing import Process
    
    processes = [
        Process(target=runPicaiEvaluation, args=('',)),
        Process(target=runPicaiEvaluation, args=('_norm',)),
        Process(target=runPicaiEvaluation, args=('_n4',)),
        Process(target=runPicaiEvaluation, args=('_n4_norm',)),
        Process(target=runPicaiEvaluation, args=('_fat',)),
        Process(target=runPicaiEvaluation, args=('_muscle',)),
    ]
    for p in processes:
        p.start()
        
    for p in processes:
        p.join()
    
    # runPicaiEvaluation()