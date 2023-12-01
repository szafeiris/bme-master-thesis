from ..utils.CustomJSONEncoder import CustomJSONEncoder
from ..evaluator import GridSearchNestedCVEvaluation
from ..utils.notification import send_to_telegram
from ..utils.config import PATHS, Datasets
from ..services import PicaiDataService
from .Pipeline import FullCombinationPipeline

from sklearn.model_selection import StratifiedShuffleSplit

import pandas as pd
import numpy as np
import json

class FusionCombinationPipeline(FullCombinationPipeline):        
    def __step_3_evaluate__(self):
        args = {
            'patientIds': self._state['patientIds'],
            'train_idx': self._state['train_index'], 'test_idx': self._state['test_index'], 
            'radiomicFeaturesNames': self._state['radiomicFeaturesNames'],
            'featureStart': 3, 'featureStep': 5, 'featureStop': 100,
        }
       
        evaluator = GridSearchNestedCVEvaluation(**args)
        evaluationResults = evaluator.evaluateAll(self._state['X'], self._state['y'], self._dataset)
        json.dump(evaluationResults, PATHS.getEvaluationResultDir(self._dataset).open('w'), cls=CustomJSONEncoder, sort_keys=True, indent=1)
        
        # try:
            # X, y, yStrat, patientIds, radiomicFeaturesNames, sufix, train_index, test_index = getBaseData(sufix)
     
            # Configure evaluation
            # evaluator = FusionFsEvaluator()
            # evaluationResults = evaluator.evaluate(X, y, yStrat, sufix=sufix)
            # json.dump(evaluationResults, open(f'{conf.RESULTS_DIR}/fusion_evaluation{sufix}.json', 'w'), cls=NumpyArrayEncoder, sort_keys=True, indent=1)
     
        # except Exception as e:
            # log.exception(e)
            # log.error('Exception occured in fusion: ' + str(e))
            # send_to_telegram('Exception occured in fusion: ' + str(e))
            # send_to_telegram(f"{'=' * 50}\n{str(e)}\n{'=' * 50}")
            # 
        self._state = {
            **self._state,
            'evaluationResults': evaluationResults
        }
        self._saveState()