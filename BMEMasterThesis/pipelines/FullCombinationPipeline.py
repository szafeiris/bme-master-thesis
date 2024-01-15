from sklearn.discriminant_analysis import StandardScaler

from ..algorithm import decodeMethod, decodeModel
from ..visualizer import PicaiVisualizer
from ..utils.CustomJSONEncoder import CustomJSONEncoder
from ..evaluator import EvaluationCombination, GridSearchNestedCVEvaluation
from ..utils.notification import send_to_telegram
from ..utils.config import PATHS, Datasets
from ..services import PicaiDataService
from .Pipeline import Pipeline

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline as sklearnPipeline


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
import shap

class FullCombinationPipeline(Pipeline):
    def __init__(self, dataset: str = None, dataService: PicaiDataService = PicaiDataService()) -> None:
        super().__init__(dataset, dataService)
        self._dataService = dataService
        if self._getCachedStateFile().exists():
            self._loadState()

    def __step_1_extractRadiomics__(self):
        self.radiomicsFile_ = PATHS.getRadiomicFile(self._dataset)
        if self.radiomicsFile_.exists() and self.radiomicsFile_.is_file():
            radiomics = pd.read_csv(self.radiomicsFile_)
            self._logger.info(f'Radiomics for `{self._dataset}` are loaded successfully')
            
            self._state = {
                **self._state,
                'radiomicsFile': self.radiomicsFile_,
                'radiomics': radiomics
            }
            
            self._saveState()
            return
        
        normalizeScale = None 
        if (self._dataset in [ Datasets.FAT_NORMALIZED,
                               Datasets.MUSCLE_NORMALIZED,
                               Datasets.N4_NORMALIZED,
                               Datasets.ORIGINAL_NORMALIZED
                             ]):
                normalizeScale = self.__normallizeScale
             
        if self.__isFixedBinWidth:
            binWidth, globalMin = self._dataService.generateBinWidth(self._dataset, self.__binCount, normalizeScale)
            normalizedGlobalMin = 0 if globalMin > 0 else -globalMin
            self._logger.info(f'Bin width: {binWidth}, Global Minimum (Normalized): {globalMin} ({normalizedGlobalMin})')
            
            radiomics = self._dataService.extractRadiomics(self._dataset, 
                                                           self.radiomicsFile_, 
                                                           binWidth=binWidth, 
                                                           shiftValue=normalizedGlobalMin, 
                                                           normalizeScale=normalizeScale)
        else:
            radiomics = self._dataService.extractRadiomics(self._dataset, 
                                                           self.radiomicsFile_,
                                                           binCount=self.__binCount,
                                                           normalizeScale=normalizeScale)
        
        self._state = {
            **self._state,
            'radiomicsFile': self.radiomicsFile_,
            'radiomics': radiomics
        }
        
        self._saveState()
        
    def __step_2_readData__(self):
        radiomicFeatures = self._state['radiomics']
        picaiMetadata = self._dataService.getMetadata()    
        
        jointDfs = pd.merge(picaiMetadata, radiomicFeatures, on='Patient_Id')
        conditions = [
            (jointDfs['Label'] == 2) & (jointDfs['Manufacturer'] == 'Philips Medical Systems'),
            (jointDfs['Label'] == 2) & (jointDfs['Manufacturer'] == 'SIEMENS'),
            (jointDfs['Label'] > 2)  & (jointDfs['Manufacturer'] == 'Philips Medical Systems'),
            (jointDfs['Label'] > 2)  & (jointDfs['Manufacturer'] == 'SIEMENS'),
        ]
    
        jointDfs['StratifiedLabels'] = np.select(conditions, [0, 1, 2, 3])
        yStrat = jointDfs['StratifiedLabels'].to_numpy()
    
        # Get features and original labels
        patientIds = radiomicFeatures.pop('Patient_Id').to_list()
        labels = radiomicFeatures.pop('Label')
        radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    
        X = radiomicFeatures.to_numpy()
        y = np.copy(labels)
        y[y == 2] = 0   # 0: ISUP = 2,
        y[y > 2] = 1    # 1: ISUP > 2
    
        if not PATHS.PICAI_INDICES_FILE.exists():
            stratifiedShuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.35, random_state=42)
            stratifiedShuffleSplit.get_n_splits(X, y)
            train_index, test_index = next(stratifiedShuffleSplit.split(X, yStrat))
            
            indicesData = {
                'train_idx': list([int(i) for i in train_index]),
                'test_idx': list([int(i) for i in test_index]),
            }
            
            json.dump(indicesData, PATHS.PICAI_INDICES_FILE.open('w'))
        else:
            indicesData = json.load(PATHS.PICAI_INDICES_FILE.open())
            train_index = np.asarray(indicesData['train_idx'])
            test_index = np.asarray(indicesData['test_idx'])

        self._state = {
            **self._state,
            'radiomicFeaturesNames': radiomicFeaturesNames,
            'X': X, 'y': y, 'yStrat': yStrat,
            'patientIds': patientIds,
            'train_index': train_index,
            'test_index': test_index,
        }
        
        self._saveState()
        
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
        
        self._state = {
            **self._state,
            'evaluationResults': evaluationResults
        }
        self._saveState()
    
    def __step_4_createScores__(self):
        yTrue = self._state['y'][self._state['test_index']]
        self._dataService.generateScoresFromResults(self._dataset, yTrue)
        scores = self._dataService.getScores(self._dataset)
        
        self._state = {
            **self._state,
            'scores': scores
        }
        self._saveState()
        
    def __step_5_visualize__(self):
        bestMethod, bestModel = PicaiVisualizer(dataService=self._dataService).visualizeDataset(self._dataset)
        self._state = {
            **self._state,
            'bestMethod': bestMethod,
            'bestModel': bestModel
        }
        self._saveState()
    
    def __step_6_calculate_SHAP_values__(self):
        shapValuePlotDir = PATHS.getShapValuePlotDir(self._dataset)
        shapValueFile = PATHS.getShapValueFile(self._dataset)
        if not shapValueFile.exists():
            bestMethod, bestModel = self._state['bestMethod'], self._state['bestModel']
            bestMethodResultFile = PATHS.getResultsForCombinationDir(self._dataset, bestMethod, bestModel)
            bestMethodParams = json.load(bestMethodResultFile.open())['best_method_params']
            method = decodeMethod(bestMethod)
            method.set_params(**bestMethodParams)
            pipeline = sklearnPipeline([
                ('standard_scaler', StandardScaler()),
                ('feature_selector', method),
                ('classifier', decodeModel(bestModel))
            ])

            pipeline.fit(self._state['X'][self._state['train_index']], self._state['y'][self._state['train_index']])
            explainer = shap.Explainer(pipeline.predict, self._state['X'][self._state['train_index']])
            shap_values = explainer.shap_values(self._state['X'][self._state['test_index']])
            json.dump(shap_values, shapValueFile.open('w'), cls=CustomJSONEncoder, indent=1)
        else:
            shap_values = json.load(shapValueFile.open())
        
        shap.summary_plot(shap_values, self._state['X'][self._state['test_index']], feature_names=self._state['radiomicFeaturesNames'], show=False)
        plt.tight_layout()
        plt.savefig(shapValuePlotDir.joinpath('.summary_plot.png'), format='png')
        plt.close()
        
        self._state = {
            **self._state,
            'shapValues': shap_values
        }
        self._saveState()
        
    def _unpackArgs(self, **kwargs):
        self.__isFixedBinWidth = kwargs['isFixedBinWidth'] if 'isFixedBinWidth' in kwargs else True
        self.__binCount = kwargs['binCount'] if 'binCount' in kwargs and isinstance(kwargs['binCount'], int) and kwargs['binCount']  > 0 else 32
        self.__normallizeScale = kwargs['normallizeScale'] if 'normallizeScale' in kwargs and isinstance(kwargs['normallizeScale'], int) and kwargs['normallizeScale']  > 0 else 100
        
    def run(self, **kwargs) -> None:
        try:
            self._logger.debug(f'Starting {self.__class__.__name__} for {self._dataset}')
            self._unpackArgs(**kwargs)
            self.__step_1_extractRadiomics__()
            self.__step_2_readData__()
            self.__step_3_evaluate__()
            self.__step_4_createScores__()
            self.__step_5_visualize__()
            self.__step_6_calculate_SHAP_values__()
            self._logger.debug(f'Pipeline {self.__class__.__name__}({self._dataset}) ended')
        except KeyboardInterrupt:
            self._logger.warning('Pipeline finished unexpectedly after keyboard interupt')
        except Exception as e:
            self._logger.error(f'An error occured during the execution of pipeline: {self.__class__.__name__}.')
            self._logger.exception(e)
            send_to_telegram(f'An error occured during the execution of pipeline: {self.__class__.__name__} [{str(e)}]')
        finally:
            self._saveState()
            del self._state