from pathlib import Path
from typing import List, Optional
from pydantic_settings import BaseSettings


class Configuration(BaseSettings):
    BASE_DIR: str = "."
    
    # Logging settings
    LOG_LEVEL: str = "DEBUG"
    LOG_INTERVAL: str = "d"
    LOG_INTERVAL_COUNT: int = 1
    LOG_BACKUP_COUNT: int = 10
    LOG_FILENAME: str = "bme-master-thesis.log"

    # Pyradiomics Params.yml files
    PYRADIOMICS_PARAMS_FILE: str = "./resources/Params.yaml"

    # Telegram settings
    TELEGRAM_TOKEN: str = ""
    TELEGRAM_CHAT_ID: Optional[int] = None
    TELEGRAM_SEND: bool = False
    
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'

class Datasets:
    ORIGINAL: str = "original"
    ORIGINAL_NORMALIZED: str = "original_norm"
    N4: str = "n4"
    N4_NORMALIZED: str = "n4_norm"
    FAT_NORMALIZED: str = "fat"
    MUSCLE_NORMALIZED: str = "muscle"
    FAT_PIECEWISE_NORMALIZED: str = "fat_piecewise"
    MUSCLE_PIECEWISE_NORMALIZED: str = "muscle_piecewise"
    
    _ALL_DATASETS: List[str] = [ ORIGINAL, ORIGINAL_NORMALIZED, N4, N4_NORMALIZED, FAT_NORMALIZED, MUSCLE_NORMALIZED, FAT_PIECEWISE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
    _ORIGINAL_DATASETS: List[str] = [ ORIGINAL, ORIGINAL_NORMALIZED ]
    _N4_DATASETS: List[str] = [ N4, N4_NORMALIZED ]
    _FAT_DATASETS: List[str] = [ FAT_NORMALIZED, FAT_PIECEWISE_NORMALIZED ]
    _MUSCLE_DATASETS: List[str] = [ MUSCLE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
    _NORMALIZED_DATASETS: List[str] = [ ORIGINAL_NORMALIZED, N4_NORMALIZED, FAT_NORMALIZED, MUSCLE_NORMALIZED, FAT_PIECEWISE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
    _PIECEWISE_DATASETS: List[str] = [ FAT_PIECEWISE_NORMALIZED, MUSCLE_PIECEWISE_NORMALIZED ]
        
    def prettifyDataset(dataset: str):
        return dataset.upper().replace('_', ' ').replace('NORM', '(Normalized)').replace('PIECEWISE', '(Piece-Wise)')
    
    def simplifyDataset(dataset: str):
        return dataset.replace(' ', '_').replace('(Normalized)', 'NORM').replace('(Piece-Wise)', 'PIECEWISE').lower()
    
    
class Paths:
    def __init__(self, base_dir: str = ".") -> None:
        self.BASE_DIR = Path(base_dir)
        self.DATA_DIR = self.BASE_DIR.joinpath("data")
        self.LOGS_DIR = self.BASE_DIR.joinpath("logs")
        self.CACHE_DIR = self.BASE_DIR.joinpath(".cache")

        self.LOG_FILE = self.LOGS_DIR.joinpath(configuration.LOG_FILENAME)
                
        self.PICAI_DIR = self.DATA_DIR.joinpath("PICAI")
        self.PICAI_IMAGES_DIR = self.PICAI_DIR.joinpath("images")
        self.PICAI_MASKS_DIR = self.PICAI_DIR.joinpath("masks")
        self.PICAI_PATIENTS_ID_FILE = self.PICAI_DIR.joinpath("valid-patient-ids.txt")
        self.PICAI_METADATA_FILE = self.PICAI_DIR.joinpath("picai_metadata.csv")
        self.PICAI_INDICES_FILE = self.PICAI_DIR.joinpath("picai_indices.json")
        self.PICAI_RADIOMICS_NAMES_FILE = self.PICAI_DIR.joinpath("picai_feature_names.json")
        
        self.RADIOMICS_DIR = self.DATA_DIR.joinpath("radiomics")
        self.RESULTS_DIR = self.DATA_DIR.joinpath("results")
        self.RANGES_DIR = self.DATA_DIR.joinpath("ranges")
        self.ANALYSIS_DIR = self.DATA_DIR.joinpath("analysis")
    
    def getDatasetImagesDir(self, dataset: str = ""):
        dataset = dataset.strip()
        return self.PICAI_IMAGES_DIR.joinpath(self.transformDataset(dataset))
    
    def getRadiomicFile(self, dataset: str = ""):
        dataset = dataset.strip()
        self.RADIOMICS_DIR.mkdir(exist_ok=True)
        return self.RADIOMICS_DIR.joinpath(f"picai_radiomic_features.{dataset}.csv")
    
    def getResultsDir(self, dataset: str = ""):
        dataset = dataset.strip()
        self.RESULTS_DIR.mkdir(exist_ok=True)
        path = self.RESULTS_DIR.joinpath(self.transformDataset(dataset))
        path.mkdir(exist_ok=True)
        return path
    
    def getAnalysisDir(self, dataset: str = ""):
        dataset = dataset.strip()
        self.ANALYSIS_DIR.mkdir(exist_ok=True)
        path = self.ANALYSIS_DIR.joinpath(self.transformDataset(dataset))
        path.mkdir(exist_ok=True)
        path.joinpath('plots').mkdir(exist_ok=True)
        path.joinpath('plots').joinpath('classifiers').mkdir(exist_ok=True)
        path.joinpath('plots').joinpath('methods').mkdir(exist_ok=True)
        return path
    
    def getShapValueFile(self, dataset: str):
        return self.getAnalysisDir(dataset).joinpath(f'shap_values.{dataset.lower()}.shapv')
    
    def getShapValuePlotDir(self, dataset: str):
        path =  self.getAnalysisDir(dataset).joinpath('shap_values')
        path.mkdir(exist_ok=True)
        return path
    
    def getScoresCsvFile(self, dataset: str = ""):
        return self.RESULTS_DIR.joinpath(f'scores.{self.transformDataset(dataset)}.csv')
    
    def getResultsForCombinationDir(self, dataset: str = "", method: str = "", model: str = ""):
        return self.getResultsDir(dataset).joinpath(f'{method}_{model}.json')
    
    def getEvaluationResultDir(self, dataset: str = ""):
        dataset = dataset.strip()
        return self.RESULTS_DIR.joinpath(f'evaluation.{self.transformDataset(dataset)}.json')
    
    def getRangesFile(self, dataset: str = ""):
        dataset = dataset.strip()
        self.RANGES_DIR.mkdir(exist_ok=True)
        return self.RANGES_DIR.joinpath(f"ranges.{dataset}.json")
    
    def transformDataset(self, dataset: str = ''):
        return 'original' if dataset == '' else f'original_norm' if dataset == 'norm' else dataset
    
    def inverseTransformDataset(self, dataset: str = ''):
        return '' if dataset == 'original' else f'norm' if dataset == 'original_norm' else dataset
    
    def getMaskPathByPatientId(self, patientId: str = ''):
        maskPath = self.PICAI_MASKS_DIR.joinpath(f'{patientId}.nii.gz')
        if maskPath.exists() and maskPath.is_file():
            return maskPath
        raise ValueError(f'Mask for patient with id: `{patientId}` does not exists')
    
    def getImagePathByDatasetAndPatientId(self, dataset: str = '', patientId: str = ''):
        maskPath = self.getDatasetImagesDir(dataset).joinpath(f'{patientId}_t2w.nii.gz')
        if maskPath.exists() and maskPath.is_file():
            return maskPath
        raise ValueError(f'Image for patient with id: `{patientId}` in dataset: `{dataset}` does not exists')

    

configuration = Configuration()
PATHS = Paths(configuration.BASE_DIR)
