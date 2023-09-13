from typing import Optional
from pydantic_settings import BaseSettings

from pathlib import Path

from logging.config import dictConfig
import logging

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

configuration = Configuration()

class Datasets:
    ORIGINAL = ""
    ORIGINAL_NORMALIZED = "norm"
    N4 = "n4"
    N4_NORMALIZED = "n4_norm"
    FAT_NORMALIZED = "fat"
    MUSCLE_NORMALIZED = "muscle"
    
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
        
        self.RADIOMICS_DIR = self.DATA_DIR.joinpath("radiomics")
        self.RESULTS_DIR = self.DATA_DIR.joinpath("results")
        self.RANGES_DIR = self.DATA_DIR.joinpath("ranges")
    
    def getRadiomicFile(self, dataset: str = ""):
        dataset = dataset.strip()
        dataset = f"_{dataset}" if dataset != "" else dataset
        return self.RADIOMICS_DIR.joinpath(f"picai_radiomic_features{dataset}.csv")
    
    def getResultsDir(self, dataset: str = ""):
        dataset = dataset.strip()
        return self.RESULTS_DIR.joinpath(dataset)
    
    def getRangesFile(self, dataset: str = ""):
        dataset = dataset.strip()
        dataset = f"_{dataset}" if dataset != "" else dataset
        return self.RANGES_DIR.joinpath(f"ranges{dataset}.npy")

PATHS = Paths(configuration.BASE_DIR)

def getLoggingConfiguration():
    return {
                'version': 1,
                'disable_existing_loggers': True,
                'loggers': {
                    'bme-thesis': {
                        'level': configuration.LOG_LEVEL,
                        'handlers': ['console_handler', 'file_handler']
                    },
                },

                'handlers': {
                    'console_handler': {
                        'class': 'logging.StreamHandler',
                        'formatter': 'base_formatter',
                    },
                    'file_handler': {
                        'class': 'logging.handlers.TimedRotatingFileHandler',
                        'filename': PATHS.LOG_FILE,
                        'when': configuration.LOG_INTERVAL,
                        'interval': configuration.LOG_INTERVAL_COUNT, 
                        'backupCount': configuration.LOG_BACKUP_COUNT,
                        'formatter': 'file_formatter',
                    }
                },

                'formatters': {
                    'file_formatter': {
                        'format': '%(asctime)s [%(levelname)s] @ %(module)s/%(funcName)s: %(lineno)d| %(message)s',
                        'datefmt': '%d-%m-%Y %I:%M:%S'
                    },
                    'base_formatter': {
                        'format': '%(asctime)s [%(levelname)s]: %(message)s',
                        'datefmt': '%d-%m-%Y %I:%M:%S'
                    },
                }
            }

## Logging setup
dictConfig(getLoggingConfiguration())
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('radiomics').setLevel(logging.CRITICAL)
log = logging.getLogger('bme-thesis')