from .config import configuration, PATHS
from logging.config import dictConfig
import logging

LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'loggers': {
        'bme-thesis.BMEMasterThesis.utils.notification': {
            'level': logging.WARNING,
            'handlers': ['console_handler', 'file_handler'],
            'propagate': True
        },
        'bme-thesis': {
            'level': configuration.LOG_LEVEL,
            'handlers': ['console_handler', 'file_handler'],
            'propagate': False
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
            'format': '[%(asctime)s] LOGGER:%(name)s: %(levelname)s  in %(processName)s/%(threadName)s/%(module)s/%(funcName)s:%(lineno)d| %(message)s',
            'datefmt': '%d-%m-%Y %I:%M:%S'
        },
        'base_formatter': {
            'class': 'coloredlogs.ColoredFormatter',
            'format': '[%(asctime)s] %(processName)s/%(threadName)s %(levelname)s: %(message)s',
            'datefmt': '%d-%m-%Y %I:%M:%S'
        },
    }
}

## Logging setup
dictConfig(LOG_CONFIG)

def getLogger(loggerName=''):
    if loggerName != '':
        loggerName = f".{loggerName}".replace("bme-thesis", "")
    
    return logging.getLogger(f'bme-thesis{loggerName}')

log = getLogger()