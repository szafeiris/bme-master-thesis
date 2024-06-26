from bme_thesis.utils.settings import bmeThesisSettings
from bme_thesis.utils.paths import Paths

from pathlib import Path
from logging.config import dictConfig
import logging

_LOGS_PATH = Path(bmeThesisSettings.log_path).joinpath(bmeThesisSettings.log_filename)
_LOG_CONFIG = {
    'version': 1,
    'disable_existing_loggers': True,
    'loggers': {
        'bme_thesis.utils.notification': {
            'level': logging.WARNING,
            'handlers': ['console_handler', 'file_handler'],
            'propagate': False
        },
        'bme_thesis': {
            'level': bmeThesisSettings.log_level,
            'handlers': ['console_handler', 'file_handler'],
            'propagate': False
        },
        '': {
            'level': logging.WARNING,
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
            'filename': str(_LOGS_PATH),
            'when': bmeThesisSettings.log_interval,
            'interval': bmeThesisSettings.log_interval_count,
            'backupCount': bmeThesisSettings.log_backup_count,
            'formatter': 'file_formatter',
        }
    },

    'formatters': {
        'file_formatter': {
            # 'format': '[%(asctime)s] LOGGER:%(name)s: %(levelname)s  in %(processName)s/%(threadName)s/%(module)s/%(funcName)s:%(lineno)d| %(message)s',
            'format': '[%(asctime)s] %(levelname)s  in %(processName)s/%(threadName)s/%(module)s/%(funcName)s:%(lineno)d| %(message)s',
            'datefmt': '%d-%m-%Y %I:%M:%S'
        },
        'base_formatter': {
            'class': 'coloredlogs.ColoredFormatter',
            'format': '[%(asctime)s] LOGGER:%(name)s: %(levelname)s  in %(processName)s/%(threadName)s/%(module)s/%(funcName)s:%(lineno)d| %(message)s',
            'datefmt': '%d-%m-%Y %I:%M:%S'
        },
    }
}

## Logging setup
dictConfig(_LOG_CONFIG)

def getLogger(loggerName=''):
    if loggerName != '':
        loggerName = f'.{loggerName.replace("bme_thesis", "")}'
    return logging.getLogger(f'bme_thesis{loggerName}')

__all__ = [getLogger]