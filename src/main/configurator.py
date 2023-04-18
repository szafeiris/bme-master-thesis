import os 
import re
from logging.config import dictConfig
import logging

class Configurator:
    _SettingCommentDelimiter_ = '#'

    def __init__(self, confFile=f'{os.path.dirname(__file__)}/../resources/settings.conf') -> None:
        try:
            with open(confFile, 'r') as configurationFile:
                for settingPair in configurationFile:
                    settingPair = settingPair.lstrip().rstrip('\n')
                    if settingPair == '':
                        continue
                    if not settingPair.startswith(self._SettingCommentDelimiter_):
                        settingName, settingValue = settingPair.split('=')
                        settingValue = settingValue.split(self._SettingCommentDelimiter_)[0].rstrip()
                        settingValue = self.__proccessQuery(settingValue)
                        setattr(self, settingName, settingValue)
        except Exception as e:
            print('Could not load settings configuration.', f'Error: {e}')
            exit(-1)
        
        self._LOGGING_CONFIG_ = {
            'version': 1,
            'disable_existing_loggers': True,
            'loggers': {
                '': {
                    'level': self.LOG_LEVEL,
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
                    'filename': f'{os.path.dirname(__file__)}/../../logs/bme-master-thesis.log',
                    'when': self.LOG_INTERVAL,
                    'interval': int(self.LOG_INTERVAL_COUNT), 
                    'backupCount': int(self.LOG_BACKUP_COUNT),
                    'formatter': 'file_formatter',
                }
            },

            'formatters': {
                'file_formatter': {
                    'format': '%(asctime)s %(levelname)s in %(module)s/%(funcName)s:%(lineno)d| %(message)s',
                    'datefmt': '%d-%m-%Y %I:%M:%S'
                },
                'base_formatter': {
                    'format': '%(message)s',
                    'datefmt': '%d-%m-%Y %I:%M:%S'
                },


            }
        }

    def getAll(self):
        return self.__dict__

    def __contains__(self, key):
        return key in self.__dict__
    
    def __proccessQuery(self, value: str):
        if ('{' not in value) and ('}' not in value):
            if value.lower() == 'true' or value.lower() == 'false':
                return bool('True' if value.lower() == 'true' else '')

            return value

        vals = re.findall('.*{(.*)}.*', value)
        for v in vals:
            value = value.replace('{' + v + '}', getattr(self, v))
        
        return self.__proccessQuery(value)

configurator = Configurator()

## Logging setup
dictConfig(configurator._LOGGING_CONFIG_)
logging.getLogger('requests').setLevel(logging.CRITICAL)
log = logging.getLogger()