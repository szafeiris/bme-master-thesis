from src.main.configurator import configurator as conf
from src.main.data import *

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

dicomDataService = DataService(DicomReader)