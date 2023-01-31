from src.main.configurator import configurator as conf
from src.main.data import *

## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

dicomDataService = DataService(DicomReader())

# Convert folders to nifty
# dicomDataService.convertToNifty(conf.NSCLC_IMAGES_DIR, conf.NIFTI_IMAGES_DIR)

# Extract radiomics from images
dicomDataService.extractRadiomics(conf.NIFTI_IMAGES_DIR, conf.RADIOMICS_FILE)