from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
import os
from src.main.configurator import configurator as conf
from src.main.data import PicaiDataService
from src.main.algorithm import mRMR

dataService = PicaiDataService()

# Read radiomics data
radiomicFeatures = dataService.readRadiomics(conf.PICAI_RADIOMICS_FILE)

# Get patient ids, feature names
patientIds = radiomicFeatures.pop('Patient_Id').to_list()
radiomicFeaturesNames = radiomicFeatures.columns.to_list()

X = radiomicFeatures.to_numpy()
y = dataService.getDatasetLabels(conf.PICAI_NIFTI_IMAGES_DIR)

clf = make_pipeline(ReliefF(n_features_to_select=2, n_neighbors=100),
                    RandomForestClassifier(n_estimators=100))

print(np.mean(cross_val_score(clf, X, y)))