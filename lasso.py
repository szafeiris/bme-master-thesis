from src.main.configurator import configurator as conf
from src.main.data import *
from src.main.evaluation import *

from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler

from src.main.algorithm import *
import numpy as np


dataService = PicaiDataService()

# Extract radiomics from images or read them
if os.path.exists(conf.PICAI_RADIOMICS_FILE):
    radiomicFeatures = dataService.readRadiomics(conf.PICAI_RADIOMICS_FILE)
else:
    binWidth = dataService.computeBinWidth(conf.PICAI_NIFTI_IMAGES_DIR)
    # log.info(f'Bin Width is {binWidth}.')
    radiomicFeatures = dataService.extractRadiomics(conf.PICAI_NIFTI_IMAGES_DIR, conf.PICAI_RADIOMICS_FILE, binWidth=binWidth)

# Create labels for stratification
picaiMetadata = dataService.getMetadata(conf.PICAI_METADATA_PATH)    
jointDfs = pd.merge(picaiMetadata, radiomicFeatures, on='Patient_Id')
    
conditions = [
    (jointDfs['Label'] == 2) & (jointDfs['MagneticFieldStrength'] == 1.5),
    (jointDfs['Label'] == 2) & (jointDfs['MagneticFieldStrength'] == 3),
    (jointDfs['Label'] > 2)  & (jointDfs['MagneticFieldStrength'] == 1.5),
    (jointDfs['Label'] > 2)  & (jointDfs['MagneticFieldStrength'] == 3),
]
jointDfs['StratifiedLabels'] = np.select(conditions, [0, 1, 2, 3])
yStrat = jointDfs['StratifiedLabels'].to_numpy()
# log.debug(np.unique(yStrat, return_counts=True))

# Get features and original labels
patientIds = radiomicFeatures.pop('Patient_Id').to_list()
labels = radiomicFeatures.pop('Label')
radiomicFeaturesNames = radiomicFeatures.columns.to_list()
    
X = radiomicFeatures.to_numpy()
y = np.copy(labels)
y[y == 2] = 0   # 0: ISUP = 2,
y[y > 2] = 1    # 1: ISUP > 2


# featureNumbers = [select_k_best(int(a)) for a in np.arange(start=5, step=10, stop=1132)]
featureNumbers = [int(a) for a in np.arange(start=5, step=10, stop=1132)]

shuffleSplit = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
shuffleSplit.get_n_splits(X, yStrat)
train_index, test_index = next(shuffleSplit.split(X, yStrat)) 

X_train, X_test = X[train_index], X[test_index]
y_train, y_test = y[train_index], y[test_index]


## LASSO Implementation
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', Lasso()),
])

