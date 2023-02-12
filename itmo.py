from ITMO_FS.filters.univariate import select_k_best, UnivariateFilter, spearman_corr, pearson_corr
from ITMO_FS.filters.multivariate import *
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


# est = KBinsDiscretizer(n_bins=25, encode='ordinal')
# est.fit(X)
# X = est.transform(X)

## Univariate filter methods
print('spearman_corr')
ufilter = UnivariateFilter(spearman_corr, select_k_best(10))
ufilter.fit(X, y)
print(ufilter.selected_features)

print('pearson_corr')
ufilter = UnivariateFilter(pearson_corr, select_k_best(10))
ufilter.fit(X, y)
print(ufilter.selected_features)

# ## mRMR
model = MultivariateFilter('MRMR', 10)
print(model.__dict__)
model._MultivariateFilter__n_features = 2
print(model.__dict__)
model.fit(X, y)
print(model.selected_features)


# # Custom mrmr
# mrmr = mRMR()
# mrmr.fit(X, y)
# print(mrmr._selectedFeatures)

# methods = ['MIM', 'MRMR', 'JMI', 'CIFE', 'MIFS', 'CMIM', 'ICAP', 'DCSF', 'CFR', 'MRI', 'IWFS']
# for method in methods:
#     print(method)
#     model = MultivariateFilter(method, 5)
    # model.fit(X, y)
    # print(model.selected_features)
    # print(model.__dict__)
    # print('\n\n')


# ## DISR
# disr = DISRWithMassive(10)
# print(disr.fit_transform(X, y))
# print(disr)
# print(disr.selected_features)


