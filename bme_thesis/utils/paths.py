from bme_thesis.utils.settings import bmeThesisSettings

from pathlib import Path

class Paths:
    def toPath(path):
        return Path(path)
    
    def join(path1, path2):
        return Paths.toPath(path1).joinpath(path2)
    
    def transformDataset( dataset: str = ''):
        return 'original' if dataset == '' else f'original_norm' if dataset == 'norm' else dataset
    
    def inverseTransformDataset( dataset: str = ''):
        return '' if dataset == 'original' else f'norm' if dataset == 'original_norm' else dataset
    
    def dataDir():
        return Paths.toPath(bmeThesisSettings.base_path).joinpath('data')
    
    def picaiDir():
        return Paths.dataDir().joinpath('PICAI')
    
    def resultsDir():
        return Paths.dataDir().joinpath('results')
    
    def analysisDir():
        return Paths.dataDir().joinpath('analysis')
    
    def rangesDir():
        return Paths.dataDir().joinpath('ranges')
    
    def getDatasetImagesDir(dataset: str = ""):
        dataset = dataset.strip()
        return Paths.picaiDir().joinpath('images').joinpath(Paths.transformDataset(dataset))
    
    def getRadiomicFile( dataset: str = ""):
        dataset = dataset.strip()
        Paths.dataDir().joinpath('radiomics').mkdir(exist_ok=True)
        return Paths.dataDir().joinpath('radiomics').joinpath(f"picai_radiomic_features.{dataset}.csv")
    
    def getResultsDir( dataset: str = ""):
        dataset = dataset.strip()
        Paths.resultsDir().mkdir(exist_ok=True)
        path = Paths.resultsDir().joinpath(Paths.transformDataset(dataset))
        path.mkdir(exist_ok=True)
        return path
    
    def getAnalysisDir( dataset: str = ""):
        dataset = dataset.strip()
        Paths.analysisDir().mkdir(exist_ok=True)
        path = Paths.analysisDir().joinpath(Paths.transformDataset(dataset))
        path.mkdir(exist_ok=True)
        path.joinpath('plots').mkdir(exist_ok=True)
        path.joinpath('plots').joinpath('classifiers').mkdir(exist_ok=True)
        path.joinpath('plots').joinpath('methods').mkdir(exist_ok=True)
        return path
    
    def getShapExplainerFile( dataset: str):
        return Paths.getAnalysisDir(dataset).joinpath(f'shap_values.{dataset.lower()}.save')
    
    def getShapValuePlotDir( dataset: str):
        path =  Paths.getAnalysisDir(dataset).joinpath('shap_values')
        path.mkdir(exist_ok=True)
        return path
    
    def getScoresCsvFile( dataset: str = ""):
        return Paths.resultsDir().joinpath(f'scores.{Paths.transformDataset(dataset)}.csv')
    
    def getResultsForCombinationDir( dataset: str = "", method: str = "", model: str = ""):
        return Paths.getResultsDir(dataset).joinpath(f'{method}_{model}.json')
    
    def getEvaluationResultDir( dataset: str = ""):
        dataset = dataset.strip()
        return Paths.resultsDir().joinpath(f'evaluation.{Paths.transformDataset(dataset)}.json')
    
    def getRangesFile( dataset: str = ""):
        dataset = dataset.strip()
        Paths.rangesDir().mkdir(exist_ok=True)
        return Paths.rangesDir().joinpath(f"ranges.{dataset}.json")
    
    def getMaskPathByPatientId( patientId: str = ''):
        maskPath = Paths.toPath(bmeThesisSettings.masks_path).joinpath(f'{patientId}.nii.gz')
        if maskPath.exists() and maskPath.is_file():
            return maskPath
        raise ValueError(f'Mask for patient with id: `{patientId}` does not exists')
    
    def getImagePathByDatasetAndPatientId( dataset: str = '', patientId: str = ''):
        maskPath = Paths.getDatasetImagesDir(dataset).joinpath(f'{patientId}_t2w.nii.gz')
        if maskPath.exists() and maskPath.is_file():
            return maskPath
        raise ValueError(f'Image for patient with id: `{patientId}` in dataset: `{dataset}` does not exists')