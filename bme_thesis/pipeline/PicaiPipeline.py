
from bme_thesis.data.feature_extractor.MultiLabelRadiomicExtractor import MultiLabelRadiomicExtractor
from bme_thesis.data.readers.SitkDataReader import SitkDataReader
from bme_thesis.logger import getLogger
from bme_thesis.pipeline.context import PicaiPipelineContext
from bme_thesis.utils.settings import bmeThesisSettings
from bme_thesis.utils.paths import Paths


class PicaiPipeline:
    def __init__(self, dataset: str = None) -> None:
        self.log = getLogger(self.__class__.__name__)
        self.radiomicsExtractor = MultiLabelRadiomicExtractor(bmeThesisSettings.pyradiomics_params_file)
        self.dataReader = SitkDataReader()
        self.ctx = PicaiPipelineContext(dataset=dataset)
        
        if (not isinstance(dataset, str)) or (dataset is None):
            raise AttributeError('`dataset` should be a string which cannot be empty or null')
        self.ctx.dataset = Paths.transformDataset(dataset)
    
    def run(self, *args, **kwargs) -> None:
            raise NotImplementedError()