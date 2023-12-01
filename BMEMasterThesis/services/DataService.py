from BMEMasterThesis.services.utils import DataReader
from BMEMasterThesis.extractor import BasicRadiomicExtractor

from pathlib import Path

class DataService:
    def __init__(self, dataReader: DataReader = None, radiomicsExtractor: BasicRadiomicExtractor = None) -> None:
        self._dataReader = dataReader
        self._radiomicsExtractor = radiomicsExtractor
