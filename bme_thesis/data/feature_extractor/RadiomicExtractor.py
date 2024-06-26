from abc import abstractmethod, ABC
from typing import Dict, List

from bme_thesis.logger import getLogger


class RadiomicExtractor(ABC):
    def __init__(self, paramFile: str) -> None:
        self.paramFile = paramFile
        self.log = getLogger(__name__)
    
    
    @abstractmethod
    def extract(self, csvData, **kwargs) -> List[Dict[str, any]]:
        raise NotImplementedError(f'Extraction is not implemented in class {__name__}')