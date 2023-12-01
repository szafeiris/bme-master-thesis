from ..utils.config import PATHS
from ..services import DataService
from ..utils.log import getLogger
from ..utils.CustomJSONEncoder import CustomJSONEncoder

import json

class Pipeline:
    def __init__(self, dataset: str = None, dataService: DataService = None) -> None:
        if (not isinstance(dataset, str)) or (dataset is None):
            raise AttributeError('`dataset` should be a string which cannot be empty or null')
        self._logger = getLogger(self.__class__.__name__)
        self._dataset = PATHS.transformDataset(dataset)
        self._dataService = dataService
        self._state = {}
    
    def _getCachedStateFilename(self):
        cachedStatePath = PATHS.CACHE_DIR.joinpath(f'{self.__class__.__name__}.{self._dataset}')
        cachedStatePath.mkdir(exist_ok= True)
        cachedStateFile = cachedStatePath.joinpath('.cached_state')
        return cachedStateFile
    
    def _saveState(self):
        cachedStateFile = self._getCachedStateFilename()
        json.dump(self._state, cachedStateFile.open('w'), cls=CustomJSONEncoder, sort_keys=True, indent=10)
    
    def _loadState(self):
        cachedStateFile = self._getCachedStateFilename()
        self._state = json.load(cachedStateFile.open())
        
    def run(self, *args, **kwargs) -> None:
        raise NotImplementedError()
