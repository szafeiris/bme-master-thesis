import abc
from pathlib import Path
from typing import List
from numpy.typing import NDArray
import numpy as np

from bme_thesis.logger import getLogger

class DataReader:
    def __init__(self) -> None:
        self.log = getLogger(__name__)
        
    @abc.abstractmethod
    def read(self, path: str | Path = '', returnObject: bool = False) -> NDArray:
        raise NotImplementedError()

    def readDirectory(self, path: str | Path = '', returnObject: bool = False) -> NDArray:
        if isinstance(path, str):
            path = Path(path)
        
        data = []
        try:
            for file in path.glob('*'):
                if file.is_file():
                    data.append(self.read(file, returnObject))
        except Exception as e:
            self.log.error(f'Could not read from folder.')
            self.log.exception(e)
            raise e
        finally:
            return np.array(data)

    def readDirectories(self, path: str | Path | List[str] | List[Path] = '', returnObject: bool = False) -> NDArray:
        if isinstance(path, str):
            path = Path(path)
            
            directories = list(path.glob('*'))
            return self.readDirectories(directories, returnObject)

        elif isinstance(path, list) and (isinstance(path[0], str) or isinstance(path[0], Path)):
            data = []
            try:    
                for p in path:
                    data.append(self.readDirectory(p, returnObject))
            except Exception as e:
                self.log.error(f'Could not read from multiple folders.')
                self.log.exception(e)
                raise e
            finally:
                return np.asarray(data, dtype=object)
        else:
            raise AttributeError('`path` should be a string or a list of strings')