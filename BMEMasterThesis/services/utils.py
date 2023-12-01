from typing import List

from ..utils.log import getLogger

from numpy.typing import NDArray
import pydicom
import nibabel as nib
import abc
import numpy as np
from pathlib import Path
import SimpleITK as sitk



class DataReader:
    def __init__(self) -> None:
        self.log = getLogger(__name__)
        
    @abc.abstractmethod
    def read(self, path: str | Path = '') -> NDArray:
        if isinstance(path, str):
            path = Path(path)

    def readDirectory(self, path: str | Path = '') -> NDArray:
        if isinstance(path, str):
            path = Path(path)
        
        data = []
        try:
            for file in path.glob('*'):
                if file.is_file():
                    data.append(self.read(file))
        except Exception as e:
            self.log.error(f'Could not read from folder.')
            self.log.exception(e)
            raise e
        finally:
            return np.array(data)

    def readDirectories(self, path: str | Path | List[str] | List[Path] = '') -> NDArray:
        if isinstance(path, str):
            path = Path(path)
            
            directories = list(path.glob('*'))
            return self.readDirectories(directories)

        elif isinstance(path, list) and (isinstance(path[0], str) or isinstance(path[0], Path)):
            data = []
            try:    
                for dcm in path:
                    data.append(self.readDirectory(dcm))
            except Exception as e:
                self.log.error(f'Could not read from multiple folders.')
                self.log.exception(e)
                raise e
            finally:
                return np.asarray(data, dtype=object)
        else:
            raise AttributeError('`path` should be a string or a list of strings')


class DicomReader(DataReader):
    def read(self, path='') -> NDArray:
        try:
            dcmImage = pydicom.dcmread(path)
            return dcmImage.pixel_array
        except Exception as e:
            self.log.error(f'Could not read dicom file.')
            self.log.exception(e)
            raise e

class NiftyReader(DataReader):
    def read(self, path='') -> NDArray:
        try:
            niftiImage = nib.load(path)
            niftiImageFinal = niftiImage.get_fdata()
            return niftiImageFinal
        except Exception as e:
            self.log.error(f'Could not read nifty file.')
            self.log.exception(e)
            raise e

class SitkReader(DataReader):
    def read(self, path='') -> NDArray:
        try:
            image = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            self.log.error(f'Could not read image file with SimpleITK.')
            self.log.exception(e)
            raise e

