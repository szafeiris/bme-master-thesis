from . import configurator as conf
import abc
import numpy as np
import glob
import os
import pydicom
import nibabel as nib
import SimpleITK as sitk


## Logging setup
from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger()

class DataReader:
    @abc.abstractmethod
    def read(self, path='') -> np.array:
        pass

    def readDirectory(self, path='') -> np.ndarray:
        if path[-1] != '*':
            path = os.path.join(path, '*')
        
        data = []
        try:    
            for dcm in glob.glob(path):
                if os.path.isfile(dcm):
                    data.append(self.read(dcm))
        except Exception as e:
            log.error(f'Could not read from folder.')
            log.exception(e)
            raise e
        finally:
            return np.array(data)

    def readDirectories(self, path='') -> np.ndarray:
        if isinstance(path, str):
            if path[-1] != '*':
                path = os.path.join(path, '*')
            
            directories = glob.glob(path)
            self.readDirectories(directories)

        elif isinstance(path, list) and isinstance(path[0], str):
            data = []
            try:    
                for dcm in path:
                    data.append(self.readDirectory(dcm))
            except Exception as e:
                log.error(f'Could not read from multiple folders.')
                log.exception(e)
                raise e
            finally:
                return np.array(data)
        else:
            raise AttributeError('`path` should be a string or a list of strings')

class DicomReader(DataReader):
    def read(self, path='') -> np.array:
        log.debug('Read dicom file')
        try:
            dcmimg = pydicom.dcmread(path)
            return dcmimg.pixel_array
        except Exception as e:
            log.error(f'Could not read dicom file.')
            log.exception(e)
            raise e
    
    def readSegmentation(self, path=''):
        log.debug('Read compact dicom segmentation file')
        try:
            dcmimg = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(dcmimg)
        except Exception as e:
            log.error(f'Could not read compact dicom file.')
            log.exception(e)
            raise e

class NiftyReader(DataReader):
    def read(self, path='') -> np.array:
        log.debug('Read nii file')
        try:
            niftiImage = nib.load(path)
            return niftiImage.get_data()
        except Exception as e:
            log.error(f'Could not read dicom file.')
            log.exception(e)
            raise e

class DataService:
    def __init__(self, dataReader: DataReader) -> None:
        self.__dataReader = dataReader
    
    def setDataReader(self, dataReader: DataReader):
        self.__dataReader = dataReader
        return self
    
    def read(self, path):
        if not isinstance(path, str):
            raise AttributeError('`path` should be a string')

        if os.path.isfile(path):
            return self.dataReader.read(path)
        elif os.path.isdir(path):
            return self.dataReader.readDirectory(path)
        
        raise RuntimeError('there is nothing to read')
    
    def readSegmentation(self, path):
        if not isinstance(self.__dataReader, DicomReader):
            raise RuntimeError('cannot read segmentation (DicomDataReader is needed)')
        
        return self.__dataReader.readSegmentation(path)
           
