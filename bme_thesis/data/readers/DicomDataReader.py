from numpy.typing import NDArray
import pydicom

from bme_thesis.data.readers.DataReader import DataReader

class DicomDataReader(DataReader):
    def read(self, path='', returnObject: bool = False) -> NDArray:
        try:
            dcmImage = pydicom.dcmread(path)
            if returnObject:
                return dcmImage
            
            return dcmImage.pixel_array
        except Exception as e:
            self.log.error(f'Could not read dicom file.')
            self.log.exception(e)
            raise e