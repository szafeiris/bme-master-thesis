from numpy.typing import NDArray
import SimpleITK as sitk

from bme_thesis.data.readers.DataReader import DataReader

class SitkDataReader(DataReader):
    def read(self, path='', returnObject: bool = False) -> NDArray:
        try:
            image = sitk.ReadImage(path)
            if returnObject:
                return image
            
            return sitk.GetArrayFromImage(image)
        except Exception as e:
            self.log.error(f'Could not read nifty file.')
            self.log.exception(e)
            raise e