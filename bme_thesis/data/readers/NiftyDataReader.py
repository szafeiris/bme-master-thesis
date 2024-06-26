from numpy.typing import NDArray
import nibabel as nib

from bme_thesis.data.readers.DataReader import DataReader

class NiftyDataReader(DataReader):
    def read(self, path='', returnObject: bool = False) -> NDArray:
        try:
            niftiImage = nib.load(path)
            if returnObject:
                return niftiImage
            
            return niftiImage.get_fdata()
        except Exception as e:
            self.log.error(f'Could not read nifty file.')
            self.log.exception(e)
            raise e