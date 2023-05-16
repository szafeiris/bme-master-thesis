import os
import dicom2nifti as d2n
import SimpleITK as sitk

class NiftyConverter:
    # TODO: Correct the check
    def __checkInput__(self, inputPath, outputPath):
        pass
        if (inputPath is None) and not (os.path.isfile(inputPath) or os.path.isdir(inputPath)):
            raise AttributeError('`inputPath` should be a valid filename')
        
        if (outputPath is None) and not (os.path.isfile(outputPath) or os.path.isdir(outputPath)):
            raise AttributeError('`outputPath` should be a valid filename')

    def convert(self, inputPath, outputPath):
        self.__checkInput__(inputPath, outputPath)
        d2n.dicom_series_to_nifti(inputPath, outputPath)

    def convertSegmentation(self, inputPath, outputPath):
        self.__checkInput__(inputPath, outputPath)
        image = sitk.ReadImage(inputPath)
        sitk.WriteImage(image, outputPath)
