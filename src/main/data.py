from . import configurator as conf, log
from .converter import NiftyConverter
from .extractor import RadiomicExtractor, MultiLabelRadiomicExtractor
import abc
import numpy as np
import glob
import os
import pydicom
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from json import JSONEncoder


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)

class DataReader:
    @abc.abstractmethod
    def read(self, path='') -> np.array:
        pass

    def readSegmentation(self, path=''):
        log.debug('Read compact segmentation file')
        try:
            img = sitk.ReadImage(path)
            return sitk.GetArrayFromImage(img)
        except Exception as e:
            log.error(f'Could not read compact file.')
            log.exception(e)
            raise e

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
            return self.readDirectories(directories)

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
                return np.asarray(data, dtype=object)
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

class NiftyReader(DataReader):
    def read(self, path='') -> np.array:
        log.debug('Read nii file')
        try:
            niftiImage = nib.load(path)
            return niftiImage.get_fdata()
        except Exception as e:
            log.error(f'Could not read dicom file.')
            log.exception(e)
            raise e

class RadiomicReader:
    def readCsv(self, path='') -> np.array:
        log.debug('Read csv radiomic file')
        try:
            return pd.read_csv(path)
        except Exception as e:
            log.error(f'Could not read dicom file.')
            log.exception(e)
            raise e

class DataService:
    def __init__(self, dataReader: DataReader, 
                       dataConverter: NiftyConverter = NiftyConverter(),
                       radiomicsExtractor = RadiomicExtractor(),
                       radiomicReader = RadiomicReader()) -> None:
        self._dataReader = dataReader
        self._dataConverter = dataConverter
        self._radiomicsExtractor = radiomicsExtractor
        self._radiomicReader = radiomicReader
    
    def setDataReader(self, dataReader: DataReader):
        self._dataReader = dataReader
        return self
    
    def setDataConverter(self, dataConverter: NiftyConverter):
        self._dataConverter = dataConverter
        return self
    
    def setRadiomicExtractor(self, radiomicExtractor: RadiomicExtractor):
        self._radiomicsExtractor = radiomicExtractor
        return self

    def setRadiomicReader(self, radiomicReader: RadiomicReader):
        self._radiomicReader = radiomicReader
        return self
    
    def read(self, path):
        if not isinstance(path, str):
            raise AttributeError('`path` should be a string')

        if os.path.isfile(path):
            return self._dataReader.read(path)
        elif os.path.isdir(path):
            return self._dataReader.readDirectory(path)
        
        raise RuntimeError('there is nothing to read')
    
    def readSegmentation(self, path):
        return self._dataReader.readSegmentation(path)
    
    @abc.abstractmethod
    def convertToNifty(self, inputPath, outputPath):
        pass
    
    @abc.abstractmethod
    def extractRadiomics(self, imageFolder, outputCsvFile=None, keepDiagnosticsFeatures = False):
        pass

    def readRadiomics(self, csvPath):
        return self._radiomicReader.readCsv(csvPath)


# class NsclcRadiogenomicsDataService(DataService):
#     def __init__(self, dataReader: DataReader = DicomReader(),
#                        dataConverter: NiftyConverter = NiftyConverter(),
#                        radiomicsExtractor=RadiomicExtractor(conf.NSCLC_PYRADIOMICS_PARAMS_FILE),
#                        radiomicReader=RadiomicReader()) -> None:
#         super().__init__(dataReader, dataConverter, radiomicsExtractor, radiomicReader)
        
#     def convertToNifty(self, inputPath, outputPath):
#         patients = glob.glob(os.path.join(inputPath, '**\\**'))
#         for patient in patients:
#             series = glob.glob(os.path.join(patient, '**'))
#             patientCode = patient.split('\\')[-2]
#             for s in series:
#                 output = os.path.join(outputPath, patientCode)
#                 if not os.path.exists(outputPath):
#                     os.mkdir(outputPath)
#                 if not os.path.exists(output):
#                     os.mkdir(output)
#                 if 'segmentation' in s:
#                     self._dataConverter.convertSegmentation(glob.glob(os.path.join(s, '**'))[0], os.path.join(output, s.split('\\')[-1] + '.nii'))
#                 else:
#                     self._dataConverter.convert(s, os.path.join(output, s.split('\\')[-1] + '.nii'))
           
#     def extractRadiomics(self, imageFolder, outputCsvFile=None, keepDiagnosticsFeatures = False):
#         csvData = {
#             'Image': [],
#             'Mask': [],
#             'Patient ID': []
#         }

#         log.info("Gathering image data...")
#         patients = glob.glob(os.path.join(imageFolder, '**'))
#         for patient in patients:
#             patientCode = patient.split('\\')[-1]
#             csvData['Patient ID'].append(patientCode)

#             series = glob.glob(os.path.join(patient, '*'))
#             for s in series:
#                 if 'segmentation' in s:
#                     csvData['Mask'].append(s)
#                 else:
#                     csvData['Image'].append(s)

#         log.info("Extracting radiomics features...")
#         radiomicFeaturesDataframe = self._radiomicsExtractor.extractFromCsv(csvData, keepDiagnosticsFeatures=keepDiagnosticsFeatures)
#         if outputCsvFile is not None:
#             log.info('Saving radiomics file.')
#             radiomicFeaturesDataframe.to_csv(outputCsvFile, index=False)

#         return radiomicFeaturesDataframe

class PicaiDataService(DataService):
    def __init__(self, dataReader: DataReader = NiftyReader(),
                       dataConverter: NiftyConverter = NiftyConverter(),
                       radiomicsExtractor=MultiLabelRadiomicExtractor(conf.PICAI_PYRADIOMICS_PARAMS_FILE),
                       radiomicReader=RadiomicReader()) -> None:
        super().__init__(dataReader, dataConverter, radiomicsExtractor, radiomicReader)
        
    def convertToNifty(self, inputPath, outputPath):
        pass
           
    def extractRadiomics(self, imageFolder, outputCsvFile=None, keepDiagnosticsFeatures=False, binWidth=11, sufix=''):
        csvData = {
            'Image': [],
            'Mask': [],
            'Patient ID': []
        }

        log.info("Gathering image data...")
        masks = glob.glob(os.path.join(os.path.join(imageFolder, 'masks'), '**'))
        for mask in masks:
            patientCode = mask.split(os.sep)[-1].replace('.nii.gz', '')
            csvData['Patient ID'].append(patientCode)

            csvData['Mask'].append(os.path.join(os.path.join(imageFolder, 'masks'), patientCode + '.nii.gz'))
            csvData['Image'].append(os.path.join(os.path.join(imageFolder, f'images{sufix}'), patientCode + '_t2w.nii.gz'))

        log.info("Extracting radiomics features...")
        
        radiomicFeaturesDataframe = self._radiomicsExtractor.extractFromCsv(csvData,
                                                                            keepDiagnosticsFeatures=keepDiagnosticsFeatures,
                                                                            settings={"binWidth": binWidth})
        if outputCsvFile is not None:
            log.info('Saving radiomics file.')
            radiomicFeaturesDataframe.to_csv(outputCsvFile, index=False)

        return radiomicFeaturesDataframe
    
    def getMetadata(self, path):
        metadata = pd.read_csv(path)
        return metadata
    
    def computeBinWidth(self, path=conf.PICAI_NIFTI_IMAGES_DIR, bins=32, sufix=''):
        if not os.path.exists(f'ranges{sufix}.npy'):
            ranges = []
            with open(f'{path}/data.txt', 'r') as idFile:
                for id in idFile:
                    id = id.rstrip('\n')
                    imageFile = f'{path}/images{sufix}/{id}_t2w.nii.gz'
                    maskFile = f'{path}/masks/{id}.nii.gz'
                    
                    image = self.read(imageFile)
                    mask = self.read(maskFile)
                    mask = mask > 0

                    tempImage = image * mask
                    tempImage = tempImage[tempImage > 0]
                    ranges.append(int(np.max(tempImage) - np.min(tempImage)))
                    
                    del image
                    del mask
                    del tempImage

            ranges = np.asarray(ranges)
            np.save(f'ranges{sufix}.npy', ranges)
        else:
            ranges = np.load(f'ranges{sufix}.npy')

        meanRange = float(np.mean(ranges))
        binWidth = meanRange/bins
    
        return round(binWidth)
        