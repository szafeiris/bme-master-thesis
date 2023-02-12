import pytest
import os
from main.data import *
from main.configurator import configurator as conf

sample_dicom_image = os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/3.000000-THORAX 1.0 B45f-95741/1-001.dcm')
sample_dicom_directory = os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/3.000000-THORAX 1.0 B45f-95741/')
sample_dicom_directories_list = [
    os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/3.000000-THORAX 1.0 B45f-95741/'),
    os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/R01-002/09-20-1990-NA-CT THORAX-55296/3.000000-CHEST 1.25 MM-41459/')
]
sample_dicom_directories_string = os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/')
sample_dicom_segmentation = os.path.join(conf.IMAGES_DIR, 'NSCLC_Radiogenomics/R01-001/09-06-1990-NA-CT CHEST ABD PELVIS WITH CON-98785/1000.000000-3D Slicer segmentation result-67652/1-1.dcm')
sample_nii_image = os.path.join(conf.IMAGES_DIR, 'nifty_files//R01-001//3.000000-THORAX 1.0 B45f-95741.nii')


def test_read_single_dicom():
    dcmReader = DicomReader()
    assert dcmReader.read(sample_dicom_image) is not None

def test_read_dcm_directory():
    dcmReader = DicomReader()
    data = dcmReader.readDirectory(sample_dicom_directory)
    assert data is not None and data.shape[0] == 304

def test_read_dcm_directories_from_array():
    dcmReader = DicomReader()
    data = dcmReader.readDirectories(sample_dicom_directories_list)
    assert data is not None  and data.shape[0] == 2

def test_read_dcm_directories_from_string():
    dcmReader = DicomReader()
    data = dcmReader.readDirectories(sample_dicom_directories_string)
    assert data is not None and data.shape[0] == 6

def test_read_dicom_segmentation():
    dcmReader = DicomReader()
    assert dcmReader.readSegmentation(sample_dicom_segmentation) is not None

def test_read_single_nifty():
    niiReader = NiftyReader()
    assert niiReader.read(sample_nii_image) is not None

def test_dataService_segmentation_works_in_dcm():
    dataService = DataService(DicomReader())
    assert dataService.readSegmentation(sample_dicom_segmentation) is not None
