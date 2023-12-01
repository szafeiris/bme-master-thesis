from BMEMasterThesis.services.utils import NiftyReader
from BMEMasterThesis.utils.config import PATHS

sample_dicom_image = PATHS.PICAI_IMAGES_DIR.joinpath('original').joinpath('10005_1000005_t2w.nii.gz')
sample_dicom_directory = PATHS.PICAI_IMAGES_DIR.joinpath('original')
sample_dicom_directories_list = [
    PATHS.PICAI_IMAGES_DIR.joinpath('original'),
    PATHS.PICAI_IMAGES_DIR.joinpath('fat')
]

def test_read_single_dicom():
    niftyReader = NiftyReader()
    assert niftyReader.read(sample_dicom_image) is not None

def test_read_dcm_directory():
    niftyReader = NiftyReader()
    data = niftyReader.readDirectory(sample_dicom_directory)
    assert data is not None and data.shape[0] == 220

def test_read_dcm_directories_from_array():
    niftyReader = NiftyReader()
    data = niftyReader.readDirectories(sample_dicom_directories_list)
    assert data is not None  and data.shape[0] == 2
