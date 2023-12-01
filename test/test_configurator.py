import pytest
from BMEMasterThesis.utils.config import configuration as conf

def test_conf_in_operator_exists():
    
    assert hasattr(conf, 'LOG_LEVEL')

def test_conf_in_operator_not_exists():
    assert 'blah' not in conf

def test_conf_attribute_exists():
    assert conf.LOG_LEVEL

def test_conf_attribute_not_exists():
    with pytest.raises(AttributeError):
        conf.blah