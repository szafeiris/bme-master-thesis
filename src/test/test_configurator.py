import pytest
from main.configuration import configurator as conf

def test_conf_in_operator_exists():
    assert 'LOG_LEVEL' in conf

def test_conf_in_operator_not_exists():
    assert 'blah' not in conf

def test_conf_attribute_exists():
    assert conf.LOG_LEVEL

def test_conf_attribute_not_exists():
    with pytest.raises(AttributeError):
        conf.blah