from src.main.configurator import configurator as conf

from logging.config import dictConfig
import logging

dictConfig(conf._LOGGING_CONFIG_)
log = logging.getLogger('main.configurator')

print(conf.LOG_LEVEL)
print(conf.getAll())
print('blah' in conf)
print('LOG_LEVEL' in conf)

log.debug('Hello world!')
log.info('Hello world!')
log.error('Hello world!')
log.critical('Hello world!')
