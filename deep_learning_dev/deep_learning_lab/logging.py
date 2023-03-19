import logging, os, sys


LOGGER_LEVEL = 'DEBUG'
LOGGER_DIR = 'logs'

os.makedirs(LOGGER_DIR, exist_ok= True)


def getLogger(name):
    log_filename = name
    sub_package_index = name.find('.')
    if sub_package_index != -1:
        log_filename = name[:sub_package_index]
        name = name[sub_package_index+1:]
        module_index = name.rfind('.')
        if module_index != -1:
            log_filename = name[:module_index]
            name = name[module_index+1:]
    
    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(LOGGER_DIR, log_filename), when= 'd')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    
    logger.setLevel(LOGGER_LEVEL)
    return logger
