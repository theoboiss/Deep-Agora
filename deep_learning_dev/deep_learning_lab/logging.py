"""
This module provides a method for creating a logger object that can be used for logging purposes.

Functions:
- getLogger(name: str) -> logging.Logger:
  Returns a logger object that writes to a daily rotating file under the name of the package.

Global Variables:
- LOGGER_LEVEL (str): The level of logging to use. Default is 'DEBUG'.
- LOGGER_DIR (str): The directory where log files will be stored. Default is 'logs'.
"""
import logging, os


LOGGER_LEVEL = 'DEBUG' #: The logging level used for the logger.
LOGGER_DIR = 'logs' #: The name of the directory where log files will be stored.


def getLogger(name: str) -> logging.Logger:
    """
    Returns a logger object that can be used for logging purposes. The logger writes to a daily rotating file under the
    name of the package.

    Args:
        name (str): The name for the logger, usually __name__.

    Returns:
        A logger object that can be used for logging purposes.
    """
    os.makedirs(LOGGER_DIR, exist_ok= True)

    # Get the log filename based on the package name
    log_filename = name
    sub_package_index = name.find('.')
    if sub_package_index != -1:
        log_filename = name[:sub_package_index]
        name = name[sub_package_index+1:]
        module_index = name.rfind('.')
        if module_index != -1:
            log_filename = name[:module_index]
            name = name[module_index+1:]
    
    # Set up the logger
    handler = logging.handlers.TimedRotatingFileHandler(os.path.join(LOGGER_DIR, log_filename), when= 'd')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.addHandler(handler)
    
    logger.setLevel(LOGGER_LEVEL)
    return logger
