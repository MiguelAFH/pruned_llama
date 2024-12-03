import logging
import os
from logging.handlers import TimedRotatingFileHandler

def setup_logger(log_dir='logs', log_file='run.log', log_level=logging.INFO):
    '''
    Set up a logger that writes logs to a specified directory.
    '''
    
    os.makedirs(log_dir, exist_ok=True)
    
    log_path = os.path.join(log_dir, log_file)
    
    logger = logging.getLogger('Logger')
    logger.setLevel(log_level)
    
    file_handler = TimedRotatingFileHandler(log_path, when='midnight', interval=1)
    file_handler.suffix = '%Y-%m-%d'  # Append date to log file name
    file_handler.setLevel(log_level)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
