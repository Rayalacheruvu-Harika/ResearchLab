import logging
from pathlib import Path
from config import CONFIG

def setup_logger(name):
    log_dir = Path(CONFIG['logging']['file']).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, CONFIG['logging']['level']))
    
    if logger.hasHandlers():
        return logger
    
    file_handler = logging.FileHandler(CONFIG['logging']['file'])
    file_handler.setLevel(logging.DEBUG)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter(CONFIG['logging']['format'])
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
