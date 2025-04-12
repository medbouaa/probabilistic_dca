# src/probabilistic_dca/logging_setup.py

import logging
from probabilistic_dca.config import LOGGING_LEVEL, LOG_FILE

def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, LOGGING_LEVEL.upper(), "INFO"))

    # Prevent adding handlers multiple times
    if not logger.handlers:
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, LOGGING_LEVEL.upper(), "INFO"))
        console_format = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_format)

        # File handler
        file_handler = logging.FileHandler(LOG_FILE)
        file_handler.setLevel(getattr(logging, LOGGING_LEVEL.upper(), "INFO"))
        file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
        file_handler.setFormatter(file_format)

        # Add handlers
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    return logger
