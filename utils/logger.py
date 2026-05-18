"""Logging utilities."""

import os
import sys
import logging
from datetime import datetime


def setup_logger(work_dir, name='train'):
    """Create logger that writes to both console and file."""
    os.makedirs(work_dir, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(work_dir, f'{name}_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        '[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
