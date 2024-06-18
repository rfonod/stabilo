# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import sys
import yaml
import time
import numpy as np
import logging

def setup_logger(name: str, log_file: str = None, level: int = logging.INFO) -> logging.Logger:
    """
    Setup the logger
    """
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger

def timer(profiling: bool = False):
    """
    Decorator function to measure the execution time of a function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            if not profiling:
                return func(*args, **kwargs)
            start_time = time.time()
            result = func(*args, **kwargs)
            print(f"{func.__name__:<35} execution time: {1000*(time.time() - start_time):>10.2f} ms")
            return result
        return wrapper
    return decorator


def load_config(cfg_filepath: str, logger: logging.Logger = None) -> dict:
    """
    Load the configuration file
    """
    try:
        with open(cfg_filepath, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        if logger is not None:
            logger.error(f"Configuration file {cfg_filepath} not found.")
        sys.exit(1)
    return config

def xywh2four(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from [xc, yc, w, h] to four point format [x1, y1, x2, y2, x3, y3, x4, y4].
    """
    x_c, y_c, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c - 0.5 * h
    x3 = x_c + 0.5 * w
    y3 = y_c + 0.5 * h
    x4 = x_c - 0.5 * w
    y4 = y_c + 0.5 * h

    return np.column_stack((x1, y1, x2, y2, x3, y3, x4, y4))

def four2xywh(boxes: np.ndarray) -> np.ndarray:
    """
    Convert bounding boxes from four point format [x1, y1, x2, y2, x3, y3, x4, y4] to [xc, yc, w, h].
    """ 
    x1, y1, x2, y2, x3, y3, x4, y4 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3], boxes[:, 4], boxes[:, 5], boxes[:, 6], boxes[:, 7]

    x_c = 0.25 * (x1 + x2 + x3 + x4)
    y_c = 0.25 * (y1 + y2 + y3 + y4)
    w = 0.5 * (x2 - x1 + x3 - x4)
    h = 0.5 * (y3 - y2 + y4 - y1)
    
    return np.column_stack((x_c, y_c, w, h))

def detect_delimiter(filepath: str, lines_to_check: int = 5) -> str:
    """
    Detect the delimiter of a CSV file by reading a few lines
    """
    delimiters = {',': 0, ' ': 0, '\t': 0}
    with open(filepath, 'r') as file:
        for _ in range(lines_to_check):
            line = file.readline()
            if not line:
                break
            delimiters[','] += line.count(',')
            delimiters[' '] += line.count(' ')
            delimiters['\t'] += line.count('\t')
            
    return max(delimiters, key=delimiters.get)

