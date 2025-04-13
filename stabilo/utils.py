# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import logging
import sys
import time

import numpy as np
import yaml


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
    x_c, y_c, w, h = boxes.T

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
    Convert bounding boxes from any four-point format [x1, y1, x2, y2, x3, y3, x4, y4]
    to YOLO format [xc, yc, w, h], robust to any point order and rotation.
    """
    points = boxes.reshape(-1, 4, 2)

    x_min = np.min(points[:, :, 0], axis=1)
    x_max = np.max(points[:, :, 0], axis=1)
    y_min = np.min(points[:, :, 1], axis=1)
    y_max = np.max(points[:, :, 1], axis=1)

    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2

    w = x_max - x_min
    h = y_max - y_min

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

