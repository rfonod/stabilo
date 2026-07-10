#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import logging

import numpy as np

import stabilo.stabilo as stabilo_module
from stabilo import Stabilizer
from stabilo.utils import setup_logger


def test_setup_logger_idempotent():
    name = 'stabilo_test_idempotent'
    logging.getLogger(name).handlers.clear()
    logger = setup_logger(name)
    count = len(logger.handlers)
    setup_logger(name)
    setup_logger(name)
    assert len(logger.handlers) == count


def test_setup_logger_file_handler_idempotent(tmp_path):
    name = 'stabilo_test_file_idempotent'
    logging.getLogger(name).handlers.clear()
    log_file = str(tmp_path / 'stabilo.log')
    setup_logger(name, log_file=log_file)
    setup_logger(name, log_file=log_file)
    logger = logging.getLogger(name)
    file_handlers = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_logger_default_is_module_logger():
    stab = Stabilizer()
    assert stab.logger is stabilo_module.logger


def test_logger_injection():
    custom = logging.getLogger('stabilo_test_injected')
    custom.handlers.clear()
    records = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    custom.addHandler(_Capture())
    custom.setLevel(logging.WARNING)

    stab = Stabilizer(logger=custom, mask_use=True)
    assert stab.logger is custom

    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    stab.set_ref_frame(frame)
    assert any('mask' in msg.lower() for msg in records)
