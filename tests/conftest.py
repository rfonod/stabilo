#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import os

import pytest


@pytest.fixture(autouse=True, scope="session")
def _disable_update_check():
    """
    Keep the test suite hermetic: never let the PyPI update check reach the network.
    Individual version_check tests override this via monkeypatch.
    """
    os.environ.setdefault("STABILO_DISABLE_UPDATE_CHECK", "1")
