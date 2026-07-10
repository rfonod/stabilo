#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import importlib.util
import os

import cv2
import numpy as np
import pytest

from stabilo import Stabilizer

KORNIA = importlib.util.find_spec('kornia') is not None
RUN_DL = KORNIA and os.environ.get('STABILO_DL_TESTS') == '1'
requires_dl = pytest.mark.skipif(
    not RUN_DL,
    reason='set STABILO_DL_TESTS=1 (with kornia installed) to run DL end-to-end tests; they download pretrained weights',
)

SPARSE = sorted(Stabilizer.SPARSE_DL_DETECTORS)


@pytest.fixture
def images():
    cur_frame = cv2.imread('tests/ND_before.jpg')
    ref_frame = cv2.imread('tests/ND_after.jpg')
    return cur_frame, ref_frame


# --- validation tests that do not require kornia (raise before any DL import) ---


@pytest.mark.parametrize('detector_name', SPARSE + ['loftr'])
def test_gpu_dl_detector_raises(detector_name):
    with pytest.raises(ValueError, match='CUDA'):
        Stabilizer(detector_name=detector_name, gpu=True)


def test_lightglue_incompatible_detector():
    with pytest.raises(ValueError, match='lightglue'):
        Stabilizer(matcher_name='lightglue', detector_name='orb')


def test_invalid_device():
    with pytest.raises(ValueError, match='Invalid device'):
        Stabilizer(device='tpu')


def test_invalid_loftr_confidence():
    with pytest.raises(ValueError, match='loftr_confidence'):
        Stabilizer(loftr_confidence=1.5)


# --- end-to-end tests requiring kornia (weights download on first run) ---


@requires_dl
@pytest.mark.parametrize('detector_name', SPARSE)
@pytest.mark.parametrize('matcher_name', ['bf', 'flann'])
def test_sparse_dl_stabilize(images, detector_name, matcher_name):
    cur_frame, ref_frame = images
    stab = Stabilizer(
        detector_name=detector_name,
        matcher_name=matcher_name,
        downsample_ratio=1.0,
        device='cpu',
        mask_use=False,
    )
    stab.set_ref_frame(ref_frame)
    stab.stabilize(cur_frame)
    assert stab.get_cur_trans_matrix() is not None


@requires_dl
def test_loftr_stabilize(images):
    cur_frame, ref_frame = images
    stab = Stabilizer(detector_name='loftr', downsample_ratio=0.5, device='cpu', mask_use=False)
    assert stab.matcher is None
    stab.set_ref_frame(ref_frame)
    stab.stabilize(cur_frame)
    assert stab.get_cur_trans_matrix() is not None


@requires_dl
@pytest.mark.parametrize('detector_name', sorted(Stabilizer.LIGHTGLUE_FEATURE_NAMES))
def test_lightglue_stabilize(images, detector_name):
    cur_frame, ref_frame = images
    stab = Stabilizer(
        detector_name=detector_name,
        matcher_name='lightglue',
        downsample_ratio=1.0,
        device='cpu',
        mask_use=False,
    )
    stab.set_ref_frame(ref_frame)
    stab.stabilize(cur_frame)
    assert stab.get_cur_trans_matrix() is not None


@requires_dl
def test_sparse_dl_mask_filtering(images):
    cur_frame, ref_frame = images
    h, w = ref_frame.shape[:2]
    box = np.array([[w / 2, h / 2, w, h]])  # xywh mask covering the whole frame
    stab = Stabilizer(detector_name='xfeat', downsample_ratio=1.0, device='cpu')
    stab.set_ref_frame(ref_frame, box, box_format='xywh')
    assert stab.ref_kpts is None or len(stab.ref_kpts) == 0
