#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import cv2
import numpy as np
import pytest

from stabilo import Stabilizer


@pytest.fixture
def default_stabilizer():
    return Stabilizer(downsample_ratio = 1.0, viz = True)

@pytest.fixture
def images():
    cur_frame = cv2.imread('tests/ND_before.jpg')
    ref_frame = cv2.imread('tests/ND_after.jpg')
    return cur_frame, ref_frame

def test_initialization(default_stabilizer):
    assert default_stabilizer.detector_name in default_stabilizer.VALID_DETECTORS
    assert default_stabilizer.matcher_name in default_stabilizer.VALID_MATCHERS
    assert default_stabilizer.filter_type in default_stabilizer.VALID_FILTER_TYPES
    assert default_stabilizer.transformation_type in default_stabilizer.VALID_TRANSFORMATION_TYPES

def test_invalid_detector():
    with pytest.raises(ValueError):
        Stabilizer(detector_name='invalid_detector')

def test_invalid_matcher():
    with pytest.raises(ValueError):
        Stabilizer(matcher_name='invalid_matcher')

def test_invalid_filter_type():
    with pytest.raises(ValueError):
        Stabilizer(filter_type='invalid_filter')

def test_invalid_transformation_type():
    with pytest.raises(ValueError):
        Stabilizer(transformation_type='invalid_transformation')

def test_invalid_ransac_method():
    with pytest.raises(ValueError):
        Stabilizer(ransac_method=999)

def test_invalid_downsample_ratio():
    with pytest.raises(ValueError):
        Stabilizer(downsample_ratio=1.5)

def test_invalid_max_features():
    with pytest.raises(ValueError):
        Stabilizer(max_features=-10)

def test_invalid_ref_multiplier():
    with pytest.raises(ValueError):
        Stabilizer(ref_multiplier=0.5)

def test_invalid_filter_ratio():
    with pytest.raises(ValueError):
        Stabilizer(filter_ratio=1.5)

def test_invalid_ransac_max_iter():
    with pytest.raises(ValueError):
        Stabilizer(ransac_max_iter=-100)

def test_invalid_ransac_epipolar_threshold():
    with pytest.raises(ValueError):
        Stabilizer(ransac_epipolar_threshold=-1.0)

def test_invalid_ransac_confidence():
    with pytest.raises(ValueError):
        Stabilizer(ransac_confidence=1.5)

def test_set_ref_frame(default_stabilizer, images):
    _, ref_frame = images
    default_stabilizer.set_ref_frame(ref_frame)
    assert default_stabilizer.ref_frame is not None

def test_frame_set(default_stabilizer, images):
    cur_frame, ref_frame = images
    default_stabilizer.set_ref_frame(ref_frame)
    default_stabilizer.stabilize(cur_frame)
    assert default_stabilizer.cur_frame is not None
    assert default_stabilizer.ref_frame is not None
    assert default_stabilizer.cur_frame_gray is not None
    assert default_stabilizer.ref_frame_gray is not None

def test_stabilize(default_stabilizer, images):
    cur_frame, ref_frame = images
    default_stabilizer.set_ref_frame(ref_frame)
    default_stabilizer.stabilize(cur_frame)
    assert default_stabilizer.cur_trans_matrix is not None
    assert default_stabilizer.trans_matrix_last_known is not None
    assert default_stabilizer.cur_kpts is not None
    assert default_stabilizer.ref_kpts is not None
    assert default_stabilizer.cur_desc is not None
    assert default_stabilizer.ref_desc is not None

def test_stabilize_with_boxes(default_stabilizer, images):
    cur_frame, ref_frame = images
    ref_boxes = np.array([[100, 140, 20, 30], [350, 840, 20, 60]])
    cur_boxes = np.array([[320, 240, 30, 10], [450, 140, 10, 40]])
    default_stabilizer.set_ref_frame(ref_frame, ref_boxes)
    default_stabilizer.stabilize(cur_frame, cur_boxes)
    transformed_boxes = default_stabilizer.transform_cur_boxes()
    assert transformed_boxes is not None
    assert transformed_boxes.shape == cur_boxes.shape

def test_warp_cur_frame(default_stabilizer, images):
    cur_frame, ref_frame = images
    default_stabilizer.set_ref_frame(ref_frame)
    default_stabilizer.stabilize(cur_frame)
    warped_frame = default_stabilizer.warp_cur_frame()
    assert warped_frame is not None
    assert warped_frame.shape == cur_frame.shape

def test_create_binary_mask(default_stabilizer, images):
    _, ref_frame = images
    boxes = np.array([[20, 40, 10, 20], [120, 340, 20, 15]])
    default_stabilizer.set_ref_frame(ref_frame, boxes)
    mask = default_stabilizer.create_binary_mask(boxes, 'xywh')
    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])

def test_get_basic_info(default_stabilizer):
    info = default_stabilizer.get_basic_info()
    assert isinstance(info, dict)
    assert 'detector_name' in info
    assert 'matcher_name' in info
    assert 'filter_type' in info
    assert 'transformation_type' in info
    assert 'clahe' in info
    assert 'mask_use' in info
