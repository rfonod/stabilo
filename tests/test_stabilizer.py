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

def test_create_binary_mask_xywha(default_stabilizer, images):
    """Test creating masks from oriented bounding boxes in xywha format"""
    _, ref_frame = images
    # Create oriented boxes: [xc, yc, width, height, angle_degrees]
    boxes_xywha = np.array([
        [100, 100, 50, 30, 45],  # Rotated 45 degrees
        [300, 200, 40, 60, 90],  # Rotated 90 degrees
        [200, 300, 30, 30, 0]    # No rotation
    ])
    default_stabilizer.set_ref_frame(ref_frame)
    mask = default_stabilizer.create_binary_mask(boxes_xywha, 'xywha')
    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])
    assert mask.dtype == np.uint8
    # Check that mask has excluded regions (0 values)
    assert np.min(mask) == 0
    assert np.max(mask) == 255

def test_create_binary_mask_rotated_four(default_stabilizer, images):
    """Test creating masks from rotated boxes in four-point format"""
    _, ref_frame = images
    # Create a rotated box using four corner points (45-degree rotation)
    xc, yc, w, h, angle = 150, 150, 60, 40, 45
    angle_rad = np.deg2rad(angle)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)

    # Calculate rotated corners
    dx = [-w/2, w/2, w/2, -w/2]
    dy = [-h/2, -h/2, h/2, h/2]
    x_pts = [xc + dx[i] * cos_a - dy[i] * sin_a for i in range(4)]
    y_pts = [yc + dx[i] * sin_a + dy[i] * cos_a for i in range(4)]

    boxes_four = np.array([[x_pts[0], y_pts[0], x_pts[1], y_pts[1],
                           x_pts[2], y_pts[2], x_pts[3], y_pts[3]]])

    default_stabilizer.set_ref_frame(ref_frame)
    mask = default_stabilizer.create_binary_mask(boxes_four, 'four')
    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])
    # Check that mask has excluded regions
    assert np.min(mask) == 0
    assert np.max(mask) == 255

def test_transform_boxes_xywha(default_stabilizer, images):
    """Test transforming oriented bounding boxes"""
    cur_frame, ref_frame = images
    # Create oriented boxes in xywha format
    boxes_xywha = np.array([
        [100, 100, 50, 30, 30],
        [200, 150, 40, 60, 60]
    ])

    default_stabilizer.set_ref_frame(ref_frame)
    default_stabilizer.stabilize(cur_frame)

    # Transform boxes from xywha to four format
    transformed = default_stabilizer.transform_boxes(
        boxes_xywha,
        default_stabilizer.cur_trans_matrix,
        in_box_format='xywha',
        out_box_format='four'
    )

    assert transformed is not None
    assert transformed.shape == (2, 8)  # 2 boxes, 8 coordinates each

def test_is_box_rotated_utility():
    """Test the is_box_rotated utility function"""
    from stabilo.utils import is_box_rotated

    # Axis-aligned box
    box_aligned = np.array([10, 10, 50, 10, 50, 40, 10, 40])
    assert not is_box_rotated(box_aligned)

    # Rotated box (45 degrees)
    xc, yc, w, h = 100, 100, 60, 40
    angle = np.deg2rad(45)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    dx = [-w/2, w/2, w/2, -w/2]
    dy = [-h/2, -h/2, h/2, h/2]
    x_pts = [xc + dx[i] * cos_a - dy[i] * sin_a for i in range(4)]
    y_pts = [yc + dx[i] * sin_a + dy[i] * cos_a for i in range(4)]
    box_rotated = np.array([x_pts[0], y_pts[0], x_pts[1], y_pts[1],
                           x_pts[2], y_pts[2], x_pts[3], y_pts[3]])
    assert is_box_rotated(box_rotated)

def test_xywha2four_conversion():
    """Test conversion from xywha to four-point format"""
    from stabilo.utils import xywha2four

    # Test with axis-aligned box (0 degrees)
    boxes_xywha = np.array([[100, 100, 60, 40, 0]])
    boxes_four = xywha2four(boxes_xywha)
    assert boxes_four.shape == (1, 8)

    # Expected corners for axis-aligned box
    expected = np.array([[70, 80, 130, 80, 130, 120, 70, 120]])
    np.testing.assert_array_almost_equal(boxes_four, expected, decimal=1)

    # Test with 90-degree rotation
    boxes_xywha_90 = np.array([[100, 100, 60, 40, 90]])
    boxes_four_90 = xywha2four(boxes_xywha_90)
    assert boxes_four_90.shape == (1, 8)
    # After 90-degree rotation, width and height swap
    # The corners should represent a rotated rectangle

def test_stabilize_with_obb_masks(default_stabilizer, images):
    """Test stabilization with oriented bounding box masks"""
    cur_frame, ref_frame = images
    # Use oriented boxes as exclusion masks
    ref_boxes_xywha = np.array([[100, 140, 30, 20, 30]])
    cur_boxes_xywha = np.array([[320, 240, 40, 25, 45]])

    default_stabilizer.set_ref_frame(ref_frame, ref_boxes_xywha, box_format='xywha')
    default_stabilizer.stabilize(cur_frame, cur_boxes_xywha, box_format='xywha')

    assert default_stabilizer.cur_trans_matrix is not None
    assert default_stabilizer.ref_kpts is not None
    assert default_stabilizer.cur_kpts is not None

