#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import cv2
import numpy as np
import pytest

from stabilo import Stabilizer


@pytest.fixture
def default_stabilizer():
    return Stabilizer(downsample_ratio=1.0, viz=True)


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
    boxes_xywha = np.array(
        [
            [100, 100, 50, 30, 45],  # Rotated 45 degrees
            [300, 200, 40, 60, 90],  # Rotated 90 degrees
            [200, 300, 30, 30, 0],  # No rotation
        ]
    )
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
    dx = [-w / 2, w / 2, w / 2, -w / 2]
    dy = [-h / 2, -h / 2, h / 2, h / 2]
    x_pts = [xc + dx[i] * cos_a - dy[i] * sin_a for i in range(4)]
    y_pts = [yc + dx[i] * sin_a + dy[i] * cos_a for i in range(4)]

    boxes_four = np.array([[x_pts[0], y_pts[0], x_pts[1], y_pts[1], x_pts[2], y_pts[2], x_pts[3], y_pts[3]]])

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
    boxes_xywha = np.array([[100, 100, 50, 30, 30], [200, 150, 40, 60, 60]])

    default_stabilizer.set_ref_frame(ref_frame)
    default_stabilizer.stabilize(cur_frame)

    # Transform boxes from xywha to four format
    transformed = default_stabilizer.transform_boxes(
        boxes_xywha, default_stabilizer.cur_trans_matrix, in_box_format='xywha', out_box_format='four'
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
    dx = [-w / 2, w / 2, w / 2, -w / 2]
    dy = [-h / 2, -h / 2, h / 2, h / 2]
    x_pts = [xc + dx[i] * cos_a - dy[i] * sin_a for i in range(4)]
    y_pts = [yc + dx[i] * sin_a + dy[i] * cos_a for i in range(4)]
    box_rotated = np.array([x_pts[0], y_pts[0], x_pts[1], y_pts[1], x_pts[2], y_pts[2], x_pts[3], y_pts[3]])
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


def test_create_binary_mask_polygon_flattened(default_stabilizer, images):
    _, ref_frame = images
    polygon_boxes = np.array(
        [
            [60, 60, 120, 60, 120, 120, 60, 120],
            [200, 200, 260, 210, 240, 270, 190, 260],
        ]
    )

    default_stabilizer.set_ref_frame(ref_frame)
    mask = default_stabilizer.create_binary_mask(polygon_boxes, 'polygon')

    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])
    assert mask.dtype == np.uint8
    assert np.min(mask) == 0
    assert np.max(mask) == 255


def test_create_binary_mask_polygon_xy_pairs(default_stabilizer, images):
    _, ref_frame = images
    polygon_xy_pairs = np.array(
        [
            [[100, 100], [140, 100], [140, 140], [100, 140]],
            [[300, 300], [340, 310], [320, 360], [280, 350]],
        ],
        dtype=np.float32,
    )

    default_stabilizer.set_ref_frame(ref_frame)
    mask = default_stabilizer.create_binary_mask(polygon_xy_pairs, 'polygon')

    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])
    assert mask.dtype == np.uint8
    assert np.min(mask) == 0
    assert np.max(mask) == 255


def test_create_binary_mask_circle(default_stabilizer, images):
    _, ref_frame = images
    circle_masks = np.array(
        [
            [120, 120, 25],
            [360, 240, 40],
        ]
    )

    default_stabilizer.set_ref_frame(ref_frame)
    mask = default_stabilizer.create_binary_mask(circle_masks, 'circle')

    assert mask is not None
    assert mask.shape == (ref_frame.shape[0], ref_frame.shape[1])
    assert mask.dtype == np.uint8
    assert np.min(mask) == 0
    assert np.max(mask) == 255


# ---------------------------------------------------------------------------
# Exact reproduction of an external hand-rolled SIFT -> RootSIFT -> BFMatcher
# (knn, k=2) -> Lowe ratio -> findHomography routine (used by geo-trax).
# ---------------------------------------------------------------------------

# Shared parameters between the reference routine and the mirrored Stabilizer.
_REPRO_MAX_FEATURES = 2000
_REPRO_FILTER_RATIO = 0.75
_REPRO_RSIFT_EPS = 1e-8
_REPRO_RANSAC_METHOD = cv2.USAC_MAGSAC  # 38 - MAGSAC++
_REPRO_RANSAC_CONFIDENCE = 0.999
_REPRO_RANSAC_THRESHOLD = 3.0
_REPRO_RANSAC_MAX_ITER = 10000


def _reference_homography(img_src, img_dst):
    """
    Reference hand-rolled registration routine. Returns H such that dst_pts ~ H . src_pts.
    """
    img_src_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)
    img_dst_gray = cv2.cvtColor(img_dst, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create(_REPRO_MAX_FEATURES, enable_precise_upscale=True)  # one instance, both images
    kpt_src, desc_src = sift.detectAndCompute(img_src_gray, None)
    kpt_dst, desc_dst = sift.detectAndCompute(img_dst_gray, None)

    # RootSIFT (L1-normalize then sqrt) -- genuinely written back to each descriptor array
    desc_src = desc_src / (desc_src.sum(axis=1, keepdims=True) + _REPRO_RSIFT_EPS)
    desc_src = np.sqrt(desc_src)
    desc_dst = desc_dst / (desc_dst.sum(axis=1, keepdims=True) + _REPRO_RSIFT_EPS)
    desc_dst = np.sqrt(desc_dst)

    bf = cv2.BFMatcher()  # NORM_L2, crossCheck=False
    matches = bf.knnMatch(desc_src, desc_dst, k=2)  # query = SOURCE descriptors
    good = []
    for pair in matches:
        if len(pair) == 2:
            m, n = pair
            if m.distance < _REPRO_FILTER_RATIO * n.distance:
                good.append(m)

    pts_src = np.float32([kpt_src[m.queryIdx].pt for m in good]).reshape(-1, 2)
    pts_dst = np.float32([kpt_dst[m.trainIdx].pt for m in good]).reshape(-1, 2)

    H, _ = cv2.findHomography(
        pts_src,
        pts_dst,
        method=_REPRO_RANSAC_METHOD,
        confidence=_REPRO_RANSAC_CONFIDENCE,
        ransacReprojThreshold=_REPRO_RANSAC_THRESHOLD,
        maxIters=_REPRO_RANSAC_MAX_ITER,
    )
    return H


def _mirror_stabilizer():
    """Build a Stabilizer configured to mirror the reference routine exactly."""
    return Stabilizer(
        detector_name='rsift',
        matcher_name='bf',
        filter_type='ratio',
        transformation_type='projective',
        clahe=False,
        mask_use=False,
        downsample_ratio=1.0,
        max_features=_REPRO_MAX_FEATURES,
        ref_multiplier=1.0,
        filter_ratio=_REPRO_FILTER_RATIO,
        rsift_eps=_REPRO_RSIFT_EPS,
        sift_enable_precise_upscale=True,
        match_query_frame='current',
        ransac_method=_REPRO_RANSAC_METHOD,
        ransac_confidence=_REPRO_RANSAC_CONFIDENCE,
        ransac_epipolar_threshold=_REPRO_RANSAC_THRESHOLD,
        ransac_max_iter=_REPRO_RANSAC_MAX_ITER,
    )


def test_exact_reproduction_of_reference_routine(images):
    img_src, img_dst = images  # ND_before as source, ND_after as destination

    H_ref = _reference_homography(img_src, img_dst)
    assert H_ref is not None

    stab = _mirror_stabilizer()
    stab.set_ref_frame(img_dst)  # destination is the reference frame
    stab.stabilize(img_src)  # source is the current frame
    H_stab = stab.get_cur_trans_matrix()

    assert H_stab is not None
    # H maps src -> dst, identical to the reference routine, bit-for-bit.
    np.testing.assert_allclose(H_stab, H_ref, atol=0, rtol=0)


def test_default_path_backward_compatible(images):
    """The new options at their documented defaults must be exact no-ops."""
    cur_frame, ref_frame = images

    stab_implicit = Stabilizer(downsample_ratio=1.0)
    stab_implicit.set_ref_frame(ref_frame)
    stab_implicit.stabilize(cur_frame)

    stab_explicit = Stabilizer(
        downsample_ratio=1.0,
        sift_enable_precise_upscale=False,
        rsift_eps=1e-8,
        match_query_frame='reference',
    )
    stab_explicit.set_ref_frame(ref_frame)
    stab_explicit.stabilize(cur_frame)

    np.testing.assert_array_equal(stab_implicit.get_cur_trans_matrix(), stab_explicit.get_cur_trans_matrix())


def test_invalid_match_query_frame():
    with pytest.raises(ValueError):
        Stabilizer(match_query_frame='invalid')


def test_match_quality_getters(images):
    """Inlier count, match count, and keypoint counts are available without viz mode."""
    cur_frame, ref_frame = images
    stab = Stabilizer(downsample_ratio=1.0, viz=False)
    stab.set_ref_frame(ref_frame)

    assert stab.get_cur_num_matches() is None

    stab.stabilize(cur_frame)

    inliers_count = stab.get_cur_inliers_count()
    assert isinstance(inliers_count, int)

    num_matches = stab.get_cur_num_matches()
    assert isinstance(num_matches, int)
    assert num_matches == len(stab.cur_inliers)
    assert num_matches >= inliers_count

    num_ref_kpts, num_cur_kpts = stab.get_cur_num_keypoints()
    assert isinstance(num_ref_kpts, int)
    assert isinstance(num_cur_kpts, int)
