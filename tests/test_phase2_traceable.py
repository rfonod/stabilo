#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from stabilo import Stabilizer
from stabilo.utils import detect_delimiter, is_box_rotated, xywh2four, xywha2four


@pytest.fixture
def stabilizer():
    return Stabilizer(downsample_ratio=1.0, viz=False, benchmark=False)


@pytest.fixture
def blank_frame():
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_tc_01_invalid_detector_rejected():
    with pytest.raises(ValueError):
        Stabilizer(detector_name='invalid_detector')


def test_tc_02_downsample_ratio_lower_boundary_rejected():
    with pytest.raises(ValueError):
        Stabilizer(downsample_ratio=0.0)


def test_tc_03_downsample_ratio_upper_boundary_accepted():
    instance = Stabilizer(downsample_ratio=1.0)
    assert instance.downsample_ratio == 1.0


def test_tc_04_xywha_to_four_axis_aligned_conversion():
    boxes_xywha = np.array([[100, 100, 60, 40, 0]])
    boxes_four = xywha2four(boxes_xywha)
    expected = np.array([[70, 80, 130, 80, 130, 120, 70, 120]])
    np.testing.assert_allclose(boxes_four, expected, atol=1e-6)


def test_tc_05_axis_aligned_box_is_not_rotated():
    box = np.array([10, 10, 50, 10, 50, 40, 10, 40], dtype=float)
    assert is_box_rotated(box) is False


def test_tc_06_detect_delimiter_returns_comma(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("a,b,c\n1,2,3\n4,5,6\n", encoding="utf-8")
    assert detect_delimiter(str(csv_path)) == ','


def test_tc_07_create_binary_mask_clips_out_of_bounds_boxes(stabilizer, blank_frame):
    stabilizer.h, stabilizer.w = blank_frame.shape[:2]
    boxes = np.array([[5, 5, 20, 20], [98, 98, 20, 20]], dtype=float)
    mask = stabilizer.create_binary_mask(boxes, 'xywh')

    assert mask.shape == (100, 100)
    assert mask.dtype == np.uint8
    assert np.min(mask) == 0
    assert np.max(mask) == 255


def test_tc_08_get_matches_returns_empty_when_descriptor_missing(stabilizer):
    assert stabilizer.get_matches(None, np.ones((2, 2), dtype=np.float32)) == []


def test_tc_09_get_matches_none_filter_uses_match_output(stabilizer):
    stabilizer.filter_type = 'none'
    fake_matches = [
        SimpleNamespace(distance=1.0, queryIdx=0, trainIdx=0),
        SimpleNamespace(distance=2.0, queryIdx=1, trainIdx=1),
        SimpleNamespace(distance=3.0, queryIdx=2, trainIdx=2),
    ]
    stabilizer.matcher = SimpleNamespace(match=lambda desc1, desc2, _: fake_matches)

    result = stabilizer.get_matches(np.ones((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))

    assert result == fake_matches


def test_tc_10_get_matches_distance_filter_applies_threshold(stabilizer):
    stabilizer.filter_type = 'distance'
    stabilizer.filter_ratio = 0.5
    fake_matches = [
        SimpleNamespace(distance=10.0, queryIdx=0, trainIdx=0),
        SimpleNamespace(distance=20.0, queryIdx=1, trainIdx=1),
        SimpleNamespace(distance=30.0, queryIdx=2, trainIdx=2),
    ]
    stabilizer.matcher = SimpleNamespace(match=lambda desc1, desc2, _: fake_matches)

    result = stabilizer.get_matches(np.ones((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))

    assert [m.distance for m in result] == [10.0, 20.0]


def test_tc_11_get_matches_ratio_filter_keeps_only_ratio_passes(stabilizer):
    stabilizer.filter_type = 'ratio'
    stabilizer.filter_ratio = 0.8
    ratio_matches = [
        [SimpleNamespace(distance=10.0, queryIdx=0, trainIdx=0), SimpleNamespace(distance=20.0, queryIdx=0, trainIdx=1)],
        [SimpleNamespace(distance=18.0, queryIdx=1, trainIdx=1), SimpleNamespace(distance=20.0, queryIdx=1, trainIdx=2)],
        [SimpleNamespace(distance=4.0, queryIdx=2, trainIdx=2)],
    ]
    stabilizer.matcher = SimpleNamespace(knnMatch=lambda desc1, desc2, k=2: ratio_matches)

    result = stabilizer.get_matches(np.ones((3, 2), dtype=np.float32), np.ones((3, 2), dtype=np.float32))

    assert len(result) == 1
    assert result[0].distance == 10.0


def test_tc_12_get_matches_returns_empty_on_opencv_error(stabilizer):
    stabilizer.filter_type = 'none'

    def raise_cv_error(desc1, desc2, _):
        raise cv2.error("forced failure")

    stabilizer.matcher = SimpleNamespace(match=raise_cv_error)
    result = stabilizer.get_matches(np.ones((1, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32))
    assert result == []


def test_tc_13_calculate_transformation_matrix_reuses_last_known_when_points_insufficient(stabilizer):
    stabilizer.ref_pts = np.zeros((3, 2), dtype=np.float32)
    stabilizer.cur_pts = np.zeros((3, 2), dtype=np.float32)
    stabilizer.trans_matrix_last_known = np.array([[1.0, 0.0, 7.0], [0.0, 1.0, 9.0], [0.0, 0.0, 1.0]])

    stabilizer.calculate_transformation_matrix()

    np.testing.assert_array_equal(stabilizer.cur_trans_matrix, stabilizer.trans_matrix_last_known)


def test_tc_14_calculate_transformation_matrix_updates_current_and_last_known(stabilizer):
    stabilizer.ref_pts = np.array([[0, 0], [10, 0], [0, 10], [10, 10]], dtype=np.float32)
    stabilizer.cur_pts = np.array([[1, 1], [11, 1], [1, 11], [11, 11]], dtype=np.float32)

    expected_matrix = np.array([[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    expected_inliers = np.ones((4, 1), dtype=np.uint8)

    def fake_transformer(cur_pts, ref_pts, **kwargs):
        return expected_matrix, expected_inliers

    stabilizer.transformer = fake_transformer
    stabilizer.calculate_transformation_matrix()

    np.testing.assert_array_equal(stabilizer.cur_trans_matrix, expected_matrix)
    np.testing.assert_array_equal(stabilizer.trans_matrix_last_known, expected_matrix)


def test_tc_15_transform_boxes_returns_none_for_none_input(stabilizer):
    assert stabilizer.transform_boxes(None, np.eye(3), 'xywh', 'four') is None


def test_tc_16_transform_boxes_xywh_to_four_with_identity_preserves_geometry(stabilizer):
    boxes = np.array([[20, 30, 10, 8], [50, 60, 12, 6]], dtype=np.float32)
    transformed = stabilizer.transform_boxes(boxes, np.eye(3, dtype=np.float32), 'xywh', 'four')
    expected = xywh2four(boxes)

    np.testing.assert_allclose(transformed, expected, atol=1e-6)
