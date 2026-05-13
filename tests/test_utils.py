#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from stabilo.utils import detect_delimiter, four2xywh, is_box_rotated, load_config, xywh2four, xywha2four


def test_xywh_four_roundtrip():
    boxes_xywh = np.array([
        [100, 120, 40, 20],
        [320, 240, 60, 80],
    ], dtype=np.float32)
    boxes_four = xywh2four(boxes_xywh)
    recovered = four2xywh(boxes_four)
    np.testing.assert_allclose(recovered, boxes_xywh, rtol=0, atol=1e-6)


def test_four2xywh_robust_to_point_order():
    box_corners = np.array([[10, 10, 60, 10, 60, 40, 10, 40]], dtype=np.float32)
    reordered = np.array([[60, 40, 10, 10, 10, 40, 60, 10]], dtype=np.float32)
    np.testing.assert_allclose(four2xywh(box_corners), four2xywh(reordered), rtol=0, atol=1e-6)


def test_xywha2four_axis_aligned():
    box_xywha = np.array([[100, 100, 60, 40, 0]], dtype=np.float32)
    converted = xywha2four(box_xywha)
    expected = np.array([[70, 80, 130, 80, 130, 120, 70, 120]], dtype=np.float32)
    np.testing.assert_allclose(converted, expected, rtol=0, atol=1e-6)


def test_is_box_rotated_axis_aligned_and_rotated():
    aligned = np.array([10, 10, 50, 10, 50, 40, 10, 40], dtype=np.float32)
    rotated = np.array([100, 80, 130, 100, 100, 130, 70, 110], dtype=np.float32)
    assert not is_box_rotated(aligned)
    assert is_box_rotated(rotated)


def test_detect_delimiter(tmp_path):
    comma_file = tmp_path / "comma.csv"
    space_file = tmp_path / "space.txt"
    tab_file = tmp_path / "tab.tsv"

    comma_file.write_text("a,b,c\n1,2,3\n")
    space_file.write_text("a b c\n1 2 3\n")
    tab_file.write_text("a\tb\tc\n1\t2\t3\n")

    assert detect_delimiter(str(comma_file)) == ','
    assert detect_delimiter(str(space_file)) == ' '
    assert detect_delimiter(str(tab_file)) == '\t'


def test_load_config(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text("mask_use: true\nmax_features: 1234\n")
    config = load_config(str(cfg_file))
    assert config["mask_use"] is True
    assert config["max_features"] == 1234
