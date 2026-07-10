#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import argparse

import cv2
import numpy as np
import pytest

from stabilo import Stabilizer, __version__
from stabilo.cli import main
from stabilo.cli.utils import separate_cli_arguments
from stabilo.cli.video import render_stabilization_visuals


def test_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--help'])
    assert exc.value.code == 0
    assert 'video' in capsys.readouterr().out


def test_version_output(capsys):
    with pytest.raises(SystemExit) as exc:
        main(['--version'])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


@pytest.mark.parametrize('sub', ['video', 'tracks', 'config'])
def test_subcommand_help_exits_zero(capsys, sub):
    with pytest.raises(SystemExit) as exc:
        main([sub, '--help'])
    assert exc.value.code == 0


def test_config_show(capsys, monkeypatch):
    monkeypatch.setenv('STABILO_DISABLE_UPDATE_CHECK', '1')
    main(['config', 'show'])
    out = capsys.readouterr().out
    assert 'detector_name' in out


def test_config_copy(tmp_path, monkeypatch):
    monkeypatch.setenv('STABILO_DISABLE_UPDATE_CHECK', '1')
    dest = tmp_path / 'custom.yaml'
    main(['config', 'copy', '--output', str(dest)])
    assert dest.exists()
    assert 'detector_name' in dest.read_text()


def test_video_requires_output_flag():
    with pytest.raises(SystemExit) as exc:
        main(['video', 'nonexistent.mp4'])
    assert exc.value.code != 0


def test_missing_command_errors():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_separate_cli_arguments():
    ns = argparse.Namespace(
        input='x.mp4',
        no_mask=True,
        viz=False,
        save_viz=True,
        custom_config=None,
        detector_name='orb',
        downsample_ratio=None,
    )
    _, kwargs = separate_cli_arguments(ns)
    assert kwargs['mask_use'] is False
    assert kwargs['viz'] is True
    assert kwargs['detector_name'] == 'orb'
    assert 'downsample_ratio' not in kwargs


def test_explicit_cli_flags_override_custom_config(tmp_path):
    config_path = tmp_path / 'custom.yaml'
    config_path.write_text("detector_name: 'sift'\nmax_features: 1000\nclahe: true\n")

    ns = argparse.Namespace(
        input='x.mp4',
        no_mask=False,
        viz=False,
        save_viz=False,
        custom_config=config_path,
        detector_name='orb',  # explicitly passed on the CLI -> must win over the config file
        max_features=None,  # not passed on the CLI -> config file value applies
        clahe=None,  # not passed on the CLI -> config file value applies
    )
    _, kwargs = separate_cli_arguments(ns)
    assert kwargs['detector_name'] == 'orb'
    assert kwargs['max_features'] == 1000
    assert kwargs['clahe'] is True


def test_custom_config_mask_use_and_viz_respected_when_flags_not_passed(tmp_path):
    """Regression: --no-mask/--viz/--save-viz used to unconditionally override mask_use/viz
    even when never passed on the CLI, defeating a --custom-config value."""
    config_path = tmp_path / 'custom.yaml'
    config_path.write_text("mask_use: false\nviz: true\n")

    ns = argparse.Namespace(
        input='x.mp4',
        no_mask=False,  # not passed on the CLI -> config file's mask_use applies
        viz=False,  # not passed on the CLI -> config file's viz applies
        save_viz=False,
        custom_config=config_path,
    )
    _, kwargs = separate_cli_arguments(ns)
    assert kwargs['mask_use'] is False
    assert kwargs['viz'] is True


def test_explicit_no_mask_and_viz_flags_still_override_custom_config(tmp_path):
    config_path = tmp_path / 'custom.yaml'
    config_path.write_text("mask_use: true\nviz: false\n")

    ns = argparse.Namespace(
        input='x.mp4',
        no_mask=True,  # explicitly passed -> must win over the config file
        viz=True,  # explicitly passed -> must win over the config file
        save_viz=False,
        custom_config=config_path,
    )
    _, kwargs = separate_cli_arguments(ns)
    assert kwargs['mask_use'] is False
    assert kwargs['viz'] is True


def test_cli_update_check_fires_once_end_to_end(tmp_path, monkeypatch):
    """Regression: main() used to call check_for_updates() directly (not the *_once variant),
    so a Stabilizer constructed during the same run fired the update check a second time."""
    import cv2

    import stabilo.version_check as vc

    monkeypatch.delenv('STABILO_DISABLE_UPDATE_CHECK', raising=False)
    monkeypatch.setattr(vc, '_cache_dir', lambda: tmp_path / 'cache')
    vc._state['checked'] = False
    calls = []
    monkeypatch.setattr(vc, 'check_for_updates', lambda logger=None, blocking=False: calls.append(1))

    video_path = tmp_path / 'clip.mp4'
    w, h, n = 128, 96, 6
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (w, h))
    if not writer.isOpened():
        pytest.skip('no available VideoWriter codec')
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    writer.release()
    if not video_path.exists():
        pytest.skip('VideoWriter produced no file')

    try:
        main(['video', str(video_path), '--save', '--no-mask', '--downsample-ratio', '1.0'])
        assert len(calls) == 1
    finally:
        vc._state['checked'] = False


def test_video_end_to_end(tmp_path, monkeypatch):
    import cv2

    monkeypatch.setenv('STABILO_DISABLE_UPDATE_CHECK', '1')
    video_path = tmp_path / 'clip.mp4'
    w, h, n = 128, 96, 6
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (w, h))
    if not writer.isOpened():
        pytest.skip('no available VideoWriter codec')
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))
    writer.release()
    if not video_path.exists():
        pytest.skip('VideoWriter produced no file')

    main(['video', str(video_path), '--save', '--no-mask', '--downsample-ratio', '1.0'])
    assert (tmp_path / 'clip_stab.mp4').exists()


def _failed_estimation_stabilizer():
    """
    A Stabilizer left in the state produced when RANSAC has too few correspondences to estimate
    a transformation matrix.
    """
    frame = cv2.imread('tests/ND_before.jpg')
    frame = cv2.resize(frame, (1280, 720))

    stab = Stabilizer(viz=True, mask_use=False, downsample_ratio=1.0)
    stab.set_ref_frame(frame)

    stab.cur_frame = frame
    stab.cur_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    stab.cur_mask = None
    stab.ref_pts = np.float32([[1, 1], [2, 2], [3, 3]])
    stab.cur_pts = np.float32([[4, 4], [5, 5], [6, 6]])
    stab.calculate_transformation_matrix()
    return stab, frame


def test_failed_estimation_leaves_none_inliers_count():
    stab, _ = _failed_estimation_stabilizer()
    assert stab.cur_inliers_count is None
    assert stab.get_cur_inliers_count() is None


def test_render_visuals_survives_failed_estimation():
    """Regression: the outlier count used to do int - None and raise TypeError."""
    stab, frame = _failed_estimation_stabilizer()
    args = argparse.Namespace(debug=False, no_lines=False, no_boxes=True, ref_frame=0)

    imgs = render_stabilization_visuals(stab, frame.copy(), frame.copy(), None, None, 7, args, 'xywh')

    assert imgs is not None and imgs.ndim == 3


def test_dl_resolution_warning(caplog):
    """A large frame with a memory-hungry detector warns and suggests a downsample_ratio."""
    stab = Stabilizer(mask_use=False)
    stab.detector_name = 'loftr'
    stab.h, stab.w = 2160, 3840
    stab.downsample_ratio = 0.5

    with caplog.at_level('WARNING'):
        stab._warn_dl_resolution()

    assert 'loftr' in caplog.text and 'downsample_ratio <=' in caplog.text
    assert 'quadratically' in caplog.text


def test_dl_resolution_warning_silent_when_small(caplog):
    stab = Stabilizer(mask_use=False)
    stab.detector_name = 'xfeat'
    stab.h, stab.w = 720, 1280
    stab.downsample_ratio = 0.5

    with caplog.at_level('WARNING'):
        stab._warn_dl_resolution()

    assert caplog.text == ''


def test_dl_resolution_warning_skipped_for_classical(caplog):
    stab = Stabilizer(mask_use=False)
    stab.h, stab.w = 2160, 3840

    with caplog.at_level('WARNING'):
        stab._warn_dl_resolution()

    assert caplog.text == ''
