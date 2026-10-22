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


# --- resource-error propagation (needs kornia importable, but no weights) ---

requires_kornia = pytest.mark.skipif(not KORNIA, reason='kornia is not installed')


class _RaisingDetector:
    wants = 'gray'

    def __init__(self, error):
        self.error = error

    def detectAndCompute(self, image, mask):
        raise self.error


def _dl_features_with(error, ref_frame=True):
    stab = Stabilizer(detector_name='orb', downsample_ratio=1.0, mask_use=False, clahe=False)
    stab.detector_ref = stab.detector_cur = _RaisingDetector(error)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return stab._get_features_and_descriptors_dl(frame, None, ref_frame)


@requires_kornia
@pytest.mark.parametrize(
    'make_error',
    [
        lambda: pytest.importorskip('torch').OutOfMemoryError('CUDA out of memory. Tried to allocate 20.00 GiB'),
        lambda: RuntimeError('MPS backend out of memory (MPS allocated: 18.00 GB, other allocations: 1.2 GB)'),
        lambda: RuntimeError('Invalid buffer size: 24.00 GB'),
        lambda: RuntimeError("DefaultCPUAllocator: can't allocate memory: you tried to allocate 68719476736 bytes."),
        lambda: RuntimeError('DefaultCPUAllocator: not enough memory: you tried to allocate 68719476736 bytes.'),
    ],
    ids=['torch-oom', 'mps-oom', 'mps-buffer', 'cpu-linux', 'cpu-windows'],
)
@pytest.mark.parametrize('ref_frame', [True, False])
def test_dl_resource_error_propagates(make_error, ref_frame):
    error = make_error()
    with pytest.raises(type(error)) as excinfo:
        _dl_features_with(error, ref_frame)
    assert excinfo.value is error


@requires_kornia
@pytest.mark.parametrize(
    'error',
    [
        RuntimeError('The size of tensor a (64) must match the size of tensor b (32) at non-singleton dimension 1'),
        RuntimeError('cannot reshape tensor of 0 elements into shape [-1, 0]'),
        ValueError('Expected more than 1 value per channel'),
    ],
)
def test_dl_ordinary_failure_is_still_caught(error):
    assert _dl_features_with(error) == (None, None, None)


class _FakeLoFTR:
    """
    Stand-in for kornia's LoFTR: records the input shapes and returns one match inside the original
    image and one in the bottom-right padding.
    """

    def __init__(self):
        self.shapes = []

    def eval(self):
        return self

    def to(self, device):
        return self

    def __call__(self, data):
        torch = pytest.importorskip('torch')
        self.shapes.append((tuple(data['image0'].shape), tuple(data['image1'].shape)))
        h, w = data['image0'].shape[-2:]
        xy = torch.tensor([[10.0, 12.0], [w - 1.0, h - 1.0]])
        return {'keypoints0': xy, 'keypoints1': xy.clone(), 'confidence': torch.tensor([0.9, 0.8])}


@requires_kornia
@pytest.mark.parametrize('shape', [(389, 691), (432, 768), (540, 960)])
def test_loftr_input_is_padded_to_a_multiple_of_eight(shape):
    torch = pytest.importorskip('torch')
    dl = pytest.importorskip('stabilo.dl')

    model = _FakeLoFTR()
    image = torch.rand(1, 1, *shape)
    ref_xy, cur_xy, conf = dl._LoFTRMatcher(model, torch.device('cpu')).match(image, image)
    for padded in model.shapes[0]:
        assert padded[-2] % 8 == 0 and padded[-1] % 8 == 0
        assert padded[-2] - shape[0] < 8 and padded[-1] - shape[1] < 8
    divisible = shape[0] % 8 == 0 and shape[1] % 8 == 0
    assert len(ref_xy) == len(cur_xy) == len(conf) == (2 if divisible else 1)
    np.testing.assert_array_equal(ref_xy[0], [10.0, 12.0])


@requires_kornia
def test_pad_to_multiple_keeps_the_image_at_the_origin():
    torch = pytest.importorskip('torch')
    dl = pytest.importorskip('stabilo.dl')

    image = torch.rand(1, 1, 389, 691)
    padded = dl.pad_to_multiple(image, 8)
    assert tuple(padded.shape) == (1, 1, 392, 696)
    assert torch.equal(padded[..., :389, :691], image)
    assert padded[..., 389:, :].abs().sum() == 0 and padded[..., :, 691:].abs().sum() == 0
    assert dl.pad_to_multiple(padded, 8) is padded


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
@pytest.mark.parametrize('downsample_ratio', [0.75, 1.0])
def test_loftr_recovers_a_known_homography_at_a_size_not_divisible_by_eight(images, downsample_ratio):
    """724x503 and 543x377 are not multiples of LoFTR's coarse stride of 8."""
    _, ref_frame = images
    h, w = ref_frame.shape[:2]
    true = np.array([[1.0, -0.01, 12.0], [0.01, 1.0, -8.0], [0.0, 0.0, 1.0]])
    cur_frame = cv2.warpPerspective(ref_frame, true, (w, h))
    stab = Stabilizer(detector_name='loftr', downsample_ratio=downsample_ratio, device='cpu', mask_use=False)
    stab.set_ref_frame(ref_frame)
    stab.stabilize(cur_frame)
    corners = np.array([[[0, 0], [w, 0], [w, h], [0, h]]], dtype=np.float64)
    estimated = cv2.perspectiveTransform(corners, stab.get_cur_trans_matrix())
    expected = cv2.perspectiveTransform(corners, np.linalg.inv(true))
    assert np.linalg.norm(estimated - expected, axis=2).mean() < 1.0


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
