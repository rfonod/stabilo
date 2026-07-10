# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
dl.py - Deep-learning (kornia) feature detectors, descriptors, and matchers for stabilo.

This module is imported lazily and only when a deep-learning detector or matcher is selected.
It depends on kornia and torch, which are core stabilo dependencies (Python >= 3.11).

Sparse detectors ('xfeat', 'disk', 'dedode', 'keynet') are wrapped to expose a
`detectAndCompute(image, mask)` method returning OpenCV keypoints and float32 descriptors, so
they plug into the existing 'bf'/'flann' matching path. 'loftr' is detector-free and produces
matched point pairs directly; 'lightglue' is a learned matcher for the sparse detectors.
"""

import warnings
from contextlib import contextmanager
from pathlib import Path

import cv2
import kornia.feature as KF
import numpy as np
import torch

# kornia's DeDoDe calls torch.autocast('cuda') unconditionally and warns on every forward off CUDA
warnings.filterwarnings("ignore", message=".*Disabling autocast.*", category=UserWarning)


def weights_cache_dir() -> Path:
    """
    Directory where torch.hub caches the pretrained weights used by the kornia models.
    """
    return Path(torch.hub.get_dir()) / 'checkpoints'


@contextmanager
def report_weights_cache(logger, label: str):
    """
    Log whether the weights built inside this context were downloaded or reused from the local cache.
    """
    cache_dir = weights_cache_dir()

    def listing():
        return {path.name for path in cache_dir.glob('*')} if cache_dir.is_dir() else set()

    before = listing()
    yield
    if logger is None:
        return
    downloaded = sorted(listing() - before)
    if downloaded:
        logger.info(
            f"Downloaded {label} weights ({', '.join(downloaded)}) to {cache_dir}. "
            f"Later runs reuse this cache offline; set TORCH_HOME to relocate it."
        )
    else:
        logger.info(f"Reusing cached {label} weights from {cache_dir}.")


def resolve_device(device: str) -> torch.device:
    """
    Resolve a torch device from a device string ('auto', 'cpu', 'cuda', 'mps').
    """
    mps_available = getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available()
    if device == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        if mps_available:
            return torch.device('mps')
        return torch.device('cpu')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError("device='cuda' was requested, but CUDA is not available to torch")
    if device == 'mps' and not mps_available:
        raise ValueError("device='mps' was requested, but MPS is not available to torch")
    return torch.device(device)


def to_tensor_gray(img_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an HxW uint8 grayscale image to a (1, 1, H, W) float tensor in [0, 1].
    """
    return (torch.from_numpy(img_u8).float()[None, None] / 255.0).to(device)


def to_tensor_rgb(img_u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    Convert an HxWx3 uint8 RGB image to a (1, 3, H, W) float tensor in [0, 1].
    """
    return (torch.from_numpy(img_u8).float().permute(2, 0, 1)[None] / 255.0).to(device)


def keypoints_to_cv2(xy: np.ndarray, scores=None) -> list:
    """
    Convert an (N, 2) array of point coordinates to a list of cv2.KeyPoint.
    """
    if scores is None:
        return [cv2.KeyPoint(float(x), float(y), 1.0) for x, y in xy]
    return [cv2.KeyPoint(float(x), float(y), 1.0, response=float(s)) for (x, y), s in zip(xy, scores, strict=False)]


def filter_by_mask(xy: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Boolean array selecting points that fall on a non-zero pixel of an HxW uint8 mask.
    """
    if mask is None or len(xy) == 0:
        return np.ones(len(xy), dtype=bool)
    h, w = mask.shape[:2]
    xi = np.clip(np.round(xy[:, 0]).astype(int), 0, w - 1)
    yi = np.clip(np.round(xy[:, 1]).astype(int), 0, h - 1)
    return mask[yi, xi] != 0


class _SparseDLDetector:
    """
    Base wrapper adapting a kornia sparse detector to a cv2-style detectAndCompute API.
    """

    wants = 'gray'

    def __init__(self, model, device, num_features):
        self.model = model.eval().to(device)
        self.device = device
        self.num_features = num_features

    def _to_tensor(self, img):
        if self.wants == 'gray':
            return to_tensor_gray(img, self.device)
        return to_tensor_rgb(img, self.device)

    def _run(self, tensor):
        raise NotImplementedError

    def detectAndCompute(self, img, mask=None):
        tensor = self._to_tensor(img)
        with torch.inference_mode():
            xy, scores, desc = self._run(tensor)
        if desc is None or len(xy) == 0:
            return [], None
        keep = filter_by_mask(xy, mask)
        xy, desc = xy[keep], desc[keep]
        scores = scores[keep] if scores is not None else None
        if len(xy) == 0:
            return [], None
        return keypoints_to_cv2(xy, scores), np.ascontiguousarray(desc, dtype=np.float32)


class _XFeatDetector(_SparseDLDetector):
    wants = 'rgb'

    def _run(self, tensor):
        feats = self.model.detectAndCompute(tensor, top_k=self.num_features)[0]
        xy = feats['keypoints'].cpu().numpy()
        desc = feats['descriptors'].cpu().numpy()
        scores = feats['scores'].cpu().numpy() if 'scores' in feats else None
        return xy, scores, desc


class _DISKDetector(_SparseDLDetector):
    wants = 'rgb'

    def _run(self, tensor):
        features = self.model(tensor, n=self.num_features, pad_if_not_divisible=True)[0]
        return (
            features.keypoints.cpu().numpy(),
            features.detection_scores.cpu().numpy(),
            features.descriptors.cpu().numpy(),
        )


class _DeDoDeDetector(_SparseDLDetector):
    wants = 'rgb'

    def _run(self, tensor):
        keypoints, scores, descriptions = self.model(tensor, n=self.num_features)
        return keypoints[0].cpu().numpy(), scores[0].cpu().numpy(), descriptions[0].cpu().numpy()


class _KeyNetDetector(_SparseDLDetector):
    wants = 'gray'

    def _run(self, tensor):
        lafs, responses, descs = self.model(tensor)
        xy = KF.get_laf_center(lafs)[0].cpu().numpy()
        scores = responses.reshape(-1).cpu().numpy()
        return xy, scores, descs[0].cpu().numpy()


class _LoFTRMatcher:
    """
    Detector-free LoFTR wrapper producing matched point pairs for an image pair.
    """

    def __init__(self, model, device):
        self.model = model.eval().to(device)
        self.device = device

    def match(self, ref_tensor, cur_tensor):
        with torch.inference_mode():
            out = self.model({'image0': ref_tensor.to(self.device), 'image1': cur_tensor.to(self.device)})
        return (
            out['keypoints0'].cpu().numpy(),
            out['keypoints1'].cpu().numpy(),
            out['confidence'].cpu().numpy(),
        )


class _LightGlueMatcher:
    """
    Learned LightGlue matcher operating on cv2 keypoints + float32 descriptors.
    """

    def __init__(self, matcher, device):
        self.matcher = matcher.eval().to(device)
        self.device = device

    def _lafs(self, kpts):
        xy = torch.tensor([k.pt for k in kpts], dtype=torch.float32, device=self.device).reshape(1, -1, 2)
        n = xy.shape[1]
        scale = torch.ones(1, n, 1, 1, device=self.device)
        ori = torch.zeros(1, n, 1, device=self.device)
        return KF.laf_from_center_scale_ori(xy, scale, ori)

    def match(self, desc1, desc2, kpts1, kpts2, hw):
        if len(kpts1) == 0 or len(kpts2) == 0:
            return []
        d1 = torch.from_numpy(np.ascontiguousarray(desc1, dtype=np.float32)).to(self.device)
        d2 = torch.from_numpy(np.ascontiguousarray(desc2, dtype=np.float32)).to(self.device)
        lafs1, lafs2 = self._lafs(kpts1), self._lafs(kpts2)
        with torch.inference_mode():
            dists, idxs = self.matcher(d1, d2, lafs1, lafs2, hw, hw)
        idxs = idxs.cpu().numpy()
        dists = dists.cpu().numpy().reshape(-1)
        return [cv2.DMatch(int(q), int(t), float(dist)) for (q, t), dist in zip(idxs, dists, strict=False)]


def create_dl_detectors(stab):
    """
    Build the (current, reference) sparse deep-learning detector pair for a Stabilizer.
    """
    device = stab._torch_device
    name = stab.detector_name
    n_cur = stab.max_features
    n_ref = round(stab.ref_multiplier * stab.max_features)

    def build(n):
        if name == 'xfeat':
            return _XFeatDetector(KF.XFeat.from_pretrained(top_k=n), device, n)
        if name == 'disk':
            return _DISKDetector(KF.DISK.from_pretrained(stab.disk_weights), device, n)
        if name == 'dedode':
            # DeDoDe defaults to float16 AMP, which only works on CUDA
            amp_dtype = torch.float16 if device.type == 'cuda' else torch.float32
            return _DeDoDeDetector(
                KF.DeDoDe.from_pretrained(
                    detector_weights=stab.dedode_detector_weights,
                    descriptor_weights=stab.dedode_descriptor_weights,
                    amp_dtype=amp_dtype,
                ),
                device,
                n,
            )
        if name == 'keynet':
            return _KeyNetDetector(KF.KeyNetHardNet(num_features=n, device=device), device, n)
        raise ValueError(f"Unsupported deep-learning detector: {name}")

    with report_weights_cache(None if stab.benchmark else stab.logger, name):
        return build(n_cur), build(n_ref)


def create_loftr(stab):
    """
    Build the LoFTR matcher for a Stabilizer.
    """
    with report_weights_cache(None if stab.benchmark else stab.logger, 'loftr'):
        return _LoFTRMatcher(KF.LoFTR(pretrained=stab.loftr_weights), stab._torch_device)


def create_lightglue_matcher(stab):
    """
    Build the LightGlue matcher for a Stabilizer, keyed by the selected detector.
    """
    feature_name = stab.LIGHTGLUE_FEATURE_NAMES[stab.detector_name]
    if stab.detector_name == 'dedode':
        feature_name = 'dedodeb' if str(stab.dedode_descriptor_weights).upper().startswith('B') else 'dedodeg'
    with report_weights_cache(None if stab.benchmark else stab.logger, 'lightglue'):
        return _LightGlueMatcher(KF.LightGlueMatcher(feature_name), stab._torch_device)
