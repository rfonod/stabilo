# Stabilo — Detailed Usage Guide

> **Command-line interface**: installing stabilo provides a `stabilo` console command with `video`, `tracks`, and `config` subcommands. See [15. Command-Line Interface](#15-command-line-interface) and the [`scripts/README.md`](../scripts/README.md) for the threshold-analysis tooling.
>
> **CUDA GPU acceleration** (`gpu=True`) requires a CUDA-enabled OpenCV build; see [`docs/cuda.md`](cuda.md) for build instructions. Learning-based detectors instead run on any torch device via the `device` option.

---

## Table of Contents

- [Stabilo — Detailed Usage Guide](#stabilo--detailed-usage-guide)
  - [Table of Contents](#table-of-contents)
  - [1. Core Functionality](#1-core-functionality)
  - [2. Installation](#2-installation)
  - [3. Main Workflow](#3-main-workflow)
    - [Key constraints](#key-constraints)
  - [4. Supported Mask and Box Formats](#4-supported-mask-and-box-formats)
    - [`xywh` (default)](#xywh-default)
    - [`xywha`](#xywha)
    - [`four`](#four)
    - [`polygon`](#polygon)
    - [`circle`](#circle)
  - [5. Masking Behaviour](#5-masking-behaviour)
  - [6. Frame and Box Transformation](#6-frame-and-box-transformation)
    - [Warp a frame](#warp-a-frame)
    - [Transform bounding boxes](#transform-bounding-boxes)
  - [7. Point Transformation](#7-point-transformation)
  - [8. Configuration Parameters](#8-configuration-parameters)
  - [9. Feature Detectors](#9-feature-detectors)
  - [10. Feature Matching and Filtering](#10-feature-matching-and-filtering)
  - [11. Transformation Types and RANSAC Methods](#11-transformation-types-and-ransac-methods)
    - [Transformation types](#transformation-types)
    - [RANSAC methods (integer codes)](#ransac-methods-integer-codes)
    - [Method and transformation compatibility](#method-and-transformation-compatibility)
  - [12. Visualisation Mode](#12-visualisation-mode)
  - [13. Benchmarking Mode](#13-benchmarking-mode)
  - [14. Testing and Development](#14-testing-and-development)
    - [Tests that are skipped by default](#tests-that-are-skipped-by-default)
    - [Core-dependency smoke test](#core-dependency-smoke-test)
  - [15. Command-Line Interface](#15-command-line-interface)
    - [Using an external logger](#using-an-external-logger)

---

## 1. Core Functionality

Stabilo aligns a **current frame** to a **reference frame** using a feature-point pipeline:

1. **Pre-processing** — optionally apply CLAHE for contrast enhancement and downsample the frame.
2. **Masking** — build a binary mask from user-supplied exclusion regions so that dynamic objects (e.g., vehicles) are excluded from feature extraction.
3. **Feature detection and description** — detect and describe keypoints using a chosen classical detector (ORB, SIFT, rSIFT, BRISK, KAZE, AKAZE) or learning-based detector (XFeat, DISK, DeDoDe, KeyNet), or match densely with the detector-free LoFTR.
4. **Matching** — match descriptors between the current and reference frame using a brute-force (BF), FLANN, or learned LightGlue matcher.
5. **Match filtering** — filter matches by cross-check, Lowe's ratio test, or distance threshold.
6. **Transformation estimation** — robustly estimate a 3x3 homography (*projective*) or 2x3 affine matrix using a RANSAC-type algorithm (MAGSAC++ by default).
7. **Stabilisation** — warp the current frame, or transform tracked bounding boxes, into the reference coordinate system.

This pipeline makes Stabilo suitable for:

- video stabilisation (align all frames to a chosen anchor frame),
- trajectory stabilisation (transform per-frame detections/tracks to a fixed reference),
- any downstream analysis requiring geometric frame-to-frame consistency.

---

## 2. Installation

It is recommended to create and activate a **Python virtual environment** (Python >= 3.11 and <= 3.13) first:

```bash
python3.11 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

<details>
<summary>Alternatives: conda or uv</summary>

**[Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install):**
```bash
conda create -n stabilo python=3.11 -y
conda activate stabilo
```

**[uv](https://docs.astral.sh/uv/getting-started/installation/) (fastest; use `uv pip install` in the step below):**
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
</details>

Then, install the `stabilo` package using one of the following options:

**Option 1 — from PyPI** (recommended):
```bash
pip install stabilo
```

**Option 2 — from local source** (for development):
```bash
git clone https://github.com/rfonod/stabilo.git
cd stabilo
pip install -e '.[dev]'  # editable install with dev dependencies
```

---

## 3. Main Workflow

```python
import numpy as np
from stabilo import Stabilizer

# 1. Create a stabilizer (all arguments are optional; defaults shown below)
stabilizer = Stabilizer(
    detector_name='orb',
    matcher_name='bf',
    filter_type='ratio',
    transformation_type='projective',
    max_features=2000,
)

# 2. Set the reference frame (once)
#    Optionally supply bounding-box masks to exclude dynamic regions.
stabilizer.set_ref_frame(ref_frame, ref_boxes, box_format='xywh')

# 3. Stabilize each subsequent frame
stabilizer.stabilize(cur_frame, cur_boxes, box_format='xywh')

# 4. Retrieve results
stabilized_frame  = stabilizer.warp_cur_frame()              # warped frame
stabilized_boxes  = stabilizer.transform_cur_boxes()         # transformed boxes (xywh)
trans_matrix      = stabilizer.get_cur_trans_matrix()        # 3x3 or 2x3 matrix
```

### Key constraints

- `set_ref_frame(...)` **must** be called before `stabilize(...)`.
- All frames passed to `stabilize(...)` are aligned to the single reference frame; call `set_ref_frame(...)` again to change the anchor.
- If a transformation cannot be estimated (too few matches), Stabilo falls back to the last known transformation matrix (or `None` if no previous matrix exists).

---

## 4. Supported Mask and Box Formats

The `box_format` argument is accepted by `set_ref_frame`, `stabilize`, and `create_binary_mask`. It controls how the `boxes` array is interpreted.
The `transform_boxes(...)` method uses `in_box_format` and `out_box_format` to specify the input and output box formats.

### `xywh` (default)

Axis-aligned bounding box, one row per box:

```
[x_center, y_center, width, height]
```

### `xywha`

Oriented bounding box (OBB) with an explicit rotation angle:

```
[x_center, y_center, width, height, angle_degrees]
```

`angle_degrees` is the counter-clockwise rotation from the positive x-axis.

### `four`

Four corner points (clockwise or arbitrary order), one row per box:

```
[x1, y1, x2, y2, x3, y3, x4, y4]
```

Stabilo automatically detects whether the box is axis-aligned or rotated. For axis-aligned boxes fast rectangular masking is used; rotated boxes use polygon filling.

### `polygon`

Arbitrary convex or non-convex polygons with **N >= 3** vertices. Two input shapes are accepted:

- **Flattened row** — one polygon per row: `[x1, y1, x2, y2, ..., xN, yN]`
  (all rows must have the same even length >= 6)
- **Vertex-pair array** — shape `(N, 2)` for a single polygon, or shape `(P, N, 2)` for P polygons

```python
# Flattened rows (two quadrilaterals)
polygon_masks = np.array([
    [60, 60, 120, 60, 120, 120, 60, 120],
    [200, 200, 260, 210, 240, 270, 190, 260],
])
stabilizer.set_ref_frame(ref_frame, polygon_masks, box_format='polygon')
```

### `circle`

Circular exclusion regions, one row per circle:

```
[x_center, y_center, radius]
```

```python
circle_masks = np.array([[120, 120, 25], [360, 240, 40]])
stabilizer.stabilize(cur_frame, circle_masks, box_format='circle')
```

> **Note** — `polygon` and `circle` formats are supported **only for masking** (i.e., feature-extraction exclusion). They cannot be passed to `transform_cur_boxes()` / `transform_boxes()`, which require `xywh`, `xywha`, or `four`.

---

## 5. Masking Behaviour

When boxes are provided, `create_binary_mask` builds a single-channel `uint8` image (same height x width as the reference frame) where:

| Pixel value | Meaning |
|-------------|---------|
| `255` | Included in feature extraction |
| `0`   | Excluded from feature extraction |

The `mask_margin_ratio` parameter (default `0.15`) **expands** each exclusion region by 15% to add a safety margin:

| Format | How margin is applied |
|--------|-----------------------|
| `xywh` | Width and height each multiplied by `(1 + margin_ratio)` |
| `xywha` | Width and height expanded, then the OBB is converted to four corner points |
| `four` — aligned | Same as `xywh` after bounding-box extraction |
| `four` — rotated | Corner points scaled away from their centroid |
| `polygon` | Vertices scaled away from their centroid |
| `circle` | Radius multiplied by `(1 + margin_ratio)` |

Masks for the reference and current frame are **independent** — each frame may have its own set of exclusion boxes.

If `mask_use=False`, no mask is created even if boxes are supplied.

---

## 6. Frame and Box Transformation

### Warp a frame

```python
stabilized = stabilizer.warp_cur_frame()
# or warp an arbitrary frame with the current matrix:
stabilized = stabilizer.warp_frame(some_frame)
```

For `projective` transformations `cv2.warpPerspective` is used; for `affine` `cv2.warpAffine` is used.

### Transform bounding boxes

```python
# transform boxes that were passed to stabilize(...)
out_boxes = stabilizer.transform_cur_boxes(out_box_format='xywh')

# transform arbitrary boxes
out_boxes = stabilizer.transform_boxes(
    boxes,
    trans_matrix,
    in_box_format='xywha',
    out_box_format='four',
)
```

Internally, every box format is first converted to four corner points (`four`), those points are transformed with the matrix, then converted back to the requested output format.

Supported `in_box_format` / `out_box_format` values for transformation: `'xywh'`, `'xywha'`, `'four'`.

---

## 7. Point Transformation

To transform an arbitrary pixel coordinate from the current frame to the reference frame:

```python
# projective (homography)
cur_point = np.array([x, y, 1.0])
ref_point_h = stabilizer.get_cur_trans_matrix() @ cur_point
ref_point = ref_point_h[:2] / ref_point_h[2]   # divide by homogeneous coordinate

# affine (no perspective division needed — last row is [0, 0, 1])
```

---

## 8. Configuration Parameters

All parameters can be passed as keyword arguments to `Stabilizer(...)` or set via a YAML file (see `stabilo/cfg/default.yaml`). Parameters not supplied fall back to their defaults.

| Parameter | Default | Valid values | Description |
|-----------|---------|--------------|-------------|
| `detector_name` | `'orb'` | `orb`, `sift`, `rsift`, `brisk`, `kaze`, `akaze`, `xfeat`, `disk`, `dedode`, `keynet`, `loftr` | Feature detector (see section 9) |
| `matcher_name` | `'bf'` | `bf`, `flann`, `lightglue` | Feature matcher (see section 10) |
| `filter_type` | `'ratio'` | `none`, `ratio`, `distance` | Match filtering strategy |
| `filter_ratio` | `0.9` | `(0, 1]` | Lowe's ratio (for `ratio`) or distance threshold ratio (for `distance`) |
| `match_query_frame` | `'reference'` | `reference`, `current` | Which frame's descriptors form the `knnMatch` query (matching is asymmetric) |
| `transformation_type` | `'projective'` | `projective`, `affine` | Geometric model |
| `clahe` | `false` | `true`, `false` | Apply CLAHE contrast enhancement |
| `downsample_ratio` | `0.5` | `(0, 1]` | Resize factor before feature extraction |
| `max_features` | `2000` | `> 0` (int) | Maximum keypoints to detect in current frame |
| `ref_multiplier` | `2.0` | `>= 1.0` | Scale factor for keypoints in reference frame (`ref_multiplier x max_features`) |
| `sift_enable_precise_upscale` | `false` | `true`, `false` | SIFT/rSIFT only: enable precise sub-pixel upscaling at octave -1 |
| `rsift_eps` | `1e-8` | `> 0` | RootSIFT (`rsift`) L1-normalization epsilon |
| `mask_use` | `true` | `true`, `false` | Enable exclusion masking |
| `mask_margin_ratio` | `0.15` | `[0, 1]` | Fractional margin added to exclusion regions |
| `ransac_method` | `38` | see section 11 | RANSAC algorithm (`affine` supports only `4` and `8`) |
| `ransac_epipolar_threshold` | `2.0` | `> 0` | Reprojection-error threshold (pixels) |
| `ransac_max_iter` | `5000` | `> 0` (int) | Maximum RANSAC iterations |
| `ransac_confidence` | `0.999999` | `(0, 1]` | Required confidence level |
| `brisk_threshold` | `130` | `(0, 255]` | BRISK detector threshold (fallback) |
| `kaze_threshold` | `0.01` | `> 0` | KAZE detector threshold (fallback) |
| `akaze_threshold` | `0.01` | `> 0` | AKAZE detector threshold (fallback) |
| `gpu` | `false` | `true`, `false` | Use CUDA GPU acceleration for detection/matching/warping; requires a CUDA-enabled OpenCV build, see [`docs/cuda.md`](cuda.md) |
| `gpu_device_id` | `0` | `>= 0` (int) | CUDA device index to use when `gpu` is true |
| `device` | `'auto'` | `auto`, `cpu`, `cuda`, `mps` | Torch device for learning-based detectors/matchers only (`auto` picks cuda > mps > cpu) |
| `loftr_weights` | `'outdoor'` | `outdoor`, `indoor` | LoFTR pretrained weights |
| `loftr_confidence` | `0.0` | `[0, 1]` | Minimum LoFTR correspondence confidence to keep (`0.0` keeps all) |
| `disk_weights` | `'depth'` | `depth`, `epipolar` | DISK pretrained weights |
| `dedode_detector_weights` | `'L-C4-v2'` | `L-upright`, `L-C4`, `L-SO2`, `L-C4-v2` | DeDoDe detector weights |
| `dedode_descriptor_weights` | `'B-upright'` | `B-upright`, `G-upright`, `B-C4`, `B-SO2`, `G-C4`, `G-SO2` | DeDoDe descriptor weights. The `G` variants pull a 1.2 GB DINOv2 backbone and roughly double the memory for no rotation benefit, so `B` is the default |
| `logger` | `None` | `logging.Logger` | Optional external logger (constructor-only; not a YAML key) |
| `viz` | `false` | `true`, `false` | Retain intermediate data for visualisation |
| `benchmark` | `false` | `true`, `false` | Benchmarking mode (see section 13) |
| `min_good_match_count_warning` | `20` | `>= 0` | Warn if fewer than N good matches found |
| `min_inliers_match_count_warning` | `10` | `>= 0` | Warn if fewer than N inliers found |

---

## 9. Feature Detectors

**Classical detectors** (OpenCV):

| Name | Type | Notes |
|------|------|-------|
| `orb` | Binary | Fast; uses Hamming distance. Default. |
| `sift` | Float | Scale and rotation invariant; uses L2 distance. |
| `rsift` | Float | RootSIFT variant of SIFT with improved matching quality. |
| `brisk` | Binary | Threshold auto-derived from `max_features` via a pre-fitted model; falls back to `brisk_threshold`. |
| `kaze` | Float | Non-linear scale-space detector; threshold auto-derived similarly. |
| `akaze` | Binary | Accelerated KAZE; threshold auto-derived similarly. |

For BRISK, KAZE, and AKAZE, Stabilo ships pre-fitted linear regression models that translate `max_features` into the appropriate detector threshold. These models are stored under `stabilo/thresholds/models/` and are selected based on `mask_use` and `clahe` settings. If no model is available the fallback thresholds (`brisk_threshold`, etc.) are used.

**Learning-based detectors** ([kornia](https://kornia.readthedocs.io/), core dependency; pretrained weights download on first use and run on the `device`):

| Name | Type | Rotation invariant | Notes |
|------|------|:---:|-------|
| `xfeat` | Float (64-d) | ❌ | Fast learned sparse detector/descriptor, and by far the lightest of the learned models. Works with `bf`/`flann`. |
| `disk` | Float (128-d) | ❌ | Robust learned sparse features. Works with `bf`/`flann`/`lightglue`. |
| `dedode` | Float (256-d) | ❌ | Decoupled detect-and-describe; strong but slower. Works with `bf`/`flann`/`lightglue`. Weights via `dedode_detector_weights` / `dedode_descriptor_weights`. |
| `keynet` | Float (128-d) | ✅ | KeyNet detector + HardNet descriptor, with an OriNet orientation estimator. The only rotation-invariant learned option. Works with `bf`/`flann`/`lightglue`. |
| `loftr` | Detector-free | ❌ | Semi-dense image-pair matching; produces correspondences directly, so `matcher_name` and `filter_type` are ignored. Weights via `loftr_weights`; filter correspondences with `loftr_confidence`. |

Learning-based detectors receive the same CLAHE, downsampling, and exclusion-mask handling as the classical ones: masks are applied by discarding keypoints (or LoFTR correspondences) that fall in excluded regions, and keypoint coordinates are rescaled back to full resolution after downsampled detection. `max_features` caps the number of detected keypoints. The `brisk_threshold`/`kaze_threshold`/`akaze_threshold` models do not apply. Select the compute device with `device` (`auto` picks cuda > mps > cpu). Note that `gpu=True` (OpenCV CUDA) is incompatible with learning-based detectors; use `device='cuda'` instead.

Pretrained weights are downloaded once, on first use, into torch's hub cache (`~/.cache/torch/hub/checkpoints` on Linux and macOS, `%USERPROFILE%\.cache\torch\hub\checkpoints` on Windows). Set the `TORCH_HOME` environment variable to relocate it. Stabilo logs whether the weights were downloaded or reused from that cache, so a run that stalls on a first-use download is easy to recognise.

### Rotation invariance

`xfeat`, `disk`, `dedode`, and `loftr` are trained on upright imagery and do not estimate a keypoint orientation. Matching against the reference frame degrades as the current frame rotates, and collapses beyond roughly 30°. Measured on a synthetically rotated frame pair, the mean reprojection error of the recovered homography is:

| Detector | 0° | 10° | 30° | 90° |
|---|---|---|---|---|
| `orb`, `sift` | 0.0 px | 0.4 px | 0.6 px | 1.2 px |
| `keynet` | 0.0 px | 0.7 px | 1.3 px | 2.7 px |
| `xfeat` | 0.0 px | 0.9 px | 1.8 px | fails |
| `disk` | 0.0 px | 0.4 px | 11.8 px | fails |
| `dedode` | 0.0 px | 0.5 px | 1.1 px | fails |
| `loftr` | 0.1 px | 0.5 px | 1.3 px | fails |

For footage with large in-plane rotation, such as a drone orbiting or yawing about its own axis, use a classical detector or `keynet`. Note that DeDoDe's `C4`/`SO2` weight variants are *not* a workaround: they require a steerer to be applied at match time, which kornia does not do.

### Choosing a downsample ratio

The learned models allocate memory in proportion to the *processed* frame size, which is `downsample_ratio²` times the input resolution. They are much hungrier than the classical detectors: at roughly 0.5 MP, `disk`, `dedode`, `keynet`, and `loftr` each peak at several GB of RAM on CPU, while `xfeat` stays a few hundred MB. `loftr` is the extreme case because its coarse matching builds an N×N confidence matrix over N = (H/8)·(W/8) tokens, so its cost grows quadratically: a 4K frame at the default `downsample_ratio: 0.5` yields a 3.9 GB matrix (before intermediates), and at full resolution over 60 GB.

Stabilo emits a warning, but does not cap the resolution, when the processed frame exceeds the guideline below:

| Detector | Guideline (processed frame) | Suggested for a 4K input (3840×2160) |
|---|---|---|
| `xfeat` | 2.0 MP | `--downsample-ratio 0.49` |
| `disk`, `dedode`, `keynet` | 0.5 MP | `--downsample-ratio 0.25` |
| `loftr` | 0.3 MP | `--downsample-ratio 0.19` |

These are guidelines for a machine with a few GB of free RAM, not hard limits; a large GPU tolerates more. Lower `downsample_ratio` further if the process is killed or the machine begins to swap.

---

## 10. Feature Matching and Filtering

Three matchers are supported:

- **`bf`** — OpenCV `BFMatcher`. Uses `crossCheck=True` for `filter_type='none'` and `'distance'`; `crossCheck=False` for `filter_type='ratio'`.
- **`flann`** — OpenCV `FlannBasedMatcher`. Uses LSH index for binary descriptors and KD-Tree for float descriptors.
- **`lightglue`** — learned [kornia](https://kornia.readthedocs.io/) `LightGlueMatcher`, compatible with the `disk`, `dedode`, and `keynet` detectors only. It performs its own filtering, so `filter_type` is ignored. (The detector-free `loftr` needs no matcher.)

Three filtering strategies:

| `filter_type` | Behaviour |
|---------------|-----------|
| `none` | Keep all matches returned by the matcher |
| `ratio` | Lowe's ratio test: keep match `m` where `m.distance < filter_ratio x n.distance` |
| `distance` | Distance threshold: keep matches below `min_dist + (max_dist - min_dist) x filter_ratio` |

`knnMatch` is asymmetric, so the set of surviving matches depends on which descriptors are used as the query. By default (`match_query_frame='reference'`) the reference frame's descriptors are the query; set `match_query_frame='current'` to query with the current frame's descriptors instead. The estimated transform still maps current → reference in both cases.

---

## 11. Transformation Types and RANSAC Methods

### Transformation types

| `transformation_type` | Matrix shape | Function |
|-----------------------|-------------|----------|
| `projective` | 3 x 3 | `cv2.findHomography` |
| `affine` | 2 x 3 | `cv2.estimateAffinePartial2D` (4-DOF similarity) |

Use `projective` (default) when the camera undergoes any motion (pan, tilt, zoom, rotation). Use `affine` to restrict the estimate to a 4-DOF **similarity** transform: rotation, uniform scale, and translation. Despite the name, `cv2.estimateAffinePartial2D` does not estimate shear or non-uniform scale; a general 6-DOF affine would need `cv2.estimateAffine2D`, which stabilo does not wire up.

### RANSAC methods (integer codes)

| Code | Method | `projective` | `affine` |
|------|--------|:------------:|:--------:|
| 4 | LMEDS (`cv2.LMEDS`) | yes | yes |
| 8 | RANSAC (`cv2.RANSAC`) | yes | yes |
| 16 | RHO (`cv2.RHO`) | yes | no |
| 32 | USAC-Default, LO-RANSAC (`cv2.USAC_DEFAULT`) | yes | no |
| 33 | USAC-Parallel, LO-RANSAC run in parallel (`cv2.USAC_PARALLEL`) | yes | no |
| 35 | USAC-Fast, LO-RANSAC with fewer local-optimization iterations (`cv2.USAC_FAST`) | yes | no |
| 36 | USAC-Accurate, GC-RANSAC (`cv2.USAC_ACCURATE`) | yes | no |
| 37 | PROSAC (`cv2.USAC_PROSAC`) | yes | no |
| **38** | **MAGSAC++ (`cv2.USAC_MAGSAC`), default** | yes | no |

Codes 32, 33, and 35 are three LO-RANSAC configurations that differ in how local optimization is scheduled, per OpenCV's [USAC tutorial](https://docs.opencv.org/4.x/de/d3f/tutorial_usac.html). None of them is DEGENSAC: that is a degeneracy check used for fundamental-matrix estimation, which OpenCV does not apply to homographies.

### Method and transformation compatibility

`affine` maps to `cv2.estimateAffinePartial2D`, which implements only LMEDS and RANSAC; every other code raises inside OpenCV. Stabilo rejects the combination when the `Stabilizer` is constructed, rather than failing on every frame:

```python
Stabilizer(transformation_type='affine')                    # ValueError: the default ransac_method 38 is projective-only
Stabilizer(transformation_type='affine', ransac_method=8)   # OK
```

---

## 12. Visualisation Mode

Set `viz=True` to have Stabilo retain intermediate data on the instance after each `stabilize(...)` call:

| Attribute | Contents |
|-----------|----------|
| `ref_mask` | Binary mask used for the reference frame |
| `cur_mask` | Binary mask used for the current frame |
| `ref_frame_gray` | Grayscale reference frame |
| `cur_frame_gray` | Grayscale current frame |
| `ref_kpts` | Reference keypoints |
| `cur_kpts` | Current keypoints |
| `ref_pts` | Matched reference keypoint coordinates |
| `cur_pts` | Matched current keypoint coordinates |
| `cur_inliers` | Boolean inlier mask for matched points |
| `cur_inliers_count` | Number of inlier matches |

These attributes are used by the `stabilo video --viz` command to render side-by-side stabilisation visualisations.

Match-quality statistics are also exposed through dedicated getters:
- `get_cur_inliers_count() -> int | None` — number of inliers (or `None` if estimation failed or has not run yet).
- `get_cur_num_matches() -> int | None` — total number of good matches fed to the estimator (denominator of inliers/total), or `None` if `stabilize()` has not yet been called.
- `get_cur_num_keypoints() -> tuple` — `(num_reference_keypoints, num_current_keypoints)`.

---

## 13. Benchmarking Mode

Set `benchmark=True` when running systematic parameter evaluation:

- If transformation estimation fails or produces `None`, the identity matrix (`np.eye(3)`) is used instead of the last known matrix.
- Log warnings about missing matches and fallback matrices are suppressed for cleaner batch output.

See [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) for a dedicated benchmarking and hyperparameter-tuning framework built on top of Stabilo.

---

## 14. Testing and Development

```bash
# run the full test suite
pytest

# run lint checks
ruff check .

# install in editable mode with development dependencies
pip install -e '.[dev]'
```

Run `pytest` from the repository root: the fixtures load images through relative paths such as `tests/ND_before.jpg`, so the suite fails from anywhere else.

### Tests that are skipped by default

Three groups of tests stay skipped unless their prerequisites are met:

| Group | Runs when | Why it is gated |
|-------|-----------|-----------------|
| Learning-based end-to-end tests (`tests/test_dl.py`) | `kornia` is importable **and** `STABILO_DL_TESTS=1` is set | Each detector downloads pretrained weights on first run (hundreds of MB into torch's hub cache) and needs far more time and memory than the classical path. Gating them keeps a plain `pytest` offline, fast, and safe to run in CI. |
| CUDA tests (`tests/test_stabilizer.py`) | OpenCV reports a CUDA-enabled device | Stock `opencv-python` wheels ship no CUDA algorithms, see [`docs/cuda.md`](cuda.md). |
| Codec-dependent CLI tests (`tests/test_cli.py`) | The OpenCV build can actually encode a video file | The end-to-end render tests skip themselves when no `VideoWriter` codec is available, or when the writer produces no file, which happens on minimal or headless OpenCV builds. |

The learning-based gate is an explicit environment variable rather than an "enable if kornia is installed" check because `kornia` and `torch` became core dependencies in v1.4.0: they are always present, so an importability check would download weights for everyone running the suite. Opt in with:

```bash
STABILO_DL_TESTS=1 pytest tests/test_dl.py
```

Once the weights are cached, later runs work offline. The validation-only tests in `tests/test_dl.py` (invalid device, incompatible matcher, and so on) always run, because argument validation raises before kornia is ever imported.

### Core-dependency smoke test

`.github/scripts/smoke_core.py` synthesizes a small video plus a matching bounding-box file and drives both `stabilo video` and `stabilo tracks` over them. It needs no development dependency, so it can validate an install made with a plain `pip install .`:

```bash
pip install .
python .github/scripts/smoke_core.py --core-only
```

`--core-only` additionally asserts that the `extras` dependency (matplotlib) is absent. CI runs this as the `core-install` job on every push and pull request.

---

## 15. Command-Line Interface

Installing stabilo provides a `stabilo` console command:

```bash
stabilo video <input> [options]    # stabilize a video relative to a reference frame
stabilo tracks <input> [options]   # stabilize per-frame object annotations (tracks)
stabilo config show                # print the default configuration
stabilo config copy [--output PATH]  # write an editable copy (default: ./custom.yaml)
```

Run `stabilo video --help` or `stabilo tracks --help` for the full list of options; both expose every stabilo parameter (detector, matcher, RANSAC, masking, device, etc.).

To customize parameters, run `stabilo config copy`, edit the generated `custom.yaml`, and pass it with `--custom-config custom.yaml`:

```bash
stabilo config copy
# edit custom.yaml ...
stabilo video data/video.mp4 --save --output data/out/ --custom-config custom.yaml
```

Resolution order is **explicit CLI flags > custom config file > built-in defaults**: a key set in the file passed to `--custom-config` fills in anything not explicitly passed as a CLI flag, and both override `stabilo/cfg/default.yaml`.

Each CLI invocation performs a best-effort check for a newer stabilo release on PyPI and prints a one-line notice if one is available. Set `STABILO_DISABLE_UPDATE_CHECK=1` to disable it.

### Using an external logger

`Stabilizer` accepts an optional `logger` so an embedding application can route stabilo's log records through its own logging setup:

```python
import logging
from stabilo import Stabilizer

my_logger = logging.getLogger("my_app.stabilo")
stabilizer = Stabilizer(logger=my_logger)
```

If omitted, stabilo uses its own module logger.
