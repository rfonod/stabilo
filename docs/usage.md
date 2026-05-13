# Stabilo — Detailed Usage Guide

> **Utility scripts** are also available to demonstrate Stabilo in practice. See the [`scripts/README.md`](../scripts/README.md) for full documentation of `stabilize_video.py`, `stabilize_boxes.py`, and the threshold-analysis tooling.

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
  - [12. Visualisation Mode](#12-visualisation-mode)
  - [13. Benchmarking Mode](#13-benchmarking-mode)
  - [14. Testing and Development](#14-testing-and-development)

---

## 1. Core Functionality

Stabilo aligns a **current frame** to a **reference frame** using a feature-point pipeline:

1. **Pre-processing** — optionally apply CLAHE for contrast enhancement and downsample the frame.
2. **Masking** — build a binary mask from user-supplied exclusion regions so that dynamic objects (e.g., vehicles) are excluded from feature extraction.
3. **Feature detection and description** — detect and describe keypoints using a chosen detector (ORB, SIFT, rSIFT, BRISK, KAZE, AKAZE).
4. **Matching** — match descriptors between the current and reference frame using a brute-force (BF) or FLANN matcher.
5. **Match filtering** — filter matches by cross-check, Lowe's ratio test, or distance threshold.
6. **Transformation estimation** — robustly estimate a 3x3 homography (*projective*) or 2x3 affine matrix using a RANSAC-type algorithm (MAGSAC++ by default).
7. **Stabilisation** — warp the current frame, or transform tracked bounding boxes, into the reference coordinate system.

This pipeline makes Stabilo suitable for:

- video stabilisation (align all frames to a chosen anchor frame),
- trajectory stabilisation (transform per-frame detections/tracks to a fixed reference),
- any downstream analysis requiring geometric frame-to-frame consistency.

---

## 2. Installation

```bash
# create and activate a virtual environment (Python >= 3.9)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install from PyPI
pip install stabilo

# or from local source in editable mode
pip install -e '.[dev]'
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
stabilized_frame  = stabilizer.warp_cur_frame()             # warped frame
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
| `detector_name` | `'orb'` | `orb`, `sift`, `rsift`, `brisk`, `kaze`, `akaze` | Feature detector |
| `matcher_name` | `'bf'` | `bf`, `flann` | Feature matcher |
| `filter_type` | `'ratio'` | `none`, `ratio`, `distance` | Match filtering strategy |
| `filter_ratio` | `0.9` | `(0, 1]` | Lowe's ratio (for `ratio`) or distance threshold ratio (for `distance`) |
| `transformation_type` | `'projective'` | `projective`, `affine` | Geometric model |
| `clahe` | `false` | `true`, `false` | Apply CLAHE contrast enhancement |
| `downsample_ratio` | `0.5` | `(0, 1]` | Resize factor before feature extraction |
| `max_features` | `2000` | `> 0` (int) | Maximum keypoints to detect in current frame |
| `ref_multiplier` | `2.0` | `>= 1.0` | Scale factor for keypoints in reference frame (`ref_multiplier x max_features`) |
| `mask_use` | `true` | `true`, `false` | Enable exclusion masking |
| `mask_margin_ratio` | `0.15` | `[0, 1]` | Fractional margin added to exclusion regions |
| `ransac_method` | `38` | see section 11 | RANSAC algorithm |
| `ransac_epipolar_threshold` | `2.0` | `> 0` | Reprojection-error threshold (pixels) |
| `ransac_max_iter` | `5000` | `> 0` (int) | Maximum RANSAC iterations |
| `ransac_confidence` | `0.999999` | `(0, 1]` | Required confidence level |
| `brisk_threshold` | `130` | `(0, 255]` | BRISK detector threshold (fallback) |
| `kaze_threshold` | `0.01` | `> 0` | KAZE detector threshold (fallback) |
| `akaze_threshold` | `0.01` | `> 0` | AKAZE detector threshold (fallback) |
| `gpu` | `false` | `true`, `false` | Use CUDA GPU acceleration (experimental) |
| `viz` | `false` | `true`, `false` | Retain intermediate data for visualisation |
| `benchmark` | `false` | `true`, `false` | Benchmarking mode (see section 13) |
| `min_good_match_count_warning` | `20` | `>= 0` | Warn if fewer than N good matches found |
| `min_inliers_match_count_warning` | `10` | `>= 0` | Warn if fewer than N inliers found |

---

## 9. Feature Detectors

| Name | Type | Notes |
|------|------|-------|
| `orb` | Binary | Fast; uses Hamming distance. Default. |
| `sift` | Float | Scale and rotation invariant; uses L2 distance. |
| `rsift` | Float | RootSIFT variant of SIFT with improved matching quality. |
| `brisk` | Binary | Threshold auto-derived from `max_features` via a pre-fitted model; falls back to `brisk_threshold`. |
| `kaze` | Float | Non-linear scale-space detector; threshold auto-derived similarly. |
| `akaze` | Binary | Accelerated KAZE; threshold auto-derived similarly. |

For BRISK, KAZE, and AKAZE, Stabilo ships pre-fitted linear regression models that translate `max_features` into the appropriate detector threshold. These models are stored under `stabilo/thresholds/models/` and are selected based on `mask_use` and `clahe` settings. If no model is available the fallback thresholds (`brisk_threshold`, etc.) are used.

---

## 10. Feature Matching and Filtering

Two matchers are supported:

- **`bf`** — OpenCV `BFMatcher`. Uses `crossCheck=True` for `filter_type='none'` and `'distance'`; `crossCheck=False` for `filter_type='ratio'`.
- **`flann`** — OpenCV `FlannBasedMatcher`. Uses LSH index for binary descriptors and KD-Tree for float descriptors.

Three filtering strategies:

| `filter_type` | Behaviour |
|---------------|-----------|
| `none` | Keep all matches returned by the matcher |
| `ratio` | Lowe's ratio test: keep match `m` where `m.distance < filter_ratio x n.distance` |
| `distance` | Distance threshold: keep matches below `min_dist + (max_dist - min_dist) x filter_ratio` |

---

## 11. Transformation Types and RANSAC Methods

### Transformation types

| `transformation_type` | Matrix shape | Function |
|-----------------------|-------------|----------|
| `projective` | 3 x 3 | `cv2.findHomography` |
| `affine` | 2 x 3 | `cv2.estimateAffinePartial2D` |

Use `projective` (default) when the camera undergoes any motion (pan, tilt, zoom, rotation). Use `affine` when you need to restrict to similarity/affine motions.

### RANSAC methods (integer codes)

| Code | Method |
|------|--------|
| 4 | LMEDS |
| 8 | RANSAC |
| 16 | RHO |
| 32 | DEGENSAC (`cv2.USAC_DEFAULT`) |
| 33 | DEGENSAC variant (`cv2.USAC_PARALLEL`) |
| 35 | LO-RANSAC (`cv2.USAC_FAST`) |
| 36 | GC-RANSAC (`cv2.USAC_ACCURATE`) |
| 37 | PROSAC (`cv2.USAC_PROSAC`) |
| **38** | **MAGSAC++ (`cv2.USAC_MAGSAC`) — default** |

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

These attributes are used by the companion utility scripts (e.g., `stabilize_video.py --viz`) to render side-by-side stabilisation visualisations.

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
