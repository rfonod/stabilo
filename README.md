# Stabilo

[![GitHub Release](https://img.shields.io/github/v/release/rfonod/stabilo?include_prereleases)](https://github.com/rfonod/stabilo/releases) [![PyPI Version](https://img.shields.io/pypi/v/stabilo)](https://pypi.org/project/stabilo/) [![PyPI - Total Downloads](https://img.shields.io/pepy/dt/stabilo?label=total%20downloads)](https://pepy.tech/project/stabilo) [![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/stabilo?color=%234c1)](https://pypi.org/project/stabilo/) [![CI](https://github.com/rfonod/stabilo/actions/workflows/ci.yml/badge.svg)](https://github.com/rfonod/stabilo/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.9--3.13-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/stabilo)](https://github.com/rfonod/stabilo/blob/main/LICENSE) [![GitHub Issues](https://img.shields.io/github/issues/rfonod/stabilo)](https://github.com/rfonod/stabilo/issues) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12117092)

**Stabilo** is a specialized Python package for stabilizing video frames or tracked object trajectories in videos, using robust homography or affine transformations. Its core functionality focuses on aligning each frame or object track to a chosen reference frame, enabling precise stabilization that mitigates disturbances like camera movements. Key features include robust keypoint-based image registration and the option to integrate user-defined masks, which exclude dynamic regions (e.g., moving objects) to enhance stabilization accuracy. Integrating seamlessly with object detection and tracking algorithms, Stabilo is ideal for high-precision applications like urban traffic monitoring, as demonstrated in the [Geo-trax](https://github.com/rfonod/geo-trax) 🚀 trajectory extraction framework. Extensive transformation and enhancement options, including multiple feature detectors and matchers, masking techniques, further expand its utility. For systematic evaluation and hyperparameter tuning, the companion tool [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯 provides a dedicated benchmarking framework. The repository also includes valuable resources like utility scripts and example videos to demonstrate its capabilities.

![Stabilization Visualization GIF](https://raw.githubusercontent.com/rfonod/stabilo/main/assets/stabilization_visualization.gif?raw=True)

## Features

- **Video Stabilization**: Align (warp) all video frames to a custom (anchor) reference frame using homography or affine transformations.
- **Trajectory Stabilization**: Transform object trajectories (e.g., bounding boxes) to a common fixed reference frame using homography or affine transformations.
- **User-Defined Masks**: Allow users to specify custom masks to exclude regions of interest during stabilization, supporting axis-aligned boxes, oriented bounding boxes (OBBs), four-point boxes, polygonal masks, and circular masks.
- **Wide Range of Algorithms**: Includes support for various feature detectors (ORB, SIFT, RSIFT, BRISK, KAZE, AKAZE), matchers (BF, FLANN), RANSAC algorithms (MAGSAC++, DEGENSAC, ...), transformation types, and pre-processing options.
- **Customizable Parameters**: Fine-tune the stabilization by adjusting parameters such as the number of keypoints, RANSAC parameters, matching thresholds, downsampling factors, etc..
- **Visualization Tools**: Generate visualizations of the stabilization process, with frame-by-frame comparisons and trajectory transformations (see the above animation).
- **Threshold Analysis**: Analyze the relationship between detection thresholds and keypoint counts for BRISK, KAZE, and AKAZE to fairly benchmark with different detectors.
- **Benchmarking and Optimization**: Fine-tune stabilization parameters with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯, which provides ground truth-free evaluation using random perturbations.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- **GPU Acceleration**: Integration of GPU acceleration to improve processing speed.
- **Bi-directional Matching**: Implementing bi-directional matching to enhance the robustness of keypoint matching.
- **Additional Feature Detectors**: Adding support for more feature detectors and matchers to provide users with a wider range of options for stabilization.

</details>

<details>
<summary><b>🔗 Related Projects</b></summary>

Stabilo integrates with and complements several specialized tools:

- **[Geo-trax](https://github.com/rfonod/geo-trax) 🚀** — End-to-end framework for extracting georeferenced vehicle trajectories from drone imagery. Uses Stabilo as its core stabilization engine to align video frames and vehicle tracks to a common reference frame.

- **[Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize) 🎯** — Benchmarking and hyperparameter optimization framework for Stabilo. Evaluates stabilization performance through ground truth-free assessment using random perturbations. Used to fine-tune Stabilo parameters.

- **[HBB2OBB](https://github.com/rfonod/hbb2obb) 📦** — Converts horizontal bounding boxes to oriented bounding boxes using SAM segmentation models. Can be used alongside Stabilo when object orientation is needed for downstream analysis.

</details>

## Installation

It is recommended to create and activate a **Python virtual environment** (Python >= 3.9 and <= 3.13) first:

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

**[uv](https://docs.astral.sh/uv/getting-started/installation/) (fastest; use `uv pip install` in step 3):**
```bash
uv venv --python 3.11
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
</details>

Then, install the stabilo library using one of the following options:

### Option 1: Install from PyPI

```bash
pip install stabilo
```

### Option 2: Install from Local Source

You can also clone the repository and install the package from the local source:

```bash
git clone https://github.com/rfonod/stabilo.git
cd stabilo && pip install .
```

If you want the changes you make in the repo to be reflected in your install, use `pip install -e .` instead of `pip install .`.

## Python API Usage

```python
from stabilo import Stabilizer 

# Create an instance of the Stabilizer class with default parameters
stabilizer = Stabilizer() 

# Set a reference frame with (optional) mask
stabilizer.set_ref_frame(ref_frame, ref_mask)

# Stabilize any frame with (optional) mask
stabilizer.stabilize(cur_frame, cur_mask)

# Get the stabilized (warped) frame 
stabilized_frame = stabilizer.warp_cur_frame()

# Transform current masks (bounding boxes) if it was provided
stabilized_boxes = stabilizer.transform_cur_boxes()

# Transform any point (pixel coordinates) from the current frame to reference frame
cur_point = np.array([x, y, 1])
ref_point = stabilizer.get_cur_trans_matrix() @ cur_point
```

### Bounding Box Formats

Stabilo supports multiple mask and bounding-box formats:

- **`xywh`**: Axis-aligned boxes `[x_center, y_center, width, height]`
- **`xywha`**: Oriented bounding boxes `[x_center, y_center, width, height, angle_degrees]`
- **`four`**: Four corner points `[x1, y1, x2, y2, x3, y3, x4, y4]` (auto-detects if rotated)
- **`polygon`**: Polygon points as flattened rows `[x1, y1, ..., xN, yN]` or `(N, 2)` point arrays
- **`circle`**: Circular masks `[x_center, y_center, radius]`

```python
import numpy as np

# Example with oriented bounding boxes (OBBs)
obb_boxes = np.array([
    [100, 150, 50, 30, 45],  # Rotated 45 degrees
    [200, 200, 60, 40, 90],  # Rotated 90 degrees
])

stabilizer.set_ref_frame(ref_frame, obb_boxes, box_format='xywha')
stabilizer.stabilize(cur_frame, cur_boxes, box_format='xywha')

# Transform boxes and get result in different format
transformed = stabilizer.transform_cur_boxes(out_box_format='four')
```

### Polygon and Circular Mask Examples

```python
# Polygon mask(s)
polygon_masks = np.array([
    [60, 60, 120, 60, 120, 120, 60, 120],
    [200, 200, 260, 210, 240, 270, 190, 260],
])
stabilizer.set_ref_frame(ref_frame, polygon_masks, box_format='polygon')

# Circular mask(s)
circle_masks = np.array([
    [120, 120, 25],
    [360, 240, 40],
])
stabilizer.stabilize(cur_frame, circle_masks, box_format='circle')
```

## Documentation

For detailed package documentation, including architecture, end-to-end workflows, mask format specifications, and API behavior notes, see:

- [`docs/usage.md`](./docs/usage.md)

## Utility Scripts

Utility scripts are provided to demonstrate the functionality of the Stabilo package. These scripts can be found in the [`scripts`](./scripts/) directory and are briefly documented in the [scripts README](./scripts/README.md).

### Stabilization Examples

- `stabilize_video.py`: Implements video stabilization relative to a reference frame.
- `stabilize_boxes.py`: Implements object trajectory stabilization relative to a reference frame.

### Threshold Analysis

- `find_threshold_models.py`: Computes regression models between detection thresholds and average keypoint counts for BRISK, KAZE, and AKAZE feature detectors.

## Citing This Work

If you use **Stabilo** in your research, software, or product, please cite the following resources appropriately:

1. **Preferred Citation:** Please cite the associated article for any use of the Stabilo package, including research, applications, and derivative work:

    ```bibtex
    @article{fonod2025advanced,
      title = {Advanced computer vision for extracting georeferenced vehicle trajectories from drone imagery},
      author = {Fonod, Robert and Cho, Haechan and Yeo, Hwasoo and Geroliminis, Nikolas},
      journal = {Transportation Research Part C: Emerging Technologies},
      volume = {178},
      pages = {105205},
      year = {2025},
      publisher = {Elsevier},
      doi = {10.1016/j.trc.2025.105205},
      url = {https://doi.org/10.1016/j.trc.2025.105205}
    }
    ```

2. **Repository Citation:** If you reference, modify, or build upon the Stabilo software itself, please also cite the corresponding Zenodo release:

    ```bibtex
    @software{fonod2026stabilo,
      author = {Fonod, Robert},
      license = {MIT},
      month = jun,
      title = {Stabilo: A Comprehensive Python Library for Video and Trajectory Stabilization with User-Defined Masks},
      url = {https://github.com/rfonod/stabilo},
      doi = {10.5281/zenodo.12117092},
      version = {1.2.3},
      year = {2026}
    }
    ```

## Contributing

Contributions from the community are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/stabilo/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
