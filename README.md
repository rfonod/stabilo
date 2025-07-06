# Stabilo

[![PyPI Version](https://img.shields.io/pypi/v/stabilo)](https://pypi.org/project/stabilo/) [![GitHub Release](https://img.shields.io/github/v/release/rfonod/stabilo?include_prereleases)](https://github.com/rfonod/stabilo/releases) [![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/) [![License](https://img.shields.io/github/license/rfonod/stabilo)](https://github.com/rfonod/stabilo/blob/main/LICENSE) [![Development Status](https://img.shields.io/badge/development-active-brightgreen)](https://github.com/rfonod/stabilo) [![Open Access](https://img.shields.io/badge/Journal-10.1016%2Fj.trc.2025.105205-blue)](https://doi.org/10.1016/j.trc.2025.105205) [![arXiv](https://img.shields.io/badge/arXiv-2411.02136-b31b1b.svg)](https://arxiv.org/abs/2411.02136) [![Archived Code](https://img.shields.io/badge/Zenodo-Software%20Archive-blue)](https://zenodo.org/doi/10.5281/zenodo.12117092) ![PyPi - Total Downloads](https://img.shields.io/pepy/dt/stabilo?label=total%20downloads) ![PyPI - Downloads per Month](https://img.shields.io/pypi/dm/stabilo?color=%234c1)

**Stabilo** is a specialized Python package for stabilizing video frames or tracked object trajectories in videos, using robust homography or affine transformations. Its core functionality focuses on aligning each frame or object track to a chosen reference frame, enabling precise stabilization that mitigates disturbances like camera movements. Key features include robust keypoint-based image registration and the option to integrate user-defined masks, which exclude dynamic regions (e.g., moving objects) to enhance stabilization accuracy. Integrating seamlessly with object detection and tracking algorithms, Stabilo is ideal for high-precision applications like urban traffic monitoring, as demonstrated in the [geo-trax](https://github.com/rfonod/geo-trax) 🚀 trajectory extraction framework. Extensive transformation and enhancement options, including multiple feature detectors and matchers, masking techniques, further expand its utility. For systematic evaluation and hyperparameter tuning, the companion tool [stabilo-optimize](https://github.com/rfonod/stabilo-optimize) 🎯 provides a dedicated benchmarking framework. The repository also includes valuable resources like utility scripts and example videos to demonstrate its capabilities.

![Stabilization Visualization GIF](https://raw.githubusercontent.com/rfonod/stabilo/main/assets/stabilization_visualization.gif?raw=True)

## Features

- **Video Stabilization**: Align (warp) all video frames to a custom (anchor) reference frame using homography or affine transformations.
- **Trajectory Stabilization**: Transform object trajectories (e.g., bounding boxes) to a common fixed reference frame using homography or affine transformations.
- **User-Defined Masks**: Allow users to specify custom masks to exclude regions of interest during stabilization.
- **Wide Range of Algorithms**: Includes support for various feature detectors (ORB, SIFT, RSIFT, BRISK, KAZE, AKAZE), matchers (BF, FLANN), RANSAC algorithms (MAGSAC++, DEGENSAC, ...), transformation types, and pre-processing options.
- **Customizable Parameters**: Fine-tune the stabilization by adjusting parameters such as the number of keypoints, RANSAC parameters, matching thresholds, downsampling factors, etc..
- **Visualization Tools**: Generate visualizations of the stabilization process, with frame-by-frame comparisons and trajectory transformations (see the above animation).
- **Threshold Analysis**: Analyze the relationship between detection thresholds and keypoint counts for BRISK, KAZE, and AKAZE to fairly benchmark with different detectors.
- **Benchmarking and Optimization**: Fine-tune stabilization parameters with [stabilo-optimize](https://github.com/rfonod/stabilo-optimize) 🎯, which provides ground truth-free evaluation using random perturbations.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- **Unit Tests**: Comprehensive unit test suite to ensure package stability and reliability.
- **Different Mask Types**: Inclusion of additional mask types (e.g., polygonal, circular) for enhanced precision in stabilization.
- **GPU Acceleration**: Integration of GPU acceleration to improve processing speed.
- **Documentation**: Detailed documentation covering the package’s functionality and usage.

</details>

## Installation

It is recommended to create and activate a **Python Virtual Environment** (Python >= 3.9) first using e.g., [Miniconda3](https://docs.anaconda.com/free/miniconda/):

```bash
conda create -n stabilo python=3.11 -y
conda activate stabilo
```

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

## Utility Scripts

Utility scripts are provided to demonstrate the functionality of the Stabilo package. These scripts can be found in the [`scripts`](./scripts/) directory and are briefly documented in the [scripts README](./scripts/README.md).

#### Stabilization Examples

- `stabilize_video.py`: Implements video stabilization relative to a reference frame.
- `stabilize_boxes.py`: Implements object trajectory stabilization relative to a reference frame.

#### Threshold Analysis

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
    @software{fonod2025stabilo,
      author = {Fonod, Robert},
      license = {MIT},
      month = apr,
      title = {Stabilo: A Comprehensive Python Library for Video and Trajectory Stabilization with User-Defined Masks},
      url = {https://github.com/rfonod/stabilo},
      doi = {10.5281/zenodo.12117092},
      version = {1.0.1},
      year = {2025}
    }
    ```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/stabilo/issues) or submit a pull request.

## License

This project is distributed under the MIT License. See the [LICENSE](LICENSE) file for more details.
