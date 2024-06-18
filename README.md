# Stabilo

![GitHub Release](https://img.shields.io/github/v/release/rfonod/stabilo?include_prereleases) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) ![GitHub](https://img.shields.io/badge/Development-Active-brightgreen)

🚧 **Development Notice** 🚧

> ⚠️ **IMPORTANT:** Stabilo is currently in its preliminary stages and under active development. Not all features are complete, and significant changes may occur. It is recommended for experimental use only. Please report any issues you encounter, and feel free to contribute to the project.

Stabilo is a Python package for stabilizing video frames or tracked object trajectories in videos using homography or affine transformations, with optional user-provided masks that define areas to ignore during stabilization. It is optimized for use in environments where accurate, frame-by-frame alignment is crucial, such as video surveillance and research applications. For instance, Stabilo can be integrated with object detection and tracking algorithms to stabilize trajectories, enhancing the performance of such systems. See for example [this project](https://github.com/rfonod/geo-trax). The package offers flexibility through a variety of transformation and enhancement options, including the ability to handle different types of feature detection and masking techniques.

## Features

- **Video Stabilization**: Align video frames to a selected reference using homography or affine transformations.
- **Trajectory Stabilization**: Apply stabilization to object trajectories using homography or affine transformations.
- **User-Defined Masks**: Allow users to specify areas to ignore during stabilization, such as bounding boxes of moving objects or regions of no interest.
- **Wide Range of Algorithms**: Support for various feature detectors (e.g., ORB, (R)SIFT, BRISK, (A)KAZE), matchers (e.g., BF, FLANN), RANSAC algorithms, transformation types, and pre-processing options.
- **Customizable Parameters**: Fine-tune the stabilization process by adjusting parameters such as the number of keypoints, RANSAC and matching thresholds, and downsampling factors.
- **Visualization Tools**: Generate visualizations of the stabilization process, including frame-by-frame comparisons.
- **Threshold Analysis**: Analyze the relationship between detection thresholds and keypoint counts for BRISK, KAZE, and AKAZE feature detectors.

<details>
<summary><b>🚀 Planned Enhancements</b></summary>

- **Benchmarking and Tuning Tools**: Develop tools to benchmark and tune the performance of the stabilization algorithms.
- **Trajectory Stabilization Script**: Create a script to stabilize object trajectories in videos. 
- **Custom Mask Encoding**: Support for more generic types of custom mask encodings.
- **Custom Reference Frame Selection**: Allow users to select a custom reference frame for stabilization.
- **GPU Acceleration**: Utilize GPU acceleration for faster processing.
- **Documentation**: Provide detailed documentation and examples for ease of use.
- **Unit Tests**: Implement comprehensive unit tests to ensure the stability and reliability of the package.
- **Deployment to PyPI**: Publish the package on PyPI for easy installation and distribution.

</details>

## Installation

First, create a **Python Virtual Environment** (Python >= 3.9) using e.g., [Miniconda3](https://docs.anaconda.com/free/miniconda/):
```bash
conda create -n stabilo python=3.9 -y
conda activate stabilo
```
    
Then, install the package using one of the following methods:

### Option 1: Install from PyPI
You can install the package from PyPI (not available yet):

<strike>

```sh
pip install stabilo
```

</strike>


### Option 2: Install from Source (recommended)
You can install the package directly from the repository:
```sh
pip install git+https://github.com/rfonod/stabilo.git
```

### Option 3: Install from Local Source

Clone the repository and install the package:

```sh
git clone https://github.com/rfonod/stabilo.git
cd stabilo
pip install .
```

If you want the changes you make in the repo to be reflected in your install, use `pip install -e .` instead of `pip install .`.

## Example Usage

```python
from stabilo import Stabilizer 

# Create an instance of the Stabilizer class with default parameters
stabilizer = Stabilizer() 

# Set a reference frame with optional mask (e.g., bounding boxes)
stabilizer.set_ref_frame(ref_frame, ref_mask)

# Stabilize any consecutive frame with optional mask
stabilizer.stabilize(frame, mask)

# Get the stabilized (warped to the reference frame) frame  
stabilized_frame = stabilizer.warp_cur_frame()

# Get the transformed bounding boxes (if mask was provided)
stabilized_boxes = stabilizer.transform_cur_boxes()
``` 

## Additional Scripts

Utility scripts are provided to demonstrate the functionality of the Stabilo package. These scripts can be found in the `scripts` directory.

#### Stabilization Examples

- `stabilize_video.py`: Demonstrates video stabilization relative to a reference frame.
- `stabilize_boxes.py`: Shows how to stabilize bounding boxes relative to a reference frame.

#### Threshold Analysis

The `find_threshold_models.py` script is designed to model the relationship between detection thresholds for BRISK, KAZE, and AKAZE feature detectors and their average keypoint counts. It outputs regression models, saves pertinent data, and generates plots for visual analysis.

To run this script, install the optional dependencies `pip install .[extras]` (or `pip install '.[extras]'` if you use zsh).

## Citing This Work

If you use this project in your academic research, commercial products, or any published material, please acknowledge its use by citing it. For the correct citation, refer to the DOI badge above, which links to the appropriate version of the release on Zenodo. Ensure you reference the version you actually used. A formatted citation can also be obtained from the top right of the [GitHub repository](https://github.com/stabilo).

```bibtex
@software{Fonod_Stabilo_2024,
author = {Fonod, Robert},
license = {MIT},
month = jun,
title = {{Stabilo}},
url = {https://github.com/rfonod/stabilo},
version = {0.1.0},
year = {2024}
}
```

## Contributing

Contributions are welcome! If you encounter any issues or have suggestions for improvements, please open a [GitHub Issue](https://github.com/rfonod/stabilo/issues) or submit a pull request. Your contributions are greatly appreciated!


## License

This project is licensed under the MIT License, an [OSI-approved](https://opensource.org/licenses/MIT) open-source license, which allows for both academic and commercial use. By citing this project, you help support its development and acknowledge the effort that went into creating it. For more details, see the [LICENSE](LICENSE) file. Thank you!
