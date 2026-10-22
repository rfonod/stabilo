# Feature detectors and matchers

Stabilo registers each frame against the reference by detecting keypoints, matching their descriptors, and fitting a transformation with RANSAC. The detector is selected with `detector_name` (`--detector-name` on the CLI) and the matcher with `matcher_name` (`--matcher-name`). Six detectors are classical OpenCV algorithms; five are learned models from [kornia](https://kornia.readthedocs.io/) that download pretrained weights on first use and run on the torch `device` (`--device cpu|cuda|mps`, `auto` picks cuda > mps > cpu).

## Comparison

--8<-- "README.md:detector-table"

Every detector keeps at most `max_features` keypoints in the current frame and `ref_multiplier x max_features` in the reference frame, the strongest by detector response.

## Two limits of the learned detectors

Most problems reported with the learned detectors come from one of these two properties. Both belong to the pretrained models, not to stabilo, and neither is something a configuration option can remove.

!!! warning "Rotation: `xfeat`, `disk`, `dedode`, and `loftr` fail past about 30 degrees"

    These four are *upright* models: they were trained on upright imagery and do not estimate a keypoint orientation. Matching degrades as the current frame rotates away from the reference and collapses beyond roughly 30 degrees of in-plane rotation; at 90 degrees all four fail outright, while ORB and SIFT still register to about 1 px.

    Footage with large in-plane rotation, such as a drone orbiting or yawing about its own axis, needs a classical detector or `keynet`, which runs an orientation estimator and is the only rotation-invariant learned option. DeDoDe's `C4` and `SO2` weight variants are not a workaround: they need a steerer applied at match time, which kornia does not do.

    The measured errors per rotation angle are in [Usage: Rotation invariance](usage.md#rotation-invariance).

!!! warning "Memory: cost grows with the processed frame size, quadratically for `loftr`"

    The learned models allocate memory in proportion to the *processed* frame, which is `downsample_ratio²` times the input pixel count, and they need far more than the classical detectors: at about 0.5 MP, `disk`, `dedode`, `keynet`, and `loftr` each peak at several GB of RAM on CPU, while `xfeat` stays at a few hundred MB.

    `loftr` grows quadratically, because its coarse matching builds an N×N matrix over N = (H/8)·(W/8) tokens. A 4K frame at the default `downsample_ratio: 0.5` already needs a 3.9 GB matrix before intermediates, and full resolution needs over 60 GB, enough to freeze a workstation.

    Lower `downsample_ratio` for large inputs. Stabilo warns when the processed frame exceeds the guideline below, but does not cap it:

    | Detector | Guideline (processed frame) | Suggested for a 4K input (3840×2160) |
    | :--- | :--- | :--- |
    | `xfeat` | 2.0 MP | `--downsample-ratio 0.49` |
    | `disk`, `dedode`, `keynet` | 0.5 MP | `--downsample-ratio 0.24` |
    | `loftr` | 0.3 MP | `--downsample-ratio 0.19` |

    Running out of memory is raised as an error rather than treated as an unregistrable frame. See [Usage: Choosing a downsample ratio](usage.md#choosing-a-downsample-ratio).

## Matchers

| `matcher_name` | Works with | Notes |
| :--- | :--- | :--- |
| `bf` (default) | every sparse detector | OpenCV brute-force matcher; Hamming distance for binary descriptors, L2 for float ones. |
| `flann` | every sparse detector | OpenCV approximate matcher; LSH index for binary descriptors, KD-tree for float ones. |
| `lightglue` | `disk`, `dedode`, `keynet` | Learned kornia matcher. It filters its own matches, so `filter_type` is ignored. It does not support `xfeat`. |

`loftr` is detector-free: it matches the image pair directly, so `filter_type` and the `bf`/`flann` choice of `matcher_name` are ignored when it is selected (`lightglue` is rejected). For `bf` and `flann`, `filter_type` chooses between no filtering, Lowe's ratio test (the default), and a distance threshold; see [Usage: Feature matching and filtering](usage.md#10-feature-matching-and-filtering).

## Choosing a detector

- **Start with `orb`**, the default. It is fast, rotation invariant, light on memory, and the only detector with an OpenCV CUDA path (`--gpu`, see [CUDA](cuda.md)).
- **Use `sift` or `rsift`** when ORB finds too few reliable matches, for example on low-texture scenes.
- **Use `keynet`** when you want a learned detector and the camera may rotate.
- **Use `xfeat`, `disk`, `dedode`, or `loftr`** only for footage that stays within about 30 degrees of the reference orientation, with a reduced `downsample_ratio`, and preferably on a GPU.

Classical and learned detectors receive the same CLAHE, downsampling, and exclusion-mask handling, and keypoints are always rescaled to full resolution before estimation. The detector-specific options (BRISK, KAZE, and AKAZE thresholds, LoFTR, DISK, and DeDoDe weights) are listed in [Usage: Configuration parameters](usage.md#8-configuration-parameters).
