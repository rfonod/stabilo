# Stabilo

**Stabilo** stabilizes video frames and tracked object trajectories against a chosen reference frame, using robust homography or affine registration with optional exclusion masks for dynamic regions. It is the registration engine behind [Geo-trax](https://github.com/rfonod/geo-trax) and is tuned with [Stabilo-Optimize](https://github.com/rfonod/stabilo-optimize).

## What it does

- **Anchored, not smoothed.** Every frame and every tracked box is aligned to one reference frame, so coordinates stay comparable across the whole video, not just between neighbouring frames.
- **Masks that ignore moving objects.** Vehicles, pedestrians, or any other dynamic region can be excluded from registration, as axis-aligned boxes, oriented boxes, four-point boxes, polygons, or circles.
- **Classical and learned features in one API.** Six classical OpenCV detectors and five learned kornia models, paired with brute-force, FLANN, or LightGlue matching; see [Feature detectors and matchers](detectors.md).
- **Two acceleration paths.** OpenCV CUDA for the classical pipeline ([CUDA](cuda.md)) and any PyTorch device for the learned models.
- **Library and CLI.** The `stabilo` command needs no code; the `Stabilizer` class embeds into your own pipeline.

## Installation

Stabilo needs Python 3.11 to 3.13:

```bash
pip install stabilo
```

This installs everything, including the learned detectors and matchers (kornia and torch). Their pretrained weights download into torch's cache the first time each model is used and are reused offline afterwards. See [Usage: Installation](usage.md#2-installation) for virtual-environment, conda, uv, and from-source instructions.

## Minimal example

```python
import cv2
from stabilo import Stabilizer

cap = cv2.VideoCapture("video.mp4")
stabilizer = Stabilizer(mask_use=False)  # no exclusion boxes in this example

ok, frame = cap.read()
stabilizer.set_ref_frame(frame)  # frame 0 is the reference

while True:
    ok, frame = cap.read()
    if not ok:
        break
    stabilizer.stabilize(frame)
    stabilized = stabilizer.warp_cur_frame()     # the frame warped onto the reference
    matrix = stabilizer.get_cur_trans_matrix()   # current -> reference transform
```

To exclude moving objects, keep the default `mask_use=True` and pass their boxes to `set_ref_frame(frame, boxes)` and `stabilize(frame, boxes)`; `transform_cur_boxes()` then maps the boxes into the reference frame. See [Usage: Main workflow](usage.md#3-main-workflow) and [Masking behaviour](usage.md#5-masking-behaviour).

The same pipeline from the command line:

```bash
stabilo video video.mp4 --save --save-viz
stabilo tracks video.mp4 --save
```

## Demo

A 14-second 4K demo of stabilo on drone footage:

<iframe src="https://www.youtube-nocookie.com/embed/BGEZCEe6yHw" title="Stabilo demo" style="width: 100%; aspect-ratio: 16 / 9; border: 0;" allow="accelerometer; encrypted-media; gyroscope; picture-in-picture; fullscreen" allowfullscreen loading="lazy"></iframe>

## Where to go next

- [Usage](usage.md): the full guide, covering mask formats, every configuration key, RANSAC methods, visualization, and benchmarking.
- [Feature detectors and matchers](detectors.md): which detector to pick, and the rotation and memory limits of the learned ones.
- [CUDA](cuda.md): building OpenCV with CUDA for `--gpu`.
- [CLI reference](cli.md) and [API reference](api.md).
- [Citation](citation.md): how to cite stabilo in your work.
