# Stabilo — Building OpenCV with CUDA Support

> Applies to any NVIDIA GPU on Linux or Windows. CUDA is NVIDIA-only: there is no macOS support (Apple no longer supports NVIDIA GPUs), so `gpu=True` cannot be built or tested on macOS.

This guide is a distilled, stabilo-specific path through a general build process that's documented more broadly elsewhere. For the full picture beyond what's needed here: OpenCV's own [configuration options reference](https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html) and [Linux install tutorial](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) cover the general CMake build process and the flags referenced throughout this doc (`WITH_CUDA`, `CUDA_ARCH_BIN`, `OPENCV_EXTRA_MODULES_PATH`); [PyImageSearch's OpenCV + CUDA + cuDNN walkthrough](https://pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/) is a widely used community reference for the same underlying build, using a manual `cmake`/`make` install rather than the `opencv-python` wheel-building approach in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system) below (see [Section 1](#1-why-a-source-build-is-required) for why this guide takes that different route).

## Table of Contents

- [Stabilo — Building OpenCV with CUDA Support](#stabilo--building-opencv-with-cuda-support)
  - [Table of Contents](#table-of-contents)
  - [1. Why a Source Build Is Required](#1-why-a-source-build-is-required)
  - [2. What Gets CUDA-Accelerated in Stabilo](#2-what-gets-cuda-accelerated-in-stabilo)
  - [3. Prerequisites](#3-prerequisites)
  - [4. Recommended Build: OpenCV 5.0.0 via the `opencv-python` Build System](#4-recommended-build-opencv-500-via-the-opencv-python-build-system)
    - [4.1 Pinning the OpenCV version (submodules) — and making the pin stick](#41-pinning-the-opencv-version-submodules--and-making-the-pin-stick)
    - [4.2 CUDA 13.x pre-flight check (skip on CUDA 12.x)](#42-cuda-13x-pre-flight-check-skip-on-cuda-12x)
    - [4.3 Configure and build](#43-configure-and-build)
  - [5. Verifying the Build](#5-verifying-the-build)
  - [6. Installing Stabilo on Top](#6-installing-stabilo-on-top)
  - [7. Enabling GPU Mode in Stabilo](#7-enabling-gpu-mode-in-stabilo)
  - [8. Performance Notes](#8-performance-notes)
  - [9. Troubleshooting](#9-troubleshooting)

---

## 1. Why a Source Build Is Required

The `opencv-python` / `opencv-contrib-python` wheels published on PyPI are **CPU-only**: they are never compiled with `WITH_CUDA=ON`. This is true even for the latest release; verified against `opencv-python==4.10.0`:

```python
import cv2
cv2.cuda.getCudaEnabledDeviceCount()   # 0
hasattr(cv2.cuda, 'SIFT_create')       # False (the cv2.cuda *namespace* exists, but no CUDA algorithms are compiled in)
```

So `Stabilizer(gpu=True)` will raise `ValueError: GPU is enabled but no CUDA-enabled device was found` (or, on a machine with an NVIDIA GPU but a stock OpenCV wheel, fail as soon as it tries to build a `cv2.cuda.*_create` detector) unless OpenCV is built from source with CUDA enabled. There is no way around this; it's an OpenCV packaging decision, not a Stabilo limitation.

**Every CUDA module lives in `opencv_contrib`, not the main `opencv` repo.** Verified directly against the `opencv/opencv` and `opencv/opencv_contrib` GitHub trees for both the `5.0.0` and `4.13.0` tags: `cudev`, `cudaarithm`, `cudafeatures2d`, `cudafilters`, `cudaimgproc`, `cudawarping`, etc. all live under `opencv_contrib/modules/`; none exist under `opencv/modules/`. This means building with contrib modules enabled is **mandatory** for `gpu=True` to work at all, not an optional extra. See [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system) for the practical consequence (the built package is named `opencv-contrib-python`, not `opencv-python`).

## 2. What Gets CUDA-Accelerated in Stabilo

| Stabilo step | CUDA-capable? | Notes |
|---|---|---|
| Feature detection (ORB, SIFT, RSIFT, BRISK, KAZE, AKAZE) | Only ORB, confirmed | `cv2.cuda.ORB_create`, from `opencv_contrib`'s `cudafeatures2d` module. Confirmed on an actual OpenCV 5.0.0 + opencv_contrib 5.0.0 build (the versions this guide targets) via the [Section 5](#5-verifying-the-build) diagnostic: `cudafeatures2d` is active, but only `ORB`/`ORB_create` show up under `cv2.cuda`; SIFT, BRISK, KAZE, and AKAZE have no CUDA Python binding at all in this module, under any naming convention, despite the module being built. RSIFT would reuse `cv2.cuda.SIFT_create` if it existed, so it's CPU-only here too. This may change in later OpenCV releases; re-run the Section 5 diagnostic to check on a different build. BRISK/KAZE/AKAZE aren't merely missing a CUDA binding, either: OpenCV 5.0.0 dropped `cv2.BRISK_create`/`cv2.KAZE_create`/`cv2.AKAZE_create` from the Python bindings entirely, so they don't work even on CPU in a venv built against 5.0.0 (see the warning in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system)). |
| Grayscale conversion, CLAHE, downsampling | Yes, confirmed | `cv2.cuda.cvtColor`/`cv2.cuda.createCLAHE` (`cudaimgproc`), `cv2.cuda.resize` (`cudawarping`). Confirmed to run without error on real hardware, chained together on the GPU before a single download, and to pass the `test_gpu_vs_cpu_produce_comparable_results` correctness check; not yet compared pixel-by-pixel against their CPU counterparts. |
| Descriptor matching (BF) | Yes, confirmed | `cv2.cuda.DescriptorMatcher` (`cudafeatures2d`) requires `cuda_GpuMat` inputs, not plain host arrays; stabilo uploads descriptors to `GpuMat` before matching when `gpu=True`. `DescriptorMatcher_createBFMatcher()` also only accepts `normType` on this build, not `crossCheck`. `get_matches` uses the async `matchAsync`/`knnMatchAsync` + `matchConvert`/`knnMatchConvert` API (confirmed present on this build via `dir()`), not the synchronous `match`/`knnMatch` also available on the GPU matcher object; this avoids an extra implicit sync inside the matcher call itself. FLANN's GPU matcher factory is still unverified end-to-end (constructs without error; not yet checked for correct match output). |
| Frame warping (`warp_cur_frame`) | Yes, but not necessarily faster | `cv2.cuda.warpPerspective` / `cv2.cuda.warpAffine` (`cudawarping`), confirmed to run without error on real hardware. On an RTX 4090, per-call GPU warp measures ~2x *slower* than CPU (6.30ms vs 3.14ms average, ORB, 3840x2160 frames; full breakdown in [Section 8](#8-performance-notes)). Reason: `warp_frame` issues a single, isolated upload -> kernel -> download round trip per call, with nothing else queued on the stream to amortize the sync/transfer cost against, unlike `get_features_and_descriptors`, which chains 4-5 GPU ops before its one sync. Even at this full 4K frame size, the two-way PCIe transfer (~24MB per frame) outweighs the CPU SIMD warp savings when it isn't pipelined with other GPU work. Detection and matching win decisively on the same measurements, so `gpu=True` is still a net win overall, just not on every stage. |
| RANSAC / homography / affine estimation | **No** | OpenCV has no CUDA equivalent of `cv2.findHomography` / `cv2.estimateAffinePartial2D`, in `opencv` or `opencv_contrib`. This always runs on CPU, GPU or not. |

## 3. Prerequisites

- An NVIDIA GPU with a recent driver installed. Check with:
  ```bash
  nvidia-smi
  ```
  The top-right corner shows a "CUDA Version: X.Y" figure. That's the *maximum* CUDA Toolkit version the installed driver supports, not a toolkit that's actually installed.

- **The CUDA Toolkit (provides `nvcc`).** A display driver alone does not include this; it's a separate install. Check first:
  ```bash
  nvcc --version
  ```
  If that already prints a version, confirm it's usable before installing anything: the toolkit version must be **≤** the "CUDA Version: X.Y" figure `nvidia-smi` reports (drivers are backward-compatible with older toolkits, not forward-compatible with newer ones), and it must be new enough to know about the GPU's architecture (recent GPUs, e.g. Ada Lovelace/RTX 40-series, need CUDA ≥ 11.8; older GPUs work with any reasonably recent toolkit). If both hold, the existing `nvcc` is fine, skip ahead to [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system).

  **CUDA Toolkit 13.x note.** CUDA 13 ships CCCL 3.x (the merged libcu++/Thrust/CUB), which removed several internal macros that older OpenCV/`opencv_contrib` releases still use, most visibly `_LIBCUDACXX_BEGIN_NAMESPACE_STD` in `opencv_contrib`'s `cudev` headers. Any pinned OpenCV/contrib version that predates the corresponding fix fails compilation of the *very first* CUDA source file with ~100 cascading errors rooted in `cudev/ptr2d/zip.hpp`. Section 4 includes a pre-flight check and a one-line patch for this; the corresponding failure signature is in [Section 9](#9-troubleshooting). CUDA 12.x toolkits are unaffected.

  If `nvcc --version` fails with "command not found", or the version check above fails, install it:
  - **Ubuntu/Debian, simplest option (distro package, no extra repo setup):**
    ```bash
    sudo apt-get update
    sudo apt-get install -y nvidia-cuda-toolkit
    nvcc --version   # confirm it now works
    ```
    For a specific newer version instead (e.g. the distro package is older than what the GPU's architecture requires), use NVIDIA's own repo: pick the OS/architecture at [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads), which generates the exact `wget`/`dpkg`/`apt-get install cuda-toolkit` commands for that system. That installer places `nvcc` under `/usr/local/cuda/bin`, which may not be on `PATH` yet; add it for the *current terminal session* (no need to edit any shell startup file, since it's only needed for running the build commands below, in this same session):
    ```bash
    export PATH="/usr/local/cuda/bin:$PATH"
    nvcc --version
    ```
    To keep it on `PATH` permanently, append that same `export` line to whichever startup file the shell in use reads: `~/.bashrc` for bash, `~/.zshrc` for zsh, `~/.config/fish/config.fish` for fish (different syntax there: `fish_add_path /usr/local/cuda/bin`). `echo $SHELL` shows which shell is active.
  - **Fedora/RHEL:** `sudo dnf install cuda-toolkit` (after adding NVIDIA's `.repo` file, generated at the same [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) page for the target distro/version).
  - **Windows:** download and run the installer from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads) (select Windows, the target version, exe (local)). It adds `nvcc` to `PATH` automatically; restart the terminal afterward and confirm with `nvcc --version`.

- **A C++ host compiler compatible with that CUDA Toolkit's `nvcc`.** Each CUDA Toolkit release only supports host compilers up to a specific maximum version; a compiler that's too new fails the build with an "unsupported GNU version" (Linux) or similarly rejected MSVC version (Windows) error. This is a common combination on newer Linux distributions, which ship newer GCC by default than older CUDA Toolkits support: for example, CUDA 12.0 supports GCC up to version 12, while Ubuntu 24.04 ships GCC 13 by default.
  - Linux: check what's installed with `gcc --version`, and cross-reference it against the compiler support table in [NVIDIA's CUDA installation guide](https://docs.nvidia.com/cuda/) for the CUDA Toolkit version in use. If the default compiler is too new, install an older `gcc`/`g++` alongside it (this does not remove or change the system default):
    ```bash
    sudo apt-get install -y gcc-12 g++-12
    ```
    Note the path to the alternate compiler (`/usr/bin/g++-12` above, adjust the version number to whatever was installed). [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system) has a `HOST_COMPILER_FLAG` variable to set it in; it's kept separate from the rest of `CMAKE_ARGS` there specifically so this doesn't require hand-editing a long multi-line string in the shell. That variable sets both `CUDA_HOST_COMPILER` and `CMAKE_CUDA_HOST_COMPILER`: OpenCV's CUDA modules mix CMake's legacy `FindCUDA.cmake` module (reads `CUDA_HOST_COMPILER`) and modern native CUDA language support (reads `CMAKE_CUDA_HOST_COMPILER`) depending on the module, and only one of the two takes effect for a given `.cu` file, so setting both covers either case without needing to know which path a specific module uses.

    **Also check what the *default* `c++`/`cc` point to** (`c++ --version`), not just `gcc --version`: on machines where `update-alternatives` has been used, the default can be an unexpectedly old compiler (e.g. GCC 9 on Ubuntu 24.04). The host-compiler flags above only steer `.cu` compilation; the bulk of OpenCV's C++ code still uses the default. Mixing a very old default C++ compiler with a newer nvcc host compiler generally links fine (same libstdc++ ABI), but it's cleaner to put everything on one toolchain. `HOST_COMPILER_FLAG` in Section 4 therefore has an optional second line adding `-DCMAKE_C_COMPILER`/`-DCMAKE_CXX_COMPILER` for exactly this case.
  - Windows: Visual Studio Build Tools, matching the MSVC version supported by the CUDA Toolkit in use (same NVIDIA installation guide has the compatibility table).
- CMake (recent version; the OpenCV 5.x build requires a fairly modern CMake). Check with `cmake --version`; if the distro's package is old, install a newer one via `pip install cmake` or from [cmake.org](https://cmake.org/download/).
- Python build tooling: `pip install --upgrade pip` (>= 19.3), plus `numpy`.
- **Video I/O support (Linux).** OpenCV's CMake step silently disables FFMPEG (and thus `cv2.VideoCapture`/`VideoWriter`, which `scripts/stabilize_video.py`/`stabilize_boxes.py` both need) if the dev headers aren't found; it does *not* fail loudly, so this is easy to miss until a video won't open. Install before building:

  ```bash
  # Debian/Ubuntu
  sudo apt-get install -y build-essential cmake git pkg-config \
      libavcodec-dev libavformat-dev libavutil-dev libswscale-dev
  ```
  (Adjust package names for the target distro, e.g. `dnf install ffmpeg-devel` on Fedora.) Image codec libs (`libjpeg-dev`, `libpng-dev`, `libtiff-dev`, `zlib1g-dev`) are optional: OpenCV falls back to building bundled copies from source if they're missing, just slower.

## 4. Recommended Build: OpenCV 5.0.0 via the `opencv-python` Build System

OpenCV 5.0.0 (stable) is the version to target. Build it through the official [`opencv-python`](https://github.com/opencv/opencv-python) repository's own packaging system rather than a bare `cmake`/`make install` of the `opencv` repo. This produces a properly-versioned wheel that `pip` understands, avoids the classic "two conflicting `cv2` installs" problem, and gives a clean `pip uninstall` path if it's ever removed later.

**`ENABLE_CONTRIB=1` is required.** As established in [Section 1](#1-why-a-source-build-is-required), every CUDA module lives in `opencv_contrib`; there is no way to get `gpu=True` working without it. The consequence: the wheel this produces is named `opencv-contrib-python`, not `opencv-python`, which does *not* satisfy `pyproject.toml`'s `opencv-python>=4.10.0` dependency as far as `pip` is concerned (they're different package names on PyPI even though both provide the `cv2` import). Section 6 below shows the `--no-deps` workaround.

**Use a dedicated virtual environment for this**, separate from the normal stabilo dev venv. Building against OpenCV 5.0.0 replaces the entire `cv2` module in that venv (CPU code paths included), and stabilo's bit-exact regression test (`test_exact_reproduction_of_reference_routine`, pinned against OpenCV 4.x SIFT/RANSAC numerics) may legitimately fail there if internal numerics shifted between major versions. That's expected in a GPU-dedicated venv, not a bug.

**`detector_name` in `{'brisk', 'kaze', 'akaze'}` does not work in this venv, at all.** OpenCV 5.0.0 dropped the Python bindings for `cv2.BRISK_create`, `cv2.KAZE_create`, and `cv2.AKAZE_create` outright, not just their `cv2.cuda` equivalents (see [Section 2](#2-what-gets-cuda-accelerated-in-stabilo)). As of stabilo 1.3.1, `Stabilizer(detector_name='brisk'|'kaze'|'akaze', ...)` raises a clear `ValueError` on construction here regardless of `gpu`, instead of the `AttributeError` it raised before. If you need those three detectors, use the normal CPU-only stabilo venv, which now pins `opencv-python<5.0.0`; this CUDA venv is for `orb` (`gpu=True`) plus `orb`/`sift`/`rsift` (`gpu=False`) only.

If the Python used to create this venv is itself a conda-provided interpreter (e.g. from a `miniconda3`/`anaconda3` install, even if conda isn't actively "activated"), the venv still resolves its C++ runtime (`libstdc++.so.6`) from that conda installation at import time. If conda's bundled `libstdc++` is older than the compiler used for this build, importing the built `cv2` fails at runtime with `GLIBCXX_3.4.3X not found`, a compile-time-looking error that's actually unrelated to anything in this guide's build steps; see [Section 9](#9-troubleshooting) if that happens.

```bash
python -m venv .venv-cuda
source .venv-cuda/bin/activate        # Windows: .venv-cuda\Scripts\activate
pip install --upgrade pip numpy

git clone --recursive https://github.com/opencv/opencv-python.git
cd opencv-python
```

### 4.1 Pinning the OpenCV version (submodules) — and making the pin stick

Two things about this repo's layout that are easy to trip over:

1. **The wrapper repo's own tags are internal build numbers (`86`, `88`, `93`, ...), not OpenCV versions.** `git tag` in the `opencv-python` checkout will not show `5.0.0`; the OpenCV version tags live inside the `opencv` and `opencv_contrib` *submodules*. Also, a brand-new OpenCV release may exist as a tag in the submodules before the wrapper repo has moved its recorded submodule pointers to it — pinning the submodules manually (below) covers that case.

2. **A bare `git -C opencv checkout <ref>` does not survive the build.** `pip`'s build hooks run `git submodule sync` + `git submodule update --init --recursive` on *every* invocation of `pip wheel .`, which resets both submodules to whatever SHAs the wrapper repo has committed — silently undoing any manual checkout. The build then proceeds against the wrapper's recorded (possibly much older) OpenCV version without any error. The fix is to record the new submodule SHAs with a local throwaway commit; `git submodule update` then becomes a no-op.

```bash
# Pin both submodules to the OpenCV 5.0.0 release
git -C opencv fetch --tags && git -C opencv checkout 5.0.0
git -C opencv_contrib fetch --tags && git -C opencv_contrib checkout 5.0.0

# REQUIRED: commit the new submodule pointers so pip's `git submodule update`
# can't revert them. Local throwaway commit; identity flags avoid needing global git config.
git add opencv opencv_contrib
git -c user.name=local -c user.email=local@local commit -m "Pin submodules to 5.0.0"
```

To pin to a branch head instead of a release tag (e.g. `origin/4.x` / `origin/5.x` to pick up post-release fixes that haven't been tagged yet, such as the CUDA 13 compatibility fixes):

```bash
git -C opencv fetch origin && git -C opencv checkout origin/4.x
git -C opencv_contrib fetch origin && git -C opencv_contrib checkout origin/4.x
git add opencv opencv_contrib
git -c user.name=local -c user.email=local@local commit -m "Pin submodules to 4.x heads"
```

Either way, **verify the pin took effect** when the build starts: the CMake configure output prints a banner like `General configuration for OpenCV 5.0.0` within the first minute. If it names a different version — or the log shows `Submodule path 'opencv': checked out '<some other sha>'` right at the start — the commit step above was skipped and pip reverted the checkout; stop the build rather than waiting 40 minutes for the wrong version.

### 4.2 CUDA 13.x pre-flight check (skip on CUDA 12.x)

If `nvcc --version` reports CUDA 13.x, check whether the pinned `opencv_contrib` contains the CCCL 3.x fix before starting the build:

```bash
grep -n '_LIBCUDACXX' opencv_contrib/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp
```

No output → the fix is in, continue to 4.3. If it prints matches (two lines, `_LIBCUDACXX_BEGIN_NAMESPACE_STD` and `_LIBCUDACXX_END_NAMESPACE_STD`), the build **will** fail on the first `.cu` file (see [Section 9](#9-troubleshooting) for the failure signature). Patch it — CUDA 13's libcu++ removed those macros, and the enclosed `tuple_size`/`tuple_element` specializations need to live in an explicit `cuda::std` namespace instead:

```bash
sed -i \
  -e 's/^_LIBCUDACXX_BEGIN_NAMESPACE_STD$/namespace cuda { namespace std {/' \
  -e 's/^_LIBCUDACXX_END_NAMESPACE_STD$/}}/' \
  opencv_contrib/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp

# Sanity-check the edit: the tuple_size/tuple_element specializations should now sit
# inside an explicit `namespace cuda { namespace std { ... }}` block.
grep -n -A2 'namespace cuda { namespace std {' \
  opencv_contrib/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp

# Commit the patch inside the submodule, then re-pin the wrapper repo — same reason
# as 4.1: without both commits, pip's `git submodule update` discards the patch.
git -C opencv_contrib add -A
git -C opencv_contrib -c user.name=local -c user.email=local@local commit -m "Fix cudev zip.hpp for CUDA 13 CCCL 3.x"
git add opencv_contrib
git -c user.name=local -c user.email=local@local commit -m "Pin patched contrib"
```

Verified working against OpenCV/contrib `4.x` heads with CUDA 13.3 (compute capability 8.9), Ubuntu 24.04, GCC 12 host compiler. Later OpenCV releases are expected to include this fix upstream, at which point the `grep` above simply comes back empty and no patch is needed.

### 4.3 Configure and build

Detect the GPU's compute capability and build the CMake flags from it. `HOST_COMPILER_FLAG` is a separate variable specifically so nothing needs to be hand-edited into the middle of the longer `CMAKE_ARGS` string below: leave it empty if [Section 3](#3-prerequisites) found the default compiler already compatible, otherwise set it to the one line shown (commented out) with the path noted there.

```bash
CUDA_ARCH=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "Detected compute capability: $CUDA_ARCH"

HOST_COMPILER_FLAG=""
# Only if Section 3 found the default host compiler incompatible, uncomment and adjust the version.
# Both variables are set because OpenCV's CUDA modules mix CMake's legacy FindCUDA.cmake module
# (reads CUDA_HOST_COMPILER) and modern native CUDA language support (reads CMAKE_CUDA_HOST_COMPILER)
# depending on the specific module, and only one of the two takes effect for a given .cu file.
# HOST_COMPILER_FLAG="-DCUDA_HOST_COMPILER=/usr/bin/g++-12 -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12"
#
# Optionally also put the plain C/C++ compilation on the same toolchain — recommended
# when the machine's default `c++` is a different major version than the nvcc host
# compiler (check with `c++ --version`; see the Section 3 note on update-alternatives):
# HOST_COMPILER_FLAG="$HOST_COMPILER_FLAG -DCMAKE_C_COMPILER=/usr/bin/gcc-12 -DCMAKE_CXX_COMPILER=/usr/bin/g++-12"

export ENABLE_CONTRIB=1
export CMAKE_ARGS="-DWITH_CUDA=ON -DCUDA_ARCH_BIN=${CUDA_ARCH} ${HOST_COMPILER_FLAG} \
  -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,features,flann,calib3d,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudafeatures2d \
  -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF"
```

`BUILD_LIST` restricts the contrib build to only the modules stabilo needs (see [Section 2](#2-what-gets-cuda-accelerated-in-stabilo)) instead of compiling all ~60 `opencv_contrib` modules, several of which pull in dependencies unrelated to stabilo's GPU requirements (VTK, Eigen, Atlas/BLAS/LAPACK, etc.) that a typical build machine won't have installed. The list above works for both the 5.0.0 tag and the 4.x branch heads (OpenCV 4.13+ already uses the renamed `features` module; on much older 4.x pins, `features` may need to be `features2d`). **If `BUILD_LIST` causes a "module not found" or missing-dependency CMake error**, drop the `-DBUILD_LIST=...` flag entirely and let it build full `opencv_contrib` instead: slower (potentially 1-2+ hours) but a safe superset that sidesteps any module-naming mismatch for a given OpenCV version.

On Windows PowerShell, the equivalent:

```powershell
$CUDA_ARCH = (nvidia-smi --query-gpu=compute_cap --format=csv,noheader | Select-Object -First 1)
Write-Host "Detected compute capability: $CUDA_ARCH"

$HostCompilerFlag = ""
# Only if Section 3 found the default host compiler incompatible, uncomment and adjust the version/path.
# Both variables are set for the same reason as the bash block above (legacy FindCUDA.cmake vs
# modern native CUDA language support each reading a different variable):
# $HostCompilerFlag = "-DCUDA_HOST_COMPILER=C:\path\to\older\cl.exe -DCMAKE_CUDA_HOST_COMPILER=C:\path\to\older\cl.exe"

$env:ENABLE_CONTRIB = "1"
$env:CMAKE_ARGS = "-DWITH_CUDA=ON -DCUDA_ARCH_BIN=$CUDA_ARCH $HostCompilerFlag -DBUILD_LIST=core,imgproc,imgcodecs,videoio,highgui,features,flann,calib3d,python3,cudev,cudaarithm,cudafilters,cudaimgproc,cudawarping,cudafeatures2d -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF"
```

**Retrying a failed or interrupted `pip wheel .`? Remove the cached build tree first, every time:**
```bash
rm -rf _skbuild
```
`pip` creates a brand-new temporary build-isolation environment (`/tmp/pip-build-env-XXXXX/`) on *every* `pip wheel .` invocation, including the one that provides NumPy's headers to the build, and deletes it once that invocation ends. CMake caches whatever it detects on a given configure (the CUDA host compiler, NumPy's include path, etc.) in `_skbuild/<platform>/cmake-build/CMakeCache.txt` and does not reliably re-detect those values on a later incremental configure, even when the underlying value (like that temp directory) no longer exists. A stale cache pointing at a deleted temp directory produces confusing failures far removed from the actual cause (a "missing" file that was never actually missing, just relocated by the next pip invocation), so it's not worth trying to reason about which specific retry needs a clean tree: always wipe it. This does mean a retry takes the full build time again rather than resuming. The same applies after changing the submodule pin (4.1) or applying the header patch (4.2): stale caches across OpenCV versions cause unrelated-looking configure errors.

Build and install the wheel (`tee` keeps a searchable copy of the very long build log — `grep -m5 'error:' build.log` beats scrolling if something fails):

```bash
pip wheel . --verbose 2>&1 | tee build.log
pip install opencv_contrib_python-*.whl
```
If the shell reports no match for that wildcard (e.g. zsh's "no matches found"), the build above did not produce a wheel; search `build.log` for the actual error rather than treating the missing file as the problem itself.

This can take anywhere from ~15 minutes to over an hour depending on the hardware; the `BUILD_LIST`/`BUILD_TESTS`/`BUILD_PERF_TESTS`/`BUILD_EXAMPLES`/`BUILD_DOCS` flags above cut this down significantly relative to a full default build. A clean rebuild after `rm -rf _skbuild` takes the full time again, since nothing is cached.

**A note on dependency versions.** This build uses whatever `numpy` is installed in `.venv-cuda`, and targets OpenCV 5.0.0, independent of the version floors declared in stabilo's `pyproject.toml` (`opencv-python>=4.10.0`, `numpy>=1.26.4`). Those floors govern the plain CPU-only `pip install stabilo` path that most users take, and are not raised by this guide: building and testing CUDA support does not require, and should not force, a newer OpenCV/numpy baseline onto users who only need the CPU path. The dedicated venv from this section, combined with the `--no-deps` install in Section 6, keeps the two fully isolated from each other.

## 5. Verifying the Build

Run this from *outside* the `opencv-python` checkout directory (`cd ~` or anywhere else first). That checkout tracks its own `cv2/` directory at its root (packaging template files: `config.py`, `load_config_py3.py`, `__init__.py`, etc.), and Python's import system checks the current directory before site-packages, so running `python -c "import cv2"` from inside the checkout silently imports that local template instead of the properly installed package, producing confusing `AttributeError`s that look like the CUDA build failed even when it didn't. Check `cv2.__file__` first if anything here looks wrong: it should point into `.venv-cuda/lib/python3.11/site-packages/cv2/__init__.py`, not into the `opencv-python` checkout.

```bash
python -c "
import cv2
print(cv2.__file__)
print(cv2.__version__)
print('CUDA devices:', cv2.cuda.getCudaEnabledDeviceCount())
info = cv2.getBuildInformation()
print([l for l in info.splitlines() if 'CUDA' in l or 'cuda' in l.lower()])
"
```

Confirm `cv2.__version__` matches the version pinned in [Section 4.1](#41-pinning-the-opencv-version-submodules--and-making-the-pin-stick) (an unexpected version here means the submodule pin didn't stick; see Section 9). There should be at least one CUDA device, and `cudev`/`cudafeatures2d`/`cudawarping`/`cudaimgproc`/`cudafilters`/`cudaarithm` listed among the enabled modules. Then check which of stabilo's detectors the build actually exposes on the GPU. This scans every attribute under `cv2.cuda` rather than checking a fixed `*_create` name list, since a detector's CUDA class might not follow that exact naming convention:

```bash
python -c "
import cv2
names = [a for a in dir(cv2.cuda) if any(k.lower() in a.lower() for k in ['ORB', 'SIFT', 'BRISK', 'KAZE'])]
print(sorted(names))
"
```

Confirmed on an actual OpenCV 5.0.0 + opencv_contrib 5.0.0 build (the versions this guide targets): the output is `['ORB', 'ORB_create']`, i.e. **only ORB** has a CUDA-wrapped Python binding, despite `cudafeatures2d` being active. SIFT, BRISK, KAZE, and AKAZE (and therefore RSIFT, which would reuse SIFT's CUDA class if one existed) have no CUDA Python binding in this module at all, so `gpu=True` is currently only useful with `detector_name='orb'`. This may differ on other OpenCV/opencv_contrib versions; re-run the check above to verify on a specific build. Whatever's missing from this list isn't CUDA-accelerated: `Stabilizer(detector_name=..., gpu=True)` will fail loudly for that detector (this is intentional, see [Section 7](#7-enabling-gpu-mode-in-stabilo)); use `gpu=False` for it instead, except for BRISK/KAZE/AKAZE, whose Python bindings OpenCV 5.0.0 removed entirely, CPU included (see the warning in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system)); `gpu=False` doesn't rescue those three on this build.

## 6. Installing Stabilo on Top

In the same `.venv-cuda` environment. Because the wheel installed in Section 4 is named `opencv-contrib-python` (not `opencv-python`), a plain `pip install stabilo` or `pip install -e '.[dev]'` would try to *also* install a separate CPU-only `opencv-python`, creating two conflicting `cv2` installs. `--no-deps` avoids that in both installation paths below by skipping dependency resolution for stabilo itself; the rest of its runtime deps are then installed by hand.

**From a local clone or fork** (repo root, editable install — needed to run the test suite or edit stabilo itself):

```bash
pip install -e . --no-deps
pip install pyyaml tqdm   # core runtime deps other than opencv and numpy (both already installed)
pip install pytest ruff matplotlib   # dev extras, for the test suite / lint
```

**From PyPI, without cloning the repo:**

```bash
pip install stabilo --no-deps
pip install pyyaml tqdm   # core runtime deps other than opencv and numpy (both already installed)
pip install matplotlib   # optional, only needed for scripts/find_thresholds/ (the `extras` install group)
```
There's no local `tests/` directory in this path, so `pytest`/`ruff` aren't relevant; use the clone/fork path above if the CUDA build needs to be tested against stabilo's test suite.

Verify nothing pulled in a second `cv2`, for either path:

```bash
pip list | grep -i opencv   # should show exactly one line: opencv-contrib-python
```

## 7. Enabling GPU Mode in Stabilo

```python
from stabilo import Stabilizer

stab = Stabilizer(gpu=True, gpu_device_id=0, detector_name='orb')
stab.set_ref_frame(ref_frame)
stab.stabilize(cur_frame)
warped = stab.warp_cur_frame()
```

Or via the CLI scripts:

```bash
python scripts/stabilize_video.py path/to/video.mp4 --gpu --save
python scripts/stabilize_boxes.py path/to/video.mp4 --gpu --save
```

Multi-GPU machines: pass `gpu_device_id` (or `--gpu-device-id`) to pick which CUDA device to use.

## 8. Performance Notes

When `gpu=True`, `Stabilizer` keeps one persistent `cv2.cuda_Stream` (`self.gpu_stream`) and four reusable `GpuMat` buffers per instance (`_gpu_frame_buf`, `_gpu_mask_buf`, `_gpu_cur_desc_buf`, `_gpu_warp_buf`) for the lifetime of the object, so no GPU memory is allocated on a per-frame basis. Every GPU operation runs on `self.gpu_stream` rather than the implicit default stream, and each of `get_features_and_descriptors`, `get_matches`, and `warp_frame` syncs that stream exactly once, right before touching its result as CPU/numpy data. `get_matches` uses the async `matchAsync`/`knnMatchAsync` + `matchConvert`/`knnMatchConvert` matcher API rather than the synchronous `match`/`knnMatch`.

Measured on an RTX 4090, OpenCV 5.0.0 + opencv_contrib 5.0.0, `detector_name='orb'`, 3840x2160 (4K) frames: GPU numbers are averaged over 5 runs (pooling every per-call sample, 1625 calls total), CPU numbers over 2 runs, which is enough for CPU since it varied under 2% run-to-run while GPU numbers on a busy machine can vary 30%+ between individual runs.

| stage | GPU | CPU | delta |
|---|---|---|---|
| `get_features_and_descriptors` | 15.36 ms | 27.74 ms | 1.8x faster |
| `get_matches` | 0.92 ms | 4.08 ms | 4.4x faster |
| `calculate_transformation_matrix` (CPU-only either way) | 2.84 ms | 4.60 ms | lower on the GPU run; RANSAC itself isn't GPU-accelerated, this reflects CUDA ORB producing a different keypoint set than CPU ORB, feeding a cheaper fit |
| `warp_cur_frame` | 6.30 ms | 3.14 ms | 2x *slower*, see the table in [Section 2](#2-what-gets-cuda-accelerated-in-stabilo) |
| end-to-end frame rate | 14.2 fps | 11.9 fps | 19% faster |

`gpu=True`'s net win comes entirely from detection and matching; the warp stage is a wash-to-slight-loss for this frame size and isn't worth chasing further unless a workload does many warps per detect/match cycle.

Reproduce a single run with `STABILO_PROFILE=1` (enables the existing `@timer`-decorated method prints without editing source):

```bash
STABILO_PROFILE=1 python scripts/stabilize_video.py path/to/video.mp4 --gpu --save --detector-name orb 1> gpu.log
STABILO_PROFILE=1 python scripts/stabilize_video.py path/to/video.mp4 --save --detector-name orb 1> cpu.log
awk '{sum[$1]+=$(NF-1); n[$1]++} END {for (s in sum) printf "%-35s avg %8.2f ms (n=%d)\n", s, sum[s]/n[s], n[s]}' gpu.log | sort
awk '{sum[$1]+=$(NF-1); n[$1]++} END {for (s in sum) printf "%-35s avg %8.2f ms (n=%d)\n", s, sum[s]/n[s], n[s]}' cpu.log | sort
```

For a steadier read on a shared or otherwise busy machine, average several runs and pool every per-call sample across all of them instead of trusting one run:

```bash
for i in $(seq 1 5); do
  STABILO_PROFILE=1 python scripts/stabilize_video.py path/to/video.mp4 --gpu --save --detector-name orb 1> gpu_run$i.log
done
awk '{sum[$1]+=$(NF-1); n[$1]++} END {for (s in sum) printf "%-35s avg %8.2f ms (n=%d)\n", s, sum[s]/n[s], n[s]}' gpu_run*.log | sort
```
(swap in `--save` without `--gpu` and a `cpu_run$i.log` name for the CPU-side average.)

## 9. Troubleshooting

- **`CMake Error: CUDA: OpenCV requires enabled 'cudev' module from 'opencv_contrib'`.** The build ran with `WITH_CUDA=ON` but without `ENABLE_CONTRIB=1`. This is mandatory, not optional; see [Section 1](#1-why-a-source-build-is-required)/[Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system). Set `export ENABLE_CONTRIB=1` before `pip wheel .`.
- **CMake can't find `nvcc` / CUDA compiler.** Ensure the CUDA Toolkit's `bin` directory is on `PATH` (Linux: `/usr/local/cuda/bin`) before running `pip wheel .`.
- **`nvcc` rejects the host compiler ("unsupported GNU version" or similar), including after setting `HOST_COMPILER_FLAG`.** Three distinct causes, in the order to check them:
  1. **Wrong variable name for the code path in use.** OpenCV's CUDA modules mix CMake's legacy `FindCUDA.cmake` module (reads `CUDA_HOST_COMPILER`) and modern native CUDA language support (reads `CMAKE_CUDA_HOST_COMPILER`), and only one of the two is read for a given `.cu` file. A build log with file names like `..._generated_gpu_mat.cu.o` and a `*.cu.o.Release.cmake` wrapper script is the signature of the legacy `FindCUDA.cmake` path, meaning `CUDA_HOST_COMPILER` (no `CMAKE_` prefix) is the variable that matters there, not `CMAKE_CUDA_HOST_COMPILER`. `HOST_COMPILER_FLAG` in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system) sets both, so this shouldn't come up when following these instructions as written, but it's the first thing to check if it was set by hand.
  2. **Missing `-D` prefix.** Check the CMake configure log for a line like `CMake Warning: Ignoring extra path from command line: "DCMAKE_CUDA_HOST_COMPILER=..."` (or `DCUDA_HOST_COMPILER=...`): that means the `-D` was dropped somewhere, commonly from editing the `CMAKE_ARGS` string by hand instead of using the separate `HOST_COMPILER_FLAG` variable, so the flag was silently discarded rather than applied.
  3. **Stale build cache.** CMake detects and caches the CUDA host compiler the first time it configures a given build directory, and reuses that cached detection on later configures regardless of a changed `CMAKE_ARGS`, even once the variable name and formatting are both correct. Remove the cached build tree and rebuild, per the note in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system): `rm -rf _skbuild` from the `opencv-python` checkout root, then `pip wheel .` again.
- **~100 CUDA compile errors on the very first `.cu` file, rooted in `opencv_contrib/modules/cudev/include/opencv2/cudev/ptr2d/zip.hpp`, starting with `error: this declaration has no storage class or type specifier` on a line reading `_LIBCUDACXX_BEGIN_NAMESPACE_STD`.** The pinned OpenCV/contrib version predates the CUDA 13 / CCCL 3.x compatibility fix: CUDA 13's libcu++ removed that macro, so it expands to nothing, the parser derails on that header, and everything downstream (`grid/detail/copy.hpp`, `gpu_mat.cu`, `gpu_mat_nd.cu`, ...) cascades into `identifier "GpuMat_" is undefined`-style noise. Only the first two `zip.hpp` errors are the real problem. Apply the patch in [Section 4.2](#42-cuda-13x-pre-flight-check-skip-on-cuda-12x) (including its two commit steps) or pin to a submodule version that contains the upstream fix, then `rm -rf _skbuild` and rebuild.
- **The build compiles the wrong OpenCV version despite checking out a different tag/branch in the submodules** (CMake's `General configuration for OpenCV X.Y.Z` banner names the old version; the log shows `Submodule path 'opencv': checked out '<sha>'` near the start). `pip`'s build hooks run `git submodule update --init --recursive` on every `pip wheel .`, resetting the submodules to the SHAs committed in the wrapper repo and silently discarding any uncommitted `git -C opencv checkout ...`. This also silently discards the Section 4.2 header patch if it wasn't committed inside the submodule. Fix: the commit steps in [Section 4.1](#41-pinning-the-opencv-version-submodules--and-making-the-pin-stick)/[4.2](#42-cuda-13x-pre-flight-check-skip-on-cuda-12x), then `rm -rf _skbuild` and rebuild.
- **`git checkout 5.0.0` (or another OpenCV version) fails with "pathspec did not match" in the `opencv-python` checkout root, or `git tag` there only shows small numbers like `86`, `93`.** Those are the wrapper repo's internal build-number tags, not OpenCV versions. The version tags live in the submodules: `git -C opencv fetch --tags && git -C opencv checkout <version>` (and the same for `opencv_contrib`), per [Section 4.1](#41-pinning-the-opencv-version-submodules--and-making-the-pin-stick).
- **`BUILD_LIST` causes a "module not found" or missing-dependency error.** Module names can shift between OpenCV releases (e.g. `features2d` → `features` in 4.13/5.x; `calib3d` split into `calib` + `3d` in parts of 5.x). Drop the `-DBUILD_LIST=...` flag entirely (full contrib build, slower but a safe superset) rather than debugging the exact module name for a given version.
- **Build reaches 100% and compiles successfully, but fails at the very end with `Exception: Not found: 'python/cv2/py.typed'`** (often preceded by `UserWarning: Typing stubs generation has failed` and a `SymbolNotFoundError` naming some `cv.*` function). `opencv-python`'s packaging step generates Python typing stubs by looking up a fixed list of functions (`findEssentialMat`, `solvePnP*`, `calibrateCamera`, `undistortPoints`, etc., all from `calib3d`) regardless of which modules a downstream project actually needs, and fails the whole build if any of them are missing. The `BUILD_LIST` in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system) already includes `calib3d` for exactly this reason; if a different symbol from a different excluded module trips the same error, either add that module to `BUILD_LIST` too or drop `BUILD_LIST` entirely. Wipe `_skbuild` before retrying either way, per the note in Section 4.
- **`fatal error: numpy/ndarrayobject.h: No such file or directory`, especially on a retry after a build that got further than this before.** A stale `_skbuild` cache pointing at NumPy headers inside a previous `pip` build-isolation temp directory that no longer exists; see the note in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system). `rm -rf _skbuild` and rebuild.
- **Build takes forever / runs out of memory.** Make sure `BUILD_LIST` is set (or, failing that, `BUILD_TESTS`/`BUILD_PERF_TESTS`/`BUILD_EXAMPLES`/`BUILD_DOCS` are all `OFF`, see Section 4). On memory-constrained machines, reduce parallel build jobs via `export MAKEFLAGS="-j2"` before `pip wheel .`.
- **CMake errors mentioning a literal `<your` or `compute` or `capability>` as separate arguments.** `CUDA_ARCH_BIN` ended up set to placeholder text instead of a real number. The Section 4 commands compute this automatically via `nvidia-smi`; if the `CMAKE_ARGS` line was edited by hand, confirm `echo "$CUDA_ARCH"` (`Write-Host $CUDA_ARCH` on Windows) prints a real number like `8.6` before it's used.
- **`AttributeError: module 'cv2' has no attribute 'cuda'` (or `'__version__'`, or any other unexpected missing attribute) even though the build reported success.** Check `cv2.__file__` first. If it points inside the `opencv-python` checkout directory (e.g. `.../opencv-python/cv2/__init__.py`) rather than into `.venv-cuda/lib/pythonX.Y/site-packages/cv2/__init__.py`, Python imported that checkout's own tracked `cv2/` packaging-template directory instead of the actually-installed package, because the current directory is checked before site-packages and the checkout happens to contain a same-named folder. Re-run from any other directory (`cd ~` first) rather than from inside the `opencv-python` checkout. See the note in [Section 5](#5-verifying-the-build).
- **Videos won't open / `cv2.VideoCapture(...).isOpened()` is `False`, even though the build succeeded.** CMake's configure step silently disabled FFMPEG (look for `-- FFMPEG is disabled. Required libraries: ... not found` in the build log) instead of failing loudly. Install the FFMPEG dev headers listed in [Section 3](#3-prerequisites) and rebuild.
- **`ImportError` / two `cv2` installs conflicting.** Run `pip uninstall opencv-python opencv-contrib-python opencv-python-headless opencv-contrib-python-headless` (all of them) in the target venv before installing the custom build, then reinstall only the custom wheel. Never have more than one `opencv-*` package installed in the same environment; this is also why Section 6 uses `pip install -e . --no-deps` for stabilo itself.
- **`ImportError: .../libstdc++.so.6: version 'GLIBCXX_3.4.3X' not found (required by .../cv2.abi3.so)`.** The venv's Python is a conda-provided interpreter, so it resolves `libstdc++.so.6` from that conda installation at runtime, and conda's bundled copy is older than what the build's compiler requires (see the note in [Section 4](#4-recommended-build-opencv-500-via-the-opencv-python-build-system)). Update it:
  ```bash
  conda install -n base -c conda-forge libstdcxx-ng -y
  ```
  (Adjust `-n base` to whichever conda environment actually provides the venv's underlying Python, if not `base`.) This upgrades a shared runtime library only; it does not require rebuilding OpenCV. Verify with `strings /path/to/conda/lib/libstdc++.so.6 | grep GLIBCXX_3.4 | tail -5` before and after to confirm the needed version is now present.
- **A specific detector's `cv2.cuda.*_create` doesn't exist even though `WITH_CUDA=ON`.** Not every feature detector has a CUDA implementation in every OpenCV release. Re-check with the Section 5 snippet; if it's `False`, that detector isn't available on GPU in that build, use `gpu=False` for it.
- **Windows: `CMAKE_ARGS`/`ENABLE_CONTRIB` not picked up.** Environment variables set with `set` in `cmd.exe` only persist for that shell session; use `$env:CMAKE_ARGS = "..."` / `$env:ENABLE_CONTRIB = "1"` in PowerShell, or `set VAR=...` immediately before `pip wheel .` in the same `cmd.exe` window.
- **`test_exact_reproduction_of_reference_routine` fails in the CUDA venv.** Expected, see the note in Section 4. This test is pinned to OpenCV 4.x CPU numerics and isn't a signal of a GPU-path bug; check the other stabilo tests (`pytest -k gpu`) instead.
