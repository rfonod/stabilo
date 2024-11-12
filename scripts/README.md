# Stabilo Scripts

This directory contains utility scripts that demonstrate the functionality of the Stabilo library. The scripts included here are `stabilize_video.py` and `stabilize_boxes.py`. Additionally, there is another script located in the find_thresholds folder, which has its own README file.

## stabilize_video.py

### Description

`stabilize_video.py` stabilizes videos using the ‘stabilo’ library. It reads a video file, stabilizes it using a reference frame stabilization method, and optionally saves the stabilized video or the visualization of the stabilization process. The stabilization is based on feature point matching between frames, followed by transformation estimation using a RANSAC-type algorithm. The script supports various feature detectors, matchers, and extensive customization through command-line options or a configuration file. It also supports CLAHE application, video downsampling, and exclusion masks (bounding boxes) for stabilization.

### Usage
```sh
python stabilize_video.py <input> [options]
```

### Options

**Arguments:**
- `input`: Path to the input video file.

**Main Options:**
- `-o OUTPUT`, `--output OUTPUT`: Output folder to save the stabilized video or visualization (default: same as input).
- `-s`, `--save`: Save the stabilized video (default: False).
- `-rf REF_FRAME`, `--ref-frame REF_FRAME`: Custom reference frame index for stabilization (default: 0).
- `-d`, `--debug`: Enable debug mode (default: False).

**Mask Options:**
- `-nm`, `--no-mask`: Do not use exclusion masks during stabilization.
- `-mp MASK_PATH`, `--mask-path MASK_PATH`: Custom mask filepath (default: input with .txt extension).
- `-mfi MASK_FRAME_IDX`, `--mask-frame-idx MASK_FRAME_IDX`: Frame number column index in the mask file (default: 0).
- `-msi MASK_START_IDX`, `--mask-start-idx MASK_START_IDX`: Start column index of the 4 bounding box parameters used as masks (default: 2).
- `me MASK_ENC`, `--mask-enc MASK_ENC`: Bounding box encoding. Choices: ‘yolo’, ‘pascal’, ‘coco’ (default: yolo).

**Visualization Options:**
- `-v`, `--viz`: Visualize the transformation process (default: False).
- `-sv`, `--save-viz`: Save the visualization of the transformation process as a video (default: False).
- `-nl`, `--no-lines`: Hide lines between matched feature points (default: False).
- `-nb`, `--no-boxes`: Hide bounding boxes on the (un-)stabilized videos (default: False).
- `-sp SPEED`, `--speed SPEED`: Visualization speed in milliseconds (0 for manual control) (default: 10).

**Stabilo Configuration:**
- `-cc CUSTOM_CONFIG`, `--custom-config CUSTOM_CONFIG`: Path to a config file that overrides the default stabilo parameters or the CLI arguments. See an example config file [here](./custom.yaml).
- [...] Most of the Stabilo parameters can be provided as command-line arguments. Examples include `--feature-type`, `--matcher-type`, and `--ransac-reproj-threshold`. For a complete list of parameters, refer to the argument parser options or the Stabilo documentation.

### Examples

1.	Stabilize a video using default settings and save the stabilized video:
```sh
python stabilize_video.py path/to/video/video.mp4 --save
```

2.	Visualize the stabilization process:
```sh
python stabilize_video.py path/to/video/video.mp4 --viz
```

3.	Save a stabilized video using a custom detector and matcher:
```sh
python stabilize_video.py path/to/video/video.mp4 --detector-name sift --matcher-name flann --save
```

4.	Apply stabilization without a mask and visualize the process:
```sh
python stabilize_video.py path/to/video/video.mp4 --no-mask --viz
```

5.	Stabilize a video using a custom reference frame and save both stabilized video and visualization:
```sh
python stabilize_video.py path/to/video/video.mp4 --ref-frame 15 --save --save-viz
```

6.	Use a custom mask filepath and specify start column index of bounding boxes:
```sh
python stabilize_video.py path/to/video/video.mp4 --mask-path path/to/mask/mask.txt --mask-start 1 --viz
```

## stabilize_boxes.py

### Description

`stabilize_boxes.py` stabilizes bounding boxes (BBs) in a video using the ‘stabilo’ library. It reads a video file and a corresponding tracks file containing BBs for each frame of the video. The script then stabilizes these BBs and saves them to a file. It also provides options to visualize the stabilized and un-stabilized BBs in real-time or save the visualization as a video. The stabilization can be performed with respect to a custom reference frame, and exclusion masks can be used to exclude certain areas from stabilization.

### Usage
```sh
python stabilize_boxes.py <input> [options]
```

**Arguments:**
- `input`: Filepath to the input video file.

**Main Options:**
- `-o OUTPUT`, `--output OUTPUT`: Output folder to save the stabilized tracks or visualization (default: same as input).
- `-s`, `--save`: Save the stabilized tracks to a file.
- `-rf REF_FRAME`, `--ref-frame REF_FRAME`: Custom reference frame index for stabilization (default: 0).

**Tracks Options:**
- `-t TRACKS`, `--tracks TRACKS`: Filepath to the input tracks file (default: input with .txt extension).
- `-bfi BOX_FRAME_IDX`, `--boxes-frame-idx BOX_FRAME_IDX`: Frame number column index in the tracks file (default: 0).
- `-bsi BOXES_START_IDX`, `--boxes-start-idx BOXES_START_IDX`: Start column index of the 4 bounding box parameters (default: 2).
- `-be BOXES_ENC`, `--boxes-enc BOXES_ENC`: Bounding box encoding. Choices: ‘yolo’, ‘pascal’, ‘coco’ (default: yolo).

**Mask Options:**
- `-nm`, `--no-mask`: Do not use exclusion masks during stabilization.
- `-mp MASK_PATH`, `--mask-path MASK_PATH`: Custom mask filepath (default: input with .txt extension).
- `-mfi MASK_FRAME_IDX`, `--mask-frame-idx MASK_FRAME_IDX`: Frame number column index in the mask file (default: 0).
- `-msi MASK_START_IDX`, `--mask-start-idx MASK_START_IDX`: Start column index of the 4 bounding box parameters used as masks (default: 2).
- `-me MASK_ENC`, `--mask-enc MASK_ENC`: Bounding box encoding. Choices: ‘yolo’, ‘pascal’, ‘coco’ (default: yolo).

**Visualization Options:**
- `-v`, `--viz`: Visualize the stabilized and un-stabilized bounding boxes in real-time (default: False).
- `-sv`, `--save-viz`: Save the visualization of the bounding boxes as a video (default: False).
- `-sp SPEED`, `--speed SPEED`: Visualization speed in milliseconds (0 for manual control) (default: 10).
- `-tl TAIL_LENGTH`, `--tail-length TAIL_LENGTH`: Number of frames to show the track tail in the visualization (default: 40).
- `-tr TAIL_RADIUS`, `--tail-radius TAIL_RADIUS`: Maximum radius in pixels for the track tail circles in the visualization (default: 12).
- `-cx CANVAS_X`, `--canvas-x CANVAS_X`: Canvas enlargement factor in both x and y directions and as a function of the original video resolution (default: 1.5).

**Stabilo Configuration:**
- `-cc CUSTOM_CONFIG`, `--custom-config CUSTOM_CONFIG`: Path to a config file that overrides the default stabilo parameters or the CLI arguments. See an example config file [here](./custom.yaml).
- [...] Most of the Stabilo parameters can be provided as command-line arguments. Examples include `--feature-type`, `--matcher-type`, and `--ransac-reproj-threshold`. For a complete list of parameters, refer to the argument parser options or the Stabilo documentation.

### Examples

1.	Stabilize tracks using default parameters with custom reference frame:
```sh
python stabilize_boxes.py path/to/video.mp4 --save --ref-frame 100
```

2.	Visualize and save stabilization process with custom speed:
```sh
python stabilize_boxes.py path/to/video.mp4 --viz --save-viz --speed 20
```

3.	Stabilize tracks without exclusion masks and save both visualization and tracks:
```sh
python stabilize_boxes.py path/to/video.mp4 --no-mask --save-viz --save
```

4.	Use custom config and mask files; save stabilized tracks:
```sh
python stabilize_boxes.py path/to/video.mp4 --save --custom-config path/to/config.yaml --mask-path path/to/mask.txt
```

5.	Save visualization with custom tail length/radius using custom config:
```sh
python stabilize_boxes.py path/to/video.mp4 --viz --save-viz --custom-config path/to/config.yaml --tail-length 50 --tail-radius 15
```