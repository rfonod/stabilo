# Video Attribution

- **Video**: [video.mp4](./video.mp4)
- **Source**: [Pexels](https://www.pexels.com/video/drone-footage-of-an-intersection-10175667/)
- **License**: [Pexels License](https://www.pexels.com/license/)  

The video file provided in this repository is a cut version of the original drone footage available on Pexels. The video is licensed under the Pexels License, which allows for free personal and commercial use, with optional attribution and permitted modifications. Refer to the Pexels license for more details.

## Additional Files

- **[video.txt](./video.txt)**: Contains unnormalized bounding boxes (BBs) in YOLO format (`x_c, y_c, w, h`) for vehicles detected in each frame, generated using the [geo-trax](https://github.com/rfonod/geo-trax) framework. Each row represents a distinct BB, with the frame number in the 0th column and the BB coordinates in columns 2–5. These BBs can serve as exclusion masks during stabilization and/or as tracks (BBs) to be stabilized.
- **[video_stab.mp4](./video_stab.mp4)**: Stabilized version of the original video, using the zero-th frame as the reference frame for stabilization.
- **[video_viz.mov](./video_viz.mov)**: Visualization of the stabilization process, showing frame transformations relative to the reference frame. Note that `video_viz.mp4`, the actual visualization video, has been replaced in the repository due to its large size. 
- **[video_stab.txt](./video_stab.txt)**: Identical to `video.txt`, but the BBs are replaced with the stabilized BBs.
- **[video_track.mp4](./video_track.mp4)**: Visualization of the original BB tracks and the stabilized BB tracks. The original BB tracks are shown in red, while the stabilized BB tracks are shown in green.

### Commands Used
To re-create `video_stab.mp4` and `video_viz.mp4`, run the following command from the root of the repository:
```sh
python scripts/stabilize_video.py data/video.mp4 -s -sv
```
To re-create `video_stab.txt` and `video_track.mp4`, run the following command from the root of the repository:
```sh
python scripts/stabilize_boxes.py data/video.mp4 -s -sv
```