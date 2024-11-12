#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stabilize_boxes.py - Stabilize bounding boxes using the stabilo library.

Description:
    This script stabilizes bounding boxes (BBs) in a video using the 'stabilo' library. It reads a video file and a
    corresponding tracks file containing BBs for each frame of the video. The script then stabilizes these BBs and
    saves the stabilized BBs to a file. It also provides options to visualize the stabilized and un-stabilized BBs
    in real-time or save the visualization as a video. The stabilization can be performed with respect to a custom
    reference frame, and exclusion masks can be used to exclude certain areas from stabilization. By default, the
    script considers the 0-th frame as the reference frame and uses the BBs as exclusion masks.

Usage:
    python stabilize_boxes.py <input> [options]

Arguments:
    input              : Filepath to the input video file.

Main Options:
    --output OUTPUT    : Output folder to save the stabilized tracks or visualization (default: same as input).
    --save             : Save the stabilized tracks to a file.
    --ref-frame REF_FRAME : Custom reference frame index for stabilization (default: 0).

Tracks Options:
    --tracks TRACKS    : Filepath to the tracks file (default: input with .txt extension).
    --boxes-frame-idx BOXES_FRAME_IDX : Frame number column index in the tracks file (default: 0).
    --boxes-start-idx BOXES_START_IDX : Start column index of the 4 BB parameters in the tracks file (default: 2).
    --boxes-enc BOXES_ENC : Bounding box encoding. Choices: 'yolo', 'pascal', 'coco' (default: yolo).

Mask Options:
    --no-mask          : Disable exclusion masks during stabilization.
    --mask-path MASK_PATH : Custom mask file for stabilization (default: same as boxes).
    --mask-frame-idx MASK_FRAME_IDX : Frame number column index in the mask file (default: 0).
    --mask-start-idx MASK_START_IDX : Start column index of the 4 BB parameters in the mask file (default: 2).
    --mask-enc MASK_ENC : Mask encoding. Choices: 'yolo', 'pascal', 'coco' (default: yolo).

Visualization Options:
    --viz              : Show the stabilized and un-stabilized tracks.
    --save-viz         : Save the visualization as a video at original FPS.
    --speed SPEED      : Visualization speed in milliseconds (0 for manual control, default: 10).
    --tail-length TAIL_LENGTH : Tail length for visualization (default: 40).
    --tail-radius TAIL_RADIUS : Tail radius for visualization (default: 12).
    --canvas-x CANVAS_X : Canvas enlargement factor (>= 1, default: 1.5).

Stabilo Configuration:
    --custom-config    : Path to a config file that overrides the default stabilo parameters or the CLI arguments below.
    --detector-name DETECT : Feature detector. Choices: 'orb', 'sift', 'rsift', 'brisk', 'kaze', 'akaze' (default: orb).
    --matcher-name MATCHER    : Feature matcher. Choices: 'bf', 'flann' (default: bf).
    --filter-type FILTER_TYPE : Type of match filter. Choices: 'none', 'ratio', 'distance' (default: ratio).
    --transformation-type TRANSFORMATION_TYPE : Transformation. Choices: 'projective', 'affine' (default: projective).
    --clahe            : Apply CLAHE to grayscale images (default: False).
    --downsample-ratio DOWNSAMPLE_RATIO : Downsample ratio for the input video (default: 0.5).
    --max-features MAX_FEATURES   : Maximum number of features to detect (default: 2000).
    --ref-multiplier REF_MULTIPLIER : Multiplier for max features in reference frame (default: 2).
    --filter-ratio FILTER_RATIO : Filter ratio for the match filter (default: 0.9).
    --ransac-method RANSAC_METHOD : RANSAC method (default: 38 (MAGSAC++)).
    --ransac-epipolar-threshold RANSAC_EPIPOLAR_THRESHOLD : RANSAC epipolar threshold (default: 2.0).
    --ransac-max-iter RANSAC_MAX_ITER : RANSAC maximum iterations (default: 5000).
    --ransac-confidence RANSAC_CONFIDENCE : RANSAC confidence (default: 0.999999).
    --mask-margin-ratio MASK_MARGIN_RATIO : Mask margin ratio (default: 0.15).

Examples:
    1. Stabilize the tracks (BBs) using the default stabilo parameters and a custom reference frame at index 100:
        python stabilize_boxes.py path/to/video.mp4 --save --ref-frame 100
    2. Visualize and save the stabilization process with custom visualization speed (20 ms):
        python stabilize_boxes.py path/to/video.mp4 --viz --save-viz --speed 20
    3. Stabilize the tracks without exclusion masks and save the visualization and stabilized tracks:
        python stabilize_boxes.py path/to/video.mp4 --no-mask --save-viz --save
    4. Stabilize the tracks using a custom config file and custom mask file. Save the stabilized tracks:
        python stabilize_boxes.py path/to/video.mp4 --save --custom-config path/to/config.yaml --mask-path path/to/mask.txt
    5. Stabilize the tracks using a custom config file and save the visualization with custom tail length and radius:
        python stabilize_boxes.py path/to/video.mp4 --viz --save-viz --custom-config path/to/config.yaml --tail-length 50 --tail-radius 15

Notes:
    - Press 'q' to quit the real-time visualization (--viz option).
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from utils import (
    close_streams,
    draw_boxes,
    draw_text,
    get_boxes_for_frame,
    get_boxes_from_tracks,
    initialize_progress_bar,
    initialize_read_streams,
    initialize_track_write_stream,
    load_exclusion_masks,
    load_tracks,
    separate_cli_arguments,
)

from stabilo import Stabilizer
from stabilo.utils import setup_logger

logger = setup_logger(__name__)

def stabilize_boxes(args, kwargs):
    """
    Stabilize a bounding boxes using the stabilo library.
    """
    reader, frame_count, w, h, fps = initialize_read_streams(args, logger)
    writer = initialize_track_write_stream(args, w, h, fps, logger)
    tracks = load_tracks(args, logger)
    boxes = get_boxes_from_tracks(tracks, args, logger)
    if args.mask_path:
        masks = load_exclusion_masks(args, logger)
    else:
        masks = boxes
        logger.info("Using the bounding boxes found in tracks as exclusion masks.")

    pbar = initialize_progress_bar(args, frame_count)

    stabilizer = Stabilizer(**kwargs)

    ref_frame_number = args.ref_frame
    boxes_stab = []
    prev_centers = []
    prev_centers_stab = []
    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_number)
        flag, ref_frame = reader.read()
        if not flag:
            logger.error(f"Failed to read the reference frame at index {ref_frame_number}")
            sys.exit(1)

        ref_mask = None if args.no_mask else get_boxes_for_frame(masks, ref_frame_number)
        stabilizer.set_ref_frame(ref_frame, ref_mask)

        frame_num = 0
        reader.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        while reader.isOpened():
            flag, frame = reader.read()
            if not flag:
                break

            boxes_frame = get_boxes_for_frame(boxes, frame_num)
            if frame_num == ref_frame_number:
                boxes_frame_stab = boxes_frame
            else:
                mask = None if args.no_mask else get_boxes_for_frame(masks, frame_num)
                stabilizer.stabilize(frame, mask)
                cur_trans_matrix = stabilizer.get_cur_trans_matrix()
                boxes_frame_stab = stabilizer.transform_boxes(boxes_frame, cur_trans_matrix)
            boxes_stab.append(boxes_frame_stab)

            if (args.viz or args.save_viz):
                img = visualize_box_movements(args, boxes_frame, boxes_frame_stab, prev_centers, prev_centers_stab, w, h, frame_num)
                if args.viz:
                    cv2.imshow('Stabilization Process Visualization', img)
                    if cv2.waitKey(args.speed) & 0xFF == ord('q'):
                        break
                if args.save_viz:
                    img = cv2.resize(img, (w, h))
                    writer.write(img)

            pbar.update(1)
            frame_num += 1

    except Exception as e:
        logger.error(f'Error processing frames: {e}')
    else:
        save_stabilized_boxes(args, tracks, boxes_stab)
    finally:
        close_streams(args, reader, pbar, writer_track=writer)

def visualize_box_movements(args, boxes, boxes_stab, prev_centers, prev_centers_stab, w, h, frame_num):
    """
    Display bounding box trajectories on a canvas.
    """
    new_h, new_w = int(h * args.canvas_x), int(w * args.canvas_x)
    img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    center_x, center_y = new_w // 2, new_h // 2
    top_left = (center_x - w // 2, center_y - h // 2)
    bottom_right = (center_x + w // 2, center_y + h // 2)
    cv2.rectangle(img, top_left, bottom_right, (211, 211, 211), 2)
    draw_text(img, 'Reference frame boundaries', pos=(top_left[0], top_left[1] - 70), scale=5, color_fg=3*(211, ))

    def adjust_boxes(boxes):
        return [(box[0] + top_left[0], box[1] + top_left[1], box[2], box[3]) for box in boxes]

    boxes_adjusted = adjust_boxes(boxes)
    boxes_stab_adjusted = adjust_boxes(boxes_stab)

    img = draw_boxes(img, boxes_adjusted, (0, 0, 255))
    img = draw_boxes(img, boxes_stab_adjusted, (0, 255, 0))

    def get_centers(boxes):
        return [(int(box[0]), int(box[1])) for box in boxes]

    prev_centers.append(get_centers(boxes_adjusted))
    prev_centers_stab.append(get_centers(boxes_stab_adjusted))

    draw_tails(img, prev_centers, (0, 0, 255), args.tail_length, args.tail_radius)
    draw_tails(img, prev_centers_stab, (0, 255, 0), args.tail_length, args.tail_radius)

    draw_text(img, f'Frame: {frame_num}', pos=(10, 10), scale=5, color_fg=(255, 255, 255))
    draw_text(img, "Press 'q' to quit.", pos=(10, img.shape[0] - 50), scale=3, color_fg=(0, 0, 255))

    return img

def draw_tails(img, points, color, max_frames=30, max_radius=13):
    """
    Draw tails on an image.
    """
    if len(points) > max_frames:
        points.pop(0)
    for i, center_points in enumerate(reversed(points)):
        radius = max(1, int(max_radius * (1 - i / max_frames)))
        for center in center_points:
            cv2.circle(img, center, radius, color, -1)

def save_stabilized_boxes(args, tracks, boxes_stab):
    """
    Save the stabilized bounding boxes to a file.
    """
    if args.save:
        if args.output:
            output_path = args.output
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = args.input.parent
        stabilized_tracks_filepath = output_path / f"{args.input.stem}_stab.txt"

        boxes_stab = np.concatenate(boxes_stab, axis=0)
        tracks_stab = np.copy(tracks)
        if boxes_stab.shape[0] < tracks_stab.shape[0]:
            boxes_stab = np.pad(boxes_stab, ((0, tracks_stab.shape[0] - boxes_stab.shape[0]), (0, 0)), mode='constant', constant_values=np.nan)
        tracks_stab[:, args.boxes_start_idx:args.boxes_start_idx + 4] = boxes_stab

        np.savetxt(stabilized_tracks_filepath, tracks_stab, fmt='%g', delimiter=',')
        logger.info(f'Saved the stabilized bounding boxes in YOLO format to {stabilized_tracks_filepath}.')

def get_cli_arguments():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Stabilize bounding boxes using the stabilo library.")

    # main options
    parser.add_argument("input", type=Path, help="input video filepath")
    parser.add_argument("--output", "-o", type=Path, help="output folder [default: same as input]")
    parser.add_argument("--save", "-s", action="store_true", help="save the stabilized tracks to a file")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="custom reference frame index")

    # tracks options
    parser.add_argument("--tracks", "-t", type=Path, help="filepath to the tracks file [default: input with .txt extension]")
    parser.add_argument("--boxes-frame-idx", "-bfi", type=int, default=0, help="frame number column index in the tracks file")
    parser.add_argument("--boxes-start-idx", "-bsi", type=int, default=2, help="start column index for bbox in the tracks file")
    parser.add_argument("--boxes-enc", "-be", type=str, default="yolo", choices=['yolo','pascal','coco'], help="bbox encoding")

    # mask options
    parser.add_argument("--no-mask", "-nm", action="store_true", help="disable exclusion masks during stabilization")
    parser.add_argument("--mask-path", "-mp", type=Path, help="custom mask file for stabilization [default: same as boxes]")
    parser.add_argument("--mask-frame-idx", "-mfi", type=int, default=0, help="frame number column index in mask file")
    parser.add_argument("--mask-start-idx", "-msi", type=int, default=2, help="start column index for bbox in mask file")
    parser.add_argument("--mask-enc", "-me", type=str, default="yolo", choices=['yolo','pascal','coco'], help="mask encoding")

    # visualization options
    parser.add_argument("--viz", "-v", action="store_true", help="show the stabilized and un-stabilized tracks")
    parser.add_argument("--save-viz", "-sv", action="store_true", help="save the visualization as a video at original FPS")
    parser.add_argument("--speed", "-sp", type=int, default=10, help="visualization speed in ms (0 for manual control)")
    parser.add_argument("--tail-length", "-tl", type=int, default=40, help="tail length for visualization")
    parser.add_argument("--tail-radius", "-tr", type=int, default=12, help="tail radius for visualization")
    parser.add_argument("--canvas-x", "-cx", type=float, default=1.5, help="canvas enlargement factor (>= 1)")

    # stabilo custom configuration file (override default stabilo parameters or the below CLI arguments)
    parser.add_argument("--custom-config", "-cc", type=Path, help="custom stabilo config file")

    # stabilo configuration options (override default stabilo parameters, see stabilo/cfg/default.yaml)
    parser.add_argument("--detector-name", "-dn", type=str, choices=['orb', 'sift', 'rsift', 'brisk', 'kaze', 'akaze'], help="detector type [default: orb]")
    parser.add_argument("--matcher-name", "-mn", type=str, choices=['bf', 'flann'], help="matcher type [default: bf]")
    parser.add_argument("--filter-type", "-ft", type=str, choices=['none', 'ratio', 'distance'], help="filter type for the match filter [default: ratio]")
    parser.add_argument("--transformation-type", "-tt", type=str, choices=['projective', 'affine'], help="transformation type [default: projective]")
    parser.add_argument("--clahe", "-c", action="store_true", help="apply CLAHE to grayscale images [default: False]")
    parser.add_argument("--downsample-ratio", "-dr", type=float, help="downsample ratio [default: 0.5]")
    parser.add_argument("--max-features", "-mf", type=int, help="max features to detect [default: 2000]")
    parser.add_argument("--ref-multiplier", "-rm", type=float, help="multiplier for max features in reference frame (ref_multiplier x max_features) [default: 2]")
    parser.add_argument("--filter-ratio", "-fr", type=float, help="filter ratio for the match filter [default: 0.9]")
    parser.add_argument("--ransac-method", "-r", type=int, help="RANSAC method [default: 38 (MAGSAC++)]")
    parser.add_argument("--ransac-epipolar-threshold", "-ret", type=float, help="RANSAC epipolar threshold [default: 2.0]")
    parser.add_argument("--ransac-max-iter", "-rmi", type=int, help="RANSAC maximum iterations [default: 5000]")
    parser.add_argument("--ransac-confidence", "-rc", type=float, help="RANSAC confidence [default: 0.999999]")
    parser.add_argument("--mask-margin-ratio", "-mmr", type=float, help="mask margin ratio [default: 0.15]")

    cli_args = parser.parse_args()

    if not (cli_args.save or cli_args.viz or cli_args.save_viz):
        parser.error("At least one of --save, --viz, or --save-viz must be specified.")

    return cli_args

if __name__ == "__main__":
    cli_args = get_cli_arguments()
    args, kwargs = separate_cli_arguments(cli_args)
    stabilize_boxes(args, kwargs)
