#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
tracks.py - Stabilize per-frame object annotations (tracks) using the stabilo library.

Description:
    This script stabilizes bounding boxes (BBs) in a video using the 'stabilo' library. It reads a video file and a
    corresponding tracks file containing BBs for each frame of the video. The script then stabilizes these BBs and
    saves the stabilized BBs to a file. It also provides options to visualize the stabilized and un-stabilized BBs
    in real-time or save the visualization as a video. The stabilization can be performed with respect to a custom
    reference frame, and exclusion masks can be used to exclude certain areas from stabilization. By default, the
    script considers the 0-th frame as the reference frame and uses the BBs as exclusion masks.

Usage:
    stabilo tracks <input> [options]

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
    --boxes-end-idx BOXES_END_IDX : Exclusive end column index for box columns (default: auto from format). Required for 'polygon' when it does not run to the last column.
    --boxes-enc BOXES_ENC : Bounding box encoding. Choices: 'yolo', 'pascal', 'coco', 'xywha', 'four' (default: yolo).

Mask Options:
    --no-mask          : Disable exclusion masks during stabilization.
    --mask-path MASK_PATH : Custom mask file for stabilization (default: same as boxes).
    --mask-frame-idx MASK_FRAME_IDX : Frame number column index in the mask file (default: 0).
    --mask-start-idx MASK_START_IDX : Start column index of the bounding box parameters in the mask file (default: 2).
    --mask-end-idx MASK_END_IDX : Exclusive end column index for mask columns (default: auto from format). Required for 'polygon' when it does not run to the last column.
    --mask-enc MASK_ENC : Mask format. Choices: 'yolo', 'pascal', 'coco', 'xywha', 'four', 'polygon', 'circle' (default: yolo).

Visualization Options:
    --viz              : Show the stabilized and un-stabilized tracks.
    --save-viz         : Save the visualization as a video at original FPS.
    --speed SPEED      : Visualization speed in milliseconds (0 for manual control, default: 10).
    --tail-length TAIL_LENGTH : Tail length for visualization (default: 40).
    --tail-radius TAIL_RADIUS : Tail radius for visualization (default: 12).
    --canvas-x CANVAS_X : Canvas enlargement factor (>= 1, default: 1.5).

Run 'stabilo tracks --help' for the full, grouped list of options. Stabilizer options not exposed as
CLI flags (detector-specific weights, LoFTR confidence, warning thresholds) can be set through a
config file passed to --custom-config; see stabilo/cfg/default.yaml for every key and its default.
Resolution order is: built-in defaults < --custom-config file < explicit CLI flags.

Examples:
    1. Stabilize the tracks (BBs) using the default stabilo parameters and a custom reference frame at index 100:
        stabilo tracks path/to/video.mp4 --save --ref-frame 100
    2. Visualize and save the stabilization process with custom visualization speed (20 ms):
        stabilo tracks path/to/video.mp4 --viz --save-viz --speed 20
    3. Stabilize the tracks without exclusion masks and save the visualization and stabilized tracks:
        stabilo tracks path/to/video.mp4 --no-mask --save-viz --save
    4. Stabilize the tracks using a custom config file and custom mask file. Save the stabilized tracks:
        stabilo tracks path/to/video.mp4 --save --custom-config path/to/config.yaml --mask-path path/to/mask.txt
    5. Stabilize the tracks using a custom config file and save the visualization with custom tail length and radius:
        stabilo tracks path/to/video.mp4 --viz --save-viz --custom-config path/to/config.yaml --tail-length 50 --tail-radius 15
    6. Stabilize the tracks using CUDA GPU acceleration (requires a CUDA-enabled OpenCV build, see docs/cuda.md):
        stabilo tracks path/to/video.mp4 --gpu --save

Notes:
    - Press 'q' to quit the real-time visualization (--viz option).
    - The learning-based detectors need considerably more memory than the classical ones. Lower
      --downsample-ratio for large frames; 'loftr' scales quadratically with the pixel count.
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from stabilo import Stabilizer

from .utils import (
    ENCODING_TO_BOX_FORMAT,
    StabiloHelpFormatter,
    add_stabilo_config_arguments,
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

EXAMPLES = """\
examples:
  stabilo tracks video.mp4 --save                         stabilize the tracks and save them
  stabilo tracks video.mp4 --save --ref-frame 100         stabilize against frame 100
  stabilo tracks video.mp4 --viz --save-viz --speed 20    preview and record the visualization
  stabilo tracks video.mp4 --no-mask --save               ignore the exclusion masks
  stabilo tracks video.mp4 -cc config.yaml --save         take defaults from a config file
  stabilo tracks video.mp4 -dn xfeat -dr 0.25 --save      learning-based detector on smaller frames
  stabilo tracks video.mp4 --gpu --save                   OpenCV CUDA (see docs/cuda.md)
"""


def stabilize_tracks(args, kwargs, logger):
    """
    Stabilize per-frame object annotations (tracks) using the stabilo library.
    """
    reader, frame_count, w, h, fps = initialize_read_streams(args, logger)
    writer = initialize_track_write_stream(args, w, h, fps, logger)
    tracks = load_tracks(args, logger)
    boxes = get_boxes_from_tracks(tracks, args, logger)
    boxes_box_format = ENCODING_TO_BOX_FORMAT[args.boxes_enc]

    if args.mask_path:
        masks = load_exclusion_masks(args, logger)
        mask_box_format = ENCODING_TO_BOX_FORMAT[args.mask_enc]
    else:
        masks = boxes
        mask_box_format = boxes_box_format
        logger.info("Using the bounding boxes found in tracks as exclusion masks.")

    stabilizer = Stabilizer(**kwargs)

    pbar = initialize_progress_bar(args, frame_count)

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
        stabilizer.set_ref_frame(ref_frame, ref_mask, box_format=mask_box_format)

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
                stabilizer.stabilize(frame, mask, box_format=mask_box_format)
                cur_trans_matrix = stabilizer.get_cur_trans_matrix()
                boxes_frame_stab = stabilizer.transform_boxes(
                    boxes_frame, cur_trans_matrix, in_box_format=boxes_box_format, out_box_format=boxes_box_format
                )
            boxes_stab.append(boxes_frame_stab)

            if args.viz or args.save_viz:
                img = visualize_box_movements(
                    args,
                    boxes_frame,
                    boxes_frame_stab,
                    prev_centers,
                    prev_centers_stab,
                    w,
                    h,
                    frame_num,
                    boxes_box_format,
                )
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
        save_stabilized_boxes(args, tracks, boxes_stab, logger)
    finally:
        close_streams(args, reader, pbar, writer_track=writer)


def visualize_box_movements(
    args, boxes, boxes_stab, prev_centers, prev_centers_stab, w, h, frame_num, boxes_box_format='xywh'
):
    """
    Display bounding box trajectories on a canvas.
    """
    new_h, new_w = int(h * args.canvas_x), int(w * args.canvas_x)
    img = np.zeros((new_h, new_w, 3), dtype=np.uint8)

    center_x, center_y = new_w // 2, new_h // 2
    top_left = (center_x - w // 2, center_y - h // 2)
    bottom_right = (center_x + w // 2, center_y + h // 2)
    cv2.rectangle(img, top_left, bottom_right, (211, 211, 211), 2)
    draw_text(img, 'Reference frame boundaries', pos=(top_left[0], top_left[1] - 70), scale=5, color_fg=3 * (211,))

    dx, dy = top_left

    def adjust_boxes(bxs):
        if bxs is None or len(bxs) == 0:
            return bxs
        result = np.array(bxs, dtype=float)
        if boxes_box_format in ('xywh', 'xywha'):
            result[:, 0] += dx
            result[:, 1] += dy
        elif boxes_box_format == 'four':
            result[:, ::2] += dx
            result[:, 1::2] += dy
        return result

    boxes_adjusted = adjust_boxes(boxes)
    boxes_stab_adjusted = adjust_boxes(boxes_stab)

    img = draw_boxes(img, boxes_adjusted, (0, 0, 255), box_format=boxes_box_format)
    img = draw_boxes(img, boxes_stab_adjusted, (0, 255, 0), box_format=boxes_box_format)

    def get_centers(bxs):
        if bxs is None or len(bxs) == 0:
            return []
        if boxes_box_format in ('xywh', 'xywha'):
            return [(int(b[0]), int(b[1])) for b in bxs]
        elif boxes_box_format == 'four':
            return [(int(np.mean(b[::2])), int(np.mean(b[1::2]))) for b in bxs]
        return []

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


def save_stabilized_boxes(args, tracks, boxes_stab, logger):
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
            boxes_stab = np.pad(
                boxes_stab,
                ((0, tracks_stab.shape[0] - boxes_stab.shape[0]), (0, 0)),
                mode='constant',
                constant_values=np.nan,
            )
        num_box_cols = boxes_stab.shape[1]
        tracks_stab[:, args.boxes_start_idx : args.boxes_start_idx + num_box_cols] = boxes_stab

        np.savetxt(stabilized_tracks_filepath, tracks_stab, fmt='%g', delimiter=',')
        logger.info(f'Saved the stabilized bounding boxes in YOLO format to {stabilized_tracks_filepath}.')


def configure_parser(subparsers):
    """
    Register the 'tracks' subcommand and its arguments.
    """
    parser = subparsers.add_parser(
        "tracks",
        help="stabilize per-frame object annotations (tracks) with respect to a reference frame",
        description="Stabilize per-frame object annotations (tracks) using the stabilo library.",
        epilog=EXAMPLES,
        formatter_class=StabiloHelpFormatter,
    )

    group = parser.add_argument_group("input and output")
    group.add_argument("input", type=Path, help="input video filepath")
    group.add_argument("--output", "-o", type=Path, help="output folder [default: same as input]")
    group.add_argument("--save", "-s", action="store_true", help="save the stabilized tracks to a file")
    group.add_argument("--ref-frame", "-rf", type=int, default=0, help="custom reference frame index [default: 0]")

    group = parser.add_argument_group("tracks to stabilize")
    group.add_argument(
        "--tracks", "-t", type=Path, help="filepath to the tracks file [default: input with .txt extension]"
    )
    group.add_argument(
        "--boxes-frame-idx", "-bfi", type=int, default=0, help="frame number column index in the tracks file"
    )
    group.add_argument(
        "--boxes-start-idx", "-bsi", type=int, default=2, help="start column index for bbox in the tracks file"
    )
    group.add_argument(
        "--boxes-end-idx",
        "-bei",
        type=int,
        default=None,
        help="exclusive end column index for box columns [default: auto from format, required for 'polygon' that does not span to the last column]",
    )
    group.add_argument(
        "--boxes-enc",
        "-be",
        type=str,
        default="yolo",
        choices=['yolo', 'pascal', 'coco', 'xywha', 'four'],
        help="bbox encoding [default: yolo]",
    )

    group = parser.add_argument_group(
        "exclusion masks", "Regions (e.g. moving vehicles) excluded from feature detection."
    )
    group.add_argument("--no-mask", "-nm", action="store_true", help="disable exclusion masks during stabilization")
    group.add_argument(
        "--mask-path", "-mp", type=Path, help="custom mask file for stabilization [default: same as boxes]"
    )
    group.add_argument("--mask-frame-idx", "-mfi", type=int, default=0, help="frame number column index in mask file")
    group.add_argument("--mask-start-idx", "-msi", type=int, default=2, help="start column index for bbox in mask file")
    group.add_argument(
        "--mask-end-idx",
        "-mei",
        type=int,
        default=None,
        help="exclusive end column index for mask columns [default: auto from format, required for 'polygon' that does not span to the last column]",
    )
    group.add_argument(
        "--mask-enc",
        "-me",
        type=str,
        default="yolo",
        choices=['yolo', 'pascal', 'coco', 'xywha', 'four', 'polygon', 'circle'],
        help="mask format [default: yolo]",
    )

    group = parser.add_argument_group("visualization")
    group.add_argument("--viz", "-v", action="store_true", help="show the stabilized and un-stabilized tracks")
    group.add_argument(
        "--save-viz", "-sv", action="store_true", help="save the visualization as a video at original FPS"
    )
    group.add_argument("--speed", "-sp", type=int, default=10, help="visualization speed in ms (0 for manual control)")
    group.add_argument("--tail-length", "-tl", type=int, default=40, help="tail length for visualization")
    group.add_argument("--tail-radius", "-tr", type=int, default=12, help="tail radius for visualization")
    group.add_argument("--canvas-x", "-cx", type=float, default=1.5, help="canvas enlargement factor (>= 1)")

    add_stabilo_config_arguments(parser)

    parser.set_defaults(func=run)
    return parser


def run(args, logger):
    """
    Execute the 'tracks' subcommand.
    """
    _, kwargs = separate_cli_arguments(args)
    kwargs['logger'] = logger
    stabilize_tracks(args, kwargs, logger)
