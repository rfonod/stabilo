#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
video.py - Stabilize a video using the stabilo library.

Description:
    This script stabilizes videos using the 'stabilo' library. It reads a video file, stabilizes it using a reference
    frame stabilization method, and optionally saves the stabilized video or the visualization of the stabilization
    process. The stabilization is based on feature point matching between frames, followed by transformation estimation
    using RANSAC. The script supports various feature detectors, matchers, and extensive customization through
    command-line options or a configuration file. It also supports CLAHE application, video downsampling, and exclusion
    masks (bounding boxes) for stabilization.

Usage:
    stabilo video <input> [options]

Run 'stabilo video --help' for the full, grouped list of options. Stabilizer options not exposed as
CLI flags (detector-specific weights, LoFTR confidence, warning thresholds) can be set through a
config file passed to --custom-config; see stabilo/cfg/default.yaml for every key and its default.
Resolution order is: built-in defaults < --custom-config file < explicit CLI flags.

Examples:
    1. Stabilize a video using default settings and save the stabilized video:
       stabilo video path/to/video/video.mp4 --save

    2. Visualize the stabilization process:
       stabilo video path/to/video/video.mp4 --viz

    3. Save a stabilized video using a custom detector and matcher:
       stabilo video path/to/video/video.mp4 --detector-name sift --matcher-name flann --save

    4. Apply stabilization without a mask and visualize the process:
       stabilo video path/to/video/video.mp4 --no-mask --viz

    5. Stabilize a video using a custom reference frame and save the stabilized video and visualization:
       stabilo video path/to/video/video.mp4 --ref-frame 15 --save --save-viz

    6. Use a custom mask filepath and specify start column index of the bounding boxes:
        stabilo video path/to/video/video.mp4 --mask-path path/to/mask/mask.txt --mask-start 1 --viz

    7. Apply stabilization with a custom configuration file:
        stabilo video path/to/video/video.mp4 --custom-config path/to/config/config.yaml --save

    8. Stabilize a video using a learning-based detector on downsampled frames:
        stabilo video path/to/video/video.mp4 --detector-name xfeat --downsample-ratio 0.25 --save

    9. Stabilize a video using CUDA GPU acceleration (requires a CUDA-enabled OpenCV build, see docs/cuda.md):
        stabilo video path/to/video/video.mp4 --gpu --save

Notes:
    - Press 'q' to quit the real-time visualization (--viz option).
    - The learning-based detectors need considerably more memory than the classical ones. Lower
      --downsample-ratio for large frames; 'loftr' scales quadratically with the pixel count.
"""

import argparse
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
    initialize_progress_bar,
    initialize_read_streams,
    initialize_write_streams,
    load_exclusion_masks,
    separate_cli_arguments,
)

COLOURS = np.random.randint(0, 256, (100, 3))

# Mask formats that can be geometrically transformed (polygon/circle are masking-only)
TRANSFORMABLE_BOX_FORMATS = ('xywh', 'xywha', 'four')

EXAMPLES = """\
examples:
  stabilo video video.mp4 --save                          stabilize and save the result
  stabilo video video.mp4 --viz                           preview the stabilization live
  stabilo video video.mp4 -dn sift -mn flann --save       pick a classical detector and matcher
  stabilo video video.mp4 --no-mask --viz                 ignore the exclusion masks
  stabilo video video.mp4 --ref-frame 15 --save-viz       stabilize against frame 15
  stabilo video video.mp4 -cc config.yaml --save          take defaults from a config file
  stabilo video video.mp4 -dn xfeat -dr 0.25 --save       learning-based detector on smaller frames
  stabilo video video.mp4 --gpu --save                    OpenCV CUDA (see docs/cuda.md)
"""


def stabilize_video(args, kwargs, logger):
    """
    Stabilize a video using the stabilo library.
    """
    reader, frame_count, w, h, fps = initialize_read_streams(args, logger)
    writer_vid, writer_viz = initialize_write_streams(args, w, h, fps, logger)
    masks = load_exclusion_masks(args, logger)

    stabilizer = Stabilizer(**kwargs)

    pbar = initialize_progress_bar(args, frame_count)

    frame_num = 0
    ref_frame_number = args.ref_frame
    mask_box_format = ENCODING_TO_BOX_FORMAT[args.mask_enc]

    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_number)
        flag, ref_frame = reader.read()
        if not flag:
            logger.error(f"Failed to read the reference frame at index {ref_frame_number}")
            sys.exit(1)

        ref_mask = None if args.no_mask else get_boxes_for_frame(masks, ref_frame_number)
        stabilizer.set_ref_frame(ref_frame, ref_mask, box_format=mask_box_format)

        reader.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while reader.isOpened():
            flag, frame = reader.read()
            if not flag:
                break

            mask = None if args.no_mask else get_boxes_for_frame(masks, frame_num)
            if frame_num == ref_frame_number:
                frame_stab = frame
                boxes_stab = mask
            else:
                stabilizer.stabilize(frame, mask, box_format=mask_box_format)
                frame_stab = stabilizer.warp_cur_frame()
                boxes_stab = (
                    stabilizer.transform_cur_boxes(out_box_format=mask_box_format)
                    if mask_box_format in TRANSFORMABLE_BOX_FORMATS
                    else None
                )

            if writer_vid is not None and frame_stab is not None:
                writer_vid.write(frame_stab)

            if (args.viz or args.save_viz) and frame_stab is not None:
                imgs = render_stabilization_visuals(
                    stabilizer, frame, frame_stab, mask, boxes_stab, frame_num, args, mask_box_format
                )
                if args.viz:
                    cv2.imshow('Stabilization Process Visualization', imgs)
                    if cv2.waitKey(args.speed) & 0xFF == ord('q'):
                        break
                if writer_viz is not None:
                    imgs = cv2.resize(imgs, (w, h))
                    writer_viz.write(imgs)

            pbar.update(1)
            frame_num += 1

    except KeyboardInterrupt:
        logger.warning('Interrupted by user.')
    except Exception as e:
        logger.error(f'Error processing frames: {e}')
    finally:
        close_streams(args, reader, pbar, writer_vid, writer_viz)


def render_stabilization_visuals(
    stabilizer, frame, frame_stab, boxes, boxes_stab, frame_num, args, mask_box_format='xywh'
):
    """
    Illustrate the stabilization process with feature points, lines, and bounding boxes.
    """

    def draw_mask(img, mask):
        if mask is not None:
            img = cv2.bitwise_and(img, img, mask=mask)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    def draw_points(img, points):
        if points is not None:
            for i, pt in enumerate(points):
                x, y = pt.ravel()
                cv2.circle(img, (int(x), int(y)), 9, COLOURS[i % 100].tolist(), 6)
        return img

    def draw_lines(img, ref_pts, cur_pts, alpha=0.4):
        overlay = img.copy()
        lines = {'inliers': [], 'outliers': []}
        inliers = stabilizer.cur_inliers
        if ref_pts is not None and cur_pts is not None and inliers is not None:
            for i, (pt1, pt2) in enumerate(zip(ref_pts, cur_pts, strict=False)):
                if i >= len(inliers):
                    break
                if inliers[i]:
                    lines['inliers'].append((pt1, pt2, [0, 255, 0]))
                else:
                    lines['outliers'].append((pt1, pt2, [0, 0, 255]))

            for line in lines['outliers']:
                x1, y1 = line[0].ravel()
                x2, y2 = line[1].ravel()
                cv2.line(overlay, (int(x1), int(y1)), (int(x2 + ref_frame.shape[1]), int(y2)), line[2], 2, cv2.LINE_AA)

            for line in lines['inliers']:
                x1, y1 = line[0].ravel()
                x2, y2 = line[1].ravel()
                cv2.line(overlay, (int(x1), int(y1)), (int(x2 + ref_frame.shape[1]), int(y2)), line[2], 2, cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        match_count = len(lines['inliers']) + len(lines['outliers'])
        return img, (match_count or None)

    ref_frame = stabilizer.ref_frame_gray
    ref_frame = draw_mask(ref_frame, stabilizer.ref_mask)

    if stabilizer.cur_frame_gray is not None:
        cur_frame = stabilizer.cur_frame_gray
    else:
        cur_frame = np.full(stabilizer.ref_frame_gray.shape, 0, dtype=np.uint8)
    cur_frame = draw_mask(cur_frame, stabilizer.cur_mask)
    if args.debug:
        ref_frame = draw_points(ref_frame, stabilizer.ref_pts)
        cur_frame = draw_points(cur_frame, stabilizer.cur_pts)

    imgs_upper = np.hstack((ref_frame, cur_frame))
    if not args.no_lines:
        imgs_upper, match_count = draw_lines(imgs_upper, stabilizer.ref_pts, stabilizer.cur_pts)
        inliers_count = stabilizer.get_cur_inliers_count()
        outliers_count = None if match_count is None or inliers_count is None else match_count - inliers_count
        pos_inliers = (250 + ref_frame.shape[1] // 2, 15)
        pos_outliers = (ref_frame.shape[1] // 2 + 1050, 15)
        draw_text(
            imgs_upper,
            f"Inliers: {inliers_count if inliers_count is not None else 'N/A'}",
            pos=pos_inliers,
            color_fg=3 * (255,),
        )
        draw_text(
            imgs_upper,
            f"Outliers: {outliers_count if outliers_count is not None else 'N/A'}",
            pos=pos_outliers,
            color_fg=3 * (255,),
        )
    draw_text(imgs_upper, f"Ref. frame = {args.ref_frame}", scale=8, color_fg=(255, 255, 255), pos=(0, 10))
    draw_text(imgs_upper, f"Frame {frame_num}", scale=8, color_fg=(255, 255, 255), pos=(ref_frame.shape[1], 10))

    if not args.no_boxes:
        frame = draw_boxes(frame, boxes, color=(0, 0, 255), box_format=mask_box_format)
        frame_stab = draw_boxes(frame_stab, boxes_stab, color=(0, 255, 0), box_format=mask_box_format)
    draw_text(frame, f'Source video frame {frame_num}', scale=6, color_fg=(255, 255, 255))
    draw_text(frame_stab, f'Stabilized video frame {frame_num}', scale=6, color_fg=(255, 255, 255))

    imgs_lower = np.hstack((frame_stab, frame if stabilizer.cur_frame is not None else np.zeros_like(frame_stab)))
    pos_quitting = (imgs_lower.shape[1] - 600, imgs_lower.shape[0] - 60)
    draw_text(imgs_lower, "Press 'q' to quit.", pos=pos_quitting, scale=4, color_fg=(0, 0, 255))

    return np.vstack((imgs_upper, imgs_lower))


def configure_parser(subparsers):
    """
    Register the 'video' subcommand and its arguments.
    """
    parser = subparsers.add_parser(
        "video",
        help="stabilize a video with respect to a reference frame",
        description="Stabilize a video with respect to a reference frame using the stabilo library.",
        epilog=EXAMPLES,
        formatter_class=StabiloHelpFormatter,
    )

    group = parser.add_argument_group("input and output")
    group.add_argument("input", type=Path, help="input video filepath")
    group.add_argument("--output", "-o", type=Path, help="output folder [default: same as input]")
    group.add_argument("--save", "-s", action="store_true", help="save the stabilized video")
    group.add_argument("--ref-frame", "-rf", type=int, default=0, help="custom reference frame index [default: 0]")
    group.add_argument("--debug", "-d", action="store_true", help="enable debug mode")

    group = parser.add_argument_group(
        "exclusion masks", "Regions (e.g. moving vehicles) excluded from feature detection."
    )
    group.add_argument("--no-mask", "-nm", action="store_true", help="disable exclusion masks during stabilization")
    group.add_argument("--mask-path", "-mp", type=Path, help="custom mask file [default: input with .txt extension]")
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
    group.add_argument(
        "--viz", "-v", action="store_true", help="visualize the transformation process (press 'q' to quit)"
    )
    group.add_argument(
        "--save-viz", "-sv", action="store_true", help="save the visualization as a video at original FPS"
    )
    group.add_argument("--no-lines", "-nl", action="store_true", help="hide lines between matched feature points")
    group.add_argument(
        "--no-boxes", "-nb", action="store_true", help="hide bounding boxes on the (un-)stabilized frames"
    )
    group.add_argument("--speed", "-sp", type=int, default=10, help="visualization speed in ms (0 for manual control)")

    add_stabilo_config_arguments(parser)

    parser.set_defaults(func=run)
    return parser


def run(args, logger):
    """
    Execute the 'video' subcommand.
    """
    _, kwargs = separate_cli_arguments(args)
    kwargs['logger'] = logger
    stabilize_video(args, kwargs, logger)
