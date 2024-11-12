#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stabilize_video.py - Stabilize a video using the stabilo library.

Description:
    This script stabilizes videos using the 'stabilo' library. It reads a video file, stabilizes it using a reference
    frame stabilization method, and optionally saves the stabilized video or the visualization of the stabilization
    process. The stabilization is based on feature point matching between frames, followed by transformation estimation
    using RANSAC. The script supports various feature detectors, matchers, and extensive customization through
    command-line options or a configuration file. It also supports CLAHE application, video downsampling, and exclusion
    masks (bounding boxes) for stabilization.

Usage:
    python stabilize_video.py <input> [options]

Arguments:
    input              : Filepath to the input video file.

Main Options:
    --output OUTPUT    : Output folder to save the stabilized video or visualization (default: same as input).
    --save             : Save the stabilized video (default: False).
    --ref-frame REF_FRAME : Custom reference frame index for stabilization (default: 0).
    --debug            : Enable debug mode (default: False).

Mask Options:
    --no-mask          : Do not use exclusion masks during stabilization.
    --mask-path MASK_PATH : Custom mask filepath (default: input with .txt extension).
    --mask-frame-idx MASK_FRAME_IDX : Frame number column index in the mask file (default: 0).
    --mask-start-idx MASK_START_IDX : Start column index of the 4 bounding box parameters used as masks (default: 2).
    --mask-enc MASK_ENC : Bounding box encoding. Choices: 'yolo', 'pascal', 'coco' (default: yolo).

Visualization Options:
    --viz              : Visualize the transformation process (default: False).
    --save-viz         : Save the visualization of the transformation process as a video (default: False).
    --no-lines         : Hide lines between matched feature points (default: False).
    --no-boxes         : Hide bounding boxes on the (un-)stabilized videos (default: False).
    --speed SPEED      : Visualization speed in milliseconds (0 for manual control) (default: 10).

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
    1. Stabilize a video using default settings and save the stabilized video:
       python stabilize_video.py path/to/video/video.mp4 --save

    2. Visualize the stabilization process:
       python stabilize_video.py path/to/video/video.mp4 --viz

    3. Save a stabilized video using a custom detector and matcher:
       python stabilize_video.py path/to/video/video.mp4 --detector-name sift --matcher-name flann --save

    4. Apply stabilization without a mask and visualize the process:
       python stabilize_video.py path/to/video/video.mp4 --no-mask --viz

    5. Stabilize a video using a custom reference frame and save the stabilized video and visualization:
       python stabilize_video.py path/to/video/video.mp4 --ref-frame 15 --save --save-viz

    6. Use a custom mask filepath and specify start column index of the bounding boxes:
        python stabilize_video.py path/to/video/video.mp4 --mask-path path/to/mask/mask.txt --mask-start 1 --viz

    7. Apply stabilization with a custom configuration file:
        python stabilize_video.py path/to/video/video.mp4 --custom-config path/to/config/config.yaml --save

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
    initialize_progress_bar,
    initialize_read_streams,
    initialize_write_streams,
    load_exclusion_masks,
    separate_cli_arguments,
)

from stabilo import Stabilizer
from stabilo.utils import setup_logger

COLOURS = np.random.randint(0, 256, (100, 3))

logger = setup_logger(__name__)

def stabilize_video(args, kwargs):
    """
    Stabilize a video using the stabilo library.
    """
    reader, frame_count, w, h, fps = initialize_read_streams(args, logger)
    writer_vid, writer_viz = initialize_write_streams(args, w, h, fps, logger)
    masks = load_exclusion_masks(args, logger)
    pbar = initialize_progress_bar(args, frame_count)

    stabilizer = Stabilizer(**kwargs)

    frame_num = 0
    ref_frame_number = args.ref_frame

    try:
        reader.set(cv2.CAP_PROP_POS_FRAMES, ref_frame_number)
        flag, ref_frame = reader.read()
        if not flag:
            logger.error(f"Failed to read the reference frame at index {ref_frame_number}")
            sys.exit(1)

        ref_mask = None if args.no_mask else get_boxes_for_frame(masks, ref_frame_number)
        stabilizer.set_ref_frame(ref_frame, ref_mask)

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
                stabilizer.stabilize(frame, mask)
                frame_stab = stabilizer.warp_cur_frame()
                boxes_stab = stabilizer.transform_cur_boxes()

            if writer_vid is not None and frame_stab is not None:
                writer_vid.write(frame_stab)

            if (args.viz or args.save_viz) and frame_stab is not None:
                imgs = render_stabilization_visuals(stabilizer, frame, frame_stab, mask, boxes_stab, frame_num, args)
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

def render_stabilization_visuals(stabilizer, frame, frame_stab, boxes, boxes_stab, frame_num, args):
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
        best_match_count = 'N/A'
        overlay = img.copy()
        lines = {'inliers': [], 'outliers': []}
        if ref_pts is not None and cur_pts is not None:
            for i, (pt1, pt2) in enumerate(zip(ref_pts, cur_pts)):
                if stabilizer.cur_inliers[i]:
                    lines['inliers'].append((pt1, pt2, [0, 255, 0]))
                else:
                    lines['outliers'].append((pt1, pt2, [0, 0, 255]))
            best_match_count = i + 1 if 'i' in locals() else 'N/A'

            for line in lines['outliers']:
                x1, y1 = line[0].ravel()
                x2, y2 = line[1].ravel()
                cv2.line(overlay, (int(x1), int(y1)), (int(x2 + ref_frame.shape[1]), int(y2)), line[2], 2, cv2.LINE_AA)

            for line in lines['inliers']:
                x1, y1 = line[0].ravel()
                x2, y2 = line[1].ravel()
                cv2.line(overlay, (int(x1), int(y1)), (int(x2 + ref_frame.shape[1]), int(y2)), line[2], 2, cv2.LINE_AA)

            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

        return img, best_match_count

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
        imgs_upper, best_match_count = draw_lines(imgs_upper, stabilizer.ref_pts, stabilizer.cur_pts)
        inliers_count = stabilizer.cur_inliers_count if stabilizer.cur_inliers_count is not None else 'N/A'
        pos_inliers = (250+ref_frame.shape[1] // 2, 15)
        pos_outliers = (ref_frame.shape[1] // 2 + 1050, 15)
        draw_text(imgs_upper, f"Inliers: {inliers_count}", pos=pos_inliers, color_fg=3*(255,))
        if best_match_count != 'N/A':
            outlier_count = int(best_match_count - inliers_count)
            draw_text(imgs_upper, f"Outliers: {outlier_count}", pos=pos_outliers, color_fg=3*(255,))
        else:
            draw_text(imgs_upper, "Outliers: N/A", pos=pos_outliers, color_fg=(255, 255, 255))
    draw_text(imgs_upper, f"Ref. frame = {args.ref_frame}", scale=8, color_fg=(255, 255, 255), pos=(0, 10))
    draw_text(imgs_upper, f"Frame {frame_num}", scale=8, color_fg=(255, 255, 255), pos=(ref_frame.shape[1], 10))


    if not args.no_boxes:
        frame = draw_boxes(frame, boxes, color=(0, 0, 255))
        frame_stab = draw_boxes(frame_stab, boxes_stab, color=(0, 255,0))
    draw_text(frame, f'Source video frame {frame_num}', scale=6, color_fg=(255, 255, 255))
    draw_text(frame_stab, f'Stabilized video frame {frame_num}', scale=6, color_fg=(255, 255, 255))

    imgs_lower = np.hstack((frame_stab, frame if stabilizer.cur_frame is not None else np.zeros_like(frame_stab)))
    pos_quitting = (imgs_lower.shape[1] - 600, imgs_lower.shape[0] - 60)
    draw_text(imgs_lower, "Press 'q' to quit.", pos=pos_quitting, scale=4, color_fg=(0, 0, 255))

    return np.vstack((imgs_upper, imgs_lower))

def get_cli_arguments():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Stabilize a video using the stabilo library.")

    # main options
    parser.add_argument("input", type=Path, help="input video filepath")
    parser.add_argument("--output", "-o", type=Path, help="output folder [default: same as input]")
    parser.add_argument("--save", "-s", action="store_true", help="save the stabilized video")
    parser.add_argument("--ref-frame", "-rf", type=int, default=0, help="custom reference frame index")
    parser.add_argument("--debug", "-d", action="store_true", help="enable debug mode")

    # mask options
    parser.add_argument("--no-mask", "-nm", action="store_true", help="disable exclusion masks during stabilization")
    parser.add_argument("--mask-path", "-mp", type=Path, help="custom mask file [default: input with .txt extension]")
    parser.add_argument("--mask-frame-idx", "-mfi", type=int, default=0, help="frame number column index in mask file")
    parser.add_argument("--mask-start-idx", "-msi", type=int, default=2, help="start column index for bbox in mask file")
    parser.add_argument("--mask-enc", "-me", type=str, default="yolo", choices=['yolo','pascal','coco'], help="mask encoding")

    # visualization options
    parser.add_argument("--viz", "-v", action="store_true", help="visualize the transformation process")
    parser.add_argument("--save-viz", "-sv", action="store_true", help="save the visualization as a video at original FPS")
    parser.add_argument("--no-lines", "-nl", action="store_true", help="hide lines between matched feature points")
    parser.add_argument("--no-boxes", "-nb", action="store_true", help="hide bounding boxes on the (un-)stabilized frames")
    parser.add_argument("--speed", "-sp", type=int, default=10, help="visualization speed in ms (0 for manual control)")

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
    stabilize_video(args, kwargs)
