#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stabilize_video.py - Video Stabilization Script

Description:
    This script enhances video stability using the 'stabilo' library, applying a reference frame stabilization method 
    that's particularly effective for videos requiring accurate trajectory analysis, such as drone footage. It supports 
    various feature detectors and matchers, allows visualization during processing, and offers extensive customization 
    through command-line options or a configuration file.

Usage:
    python stabilize_video.py <input> [options]

Arguments:
    input               : Path to the input video file.
    
Script Options:
    --output OUTPUT     : Path to the output stabilized video file (default: input_stab_detector_matcher_transformation-type.suffix).
    --viz               : Visualize the transformation process.
    --save              : Save the stabilized video.
    --save-viz          : Save the visualization of the transformation process.    
    --no-mask           : Do not use a mask for stabilization.
    --mask-path MASK_PATH : Path to a mask file (default: input with .txt extension).
    --mask-start MASK_START : Start column index of YOLO format masks (default: 2). Column 0 reserved for frame number.
    --mask-encoding MASK_ENCODING : Bounding box format (default: 'yolo'). Choices: 'yolo', 'pascal', 'coco'.
    --no-lines          : Do not draw lines between points.
    --no-boxes          : Do not draw bounding boxes.
    --custom-config     : Path to a custom stabilo configuration file that overrides the CLI arguments.

Stabilo Configuration:
    --detector-name DETECTOR_NAME : Type of detector (default: orb). Choices: 'orb', 'sift', 'rsift', 'brisk', 'kaze', 'akaze'.
    --matcher-name MATCHER_NAME   : Type of matcher (default: bf). Choices: 'bf', 'flann'.
    --filter-type FILTER_TYPE     : Type of match filter (default: ratio). Choices: 'none', 'ratio', 'distance'.
    --transformation-type TRANSFORMATION_TYPE : Type of transformation (default: projective). Choices: 'projective', 'affine'.
    --clahe             : Apply CLAHE to grayscale images.
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
    1. Visualize the stabilization process:
       python stabilize_video.py path/to/video/video.mp4 --viz

    2. Save a stabilized video using a custom detector and matcher:
       python stabilize_video.py path/to/video/video.mp4 --detector-name sift --matcher-name flann --save

    3. Apply stabilization without a mask and visualize the process:
       python stabilize_video.py path/to/video/video.mp4 --no-mask --viz

    4. Customize output path and enable visualization save:
       python stabilize_video.py path/to/video/video.mp4 --output path/to/output-video/output.mp4 --save-viz

    5. Use a custom mask file and specify start column index for bounding boxes:
        python stabilize_video.py path/to/video/video.mp4 --mask-path path/to/mask/mask.txt --mask-start 1 --viz
    
    6. Apply stabilization with a custom configuration file:
        python stabilize_video.py path/to/video/video.mp4 --custom-config path/to/config/config.yaml

Notes:
    - Press 'q' during visualization to quit (with --viz).
"""

# ToDo:
# implement custom reference frame selection through CLI

import sys
import cv2
import numpy as np
import argparse
import platform
from tqdm import tqdm
from pathlib import Path
from stabilo import Stabilizer
from stabilo.utils import load_config, detect_delimiter

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])  # environment booleans
COLOURS = np.random.randint(0, 256, (100, 3))  # used to draw points on the frame

def stabilize_video(cli_args):
    """
    Stabilize a video using the stabilo library.
    """
    args, kwargs = get_args_and_kwargs(cli_args)
    stabilizer = Stabilizer(**kwargs)
    
    tracks = None if args.no_mask else load_bounding_boxes(args)
    reader, writer, writer_viz, pbar, w, h = initialize_streams(args, stabilizer.get_basic_info())
    
    try:
        frame_num = 0
        while reader.isOpened():
            flag, frame = reader.read()
            if not flag:
                break

            boxes = None if args.no_mask or tracks is None else tracks[tracks[:, 0].astype(int) == frame_num, 1:]
                
            if frame_num == 0:
                stabilizer.set_ref_frame(frame, boxes)
                stabilized_frame = frame
                stabilized_boxes = boxes
            else:
                stabilizer.stabilize(frame, boxes)
                stabilized_frame = stabilizer.warp_cur_frame()
                stabilized_boxes = stabilizer.transform_cur_boxes()

            if writer is not None and stabilized_frame is not None:
                writer.write(stabilized_frame)
            
            if args.viz or args.save_viz:
                imgs = visualize_frame(stabilizer, frame.copy(), stabilized_frame.copy(), boxes, stabilized_boxes, frame_num, args)
                if args.viz:
                    cv2.imshow('Stabilization Process Visualization', imgs)
                    if cv2.waitKey(0) & 0xFF == ord('q'):
                        break
                if writer_viz is not None:
                    imgs = cv2.resize(imgs, (w, h))
                    writer_viz.write(imgs)

            frame_num += 1
            pbar.update(1)

    except KeyboardInterrupt:
        print('Interrupted by user.')
    finally:
        close_streams(args, reader, writer, writer_viz, pbar, frame_num)

def initialize_streams(args, kwargs):
    """
    Initialize video reader, video writer, and progress bar.
    """
    if not args.input.exists():
        print(f'File {args.input} not found.')
        sys.exit(1)
    
    reader = cv2.VideoCapture(str(args.input))
    if not reader.isOpened():
        print(f'Failed to open {args.input}.')
        sys.exit(1)
    
    w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    writer, writer_viz = None, None
    if args.output is None:
        suffix = 'mp4' if MACOS else 'avi' if WINDOWS else 'mp4'
        output = args.input.parent / f"{args.input.stem}_stab_{kwargs['detector_name']}_{kwargs['matcher_name']}_{kwargs['transformation_type']}{'' if kwargs['mask_use'] else '_no_mask'}.{suffix}"
    else:
        output = args.output
        suffix = output.suffix[1:]

    fps = reader.get(cv2.CAP_PROP_FPS) # int() might be required, floats might produce error in MP4 codec
    fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'

    if args.save:
        if output.exists():
            print(f'File {output} already exists. Overwriting it.')
        writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    if args.save_viz:
        output_viz = output.parent / f'{output.stem}_VIZ.{suffix}'
        if output_viz.exists():
            print(f'File {output_viz} already exists. Overwriting it.')
        writer_viz = cv2.VideoWriter(str(output_viz), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    pbar = tqdm(total=frame_count, desc=f'Stabilizing {args.input}', unit='frames', leave=True, colour='green')

    return reader, writer, writer_viz, pbar, w, h

def close_streams(args, reader, writer, writer_viz, pbar, frame_num):
    """
    Close video reader, video writer, and progress bar.
    """
    reader.release()
    if writer is not None:
        writer.release()
    if writer_viz is not None:
        writer_viz.release()
    if args.viz:
        cv2.destroyAllWindows()
    pbar.total = frame_num
    pbar.refresh()
    pbar.close()

def load_bounding_boxes(args):
    """
    Read the bounding boxes from a .txt file.
    """
    tracks_path = args.mask_path or args.input.with_suffix('.txt')
    if not tracks_path.exists():
        print(f'File {tracks_path} not found. Make sure you have bounding boxes available for {args.input}, otherwise run with --no-mask.')
        sys.exit(1)

    delimiter = detect_delimiter(tracks_path)
    tracks = np.loadtxt(tracks_path, delimiter=delimiter, usecols=[0, *range(args.mask_start, args.mask_start + 4)])
    
    if args.mask_encoding == 'pascal':
        tracks[:, 1] = (tracks[:, 1] + tracks[:, 3]) / 2
        tracks[:, 2] = (tracks[:, 2] + tracks[:, 4]) / 2
        tracks[:, 3] -= tracks[:, 1]
        tracks[:, 4] -= tracks[:, 2]
    elif args.mask_encoding == 'coco':
        tracks[:, 1] += tracks[:, 3] / 2
        tracks[:, 2] += tracks[:, 4] / 2

    return tracks

def visualize_frame(stabilizer, frame, stabilized_frame, boxes, stabilized_boxes, frame_num, args):
    """
    Visualize the stabilization process.
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
    
    def draw_lines(img, ref_pts, cur_pts):
        best_match_count = 'N/A'
        if ref_pts is not None and cur_pts is not None:
            for i, (pt1, pt2) in enumerate(zip(ref_pts, cur_pts)):
                x1, y1 = pt1.ravel()
                x2, y2 = pt2.ravel()
                color = [0, 255, 0] if stabilizer.cur_inliers[i] else [0, 0, 255]
                cv2.line(img, (int(x1), int(y1)), (ref_frame.shape[1] + int(x2), int(y2)), color, 1, cv2.LINE_AA)
            best_match_count = i + 1 if 'i' in locals() else 'N/A'
        return img, best_match_count
    
    def draw_boxes(img, boxes):
        if boxes is not None:
            for box in boxes:
                x, y, w, h = box
                cv2.rectangle(img, (int(x - w / 2), int(y - h / 2)), (int(x + w / 2), int(y + h / 2)), (0, 255, 0), 2, cv2.LINE_AA)
        return img   

    def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), font_scale=6, font_thickness=4, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        cv2.rectangle(img, pos, (x + text_size[0], y + text_size[1]), text_color_bg, -1)
        cv2.putText(img, text, (x, y + text_size[1] + font_scale - 1), font, font_scale, text_color, font_thickness)

    ref_frame = stabilizer.ref_frame_gray
    ref_frame = draw_mask(ref_frame, stabilizer.ref_mask)
    ref_frame = draw_points(ref_frame, stabilizer.ref_pts)
    draw_text(ref_frame, "Reference frame (=0)")

    cur_frame = stabilizer.cur_frame_gray if stabilizer.cur_frame_gray is not None else np.full(stabilizer.ref_frame_gray.shape, 0, dtype=np.uint8)
    cur_frame = draw_mask(cur_frame, stabilizer.cur_mask)
    cur_frame = draw_points(cur_frame, stabilizer.cur_pts)
    draw_text(cur_frame, f"Frame {frame_num}")    
        
    imgs_upper = np.hstack((ref_frame, cur_frame))
    if not args.no_lines:
        imgs_upper, best_match_count = draw_lines(imgs_upper, stabilizer.ref_pts, stabilizer.cur_pts)
        inliers_count = stabilizer.cur_inliers_count if stabilizer.cur_inliers_count is not None else 'N/A'
        draw_text(imgs_upper, f"Number of matched points after pre-filtering: {best_match_count}", pos=(ref_frame.shape[1] // 2, 10), font_scale=4, text_color=(255, 0, 255))
        draw_text(imgs_upper, f"Number of inliers after RANSAC: {inliers_count}", pos=(ref_frame.shape[1] // 2, 80), font_scale=4, text_color=(255, 0, 255))

    if not args.no_boxes:
        frame = draw_boxes(frame, boxes)
        stabilized_frame = draw_boxes(stabilized_frame, stabilized_boxes)
    draw_text(frame, f'Original video frame {frame_num}')
    draw_text(stabilized_frame, f'Stabilized video frame {frame_num}')

    imgs_lower = np.hstack((stabilized_frame, frame if stabilizer.cur_frame is not None else np.zeros_like(stabilized_frame)))
    draw_text(imgs_lower, "Press 'q' to quit.", pos=(imgs_lower.shape[1] - 600, imgs_lower.shape[0] - 80), font_scale=4, text_color=(0, 0, 255))
    
    return np.vstack((imgs_upper, imgs_lower))

def drop_none_values(kwargs):
    """
    Drop None values from a dictionary.
    """
    return {k: v for k, v in kwargs.items() if v is not None}   

def get_args_and_kwargs(cli_args):
    """
    Get the arguments and keyword arguments from the CLI arguments.
    """
    kwargs = vars(cli_args)

    args = argparse.Namespace()
    args.input = kwargs.pop('input')
    args.output = kwargs.pop('output')
    args.mask_path = kwargs.pop('mask_path')
    args.no_mask = kwargs.pop('no_mask')
    args.mask_start = kwargs.pop('mask_start')
    args.mask_encoding = kwargs.pop('mask_encoding')
    args.save = kwargs.pop('save')
    args.viz = kwargs.pop('viz')
    args.save_viz = kwargs.pop('save_viz')
    args.no_lines = kwargs.pop('no_lines')
    args.no_boxes = kwargs.pop('no_boxes')
    args.custom_config = kwargs.pop('custom_config')

    if args.custom_config is not None:
        kwargs.update(load_config(args.custom_config))
    else:
        kwargs = drop_none_values(kwargs)
    kwargs['mask_use'] = not args.no_mask
    kwargs['viz'] = args.viz or args.save_viz

    return args, kwargs

def parse_args():
    """
    Parse the command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Stabilize a video using the stabilo library.")
    
    # script options
    parser.add_argument("input", type=Path, help="input video filepath")
    parser.add_argument("--output", "-o", type=Path, help="output video(s) filepath [if not provided, output will be saved in the same directory as input]")
    parser.add_argument("--viz", "-v", action="store_true", help="visualize the transformation process [default: False]")
    parser.add_argument("--save", "-s", action="store_true", help="save the stabilized video [default: False]")
    parser.add_argument("--save-viz", "-sv", action="store_true", help="save the visualization of the transformation process [default: False]")
    parser.add_argument("--no-mask", "-nm", action="store_true", help="do not use masks for stabilization [default: False]")
    parser.add_argument("--mask-path", "-mp", type=Path, help="custom mask filepath [default: input with .txt extension]")
    parser.add_argument("--mask-start", "-ms", type=int, default=2, help="start column index of bounding boxes used as masks. Column 0 reserved for frame number [default: 2]")
    parser.add_argument("--mask-encoding", "-me", type=str, default="yolo", choices=['yolo', 'pascal', 'coco'], help="bounding box encoding (all un-normalized) [default: yolo]")
    parser.add_argument("--no-lines", "-nl", action="store_true", help="do not draw lines between matched feature points [default: False]")
    parser.add_argument("--no-boxes", "-nb", action="store_true", help="do no draw bounding boxes on the (un-)stabilized videos [default: False]")
    parser.add_argument("--custom-config", "-cc", type=Path, help="path to a custom stabilo configuration file that overrides the CLI arguments below [default: None]")
    
    # stabilo options (default values are set in the stabilo library)
    parser.add_argument("--detector-name", "-dn", type=str, choices=['orb', 'sift', 'rsift', 'brisk', 'kaze', 'akaze'], help="detector type [default: orb]")
    parser.add_argument("--matcher-name", "-mn", type=str, choices=['bf', 'flann'], help="matcher type [default: bf]")
    parser.add_argument("--filter-type", "-ft", type=str, choices=['none', 'ratio', 'distance'], help="filter type for the match filter [default: ratio]")
    parser.add_argument("--transformation-type", "-tt", type=str, choices=['projective', 'affine'], help="transformation type [default: projective]")
    parser.add_argument("--clahe", "-c", action="store_true", help="apply CLAHE to grayscale images [default: False]")
    parser.add_argument("--downsample-ratio", "-dr", type=float, help="downsample ratio for the input video [default: 0.5]")
    parser.add_argument("--max-features", "-mf", type=int, help="maximum number of features to detect [default: 2000]")
    parser.add_argument("--ref-multiplier", "-rm", type=float, help="multiplier for max features in reference frame (ref_multiplier x max_features) [default: 2]")
    parser.add_argument("--filter-ratio", "-fr", type=float, help="filter ratio for the match filter [default: 0.9]")
    parser.add_argument("--ransac-method", "-r", type=int, help="RANSAC method [default: 38 (MAGSAC++)]")
    parser.add_argument("--ransac-epipolar-threshold", "-ret", type=float, help="RANSAC epipolar threshold [default: 2.0]")
    parser.add_argument("--ransac-max-iter", "-rmi", type=int, help="RANSAC maximum iterations [default: 5000]")
    parser.add_argument("--ransac-confidence", "-rc", type=float, help="RANSAC confidence [default: 0.999999]")
    parser.add_argument("--mask-margin-ratio", "-mmr", type=float, help="mask margin ratio [default: 0.15]")

    return parser.parse_args()

if __name__ == "__main__":
    stabilize_video(parse_args())
