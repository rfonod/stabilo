# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

import argparse
import platform
import sys

import cv2
import numpy as np
from tqdm import tqdm

from stabilo.utils import detect_delimiter, load_config

MACOS, LINUX, WINDOWS = (platform.system() == x for x in ['Darwin', 'Linux', 'Windows'])

def separate_cli_arguments(cli_args):
    """
    Get the command-line arguments and the corresponding keyword
    """
    args = argparse.Namespace(**vars(cli_args))
    kwargs = drop_none_values(vars(cli_args))

    if args.custom_config:
        kwargs.update(load_config(args.custom_config))

    kwargs['mask_use'] = not args.no_mask
    if hasattr(args, 'save_viz'):
        kwargs['viz'] = args.viz or args.save_viz

    return args, kwargs

def drop_none_values(kwargs):
    """
    Drop None values from a dictionary.
    """
    return {k: v for k, v in kwargs.items() if v is not None}

def load_tracks(args, logger):
    """
    Read the tracks containing bounding boxes from a file.
    """
    tracks_filepath = args.tracks or args.input.parent / f'{args.input.stem}.txt'

    if not tracks_filepath.exists():
        logger.error(f'{tracks_filepath} not found. Ensure a tracks file containing bboxes exists for {args.input}.')
        sys.exit(1)

    try:
        delimiter = detect_delimiter(tracks_filepath)
        tracks = np.loadtxt(tracks_filepath, delimiter=delimiter)
    except Exception as e:
        logger.error(f'Error reading {tracks_filepath}: {e}')
        sys.exit(1)
    else:
        logger.info(f'Loaded tracks from {tracks_filepath}.')
    return tracks

def load_exclusion_masks(args, logger):
    """
    Read the exclusion masks (bounding boxes) from a file.
    """
    if args.no_mask:
        logger.info('Exclusion masks disabled.')
        return None

    mask_filepath = args.mask_path or args.input.parent / f'{args.input.stem}.txt'
    if not mask_filepath.exists():
        logger.error(f'{mask_filepath} not found. Ensure you have a mask file for {args.input} or use --no-mask')
        sys.exit(1)

    try:
        delimiter = detect_delimiter(mask_filepath)
        masks = np.loadtxt(mask_filepath, delimiter=delimiter)
        columns = [args.mask_frame_idx, *range(args.mask_start_idx, args.mask_start_idx + 4)]
        boxes = get_boxes(masks, columns, args.mask_enc, logger)
    except Exception as e:
        logger.error(f'Error reading {mask_filepath}: {e}')
        sys.exit(1)
    else:
        logger.info(f'Loaded {len(boxes)} exclusion masks.')
    return boxes

def get_boxes_from_tracks(tracks, args, logger):
    """
    Get the bounding boxes from the tracks.
    """
    columns = [args.boxes_frame_idx, *range(args.boxes_start_idx, args.boxes_start_idx + 4)]
    return get_boxes(tracks, columns, args.boxes_enc, logger)

def get_boxes(boxes, columns, encoding, logger):
    """
    Get the bounding boxes from the exclusion masks or tracks.
    """
    try:
        boxes = boxes[:, columns]
        if encoding == 'pascal':
            logger.warning('Pascal VOC encoding is not supported. Converting to YOLO format.')
            boxes[:, 1] = (boxes[:, 1] + boxes[:, 3]) / 2
            boxes[:, 2] = (boxes[:, 2] + boxes[:, 4]) / 2
            boxes[:, 3] -= boxes[:, 1]
            boxes[:, 4] -= boxes[:, 2]
        elif encoding == 'coco':
            logger.warning('COCO encoding is not supported. Converting to YOLO format.')
            boxes[:, 1] += boxes[:, 3] / 2
            boxes[:, 2] += boxes[:, 4] / 2
    except Exception as e:
        logger.error(f'Error getting bounding boxes: {e}')
        sys.exit(1)
    return boxes

def get_boxes_for_frame(boxes, frame_num):
    """
    Get the exclusion masks (bounding boxes) for a specific frame.
    """
    if boxes is None:
        return None
    return boxes[boxes[:, 0].astype(int) == frame_num, 1:]

def initialize_read_streams(args, logger):
    """
    Initialize video reader and get video properties.
    """
    if not args.input.exists():
        logger.error(f'File {args.input} not found.')
        sys.exit(1)

    reader = cv2.VideoCapture(str(args.input))
    if not reader.isOpened():
        logger.error(f'Failed to open {args.input}.')
        sys.exit(1)

    frame_count = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = reader.get(cv2.CAP_PROP_FPS) # int() might be required, floats might produce error in MP4 codec

    return reader, frame_count, w, h, fps

def initialize_write_streams(args, w, h, fps, logger):
    """
    Initialize video writers for the stabilized video and the visualization.
    """
    writer_vid, writer_viz = None, None
    if args.save or args.save_viz:
        fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'
        suffix = 'mp4' if MACOS else 'avi' if WINDOWS else 'mp4'

        if args.output:
            output_path = args.output
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = args.input.parent

        if args.save:
            output_vid_filepath = output_path / f"{args.input.stem}_stab.{suffix}"
            if output_vid_filepath.exists():
                logger.warning(f'File {output_vid_filepath} already exists. Overwriting it.')
            writer_vid = cv2.VideoWriter(str(output_vid_filepath), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

        if args.save_viz:
            output_viz_filepath = output_path / f"{args.input.stem}_viz.{suffix}"
            if output_viz_filepath.exists():
                logger.warning(f'File {output_viz_filepath} already exists. Overwriting it.')
            writer_viz = cv2.VideoWriter(str(output_viz_filepath), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    return writer_vid, writer_viz

def initialize_track_write_stream(args, w, h, fps, logger):
    """
    Initialize video writer for the track visualization.
    """
    writer_track = None
    if args.save_viz:
        fourcc = 'avc1' if MACOS else 'WMV2' if WINDOWS else 'mp4v'
        suffix = 'mp4' if MACOS else 'avi' if WINDOWS else 'mp4'

        if args.output:
            output_path = args.output
            output_path.mkdir(parents=True, exist_ok=True)
        else:
            output_path = args.input.parent

        output_filepath = output_path / f"{args.input.stem}_track.{suffix}"
        if output_filepath.exists():
            logger.warning(f'File {output_filepath} already exists. Overwriting it.')
        writer_track = cv2.VideoWriter(str(output_filepath), cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))

    return writer_track

def initialize_progress_bar(args, frame_count):
    """
    Initialize the progress bar.
    """
    return tqdm(total=frame_count, desc=f'Stabilizing {args.input}', unit='frames', leave=True, colour='green')

def close_streams(args, reader, pbar, writer_vid=None, writer_viz=None, writer_track=None):
    """
    Close video reader, progress bar, video writers, and visualization window.
    """
    reader.release()
    pbar.close()
    if writer_vid is not None:
        writer_vid.release()
    if writer_viz is not None:
        writer_viz.release()
    if writer_track is not None:
        writer_track.release()
    if args.viz:
        cv2.destroyAllWindows()

def draw_boxes(img, boxes, color=(0, 255, 0), line_type=cv2.LINE_AA):
    """
    Draw bounding boxes.
    """
    if boxes is not None:
        for box in boxes:
            x_c, y_c, w, h = box
            pt1, pt2 = (int(x_c - w / 2), int(y_c - h / 2)), (int(x_c + w / 2), int(y_c + h / 2))
            cv2.rectangle(img, pt1, pt2, color, 2, line_type)
    return img

def draw_text(img, text, font=cv2.FONT_HERSHEY_PLAIN, pos=(0, 0), scale=7, thickness=5, color_fg=(0, 255, 0)):
    """
    Draw text on an image.
    """
    x, y = pos
    color_bg = (0, 0, 0)
    text_size, _ = cv2.getTextSize(text, font, scale, thickness)
    overlay = img.copy()
    cv2.rectangle(overlay, pos, (x + text_size[0], y + text_size[1]), color_bg, -1)
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.putText(img, text, (x, y + text_size[1] + scale - 1), font, scale, color_fg, thickness)
