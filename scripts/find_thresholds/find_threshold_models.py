#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
find_threshold_models.py - Find linear models for BRISK, KAZE, and AKAZE detectors

Description:
    This script develops linear regression models for feature detectors (BRISK, KAZE, AKAZE) based on their performance metrics, 
    specifically analyzing the relationship between detector thresholds and the average number of keypoints detected per image. 
    It processes images from a specified dataset directory, optionally applying a mask and CLAHE (Contrast Limited Adaptive Histogram Equalization) 
    for image preprocessing. The outcomes include saving the regression coefficients, generating plots to visualize relationships, 
    and saving raw and filtered data for extended analysis.

Usage:
    find_threshold_models.py [options]

Options:
    --dataset-dir, -dir <dir>   : Directory containing the image dataset. [default: scenes]
    --detectors, -d <detectors> : List of detectors to analyze. Choices include 'brisk', 'kaze', 'akaze'. [default: brisk, kaze, akaze]
    --mask-use, -m <mask_use>   : Specify whether to use a mask during detection. Choices: True, False. [default: [True, False]]
    --clahe-use, -c <clahe_use> : Specify whether to apply CLAHE on images. Choices: True, False. [default: [True, False]]
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Union

MASK_START = 1  # the column index where the mask values are stored in the .txt file
IMG_SUFFIXES = ['.jpg', '.jpeg', '.png', '.bmp']  # supported image formats
MASK_MARGIN_RATIO = 0.15  # the ratio of the bounding box width and height to add as a margin to the mask

# model fitting parameters - the model is fitted only for the keypoints in the given range
MIN_KEYPOINTS_NUM_FIT = 1000
MAX_KEYPOINTS_NUM_FIT = 10000


def find_all_models(args):
    """
    Find the linear models for all combinations of detectors, mask_use, and clahe.
    """
    for detector in args.detectors:
        for mask_on in args.mask_use:
            for clahe_on in args.clahe_use:
                if detector == 'brisk':
                    t_min, t_max, t_step = 65, 150, 5
                elif detector in ['akaze', 'kaze']:
                    t_min, t_max, t_step = 0.001, 0.025, 0.001
                threshold_settings = (t_min, t_max, t_step)

                print(f"Detector: {detector}, mask: {mask_on}, CLAHE: {clahe_on}")
                find_linear_model(args.dataset_dir, detector, mask_on, clahe_on, threshold_settings)


def find_linear_model(dataset_dir: Path, detector_name: str, mask_on: bool, clahe_on: bool, threshold_settings: tuple):
    """
    Find the linear model for the given data.
    
    Parameters:
    - dataset_dir (Path): Path to the directory containing the dataset
    - detector_name (str): Name of the detector
    - mask_on (bool): Whether the mask was used
    - clahe_on (bool): Whether CLAHE was used
    - threshold_settings (tuple): (threshold_min, threshold_max, threshold_step)
    """
    threshold_min, threshold_max, threshold_step = threshold_settings

    scenes_path = [s for s in dataset_dir.iterdir() if s.is_file() and s.suffix in IMG_SUFFIXES]

    data = []
    for scene_filepath in tqdm(scenes_path, desc="Loading scenes...", unit="scenes", leave=False):
        scene, mask = load_image_and_mask(scene_filepath, mask_on)
        if mask_on and mask is None:
            continue
        data.append((scene, mask))

    if len(data) == 0:
        print(f"No scenes found or no bounding boxes found for the scenes. Skipping {detector_name} detector for {mask_on} mask and {clahe_on} CLAHE.")
        return

    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)) if clahe_on else None
    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    print(f"Thresholds: {thresholds}")

    thresholds_arr, keypoints_arr = [], []

    for threshold in tqdm(thresholds, desc="Finding # keypoints...", unit="threshold", leave=False):
        if detector_name.lower() == "brisk":
            detector = cv2.BRISK_create(thresh=int(threshold))
        elif detector_name.lower() == "kaze":
            detector = cv2.KAZE_create(threshold=float(threshold))
        elif detector_name.lower() == "akaze":
            detector = cv2.AKAZE_create(threshold=float(threshold))
    
        keypoints_in_scene = [find_keypoints_for_threshold(detector, scene, mask, clahe) for scene, mask in data]

        keypoints_arr.append(np.mean(keypoints_in_scene))
        thresholds_arr.append(threshold)
    
    keypoints_arr = np.array(keypoints_arr)
    thresholds_arr = np.array(thresholds_arr)

    save_data(thresholds_arr, keypoints_arr, detector_name, mask_on, clahe_on, True)

    interval_points = (keypoints_arr >= MIN_KEYPOINTS_NUM_FIT) & (keypoints_arr <= MAX_KEYPOINTS_NUM_FIT)
    keypoints_arr = keypoints_arr[interval_points]
    thresholds_arr = thresholds_arr[interval_points]

    save_data(thresholds_arr, keypoints_arr, detector_name, mask_on, clahe_on, False)

    model = fit_model(thresholds_arr, keypoints_arr)

    save_model(model, detector_name, mask_on, clahe_on)
    
    plot_and_save(thresholds_arr, keypoints_arr, model, detector_name, mask_on, clahe_on)


def find_keypoints_for_threshold(detector, scene: np.ndarray, mask: Union[np.ndarray, None] = None, clahe: Union[cv2.CLAHE, None] = None) -> int:
    """
    Find the number of keypoints for the given threshold value.
    
    Parameters:
    - detector: cv2.FeatureDetector - Detector to use
    - scene (np.ndarray): Scene to find keypoints in
    - mask (Union[np.ndarray, None]): Mask to use
    - clahe (Union[cv2.CLAHE, None]): CLAHE object to use or None
    
    Returns:
    - keypoints_num (int): Number of keypoints found
    """
    scene = cv2.cvtColor(scene, cv2.COLOR_BGR2GRAY)
    if clahe is not None:
        scene = clahe.apply(scene)
    kpts = detector.detectAndCompute(scene, mask)[0]
    return len(kpts)


def load_image_and_mask(scene_path: Path, mask_on: bool) -> tuple[np.ndarray, Union[np.ndarray, None]]:
    """
    Load the scene and the bounding boxes.
    
    Parameters:
    - scene_path (Path): Path to the scene
    - mask_on (bool): Whether the mask should be used
    
    Returns:
    - scene (np.ndarray): Scene
    - mask (Union[np.ndarray, None]): Mask or None
    """
    scene = cv2.imread(str(scene_path))

    mask = None
    if mask_on:
        boxes_path = scene_path.with_suffix('.txt')
        if boxes_path.exists():
            boxes = np.loadtxt(str(boxes_path), delimiter=' ')
            boxes = boxes[:, MASK_START:MASK_START + 4]

            h, w = scene.shape[:2]

            if np.max(boxes) <= 1:
                boxes[:, 0] *= w
                boxes[:, 1] *= h
                boxes[:, 2] *= w
                boxes[:, 3] *= h

            mask = create_mask(boxes, w, h)
        else:
            print(f"No bounding boxes found for {scene_path.name}")

    return scene, mask


def create_mask(boxes: np.ndarray, w: int, h: int) -> np.ndarray:
    """
    Create a binary mask from the bounding boxes.
    
    Parameters:
    - boxes (np.ndarray): Bounding boxes
    - w (int): Width of the image
    - h (int): Height of the image
    
    Returns:
    - mask (np.ndarray): Binary mask
    """
    mask = np.full((h, w), 255, dtype=np.uint8)

    for box in boxes:
        xc, yc, wb, hb = box

        wb += wb * MASK_MARGIN_RATIO
        hb += hb * MASK_MARGIN_RATIO

        x1 = max(0, min(w, int(xc - wb / 2)))
        y1 = max(0, min(h, int(yc - hb / 2)))
        x2 = max(0, min(w, int(xc + wb / 2)))
        y2 = max(0, min(h, int(yc + hb / 2)))

        mask[y1:y2, x1:x2] = 0

    return mask


def fit_model(thresholds_arr: np.ndarray, keypoints_arr: np.ndarray) -> np.ndarray:
    """
    Fit the linear model to the given data.
    
    Parameters:
    - thresholds_arr (np.ndarray): Threshold values
    - keypoints_arr (np.ndarray): Number of keypoints for each threshold/scene value
    
    Returns:
    - model (np.ndarray): Coefficients of the linear model
    """
    model = np.polyfit(keypoints_arr, thresholds_arr, 1)
    print(f"Model: {model}")
    return model


def save_data(thresholds_arr: np.ndarray, keypoints_arr: np.ndarray, detector_name: str, mask_on: bool, clahe_on: bool, raw_data: bool):
    """
    Save the data to a file.
    
    Parameters:
    - thresholds_arr (np.ndarray): Threshold values
    - keypoints_arr (np.ndarray): Number of keypoints for each threshold/scene value
    - detector_name (str): Name of the detector
    - mask_on (bool): Whether the mask was used
    - clahe_on (bool): Whether CLAHE was used
    - raw_data (bool): Whether the data is raw or filtered
    """
    data_dir = Path(__file__).resolve().parent / 'results' / detector_name
    data_dir.mkdir(parents=True, exist_ok=True)

    filename = f"data_{'RAW' if raw_data else 'FILTERED'}_mask_{mask_on}_clahe_{clahe_on}.txt"
    filepath = data_dir / filename

    np.savetxt(str(filepath), np.column_stack((thresholds_arr, keypoints_arr)), delimiter=',', header="threshold, avg_num_keypoints", comments='')


def save_model(model: np.ndarray, detector_name: str, mask_on: bool, clahe_on: bool):
    """
    Save the model to a file.
    
    Parameters:
    - model (np.ndarray): Coefficients of the linear model
    - detector_name (str): Name of the detector
    - mask_on (bool): Whether the mask was used
    - clahe_on (bool): Whether CLAHE was used
    """
    model_dir = Path(__file__).resolve().parent / 'models' / detector_name
    model_dir.mkdir(parents=True, exist_ok=True)

    filename = f"model_mask_{mask_on}_clahe_{clahe_on}.txt"
    filepath = model_dir / filename

    np.savetxt(str(filepath), model)


def plot_and_save(thresholds_arr: np.ndarray, keypoints_arr: np.ndarray, model: np.ndarray, detector_name: str, mask_on: bool, clahe_on: bool):
    """
    Plot the data and save the plot to a file.
    
    Parameters:
    - thresholds_arr (np.ndarray): Threshold values
    - keypoints_arr (np.ndarray): Number of keypoints for each threshold/scene value
    - model (np.ndarray): Coefficients of the linear model
    - detector_name (str): Name of the detector
    - mask_on (bool): Whether the mask was used
    - clahe_on (bool): Whether CLAHE was used
    """
    plots_dir = Path(__file__).resolve().parent / 'plots' / detector_name
    plots_dir.mkdir(parents=True, exist_ok=True)

    filename = f"plot_mask_{mask_on}_clahe_{clahe_on}.png"
    filepath = plots_dir / filename

    plt.plot(keypoints_arr, thresholds_arr, 'o', label='data')
    keypoints_fit = np.array([MIN_KEYPOINTS_NUM_FIT, MAX_KEYPOINTS_NUM_FIT])
    plt.plot(keypoints_fit, np.polyval(model, keypoints_fit), 'r-', label='model')

    plt.xlim(MIN_KEYPOINTS_NUM_FIT, MAX_KEYPOINTS_NUM_FIT)
    plt.title(f"Thresholds for {detector_name} detector (Mask={mask_on}, CLAHE={clahe_on})")
    plt.xlabel('Number of keypoints')
    plt.ylabel(f'{detector_name} threshold value')
    plt.legend()
    plt.savefig(filepath)
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    Parse the command line arguments.
    """
    parser = argparse.ArgumentParser(description="Find brisk, kaze, or akaze threshold models for the given data")
    parser.add_argument("--dataset-dir", "-dir", type=Path, default=Path('scenes'), help="Directory containing the dataset")
    parser.add_argument("--detectors", "-d", type=str, nargs='+', default=['brisk', 'kaze', 'akaze'], help="Detectors to consider [default: brisk, kaze, akaze]") 
    parser.add_argument("--mask-use", "-m", type=bool, nargs='+', default=[True, False], help="Whether to use the mask [default: True False]")
    parser.add_argument("--clahe-use", "-c", type=bool, nargs='+', default=[True, False], help="Whether to use CLAHE [default: True False]")
    return parser.parse_args()


if __name__ == "__main__":
    find_all_models(parse_args())
