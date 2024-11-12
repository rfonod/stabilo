# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stabilo.py - Reference frame video stabilization with optional user-provided exclusion masks

This module provides the Stabilizer class for video or track stabilization using feature matching
and transformation estimation. It leverages OpenCV for core functionalities.

The class supports various feature detectors, matchers, filtering methods, and transformation types.
Fine-tuning these parameters allows customization for specific video stabilization needs.

The parameters for the feature detectors, matchers, filtering methods, and transformations can be
fine-tuned to suit specific requirements, see https://github.com/rfonod/stabilo-optimize.

Key Features:
  - Video or bounding box (tracks) stabilization with respect to a reference frame.
  - Fine-tunable parameters for feature detectors, matchers, filtering methods, and transformations.
  - Support for various feature detectors (e.g., ORB, SIFT) and matchers (e.g., BF, FLANN).
  - Projective or affine transformations for frame stabilization.
  - RANSAC-based algorithms for robust transformation matrix estimation.
  - CLAHE and pre-processing options for contrast enhancement.
  - Visualization and debugging features for keypoints, descriptors, and masks.
  - GPU acceleration for improved performance (not implemented yet).

Usage:
1. Create an instance of the 'Stabilizer' class with desired parameter configurations.
2. Set a reference frame using the 'set_ref_frame' method.
3. Stabilize any preceding or subsequent frames using the 'stabilize' method.
4. Access stabilized frames, bounding boxes, and transformation matrices using specific methods.
"""

import sys
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from .utils import four2xywh, load_config, setup_logger, timer, xywh2four

# Configure logging
logger = setup_logger(__name__)

# Define the root directory
ROOT = Path(__file__).resolve().parents[0]

# Read the default parameters from a configuration file
cfg = load_config(ROOT / "cfg" / "default.yaml", logger)

# Profiling flag
PROFILING = False


class Stabilizer:
    """
    This class implements a video stabilizer. It uses feature matching to find the transformation
    between the reference frame and the current frame, allowing stabilization of subsequent frames.
    The transformation matrix can be used to transform the current frame to the reference frame or
    to transform points from the current frame to the reference frame.
    """

    VALID_DETECTORS = ['orb', 'sift', 'rsift', 'brisk', 'kaze', 'akaze']
    VALID_MATCHERS = ['bf', 'flann']
    VALID_FILTER_TYPES = ['none', 'ratio', 'distance']
    VALID_TRANSFORMATION_TYPES = ['projective', 'affine']
    VALID_RANSAC_METHODS_DICT = {
        'cv2.LMEDS': cv2.LMEDS,                  # 4 - LMEDS
        'cv2.RANSAC': cv2.RANSAC,                # 8 - RANSAC
        'cv2.RHO': cv2.RHO,                      # 16 - RHO
        'cv2.USAC_DEFAULT': cv2.USAC_DEFAULT,    # 32 - DEGENSAC
        'cv2.USAC_PARALLEL': cv2.USAC_PARALLEL,  # 33 - DEGENSAC (with different parameters)
        'cv2.USAC_FAST': cv2.USAC_FAST,          # 35 - LO-RANSAC
        'cv2.USAC_ACCURATE': cv2.USAC_ACCURATE,  # 36 - GC-RANSAC
        'cv2.USAC_PROSAC': cv2.USAC_PROSAC,      # 37 - PROSAC
        'cv2.USAC_MAGSAC': cv2.USAC_MAGSAC       # 38 - MAGSAC++
    }

    def __init__(self, **kwargs):
        """
        Initialize the Stabilizer with user-provided or default configurations.

        Arguments:
        - detector_name: str - feature detector to use (orb, sift, rsift, brisk, kaze, akaze)
        - matcher_name: str - feature matcher to use (bf, flann)
        - filter_type: str - filter type for the feature matcher (none, ratio, distance)
        - transformation_type: str - transformation for stabilization (projective, affine)
        - clahe: bool - use CLAHE for contrast enhancement
        - downsample_ratio: float - down-sampling ratio for the frames (e.g., 0.5 for half the size)
        - max_features: int - max number of features to detect (for BRISK, KAZE, and AKAZE this is an approximation)
        - ref_multiplier: float - multiplier for max features in reference frame (ref_multiplier x max_features)
        - mask_use: bool - use mask for feature detection
        - filter_ratio: float - filtering ratio; Lowe's ratio for 'ratio' filter, distance threshold ratio for 'distance' filter
        - ransac_method: int - method for RANSAC algorithm (see above for options)
        - ransac_epipolar_threshold: float - threshold for RANSAC (e.g., 1.0)
        - ransac_max_iter: int - max iterations for RANSAC (e.g., 2000)
        - ransac_confidence: float - confidence for RANSAC (e.g., 0.999)
        - brisk_threshold: int - threshold for BRISK detector (used only if 'max_features -> threshold' model is unavailable)
        - kaze_threshold: float - threshold for KAZE detector (used only if 'max_features -> threshold' model is unavailable)
        - akaze_threshold: float - threshold for AKAZE detector (used only if 'max_features -> threshold' model is unavailable)
        - gpu: bool - use GPU acceleration (not fully implemented/tested yet)
        - viz: bool - save some features for visualization (e.g., keypoints, descriptors, masks)
        - benchmark: bool - different behavior for benchmarking purposes (e.g., re-use the last transformation if the current is None)
        - min_good_match_count_warning: int - min number of good matches to trigger a warning
        - min_inliers_match_count_warning: int - min number of inliers to trigger a warning
        """
        self._load_configuration(kwargs)
        self._validate_arguments()
        self._initialize_variables()
        self._create_feature_detectors()
        self._create_matcher()
        self._create_transformer()
        self._create_helpers()

    def _load_configuration(self, kwargs):
        """
        Load configuration parameters, using defaults if not provided.
        """
        for key, value in cfg.items():
            setattr(self, key, kwargs.get(key, value))

    def _initialize_variables(self):
        """
        Initialize internal variables for the Stabilizer.
        """
        self.ref_frame, self.cur_frame = None, None
        self.ref_frame_gray, self.cur_frame_gray = None, None
        self.ref_boxes, self.cur_boxes = None, None
        self.ref_mask, self.cur_mask = None, None
        self.ref_kpts, self.cur_kpts = None, None
        self.ref_desc, self.cur_desc = None, None
        self.ref_pts, self.cur_pts = None, None
        self.cur_trans_matrix, self.trans_matrix_last_known = None, None
        self.cur_inliers, self.cur_inliers_count = None, None
        self.h, self.w = None, None

    def _create_feature_detectors(self):
        """
        Create feature detectors and descriptor extractors based on the provided configurations.
        """
        self.detector_cur, self.detector_ref = self._create_detector(self.detector_name)
        self.norm_type = self._get_norm_type()

    def _create_detector(self, detector_name: str):
        """
        Create a feature detector based on the provided detector name.
        """
        if detector_name == "orb":
            return self._create_orb_detectors()
        elif detector_name in ["sift", "rsift"]:
            return self._create_sift_detectors()
        elif detector_name == "brisk":
            return self._create_brisk_detectors()
        elif detector_name == "kaze":
            return self._create_kaze_detectors()
        elif detector_name == "akaze":
            return self._create_akaze_detectors()

    def _create_orb_detectors(self):
        """
        Create ORB detectors and descriptor extractors.
        """
        return (
            cv2.cuda.ORB_create(self.max_features) if self.gpu else cv2.ORB_create(self.max_features),
            cv2.cuda.ORB_create(round(self.ref_multiplier * self.max_features)) if self.gpu else cv2.ORB_create(round(self.ref_multiplier * self.max_features))
        )

    def _create_sift_detectors(self):
        """
        Create SIFT detectors and descriptor extractors.
        """
        return (
            cv2.cuda.SIFT_create(self.max_features) if self.gpu else cv2.SIFT_create(self.max_features), # (enable_precise_upscale=True)
            cv2.cuda.SIFT_create(round(self.ref_multiplier * self.max_features)) if self.gpu else cv2.SIFT_create(round(self.ref_multiplier * self.max_features))
        )

    def _create_brisk_detectors(self):
        """
        Create BRISK detectors and descriptor extractors.
        """
        threshold_cur, threshold_ref = self._get_thresholds()
        return (
            cv2.cuda.BRISK.create(thresh=round(threshold_cur)) if self.gpu else cv2.BRISK_create(thresh=round(threshold_cur)),
            cv2.cuda.BRISK.create(thresh=round(threshold_ref)) if self.gpu else cv2.BRISK_create(thresh=round(threshold_ref))
        )

    def _create_kaze_detectors(self):
        """
        Create KAZE detectors and descriptor extractors.
        """
        threshold_cur, threshold_ref = self._get_thresholds()
        return (
            cv2.cuda.KAZE_create(threshold=threshold_cur) if self.gpu else cv2.KAZE_create(threshold=threshold_cur),
            cv2.cuda.KAZE_create(threshold=threshold_ref) if self.gpu else cv2.KAZE_create(threshold=threshold_ref)
        )

    def _create_akaze_detectors(self):
        """
        Create AKAZE detectors and descriptor extractors.
        """
        threshold_cur, threshold_ref = self._get_thresholds()
        return (
            cv2.cuda.AKAZE_create(threshold=threshold_cur) if self.gpu else cv2.AKAZE_create(threshold=threshold_cur),
            cv2.cuda.AKAZE_create(threshold=threshold_ref) if self.gpu else cv2.AKAZE_create(threshold=threshold_ref)
        )

    def _get_thresholds(self):
        """
        Get thresholds for BRISK, KAZE, and AKAZE based on precomputed models.
        """
        detector_name = self.detector_name.upper()
        threshold_model_filepath = ROOT / 'thresholds' / 'models' / f'{detector_name}' / f'model_mask_{self.mask_use}_clahe_{self.clahe}.txt'
        if threshold_model_filepath.exists():
            model = np.loadtxt(str(threshold_model_filepath))
            threshold_cur = model[1] + model[0] * self.max_features
            threshold_ref = model[1] + model[0] * self.max_features * self.ref_multiplier
            if not self.benchmark:
                logger.info(f"Using {detector_name} with threshold {threshold_ref} for the reference frame and {threshold_cur} for the current frame.")
        else:
            threshold_cur = self.brisk_threshold if detector_name == 'BRISK' else self.kaze_threshold if detector_name == 'KAZE' else self.akaze_threshold
            threshold_ref = threshold_cur
            if not self.benchmark:
                logger.warning(f"No threshold analysis for {detector_name}. Using default threshold.")
        return threshold_cur, threshold_ref

    def _get_norm_type(self):
        """
        Get the norm type based on the detector name.
        """
        if self.detector_name in ["orb", "brisk", "akaze"]:
            return cv2.NORM_HAMMING # N.B.: if ORB is using WTA_K == 3 or 4, cv.NORM_HAMMING2 should be used
        elif self.detector_name in ["sift", "rsift", "kaze"]:
            return cv2.NORM_L2

    def _create_matcher(self):
        """
        Create the feature matcher based on the provided configurations.
        """
        if self.matcher_name == "bf":
            self.matcher = self._create_brute_force_matcher()
        elif self.matcher_name == "flann":
            self.matcher = self._create_flann_matcher()

    def _create_brute_force_matcher(self):
        """
        Create a brute-force matcher.
        """
        return cv2.cuda.DescriptorMatcher_createBFMatcher(self.norm_type, crossCheck=(True,False)[self.filter_type=='ratio']) if self.gpu else cv2.BFMatcher(self.norm_type, crossCheck=(True,False)[self.filter_type=='ratio'])

    def _create_flann_matcher(self):
        """
        Create a FLANN-based matcher.
        """
        if self.norm_type in [cv2.NORM_HAMMING, cv2.NORM_HAMMING2]:
            index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        elif self.norm_type == cv2.NORM_L2:
            index_params = dict(algorithm=0, trees=5)
        search_params = dict(checks=100)
        return cv2.cuda.DescriptorMatcher_createFlannBasedMatcher(index_params, search_params) if self.gpu else cv2.FlannBasedMatcher(index_params, search_params)

    def _create_transformer(self):
        """
        Create the transformation matrix estimator based on the provided configurations.
        """
        if self.transformation_type == 'projective':
            self.transformer = cv2.findHomography
        elif self.transformation_type == 'affine':
            self.transformer = cv2.estimateAffinePartial2D

    def _create_helpers(self):
        """
        Create helper objects for grayscale conversion, CLAHE, and resizing.
        """
        self.grayscale_converter = cv2.cuda.cvtColor if self.gpu else cv2.cvtColor
        self.claher = cv2.cuda.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)) if self.gpu else cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))
        self.resizer = cv2.cuda.resize if self.gpu else cv2.resize

    @timer(PROFILING)
    def set_ref_frame(self, frame: np.ndarray, boxes: np.ndarray = None, box_format: str = 'xywh') -> None:
        """
        Set the reference frame and object bounding boxes.
        Calculate keypoints and descriptors for the reference frame.
        """
        self.process_frame(frame, boxes, box_format, is_reference=True)

    @timer(PROFILING)
    def stabilize(self, frame: np.ndarray, boxes: np.ndarray = None, box_format: str = 'xywh') -> None:
        """
        This method takes an un-stabilized video frame and
        calculates a transformation matrix that can transform
        this frame or boxes to the reference frame coordinates.
        """
        success = self.process_frame(frame, boxes, box_format, is_reference=False)
        if success:
            matches = self.get_matches(self.ref_desc, self.cur_desc)
            if matches and self.ref_kpts is not None and self.cur_kpts is not None:
                self.ref_pts = np.float32([self.ref_kpts[m.queryIdx].pt for m in matches]).reshape(-1, 2)
                self.cur_pts = np.float32([self.cur_kpts[m.trainIdx].pt for m in matches]).reshape(-1, 2)
            else:
                self.ref_pts = []
                self.cur_pts = []
            self.calculate_transformation_matrix()

    def process_frame(self, frame: np.ndarray, boxes: np.ndarray = None, box_format: str = 'xywh', is_reference: bool = False) -> bool:
        """
        Process the given frame and bounding boxes.
        """
        if frame is None:
            logger.error(f'{"Reference" if is_reference else "Current"} frame is invalid.')
            sys.exit(1)

        if self.mask_use:
            if boxes is None:
                logger.warning(f'Mask is set to be used, but no bounding boxes were provided for the {"reference" if is_reference else "current"} frame.')
        else:
            boxes = None # if mask is not to be used, ignore the bounding boxes even if provided

        if is_reference:
            self.h, self.w = frame.shape[:2]
            self.ref_frame, self.ref_boxes, self.ref_box_format = frame, boxes, box_format
        else:
            self.cur_frame, self.cur_boxes, self.cur_box_format = frame, boxes, box_format

        mask = None if boxes is None else self.create_binary_mask(boxes, box_format)
        kpts, desc, frame_gray = self.get_features_and_descriptors(frame, mask, is_reference)

        if is_reference:
            self.ref_kpts, self.ref_desc, self.ref_frame_gray = kpts, desc, frame_gray
        else:
            self.cur_kpts, self.cur_desc, self.cur_frame_gray = kpts, desc, frame_gray

        if self.viz:
            if is_reference:
                self.ref_mask = mask
                self.ref_pts = np.array([ref_kpt.pt for ref_kpt in self.ref_kpts], dtype=np.float32).reshape(-1, 2) if self.ref_kpts is not None else []
            else:
                self.cur_mask = mask

        return True

    @timer(PROFILING)
    def get_features_and_descriptors(self, frame: np.ndarray, mask: np.ndarray = None, ref_frame: bool = False) -> tuple:
        """
        Get the features and descriptors for the given frame.
        """
        if frame is None:
            return None, None, None

        if self.gpu:
            frame = cv2.cuda_GpuMat(frame)
            mask = cv2.cuda_GpuMat(mask) if mask is not None else None

        frame = self.grayscale_converter(frame, cv2.COLOR_BGR2GRAY)
        if self.clahe:
            frame = self.claher.apply(frame)

        frame_gray = frame if self.viz else None

        if self.downsample_ratio != 1.0:
            frame = self.resizer(frame, (0, 0), fx=self.downsample_ratio, fy=self.downsample_ratio)
            mask = self.resizer(mask, (0, 0), fx=self.downsample_ratio, fy=self.downsample_ratio) if mask is not None else None

        try:
            kpts, desc = (self.detector_ref if ref_frame else self.detector_cur).detectAndCompute(frame, mask)
        except cv2.error as e:
            logger.warning(f"Features and descriptors couldn't be found. \n Error: {e}")
            return None, None, None

        if self.detector_name == 'rsift':
            desc /= (desc.sum(axis=1, keepdims=True) + 1e-8)
            desc = np.sqrt(desc)

        if self.downsample_ratio != 1.0:
            for kpt in kpts:
                kpt.pt = (kpt.pt[0] / self.downsample_ratio, kpt.pt[1] / self.downsample_ratio)

        if self.gpu:
            kpts = self.detector_cur.convert(kpts)
            desc = self.detector_cur.convert(desc)
            frame_gray = frame.download(frame_gray)

        return kpts, desc, frame_gray

    @timer(PROFILING)
    def get_matches(self, desc1: np.ndarray, desc2: np.ndarray) -> list:
        """
        Match the given descriptors.
        """
        if desc1 is None or desc2 is None:
            logger.warning("One of the descriptors is invalid.")
            return []

        try:
            if self.filter_type == 'none':
                good_matches = self.matcher.match(desc1, desc2, None)
            elif self.filter_type == 'distance':
                matches = self.matcher.match(desc1, desc2, None)
                matches = sorted(matches, key=lambda x: x.distance)
                min_dist, max_dist = matches[0].distance, matches[-1].distance
                good_thresh = min_dist + (max_dist - min_dist) * self.filter_ratio
                good_matches = [m for m in matches if m.distance <= good_thresh]
            elif self.filter_type == 'ratio':
                matches = self.matcher.knnMatch(desc1, desc2, k=2)
                good_matches = []
                for pair in matches:
                    if len(pair) == 2:
                        m, n = pair
                        if m.distance < self.filter_ratio*n.distance:
                            good_matches.append(m)
        except cv2.error as e:
            logger.error(f"Matches couldn't be found. \n Error: {e}")
            return []

        if len(good_matches) <= self.min_good_match_count_warning:
            logger.warning(f'Only {len(good_matches)} good matches were found.')

        return list(good_matches)

    @timer(PROFILING)
    def calculate_transformation_matrix(self) -> None:
        """
        Estimate the transformation matrix using the current and reference points.
        """
        if self.ref_pts is not None and self.cur_pts is not None and len(self.ref_pts) >= 4 and len(self.cur_pts) >= 4:
            try:
                self.cur_trans_matrix, inliers = self.transformer(self.cur_pts, self.ref_pts, maxIters=self.ransac_max_iter,
                method=self.ransac_method, confidence=self.ransac_confidence, ransacReprojThreshold=self.ransac_epipolar_threshold)
            except cv2.error as e:
                logger.exception(f"Transformation matrix couldn't be calculated.\n Error: {e}")
                self.cur_trans_matrix = np.eye(3) if self.benchmark else self.trans_matrix_last_known
                inliers = np.full((len(self.cur_pts), 1), False, dtype=bool)
                inliers_count = 'N/A'
                if not self.benchmark:
                    logger.warning("Re-using the last known transformation matrix.")
            else:
                if self.cur_trans_matrix is not None:
                    self.trans_matrix_last_known = self.cur_trans_matrix
                    inliers_count = sum(inliers.ravel().tolist())
                    if inliers_count <= self.min_inliers_match_count_warning:
                        logger.warning(f'Only {inliers_count} inliers points were used to estimate the transformation matrix.')
                else:
                    logger.warning('Transformation matrix is None.')
                    self.cur_trans_matrix = np.eye(3) if self.benchmark else self.trans_matrix_last_known
                    inliers = np.full((len(self.cur_pts), 1), False, dtype=bool)
                    inliers_count = 'N/A'
                    if not self.benchmark:
                        logger.warning("Re-using the last known transformation matrix.")
        else:
            logger.warning('Not enough points to estimate the transformation matrix.')
            self.cur_trans_matrix = np.eye(3) if self.benchmark else self.trans_matrix_last_known
            if not self.benchmark:
                logger.warning("Re-using the last known transformation matrix.")
            inliers = np.full((len(self.cur_pts), 1), False, dtype=bool)
            inliers_count = 'N/A'

        if self.viz:
            self.cur_inliers = inliers
            self.cur_inliers_count = inliers_count

    @timer(PROFILING)
    def create_binary_mask(self, boxes: np.ndarray, box_format: str) -> np.ndarray:
        """
        Create a mask from the given bounding boxes.
        """
        if self.h is None or self.w is None:
            logger.error("Reference frame is not set.")
            sys.exit(1)

        if box_format == 'four':
            boxes = four2xywh(boxes)

        mask = np.full((self.h, self.w), 255, dtype=np.uint8)
        for box in boxes:
            xc, yc, wb, hb = box
            wb += wb * self.mask_margin_ratio
            hb += hb * self.mask_margin_ratio
            x1, y1, x2, y2 = int(xc - wb / 2), int(yc - hb / 2), int(xc + wb / 2), int(yc + hb / 2)
            mask[max(0, y1):min(self.h, y2), max(0, x1):min(self.w, x2)] = 0

        return mask

    @timer(PROFILING)
    def warp_cur_frame(self) -> Union[np.ndarray, None]:
        """
        Warp the current frame to the reference frame using the current transformation matrix.
        """
        return self.warp_frame(self.cur_frame)

    def warp_frame(self, frame: np.ndarray) -> Union[np.ndarray, None]:
        """
        Warp the given frame to the reference frame using the current transformation matrix.
        """
        if frame is None:
            return None
        if self.w is None or self.h is None:
            logger.error("Reference frame is not set.")
            sys.exit(1)
        if self.cur_trans_matrix is None:
            logger.warning("Transformation matrix is None.")
            return frame
        if self.transformation_type == 'projective':
            return cv2.warpPerspective(frame, self.cur_trans_matrix, (self.w, self.h))
        elif self.transformation_type == 'affine':
            return cv2.warpAffine(frame, self.cur_trans_matrix, (self.w, self.h))

    def transform_cur_boxes(self, out_box_format: str = 'xywh') -> Union[np.ndarray, None]:
        """
        Warp the current bounding boxes to the reference frame using the current transformation matrix.
        """
        return self.transform_boxes(self.cur_boxes, self.cur_trans_matrix, self.cur_box_format, out_box_format)

    def transform_boxes(self, boxes: np.ndarray, trans_matrix: np.ndarray, in_box_format: str = 'xywh', out_box_format: str = 'xywh') -> Union[np.ndarray, None]:
        """
        Transform the provided bounding boxes using the provided transformation matrix.
        """
        if boxes is None:
            return None
        if trans_matrix is None:
            return boxes

        if in_box_format == 'xywh':
            boxes = xywh2four(boxes)

        boxes = np.array([boxes]).reshape(-1, 1, 2)
        if self.transformation_type == 'projective':
            boxes = cv2.perspectiveTransform(boxes, trans_matrix)
        elif self.transformation_type == 'affine':
            boxes = cv2.transform(boxes, trans_matrix)

        if out_box_format == 'xywh':
            return four2xywh(boxes.reshape(-1, 8))
        elif out_box_format == 'four':
            return boxes.reshape(-1, 8)

    def get_cur_frame(self) -> Union[np.ndarray, None]:
        """
        Get the current frame.
        """
        return self.cur_frame

    def get_cur_boxes(self) -> Union[np.ndarray, None]:
        """
        Get the current bounding boxes.
        """
        return self.cur_boxes

    def get_cur_trans_matrix(self) -> Union[np.ndarray, None]:
        """
        Get the current transformation matrix.
        """
        return self.cur_trans_matrix

    def get_basic_info(self) -> dict:
        """
        Get basic information about the Stabilizer.
        """
        return {
            'detector_name': self.detector_name,
            'matcher_name': self.matcher_name,
            'filter_type': self.filter_type,
            'transformation_type': self.transformation_type,
            'clahe': self.clahe,
            'mask_use': self.mask_use,
        }

    def _validate_arguments(self):
        """
        Validate the arguments provided during the initialization of the Stabilizer class.
        """
        if self.detector_name not in self.VALID_DETECTORS:
            raise ValueError(f"Invalid detector: {self.detector_name}. Choose from {self.VALID_DETECTORS}")
        if self.matcher_name not in self.VALID_MATCHERS:
            raise ValueError(f"Invalid matcher: {self.matcher_name}. Choose from {self.VALID_MATCHERS}")
        if self.filter_type not in self.VALID_FILTER_TYPES:
            raise ValueError(f"Invalid filter type: {self.filter_type}. Choose from {self.VALID_FILTER_TYPES}")
        if self.transformation_type not in self.VALID_TRANSFORMATION_TYPES:
            raise ValueError(f"Invalid transformation type: {self.transformation_type}. Choose from {self.VALID_TRANSFORMATION_TYPES}")
        if self.ransac_method not in self.VALID_RANSAC_METHODS_DICT.values():
            raise ValueError(f"Invalid RANSAC method: {self.ransac_method}. Choose from {self.VALID_RANSAC_METHODS_DICT.keys()}")
        if not (0.0 < self.downsample_ratio <= 1.0):
            raise ValueError("Invalid downsample_ratio. It should be in the range (0.0, 1.0]")
        if not (0 < self.max_features) and isinstance(self.max_features, int):
            raise ValueError("Invalid max_features. It should be greater than 0 and an integer")
        if not (1 <= self.ref_multiplier):
            raise ValueError("Invalid ref_multiplier. It should be greater than or equal to 1")
        if not (0.0 < self.filter_ratio <= 1.0):
            raise ValueError("Invalid filter_ratio. It should be in the range (0.0, 1.0]")
        if not (0 < self.ransac_max_iter) and isinstance(self.ransac_max_iter, int):
            raise ValueError("Invalid ransac_max_iter. It should be greater than 0 and an integer")
        if not (0.0 < self.ransac_epipolar_threshold):
            raise ValueError("Invalid ransac_epipolar_threshold. It should be greater than 0")
        if not (0.0 < self.ransac_confidence <= 1.0):
            raise ValueError("Invalid ransac_confidence. It should be in the range (0.0, 1.0]")
        if self.gpu and not cv2.cuda.getCudaEnabledDeviceCount():
            raise ValueError("GPU is enabled but no CUDA-enabled device was found")
