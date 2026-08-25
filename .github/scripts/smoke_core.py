#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
End-to-end smoke test for a core-dependency-only install of stabilo.

Synthesizes a small video plus a matching bounding-box file, then drives both CLI
subcommands over them ('stabilo video' and 'stabilo tracks'), checking that the
stabilized video and the stabilized tracks file are produced. The point is to catch a
core install that cannot actually run, e.g. an optional dependency that silently became
mandatory, without needing pytest or the 4K sample clip in data/.

Usage:
    python .github/scripts/smoke_core.py               # run the smoke test
    python .github/scripts/smoke_core.py --core-only   # also assert matplotlib (extras) is absent
"""

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np

import stabilo

FRAMES = 12
WIDTH, HEIGHT = 640, 480
BOX = (90.0, 60.0)  # width, height of the synthetic 'vehicle' used as track and mask
OPTIONAL_MODULES = ('matplotlib',)


def synthesize_inputs(directory: Path) -> tuple[Path, Path]:
    """
    Write a textured video that pans by a known amount per frame, plus its box file.

    The box file follows the CLI defaults: frame number in column 0 and unnormalized
    YOLO boxes (x_c, y_c, w, h) in columns 2-5.
    """
    rng = np.random.default_rng(0)
    canvas = rng.integers(90, 150, (HEIGHT * 2, WIDTH * 2, 3), dtype=np.uint8)
    for _ in range(1200):
        x, y = int(rng.integers(0, WIDTH * 2)), int(rng.integers(0, HEIGHT * 2))
        size = int(rng.integers(4, 18))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        if rng.random() < 0.5:
            cv2.rectangle(canvas, (x, y), (x + size, y + size), color, -1)
        else:
            cv2.circle(canvas, (x, y), size // 2, color, -1)

    video_filepath = directory / 'smoke.mp4'
    boxes_filepath = directory / 'smoke.txt'
    writer = cv2.VideoWriter(str(video_filepath), cv2.VideoWriter_fourcc(*'mp4v'), 10, (WIDTH, HEIGHT))
    if not writer.isOpened():
        sys.exit('could not open a cv2.VideoWriter; the OpenCV build cannot encode mp4v')

    rows = []
    for frame_num in range(FRAMES):
        x0, y0 = 200 + 6 * frame_num, 160 + 4 * frame_num
        frame = canvas[y0 : y0 + HEIGHT, x0 : x0 + WIDTH].copy()
        x_c, y_c = 160.0 + 12.0 * frame_num, 240.0
        top_left = (int(x_c - BOX[0] / 2), int(y_c - BOX[1] / 2))
        bottom_right = (int(x_c + BOX[0] / 2), int(y_c + BOX[1] / 2))
        cv2.rectangle(frame, top_left, bottom_right, (20, 20, 200), -1)
        writer.write(frame)
        rows.append(f'{frame_num},1,{x_c},{y_c},{BOX[0]},{BOX[1]}')
    writer.release()

    boxes_filepath.write_text('\n'.join(rows) + '\n')
    return video_filepath, boxes_filepath


def run_cli(subcommand: str, video_filepath: Path, output_path: Path) -> None:
    """
    Run one stabilo subcommand and fail loudly if it does not exit cleanly.
    """
    executable = shutil.which('stabilo')
    if executable is None:
        sys.exit("the 'stabilo' console command is not on PATH; is the package installed?")
    command = [executable, subcommand, str(video_filepath), '--save', '--output', str(output_path)]
    print(f'\n$ {" ".join(command)}', flush=True)
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False, timeout=300)
    except subprocess.TimeoutExpired:
        sys.exit(f"'stabilo {subcommand}' did not finish within 300s")
    print(result.stdout, end='')
    print(result.stderr, end='', file=sys.stderr)
    if result.returncode != 0:
        sys.exit(f"'stabilo {subcommand}' exited with {result.returncode}")


def count_frames(video_filepath: Path) -> int:
    """
    Count decodable frames, rather than trusting the container's CAP_PROP_FRAME_COUNT metadata.
    """
    capture = cv2.VideoCapture(str(video_filepath))
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    return frames


def check_core_only() -> None:
    """
    Assert the 'extras' dependency is absent, i.e. this really is a core-only install.
    """
    present = [name for name in OPTIONAL_MODULES if importlib.util.find_spec(name) is not None]
    if present:
        sys.exit(f'not a core-only environment: {present} importable; install with "pip install ." only')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--core-only', action='store_true', help='also assert matplotlib (extras) is not installed')
    args = parser.parse_args()

    if args.core_only:
        check_core_only()

    os.environ['STABILO_DISABLE_UPDATE_CHECK'] = '1'

    print(f'stabilo {stabilo.__version__} / OpenCV {cv2.__version__} / NumPy {np.__version__}')
    stabilo.Stabilizer()

    with tempfile.TemporaryDirectory() as tmp:
        directory = Path(tmp)
        video_filepath, boxes_filepath = synthesize_inputs(directory)
        output_path = directory / 'out'

        run_cli('video', video_filepath, output_path)
        stabilized_videos = list(output_path.glob(f'{video_filepath.stem}_stab.*'))
        if not stabilized_videos:
            sys.exit(f'no stabilized video written to {output_path}')
        written = count_frames(stabilized_videos[0])
        if written != FRAMES:
            sys.exit(f'{stabilized_videos[0].name} holds {written} frames, expected {FRAMES}')

        run_cli('tracks', video_filepath, output_path)
        stabilized_tracks = output_path / f'{video_filepath.stem}_stab.txt'
        if not stabilized_tracks.exists():
            sys.exit(f'no stabilized tracks written to {stabilized_tracks}')

        before = np.loadtxt(boxes_filepath, delimiter=',')
        after = np.loadtxt(stabilized_tracks, delimiter=',')
        if after.shape != before.shape:
            sys.exit(f'stabilized tracks have shape {after.shape}, expected {before.shape}')
        if np.isnan(after).any():
            sys.exit('stabilized tracks contain NaN: the run did not produce a box for every input row')
        shift = np.abs(after[:, 2:4] - before[:, 2:4]).max()
        if shift < 1.0:
            sys.exit(f'stabilized boxes moved at most {shift:.3f} px; stabilization did nothing')

        print(
            f'\nOK: {stabilized_videos[0].name} holds {written} frames, '
            f'{len(after)} track rows moved up to {shift:.1f} px'
        )


if __name__ == '__main__':
    main()
