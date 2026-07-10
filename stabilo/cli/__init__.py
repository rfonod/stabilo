# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
stabilo command-line interface.

Provides the 'stabilo' console command with the subcommands:
  - stabilo video <input>  : stabilize a video with respect to a reference frame
  - stabilo tracks <input> : stabilize per-frame object annotations (tracks)
  - stabilo config <action>: inspect or copy the default configuration
"""

import argparse

from stabilo import __version__
from stabilo.utils import setup_logger
from stabilo.version_check import check_for_updates_once

from . import config, tracks, video


def main(argv=None):
    """
    Entry point for the 'stabilo' console command.
    """
    parser = argparse.ArgumentParser(
        prog="stabilo",
        description="Stabilize videos or per-frame object annotations with respect to a reference frame.",
    )
    parser.add_argument("--version", "-V", action="version", version=f"stabilo {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parsers = {
        "video": video.configure_parser(subparsers),
        "tracks": tracks.configure_parser(subparsers),
        "config": config.configure_parser(subparsers),
    }

    args = parser.parse_args(argv)

    if args.command in ("video", "tracks") and not (
        getattr(args, "save", False) or getattr(args, "viz", False) or getattr(args, "save_viz", False)
    ):
        parsers[args.command].error("At least one of --save, --viz, or --save-viz must be specified.")

    logger = setup_logger("stabilo.cli")
    check_for_updates_once(logger)
    args.func(args, logger)
