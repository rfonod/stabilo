# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
config.py - Inspect or copy the default stabilo configuration from the command line.

  stabilo config show           # print the default configuration to stdout
  stabilo config copy [--output PATH]  # write a copy of the default configuration for editing
"""

import shutil
from pathlib import Path

import stabilo

DEFAULT_CFG = Path(stabilo.__file__).resolve().parent / 'cfg' / 'default.yaml'


def configure_parser(subparsers):
    """
    Register the 'config' subcommand and its arguments.
    """
    parser = subparsers.add_parser(
        "config",
        help="inspect or copy the default stabilo configuration",
        description="Inspect the default stabilo configuration or copy it locally for editing.",
    )
    parser.add_argument(
        "action",
        choices=["show", "copy"],
        help="'show' prints the default configuration; 'copy' writes an editable copy locally",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="destination file for 'copy' [default: ./custom.yaml]",
    )
    parser.set_defaults(func=run)
    return parser


def run(args, logger):
    """
    Execute the 'config' subcommand.
    """
    if args.action == "show":
        print(DEFAULT_CFG.read_text())
        return

    dest = args.output or Path.cwd() / "custom.yaml"
    if dest.exists():
        logger.warning(f"File {dest} already exists. Overwriting it.")
    shutil.copyfile(DEFAULT_CFG, dest)
    logger.info(f"Wrote a copy of the default configuration to {dest}.")
    print(
        f"\nEdit {dest} and pass it to a stabilo command with --custom-config, e.g.:\n"
        f"    stabilo video <input> --save --custom-config {dest}\n"
        f"Values set in the file override the defaults; CLI flags override the file."
    )
