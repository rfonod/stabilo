"""
MkDocs hook that renders the stabilo command-line help into docs/cli.md at build time.

The page holds placeholders such as `<!-- cli-help: video -->`. Each is replaced by the output of
`stabilo video --help`, produced by calling the installed `stabilo.cli.main`, so the reference is
generated from the real parsers and cannot drift from them. COLUMNS is pinned while rendering so
argparse wraps the text identically on every machine.
"""

import contextlib
import io
import os
import re
from unittest import mock

from mkdocs.exceptions import PluginError

from stabilo.cli import main

PLACEHOLDER = re.compile(r"<!-- cli-help:(?P<command>[\w ]*)-->")
HELP_COLUMNS = "100"


def render_help(command: str) -> str:
    """
    Return the --help text of `stabilo <command>`, or of `stabilo` itself when command is empty.
    """
    buffer = io.StringIO()
    with (
        mock.patch.dict(os.environ, {"COLUMNS": HELP_COLUMNS}),
        contextlib.redirect_stdout(buffer),
        contextlib.suppress(SystemExit),
    ):
        main([*command.split(), "--help"])
    text = buffer.getvalue().rstrip()
    if not text:
        raise PluginError(f"'stabilo {command} --help' produced no output")
    return text


def on_page_markdown(markdown, page, config, files):
    """
    Replace every CLI help placeholder on the page with the corresponding help text.
    """
    return PLACEHOLDER.sub(lambda m: f"```text\n{render_help(m.group('command'))}\n```", markdown)
