# -*- coding: utf-8 -*-
# Author: Robert Fonod (robert.fonod@ieee.org)

"""
version_check.py - Non-intrusive check for a newer stabilo release on PyPI.

The check is best-effort: it never raises, is silent on any failure or when offline,
caches the last result for 24 hours, and can be disabled via the
STABILO_DISABLE_UPDATE_CHECK environment variable.
"""

import json
import os
import platform
import threading
import time
import urllib.request
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/stabilo/json"
CACHE_TTL = 24 * 3600
FETCH_TIMEOUT = 2.0
ENV_OPT_OUT = "STABILO_DISABLE_UPDATE_CHECK"

_state = {'checked': False}
_lock = threading.Lock()


def _opted_out() -> bool:
    return bool(os.environ.get(ENV_OPT_OUT))


def _cache_dir() -> Path:
    """
    Resolve the platform-specific user cache directory for stabilo.
    """
    system = platform.system()
    if system == "Darwin":
        base = Path.home() / "Library" / "Caches"
    elif system == "Windows":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    else:
        base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return base / "stabilo"


def _cache_file() -> Path:
    return _cache_dir() / "update_check.json"


def _read_cache():
    try:
        with open(_cache_file(), "r") as f:
            return json.load(f)
    except Exception:
        return None


def _write_cache(latest: str) -> None:
    try:
        cache_dir = _cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        with open(_cache_file(), "w") as f:
            json.dump({"last_check": time.time(), "latest_version": latest}, f)
    except Exception:
        pass


def _parse_version(version: str):
    """
    Parse a version string into a comparable tuple; never raises.
    """
    parts = []
    for chunk in str(version).split("."):
        digits = ""
        for ch in chunk:
            if ch.isdigit():
                digits += ch
            else:
                break
        parts.append(int(digits) if digits else 0)
    return tuple(parts)


def _fetch_latest() -> str:
    with urllib.request.urlopen(PYPI_URL, timeout=FETCH_TIMEOUT) as response:
        data = json.load(response)
    return data["info"]["version"]


def _notify_if_newer(latest: str, logger) -> None:
    from stabilo import __version__

    if latest and _parse_version(latest) > _parse_version(__version__):
        message = (
            f"A newer stabilo version ({latest}) is available on PyPI (installed: {__version__}). "
            f"Upgrade with: pip install -U stabilo (set {ENV_OPT_OUT}=1 to silence)."
        )
        if logger is not None:
            logger.warning(message)


def _do_check(logger) -> None:
    try:
        cache = _read_cache()
        if cache and (time.time() - cache.get("last_check", 0)) < CACHE_TTL:
            latest = cache.get("latest_version")
        else:
            latest = _fetch_latest()
            _write_cache(latest)
        _notify_if_newer(latest, logger)
    except Exception:
        pass


def check_for_updates(logger=None, blocking: bool = False) -> None:
    """
    Warn (once) if a newer stabilo release is available on PyPI. Never raises.

    When the cache is fresh the comparison is done in-process; otherwise the network
    fetch runs in a daemon thread unless `blocking` is True.
    """
    if _opted_out():
        return
    if blocking:
        _do_check(logger)
    else:
        threading.Thread(target=_do_check, args=(logger,), daemon=True).start()


def check_for_updates_once(logger=None) -> None:
    """
    Run check_for_updates at most once per process.
    """
    with _lock:
        if _state['checked']:
            return
        _state['checked'] = True
    check_for_updates(logger)
