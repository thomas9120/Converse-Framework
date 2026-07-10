"""CUDA DLL discovery helpers for Windows NVIDIA wheel installations.

Packages like ``nvidia-cublas-cu12`` install DLLs under
``site-packages/nvidia/<package>/bin/``, but CTranslate2 and other C
extension libraries may not search those directories automatically.
This module discovers them and adds them to the DLL search path via
``os.add_dll_directory()`` and by prepending them to ``PATH`` (needed for
libraries that resolve DLLs with a plain ``LoadLibrary`` at runtime, such
as CTranslate2).  Windows-only.

Usage::

    from converse_framework.cuda_utils import add_nvidia_dll_directories

    handles = add_nvidia_dll_directories()
    # Keep ``handles`` alive for the lifetime of the process.
    # Handles are released when they go out of scope / are garbage collected.
"""

from __future__ import annotations

import logging
import os
import site
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Known NVIDIA wheel package subdirectories that may contain DLLs.
_NVIDIA_BIN_PATTERNS = (
    "nvidia/cublas/bin",
    "nvidia/cudnn/bin",
    "nvidia/cusparse/bin",
    "nvidia/cusolver/bin",
    "nvidia/curand/bin",
)

# Tolerated suffixes for "looks like a CUDA DLL".
_CUDA_DLL_SUFFIXES = (".dll",)


def _get_search_roots() -> list[Path]:
    """Collect directories to search for NVIDIA wheel installations.

    Returns:
        Deduplicated list of :class:`Path` objects.
    """
    seen: set[Path] = set()
    roots: list[Path] = []

    for sp in site.getsitepackages():
        p = Path(sp).resolve()
        if p not in seen:
            seen.add(p)
            roots.append(p)

    # Also scan sys.path entries that look like site-packages dirs.
    for entry in sys.path:
        p = Path(entry).resolve()
        if p not in seen and p.name in ("site-packages", "dist-packages"):
            seen.add(p)
            roots.append(p)

    return roots


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def discover_nvidia_dll_dirs() -> list[Path]:
    """Search known site-packages for NVIDIA wheel DLL directories.

    Returns:
        List of :class:`Path` objects pointing to directories that contain
        at least one ``.dll`` file.  Empty if none are found.
    """
    if sys.platform != "win32":
        return []

    found: list[Path] = []
    seen: set[Path] = set()

    for root in _get_search_roots():
        for pattern in _NVIDIA_BIN_PATTERNS:
            candidate = root / pattern
            try:
                resolved = candidate.resolve(strict=False)
            except (OSError, RuntimeError):
                continue
            if resolved in seen:
                continue
            if not resolved.is_dir():
                continue
            seen.add(resolved)
            # Check if there is at least one .dll file.
            try:
                has_dll = any(
                    f.suffix.lower() in _CUDA_DLL_SUFFIXES for f in resolved.iterdir()
                )
            except (OSError, PermissionError):
                has_dll = False
            if has_dll:
                found.append(resolved)
                logger.debug("Discovered NVIDIA DLL dir: %s", resolved)

    return found


def add_nvidia_dll_directories() -> list[object]:
    """Discover and register NVIDIA DLL directories via ``os.add_dll_directory()``.

    Each returned handle keeps the directory in the DLL search path for the
    lifetime of the handle object.  Callers should keep the returned list
    alive until shutdown.

    The discovered directories are also prepended to ``os.environ["PATH"]``:
    ``os.add_dll_directory()`` only affects DLLs loaded through Python's
    import machinery, but native libraries such as CTranslate2 resolve CUDA
    DLLs (e.g. ``cublas64_12.dll``) at inference time with a plain
    ``LoadLibrary`` call that searches only the application directory and
    ``PATH``.

    Returns:
        List of handles from ``os.add_dll_directory()`` (one per discovered
        directory).  Empty if no directories are found or not on Windows.
    """
    if sys.platform != "win32":
        logger.debug("add_nvidia_dll_directories: not on Windows, skipping.")
        return []

    dirs = discover_nvidia_dll_dirs()
    handles: list[object] = []
    for d in dirs:
        try:
            handle = os.add_dll_directory(str(d))
            handles.append(handle)
            logger.info("Added DLL directory: %s", d)
        except (OSError, RuntimeError) as exc:
            logger.warning("Failed to add DLL directory %s: %s", d, exc)
    _prepend_to_path(dirs)
    return handles


def _prepend_to_path(dirs: list[Path]) -> None:
    """Prepend ``dirs`` to ``os.environ["PATH"]``, skipping entries already
    present (case-insensitive, as Windows paths are)."""
    if not dirs:
        return
    current = os.environ.get("PATH", "")
    existing = {entry.lower() for entry in current.split(os.pathsep) if entry}
    new_entries = [str(d) for d in dirs if str(d).lower() not in existing]
    if not new_entries:
        return
    os.environ["PATH"] = os.pathsep.join(
        new_entries + ([current] if current else [])
    )
    for entry in new_entries:
        logger.info("Prepended to PATH: %s", entry)


def format_nvidia_dll_diagnostic() -> str:
    """Return a human-readable diagnostic string for NVIDIA DLL discovery.

    Intended for log output or error messages to help users debug CUDA
    setup issues on Windows.

    Returns:
        Multiline string.
    """
    lines: list[str] = []
    lines.append("NVIDIA DLL discovery diagnostic")
    lines.append(f"  Platform: {sys.platform}")
    lines.append(f"  Python: {sys.version}")

    if sys.platform != "win32":
        lines.append("  CUDA DLL search is Windows-only. Skipping.")
        return "\n".join(lines)

    roots = _get_search_roots()
    lines.append(f"  Search roots ({len(roots)}):")
    for r in roots:
        lines.append(f"    - {r}")

    discovered = discover_nvidia_dll_dirs()
    lines.append(f"  Discovered DLL dirs ({len(discovered)}):")
    if not discovered:
        lines.append("    (none found)")
    else:
        for d in discovered:
            try:
                dlls = [f.name for f in d.iterdir() if f.suffix.lower() == ".dll"]
            except (OSError, PermissionError):
                dlls = ["<unreadable>"]
            lines.append(f"    - {d}")
            for name in dlls[:5]:
                lines.append(f"      -> {name}")
            if len(dlls) > 5:
                lines.append(f"      ... and {len(dlls) - 5} more")

    return "\n".join(lines)
