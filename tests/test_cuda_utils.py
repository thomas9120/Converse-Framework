"""Tests for ``converse_framework/cuda_utils.py``.

Uses monkeypatching and temp directories since real NVIDIA DLLs are not
expected in CI.
"""

from __future__ import annotations

import os
import site
import sys
from pathlib import Path

import pytest

from converse_framework import cuda_utils


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_nvidia_tree(tmp_path: Path) -> Path:
    """Create a fake site-packages tree with NVIDIA DLL directories."""
    site_pkgs = tmp_path / "site-packages"
    site_pkgs.mkdir()

    # Create nvidia/cublas/bin with a DLL
    cublas = site_pkgs / "nvidia" / "cublas" / "bin"
    cublas.mkdir(parents=True)
    (cublas / "cublas64_11.dll").write_text("fake dll")

    # Create nvidia/cudnn/bin with a DLL
    cudnn = site_pkgs / "nvidia" / "cudnn" / "bin"
    cudnn.mkdir(parents=True)
    (cudnn / "cudnn64_8.dll").write_text("fake dll")

    # Create nvidia/cusparse/bin with a DLL
    cusparse = site_pkgs / "nvidia" / "cusparse" / "bin"
    cusparse.mkdir(parents=True)
    (cusparse / "cusparse64_11.dll").write_text("fake dll")

    # Create an empty nvidia/curand/bin (no DLLs — should NOT be discovered)
    curand = site_pkgs / "nvidia" / "curand" / "bin"
    curand.mkdir(parents=True)

    # Create a non-NVIDIA dir with a DLL (should NOT be matched)
    other = site_pkgs / "some_other" / "bin"
    other.mkdir(parents=True)
    (other / "other.dll").write_text("fake")

    return site_pkgs


# ---------------------------------------------------------------------------
# Tests: discover_nvidia_dll_dirs
# ---------------------------------------------------------------------------


class TestDiscoverNvidiaDllDirs:
    def test_non_windows_returns_empty(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = cuda_utils.discover_nvidia_dll_dirs()
        assert result == []

    def test_discovers_nvidia_dlls(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])

        dirs = cuda_utils.discover_nvidia_dll_dirs()

        # Should find 3 directories (cublas, cudnn, cusparse).
        # curand has no DLLs — excluded. other/ is not under nvidia/ — excluded.
        assert len(dirs) == 3

        names = {d.name for d in dirs}
        assert "bin" in names  # all three end in .../bin
        paths = {str(d.relative_to(fake_nvidia_tree)) for d in dirs}
        assert "nvidia\\cublas\\bin" in paths or "nvidia/cublas/bin" in paths
        assert "nvidia\\cudnn\\bin" in paths or "nvidia/cudnn/bin" in paths
        assert "nvidia\\cusparse\\bin" in paths or "nvidia/cusparse/bin" in paths

    def test_empty_when_no_site_packages(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", list)
        assert cuda_utils.discover_nvidia_dll_dirs() == []

    def test_deduplicates_identical_dirs(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        # Return same root twice
        monkeypatch.setattr(
            cuda_utils,
            "_get_search_roots",
            lambda: [fake_nvidia_tree, fake_nvidia_tree],
        )
        dirs = cuda_utils.discover_nvidia_dll_dirs()
        # Should still find 3 unique directories, not double
        assert len(dirs) == 3


# ---------------------------------------------------------------------------
# Tests: add_nvidia_dll_directories
# ---------------------------------------------------------------------------


class TestAddNvidiaDllDirectories:
    def test_non_windows_noop(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "linux")
        result = cuda_utils.add_nvidia_dll_directories()
        assert result == []

    def test_invokes_add_dll_directory(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])

        added: list[str] = []

        def fake_add_dll_directory(path: str) -> object:
            added.append(path)
            return object()  # unique handle

        monkeypatch.setattr(
            os, "add_dll_directory", fake_add_dll_directory, raising=False
        )

        handles = cuda_utils.add_nvidia_dll_directories()

        assert len(handles) == 3  # one per discovered dir
        assert len(added) == 3
        # All paths should contain "nvidia"
        assert all("nvidia" in p for p in added)

    def test_handles_failure_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])

        def failing_add(path: str) -> object:
            raise OSError(f"permission denied: {path}")

        monkeypatch.setattr(os, "add_dll_directory", failing_add, raising=False)

        # Should not raise — returns empty list
        handles = cuda_utils.add_nvidia_dll_directories()
        assert handles == []

    def test_prepends_discovered_dirs_to_path(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        """CTranslate2 resolves CUDA DLLs with a plain ``LoadLibrary`` that
        only searches PATH, so discovery must also extend PATH."""
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])
        monkeypatch.setattr(
            os, "add_dll_directory", lambda path: object(), raising=False
        )
        # No drive letter: ":" is the PATH separator on POSIX CI runners.
        monkeypatch.setenv("PATH", "existing-entry")

        cuda_utils.add_nvidia_dll_directories()

        entries = os.environ["PATH"].split(os.pathsep)
        assert entries[-1] == "existing-entry"
        prepended = entries[:-1]
        assert len(prepended) == 3
        assert all("nvidia" in entry for entry in prepended)
        # Discovered dirs come before pre-existing entries.
        assert str(fake_nvidia_tree / "nvidia" / "cublas" / "bin") in prepended

    def test_path_prepend_is_idempotent(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])
        monkeypatch.setattr(
            os, "add_dll_directory", lambda path: object(), raising=False
        )
        monkeypatch.setenv("PATH", "existing-entry")

        cuda_utils.add_nvidia_dll_directories()
        first = os.environ["PATH"]
        cuda_utils.add_nvidia_dll_directories()

        assert os.environ["PATH"] == first

    def test_path_prepend_happens_even_if_add_dll_directory_fails(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])

        def failing_add(path: str) -> object:
            raise OSError("nope")

        monkeypatch.setattr(os, "add_dll_directory", failing_add, raising=False)
        monkeypatch.setenv("PATH", "existing-entry")

        cuda_utils.add_nvidia_dll_directories()

        assert os.environ["PATH"] != "existing-entry"
        assert "nvidia" in os.environ["PATH"]


# ---------------------------------------------------------------------------
# Tests: format_nvidia_dll_diagnostic
# ---------------------------------------------------------------------------


class TestFormatNvidiaDllDiagnostic:
    def test_non_windows(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "linux")
        diag = cuda_utils.format_nvidia_dll_diagnostic()
        assert "Windows-only" in diag
        assert "Linux" in diag or "linux" in diag

    def test_windows_without_dlls(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", list)
        diag = cuda_utils.format_nvidia_dll_diagnostic()
        assert "Search roots (0)" in diag
        assert "(none found)" in diag

    def test_windows_with_dlls(
        self, monkeypatch: pytest.MonkeyPatch, fake_nvidia_tree: Path
    ):
        monkeypatch.setattr(sys, "platform", "win32")
        monkeypatch.setattr(cuda_utils, "_get_search_roots", lambda: [fake_nvidia_tree])

        diag = cuda_utils.format_nvidia_dll_diagnostic()
        assert "Search roots (1)" in diag
        assert "cublas64_11.dll" in diag
        assert "cudnn64_8.dll" in diag
        assert "cusparse64_11.dll" in diag


# ---------------------------------------------------------------------------
# Tests: provider integration
# ---------------------------------------------------------------------------


class TestProviderCudaIntegration:
    """Verify FasterWhisperASRProvider calls ``add_nvidia_dll_directories``
    during ``_ensure_model()`` when auto discovery is enabled."""

    def test_ensure_model_calls_cuda_helper(self, monkeypatch: pytest.MonkeyPatch):
        from converse_framework.providers.faster_whisper import (
            FasterWhisperASRProvider,
        )

        called = False
        handles: list[object] = []

        def fake_add() -> list[object]:
            nonlocal called
            called = True
            return handles

        monkeypatch.setattr(
            "converse_framework.cuda_utils.add_nvidia_dll_directories",
            fake_add,
        )

        # Mock faster_whisper module before it's imported inside _ensure_model
        import types

        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = lambda *a, **kw: object()
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        provider = FasterWhisperASRProvider({})
        import asyncio

        asyncio.run(provider.load())

        assert called, "add_nvidia_dll_directories should have been called"
        assert provider._cuda_dll_handles is handles

    def test_ensure_model_skips_when_disabled(self, monkeypatch: pytest.MonkeyPatch):
        from converse_framework.providers.faster_whisper import (
            FasterWhisperASRProvider,
        )

        called = False

        def fake_add() -> list[object]:
            nonlocal called
            called = True
            return []

        monkeypatch.setattr(
            "converse_framework.cuda_utils.add_nvidia_dll_directories",
            fake_add,
        )

        import types

        fake_fw = types.ModuleType("faster_whisper")
        fake_fw.WhisperModel = lambda *a, **kw: object()
        monkeypatch.setitem(sys.modules, "faster_whisper", fake_fw)

        provider = FasterWhisperASRProvider({"auto_cuda_dll_dirs": False})
        import asyncio

        asyncio.run(provider.load())

        assert not called, "CUDA helper should NOT be called when disabled"


# ---------------------------------------------------------------------------
# Test: _get_search_roots
# ---------------------------------------------------------------------------


class TestGetSearchRoots:
    def test_returns_site_packages_dirs(self, monkeypatch: pytest.MonkeyPatch):
        """At minimum site.getsitepackages() should return something."""
        roots = cuda_utils._get_search_roots()
        assert isinstance(roots, list)
        assert all(isinstance(r, Path) for r in roots)
        # At least one dir should exist
        assert all(r.exists() for r in roots)

    def test_deduplicates(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        p = tmp_path / "sp"
        p.mkdir()
        monkeypatch.setattr(site, "getsitepackages", lambda: [str(p), str(p)])
        monkeypatch.setattr(sys, "path", [])
        roots = cuda_utils._get_search_roots()
        assert len(roots) == 1  # deduplicated
