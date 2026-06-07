"""
Tests for shotpeen_gui.py.

Covers:
  - App initialisation (window size, required methods present)
  - main_menu renders without error
  - browse_file / browse_directory have parent= parameter
  - Utility methods: num_of_simulations, check_file_in_folder
  - preview_file path-validation guards (no actual matplotlib window opened)
  - All public dialog methods are callable
"""

import inspect
import os
import sys

import numpy as np
import pytest

import sys as _sys, os as _os

_sys.path.insert(0, _os.path.dirname(__file__))
from helpers import SAMPLE_DATASET

# Make shotpeen_gui importable (it lives one level above tests/)
sys.path.insert(
    0,
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..")),
)
# Also make src/peen-ml importable (shotpeen_gui imports from there)
sys.path.insert(
    0,
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "src", "peen-ml")),
)

import shotpeen_gui as gui


# ===========================================================================
# 1. App initialisation
# ===========================================================================


class TestAppInit:
    def test_window_geometry_set(self, tk_root):
        app = gui.App(tk_root)  # noqa: F841
        tk_root.update_idletasks()
        geo = tk_root.geometry()
        # geometry() returns "WxH+X+Y"; check at least the WxH prefix
        assert geo.startswith("1100x820"), f"Expected window to start with 1100x820, got {geo!r}"

    def test_window_size_attribute(self, tk_root):
        app = gui.App(tk_root)
        assert app.window_size == "1100x820"

    def test_dialog_size_attribute(self, tk_root):
        app = gui.App(tk_root)
        assert app.dialog_size == "1100x860"

    def test_required_methods_present(self, tk_root):
        app = gui.App(tk_root)
        for method in [
            "main_menu",
            "train_model_dialog",
            "load_model_dialog",
            "generate_dataset_dialog",
            "browse_file",
            "browse_directory",
            "preview_file",
            "preview_deformation",
            "num_of_simulations",
            "check_file_in_folder",
            "_build_native_gen_tab",
            "_build_gaussian_gen_tab",
            "_wire_generator",
        ]:
            assert callable(getattr(app, method, None)), f"App must have callable method: {method}"


# ===========================================================================
# 2. main_menu
# ===========================================================================


class TestMainMenu:
    def test_renders_without_error(self, tk_root):
        app = gui.App(tk_root)
        # Re-calling main_menu should clear and redraw cleanly
        app.main_menu()
        # If the window was not destroyed it succeeded
        assert tk_root.winfo_exists()

    def test_main_frame_has_children(self, tk_root):
        gui.App(tk_root)
        children = tk_root.winfo_children()
        assert len(children) > 0, "main_menu must add at least one widget to root"


# ===========================================================================
# 3. browse_file / browse_directory signatures
# ===========================================================================


class TestBrowseSignatures:
    def test_browse_file_accepts_parent_kwarg(self):
        sig = inspect.signature(gui.App.browse_file)
        assert "parent" in sig.parameters, "browse_file must have a parent= parameter for window focus management"

    def test_browse_directory_accepts_parent_kwarg(self):
        sig = inspect.signature(gui.App.browse_directory)
        assert "parent" in sig.parameters, "browse_directory must have a parent= parameter for window focus management"

    def test_browse_file_parent_defaults_none(self):
        sig = inspect.signature(gui.App.browse_file)
        assert sig.parameters["parent"].default is None

    def test_browse_directory_parent_defaults_none(self):
        sig = inspect.signature(gui.App.browse_directory)
        assert sig.parameters["parent"].default is None


# ===========================================================================
# 4. num_of_simulations
# ===========================================================================


def _make_tk_root():
    """Create a hidden Tk root, or skip the test if no display is available."""
    import tkinter as tk

    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("No display available (headless CI)")
    root.withdraw()
    return root


class TestNumOfSimulations:
    def test_counts_real_sample(self):
        root = _make_tk_root()
        app = gui.App(root)
        count = app.num_of_simulations(SAMPLE_DATASET)
        root.destroy()
        assert count == 31

    def test_empty_dir_returns_zero(self, tmp_path):
        root = _make_tk_root()
        app = gui.App(root)
        count = app.num_of_simulations(str(tmp_path))
        root.destroy()
        assert count == 0

    def test_ignores_non_simulation_folders(self, tmp_path):
        (tmp_path / "Simulation_0").mkdir()
        (tmp_path / "Simulation_1").mkdir()
        (tmp_path / "not_a_sim").mkdir()
        (tmp_path / "Simulation_abc").mkdir()
        root = _make_tk_root()
        app = gui.App(root)
        count = app.num_of_simulations(str(tmp_path))
        root.destroy()
        assert count == 2


# ===========================================================================
# 5. check_file_in_folder
# ===========================================================================


class TestCheckFileInFolder:
    def test_returns_true_when_file_exists(self, tmp_path, tk_root):
        f = tmp_path / "test.npy"
        f.write_bytes(b"\x00")
        app = gui.App(tk_root)
        assert app.check_file_in_folder(str(tmp_path), "test.npy") is True

    def test_returns_false_when_file_missing(self, tmp_path, tk_root):
        app = gui.App(tk_root)
        assert app.check_file_in_folder(str(tmp_path), "ghost.npy") is False

    def test_returns_false_for_nonexistent_folder(self, tk_root):
        app = gui.App(tk_root)
        assert app.check_file_in_folder("/no/such/folder", "file.npy") is False


# ===========================================================================
# 6. preview_file validation guards
# ===========================================================================


class TestPreviewFileValidation:
    """
    preview_file() must validate paths before calling matplotlib.
    These tests intercept messagebox calls to confirm the right guard fires
    without actually rendering any plot windows.
    """

    def _patch_msgbox(self, monkeypatch, calls):
        """Redirect messagebox.showerror / showwarning into *calls* list."""
        import tkinter.messagebox as mb

        monkeypatch.setattr(mb, "showerror", lambda *a, **kw: calls.append(("error", a)))
        monkeypatch.setattr(mb, "showwarning", lambda *a, **kw: calls.append(("warning", a)))

    def test_nonexistent_path_shows_error(self, tk_root, monkeypatch):
        calls = []
        self._patch_msgbox(monkeypatch, calls)
        app = gui.App(tk_root)
        app.preview_file("/does/not/exist")
        assert any(c[0] == "error" for c in calls)

    def test_file_not_directory_shows_error(self, tk_root, tmp_path, monkeypatch):
        calls = []
        self._patch_msgbox(monkeypatch, calls)
        f = tmp_path / "file.npy"
        f.write_bytes(b"\x00")
        app = gui.App(tk_root)
        app.preview_file(str(f))
        assert any(c[0] == "error" for c in calls)

    def test_empty_directory_shows_warning(self, tk_root, tmp_path, monkeypatch):
        calls = []
        self._patch_msgbox(monkeypatch, calls)
        app = gui.App(tk_root)
        app.preview_file(str(tmp_path))
        assert any(c[0] == "warning" for c in calls)

    def test_missing_checkerboard_shows_error(self, tk_root, tmp_path, monkeypatch):
        calls = []
        self._patch_msgbox(monkeypatch, calls)
        (tmp_path / "other_file.npy").write_bytes(b"\x00")
        app = gui.App(tk_root)
        app.preview_file(str(tmp_path))
        assert any(c[0] == "error" for c in calls)

    def test_valid_path_calls_run_preview(self, tk_root, tmp_path, monkeypatch):
        """A folder with checkerboard.npy should reach run_preview (not any guard)."""
        np.save(str(tmp_path / "checkerboard.npy"), np.zeros((5, 5)))
        previewed = []
        monkeypatch.setattr(gui.App, "run_preview", lambda self, p: previewed.append(p))
        app = gui.App(tk_root)
        app.preview_file(str(tmp_path))
        tk_root.update()  # flush the root.after(0, ...) callback onto the event queue
        assert previewed == [str(tmp_path)], "Valid path should call run_preview with the folder path"


# ===========================================================================
# 7. Dialog methods are callable (smoke tests — no window interaction)
# ===========================================================================


class TestDialogsCallable:
    def test_generate_dataset_dialog_callable(self, tk_root):
        app = gui.App(tk_root)
        assert callable(app.generate_dataset_dialog)

    def test_train_model_dialog_callable(self, tk_root):
        app = gui.App(tk_root)
        assert callable(app.train_model_dialog)

    def test_load_model_dialog_callable(self, tk_root):
        app = gui.App(tk_root)
        assert callable(app.load_model_dialog)
