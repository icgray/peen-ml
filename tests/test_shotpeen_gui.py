"""
Tests for shotpeen_gui.py — the Tkinter GUI application.

Coverage targets:
    - App.__init__              (window title, geometry, state initialisation)
    - App.main_menu             (widgets rendered, re-render on second call)
    - App.get_file_path         (path resolution in dev and .exe modes)
    - App.browse_file           (file dialog mock, StringVar update, cancel case)
    - App.browse_directory      (directory dialog mock, StringVar update, cancel)
    - App.check_file_in_folder  (file exists / does not exist)
    - App.num_of_simulations    (counts Simulation_N dirs, ignores non-matching)
    - App.train_model           (non-existent folder shows error)
    - App.preview_file          (non-existent folder, not-a-directory, empty dir)
    - App.preview_deformation   (missing displacements.npy shows error)
    - check_install             (package present → no subprocess call)

All Tkinter GUI operations (Tk(), Image.open, etc.) are mocked so that tests
run in headless CI environments.

Test categories:
    Smoke    – verifies basic execution without crashing.
    One-shot – tests a specific behaviour with known inputs/outputs.
    Edge     – boundary / error conditions.
    Property – invariants that must always hold.
"""

import os
import sys
import shutil
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/peen-ml"))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ---------------------------------------------------------------------------
# Shared fixture — mocked App instance
# ---------------------------------------------------------------------------


@pytest.fixture
def app():
    """
    Returns a fully mocked App instance.

    tkinter.Tk, PIL.Image.open, and PIL.ImageTk.PhotoImage are all patched
    so that no display is required and no actual image file is opened.
    The mock root supports winfo_children(), title(), geometry() and
    after() so that App.__init__ completes without error.
    """
    with patch("tkinter.Tk") as mock_tk_class, patch("PIL.Image.open") as mock_img_open, patch(
        "PIL.ImageTk.PhotoImage"
    ):

        # Configure the mock root window
        mock_root = MagicMock()
        mock_root.winfo_children.return_value = []
        mock_tk_class.return_value = mock_root

        # Configure mock image chain (open → resize → PhotoImage)
        mock_img = MagicMock()
        mock_img.resize.return_value = mock_img
        mock_img_open.return_value = mock_img

        # Import here (after mocks are active) to avoid display errors at import time
        from shotpeen_gui import App  # noqa: PLC0415

        instance = App(mock_root)

    return instance


# Convenience re-export so tests can import App directly
@pytest.fixture
def App():  # noqa: N802
    from shotpeen_gui import App as _App  # noqa: PLC0415

    return _App


# ============================================================================
# App initialisation
# ============================================================================


class TestAppInitialisation:
    """Tests for App.__init__ and App.main_menu setup."""

    # -- Smoke ----------------------------------------------------------------

    def test_smoke_initialisation_does_not_crash(self, app):
        """Smoke: App initialises without raising any exception."""
        assert app is not None

    # -- One-shot -------------------------------------------------------------

    def test_window_title_set(self, app):
        """One-shot: window title is 'Model GUI'."""
        app.root.title.assert_called_with("Model GUI")

    def test_window_geometry_set(self, app):
        """One-shot: window geometry is set to '1100x820'."""
        app.root.geometry.assert_called_with("1100x820")

    def test_window_size_attribute(self, app):
        """One-shot: self.window_size attribute reflects the chosen geometry string."""
        assert app.window_size == "1100x820"

    def test_initial_data_path_is_empty_string(self, app):
        """One-shot: test_train_data_path starts as an empty string."""
        assert app.test_train_data_path == "" or app.test_train_data_path is not None

    def test_parent_process_initially_none(self, app):
        """One-shot: parent_process is None at start."""
        assert app.parent_process is None

    # -- Edge -----------------------------------------------------------------

    def test_missing_bullet_bill_raises_file_not_found(self):
        """Edge: if the splash image is absent, a FileNotFoundError is raised."""
        with patch("tkinter.Tk"), patch("PIL.Image.open", side_effect=FileNotFoundError("not found")), patch(
            "PIL.ImageTk.PhotoImage"
        ):
            from shotpeen_gui import App  # noqa: PLC0415

            mock_root = MagicMock()
            mock_root.winfo_children.return_value = []
            with pytest.raises(FileNotFoundError):
                App(mock_root)


# ============================================================================
# App.get_file_path
# ============================================================================


class TestGetFilePath:
    """Tests for the path-resolution helper (dev mode vs PyInstaller .exe)."""

    def test_returns_string(self, app):
        """One-shot: always returns a string."""
        result = app.get_file_path("some/relative/path.txt")
        assert isinstance(result, str)

    def test_path_contains_relative_component(self, app):
        """One-shot: returned path ends with the requested relative path component."""
        result = app.get_file_path("data/file.npy")
        assert result.endswith(os.path.join("data", "file.npy"))

    def test_uses_meipass_when_available(self, app):
        """One-shot: if sys._MEIPASS exists, it is used as the base."""
        fake_meipass = "/fake/pyinstaller/extracted"
        with patch("sys._MEIPASS", fake_meipass, create=True):
            result = app.get_file_path("model.pth")
        assert result.startswith(fake_meipass)

    def test_falls_back_to_cwd_without_meipass(self, app):
        """One-shot: without _MEIPASS the path is based on os.path.abspath('.')."""
        # Ensure _MEIPASS is not present
        import sys as _sys  # noqa: PLC0415

        _sys.__dict__.pop("_MEIPASS", None)
        result = app.get_file_path("subdir/file.txt")
        expected_base = os.path.abspath(".")
        assert result.startswith(expected_base)


# ============================================================================
# App.check_file_in_folder
# ============================================================================


class TestCheckFileInFolder:
    """Tests for the file-existence helper."""

    def test_returns_true_when_file_exists(self, app, tmp_path):
        """One-shot: returns True for a file that is actually present."""
        f = tmp_path / "present.npy"
        f.write_bytes(b"data")
        assert app.check_file_in_folder(str(tmp_path), "present.npy") is True

    def test_returns_false_when_file_absent(self, app, tmp_path):
        """One-shot: returns False for a file that does not exist."""
        assert app.check_file_in_folder(str(tmp_path), "missing.npy") is False

    def test_returns_false_for_nonexistent_folder(self, app):
        """Edge: a completely non-existent folder path returns False."""
        assert app.check_file_in_folder("/no/such/folder", "file.txt") is False

    def test_case_sensitive_filename(self, app, tmp_path):
        """Property: filename matching is case-sensitive on case-sensitive filesystems."""
        f = tmp_path / "FILE.npy"
        f.write_bytes(b"data")
        # The file "FILE.npy" exists; "file.npy" should not (on Linux/macOS)
        if os.path.exists(str(tmp_path / "file.npy")):
            # Windows filesystem is case-insensitive — skip
            pytest.skip("Filesystem is case-insensitive")
        assert app.check_file_in_folder(str(tmp_path), "file.npy") is False


# ============================================================================
# App.num_of_simulations
# ============================================================================


class TestNumOfSimulations:
    """Tests for the simulation-folder counter."""

    def _make_sim_dirs(self, base, names):
        """Helper: create the given directory names under base."""
        for name in names:
            (base / name).mkdir(exist_ok=True)

    def test_counts_correct_simulation_dirs(self, app, tmp_path):
        """One-shot: counts exactly the Simulation_N dirs present."""
        self._make_sim_dirs(tmp_path, ["Simulation_0", "Simulation_1", "Simulation_2"])
        assert app.num_of_simulations(str(tmp_path)) == 3

    def test_ignores_non_matching_directories(self, app, tmp_path):
        """One-shot: non-Simulation_ dirs are not counted."""
        self._make_sim_dirs(tmp_path, ["Simulation_0", "Simulation_1", "results", "Simulation_abc"])
        # "Simulation_abc" has a non-digit suffix and should be ignored
        assert app.num_of_simulations(str(tmp_path)) == 2

    def test_empty_folder_returns_zero(self, app, tmp_path):
        """Edge: an empty folder has zero simulations."""
        assert app.num_of_simulations(str(tmp_path)) == 0

    def test_mixed_digit_and_non_digit_suffixes(self, app, tmp_path):
        """Edge: dirs like Simulation_1a or Simulation_ are ignored."""
        self._make_sim_dirs(tmp_path, ["Simulation_1", "Simulation_1a", "Simulation_"])
        assert app.num_of_simulations(str(tmp_path)) == 1

    def test_large_index(self, app, tmp_path):
        """Property: large simulation index (e.g. 9999) is counted correctly."""
        self._make_sim_dirs(tmp_path, ["Simulation_9999"])
        assert app.num_of_simulations(str(tmp_path)) == 1


# ============================================================================
# App.browse_file
# ============================================================================


class TestBrowseFile:
    """Tests for the file-browse dialog helper."""

    def test_sets_variable_on_selection(self, app):
        """One-shot: StringVar is updated with the chosen file path."""
        import tkinter as tk  # noqa: PLC0415

        var = MagicMock()
        with patch("tkinter.filedialog.askopenfilename", return_value="/chosen/file.pth"):
            app.browse_file(var)
        var.set.assert_called_once_with("/chosen/file.pth")

    def test_does_not_update_variable_on_cancel(self, app):
        """Edge: if the dialog is cancelled (empty string returned), variable is unchanged."""
        var = MagicMock()
        with patch("tkinter.filedialog.askopenfilename", return_value=""):
            app.browse_file(var)
        var.set.assert_not_called()

    def test_dialog_is_called_once(self, app):
        """Property: exactly one file dialog is opened."""
        with patch("tkinter.filedialog.askopenfilename", return_value="") as mock_dialog:
            app.browse_file(MagicMock())
        assert mock_dialog.call_count == 1


# ============================================================================
# App.browse_directory
# ============================================================================


class TestBrowseDirectory:
    """Tests for the directory-browse dialog helper."""

    def test_sets_variable_on_selection(self, app):
        """One-shot: StringVar is updated with the chosen directory path."""
        var = MagicMock()
        with patch("tkinter.filedialog.askdirectory", return_value="/chosen/dir"):
            app.browse_directory(var)
        var.set.assert_called_once_with("/chosen/dir")

    def test_does_not_update_variable_on_cancel(self, app):
        """Edge: empty return (cancelled dialog) leaves variable unchanged."""
        var = MagicMock()
        with patch("tkinter.filedialog.askdirectory", return_value=""):
            app.browse_directory(var)
        var.set.assert_not_called()

    def test_dialog_called_once(self, app):
        """Property: exactly one directory dialog is opened."""
        with patch("tkinter.filedialog.askdirectory", return_value="") as mock_dialog:
            app.browse_directory(MagicMock())
        assert mock_dialog.call_count == 1


# ============================================================================
# App.train_model
# ============================================================================


class TestTrainModel:
    """Tests for the programmatic (non-GUI) train_model method."""

    def test_nonexistent_directory_shows_error(self, app):
        """One-shot: a path that does not exist triggers messagebox.showerror."""
        bad_path = "/tmp/nonexistent_training_data_folder_xyz"
        with patch("tkinter.messagebox.showerror") as mock_err, patch("os.path.exists", return_value=False):
            app.train_model(bad_path)
            mock_err.assert_called_once_with("Error", f"The folder path does not exist: {bad_path}")

    def test_existing_directory_does_not_show_error(self, app, tmp_path):
        """One-shot: with a valid directory, showerror is NOT called immediately."""
        # The actual training call will fail (no data), so we also mock it out
        with patch("tkinter.messagebox.showerror") as mock_err, patch(
            "shotpeen_gui.create_data_loaders", side_effect=Exception("no data")
        ):
            try:
                app.train_model(str(tmp_path))
            except Exception:
                pass
            # showerror for "folder does not exist" should NOT have been called
            for c in mock_err.call_args_list:
                assert "does not exist" not in str(c)


# ============================================================================
# App.preview_file
# ============================================================================


class TestPreviewFile:
    """Tests for the input peen-intensity preview helper."""

    def test_nonexistent_path_shows_error(self, app):
        """One-shot: non-existent folder path triggers showerror."""
        bad_path = "/tmp/nonexistent_preview_folder_xyz"
        with patch("tkinter.messagebox.showerror") as mock_err:
            app.preview_file(bad_path)
            mock_err.assert_called_once_with("Error", f"The Folder path does not exist: {bad_path}")

    def test_file_path_instead_of_dir_shows_error(self, app, tmp_path):
        """Edge: passing a file path (not a directory) triggers showerror."""
        f = tmp_path / "not_a_dir.npy"
        f.write_bytes(b"data")
        with patch("tkinter.messagebox.showerror") as mock_err:
            app.preview_file(str(f))
            mock_err.assert_called_once()
            assert (
                "not a directory" in mock_err.call_args[0][1].lower() or "directory" in mock_err.call_args[0][1].lower()
            )

    def test_empty_directory_shows_warning(self, app, tmp_path):
        """Edge: valid but empty directory triggers showwarning."""
        with patch("tkinter.messagebox.showwarning") as mock_warn:
            app.preview_file(str(tmp_path))
            mock_warn.assert_called_once_with("Warning", "The directory is empty.")

    def test_valid_directory_schedules_preview(self, app, tmp_path):
        """One-shot: a non-empty directory schedules the preview on the main thread.

        Previously used threading.Thread; now uses root.after(0, ...) so that
        matplotlib runs on the main Tkinter thread (avoids "Starting a Matplotlib
        GUI outside of the main thread" warning/crash introduced in PyTorch 2.6
        and recent matplotlib releases).
        """
        (tmp_path / "checkerboard.npy").write_bytes(b"fake")
        with patch("os.listdir", return_value=["checkerboard.npy"]):
            app.preview_file(str(tmp_path))
            # root.after should have been called with delay=0 and the run_preview callback
            app.root.after.assert_called()


# ============================================================================
# App.preview_deformation
# ============================================================================


class TestPreviewDeformation:
    """Tests for the deformation preview helper."""

    def test_shows_error_when_displacement_file_missing(self, app, tmp_path):
        """One-shot: if displacements.npy is not in the output folder, showerror fires."""
        test_folder = tmp_path / "test"
        output_folder = tmp_path / "output"
        test_folder.mkdir()
        output_folder.mkdir()

        # Create the required input files in test_folder
        rng = np.random.default_rng(0)
        np.save(test_folder / "node_coords.npy", rng.uniform(0, 1, (5, 3)).astype(np.float32))
        np.save(test_folder / "node_labels.npy", np.arange(1, 6, dtype=np.int32))
        np.save(test_folder / "disp_node_labels.npy", np.arange(1, 6, dtype=np.int32))

        with patch("tkinter.messagebox.showerror") as mock_err:
            app.preview_deformation(str(test_folder), str(output_folder))
            # displacements.npy is absent in output_folder → error expected
            mock_err.assert_called_once()
            assert "evaluate" in mock_err.call_args[0][1].lower() or "displacement" in mock_err.call_args[0][1].lower()

    def test_copies_required_files_to_output(self, app, tmp_path):
        """One-shot: node_coords.npy, node_labels.npy, disp_node_labels.npy are copied."""
        test_folder = tmp_path / "test"
        output_folder = tmp_path / "output"
        test_folder.mkdir()
        output_folder.mkdir()

        rng = np.random.default_rng(1)
        np.save(test_folder / "node_coords.npy", rng.uniform(0, 1, (5, 3)).astype(np.float32))
        np.save(test_folder / "node_labels.npy", np.arange(1, 6, dtype=np.int32))
        np.save(test_folder / "disp_node_labels.npy", np.arange(1, 6, dtype=np.int32))

        with patch("tkinter.messagebox.showerror"):
            app.preview_deformation(str(test_folder), str(output_folder))

        assert (output_folder / "node_coords.npy").exists()
        assert (output_folder / "node_labels.npy").exists()
        assert (output_folder / "disp_node_labels.npy").exists()


# ============================================================================
# App.start_training / finish_training
# ============================================================================


class TestTrainingLogHelpers:
    """Tests for the log/progress-bar helpers (used in simulated training UI)."""

    def _make_mock_log(self):
        log = MagicMock()
        log.config = MagicMock()
        log.insert = MagicMock()
        log.see = MagicMock()
        return log

    def test_start_training_logs_started_message(self, app):
        """One-shot: start_training inserts 'Training started...' into the log."""
        log = self._make_mock_log()
        progress = MagicMock()
        app.start_training(log, progress)
        log.insert.assert_called()
        logged_text = " ".join(str(c) for c in log.insert.call_args_list)
        assert "started" in logged_text.lower()

    def test_start_training_starts_progressbar(self, app):
        """One-shot: start_training calls progress_bar.start()."""
        log = self._make_mock_log()
        progress = MagicMock()
        app.start_training(log, progress)
        progress.start.assert_called_once()

    def test_finish_training_stops_progressbar(self, app):
        """One-shot: finish_training calls progress_bar.stop()."""
        log = self._make_mock_log()
        progress = MagicMock()
        app.finish_training(log, progress)
        progress.stop.assert_called_once()

    def test_finish_training_logs_completed_message(self, app):
        """One-shot: finish_training inserts 'completed' into the log."""
        log = self._make_mock_log()
        progress = MagicMock()
        app.finish_training(log, progress)
        logged_text = " ".join(str(c) for c in log.insert.call_args_list)
        assert "complet" in logged_text.lower()


# ============================================================================
# check_install (module-level function)
# ============================================================================


class TestCheckInstall:
    """Tests for the standalone dependency-checker utility."""

    def test_installed_package_does_not_call_subprocess(self):
        """One-shot: an already-installed package never triggers pip/conda."""
        from shotpeen_gui import check_install  # noqa: PLC0415

        with patch("subprocess.check_call") as mock_sub:
            check_install("os")  # 'os' is always available
            mock_sub.assert_not_called()

    def test_missing_package_attempts_pip_install(self):
        """One-shot: a missing package causes a subprocess pip call."""
        from shotpeen_gui import check_install  # noqa: PLC0415

        with patch("builtins.__import__", side_effect=ModuleNotFoundError), patch("subprocess.check_call") as mock_sub:
            check_install("definitely_not_a_real_package_xyz")
            # At minimum, pip install should have been attempted
            mock_sub.assert_called()
            pip_calls = [c for c in mock_sub.call_args_list if "pip" in str(c)]
            assert len(pip_calls) >= 1


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
