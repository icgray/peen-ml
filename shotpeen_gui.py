"""
This is the main calling and GUI script for the shot peening backend.
This script allows lay users to train and test shot peening models,
and it serves as a usage example of calling the trained models.

Workflow (from README):
- The application presents a graphical user interface (GUI) where the user can
     choose to train a new model or load an existing model.
- If the user selects "Train Model," the application opens a dialog to
     configure the training, including selecting input files
     (training and testing data) and tracking the
     training progress via a log and progress bar.
- If the user selects "Load Model," the application allows them to load an
     existing model and related files (model file, peen intensity folder, output
     path), with options to preview the input and predicted deformation.
- The script also checks and installs any required dependencies that are
     missing.

Dependencies:
- requests>=2.25.1
- numpy>=1.20.0
- matplotlib>=3.4.0
- pandas>=1.3.0
- pytorch>=1.9.0
- tkinter
- pillow

Author:
- Harshavardhan Raje
"""

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import sys
import subprocess
import shutil
import os
import threading
import torch
from PIL import Image, ImageTk

# Append src folder to path such that the called python files can be called.
sys.path.append(os.path.join(os.path.dirname(__file__), "src", "peen-ml"))
# Deviating from PEP8 to make sure that this script can call the backend
from model import train_model, create_data_loaders, create_model  # noqa: E402  # pylint: disable=wrong-import-position
from model import train_save_gui, infer_dataset_shape  # pylint: disable=wrong-import-position
from model import train_save_conv_gui, train_save_siren_gui  # pylint: disable=wrong-import-position
from model import load_and_evaluate_model_gui  # pylint: disable=wrong-import-position
from model import curved_surface_inference  # pylint: disable=wrong-import-position
from data_viz import visualize_checkerboard, visualize_all  # pylint: disable=wrong-import-position

try:
    from data_viz import visualize_stl_deformation  # pylint: disable=wrong-import-position

    _STL_VIZ_OK = True
except ImportError:
    _STL_VIZ_OK = False


# ---------------------------------------------------------------------------
# Shared style constants
# ---------------------------------------------------------------------------
STEP_FONT = ("Arial", 11, "bold")
BODY_FONT = ("Arial", 10)
HINT_FONT = ("Arial", 9)
STEP_COLOR = "#1a5276"  # dark blue step headings
HINT_COLOR = "#555555"  # gray hint text
OK_COLOR = "#1a7a4a"  # green for success status
ERR_COLOR = "#c0392b"  # red for error status
INFO_COLOR = "#2471a3"  # mid-blue for detected info


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------


class ToolTip:
    """
    Lightweight hover tooltip for any tkinter widget.

    Displays a small floating label near the mouse cursor when the pointer
    enters *widget*, and destroys it when the pointer leaves.

    Usage::

        lbl = tk.Label(root, text="Hover me")
        ToolTip(lbl, "This is the tooltip text.")

    Args:
        widget: Any tkinter widget to attach the tooltip to.
        text (str): The text to display in the tooltip.
    """

    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tip_win = None  # The Toplevel window shown on hover
        widget.bind("<Enter>", self._show, add="+")
        widget.bind("<Leave>", self._hide, add="+")

    def _show(self, event=None):
        """Create and position the tooltip window near the current cursor."""
        if self.tip_win or not self.text:
            return
        # Position just below and to the right of the cursor
        x = event.x_root + 14
        y = event.y_root + 14
        self.tip_win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # No title bar or window border
        tw.wm_geometry(f"+{x}+{y}")
        tk.Label(
            tw,
            text=self.text,
            justify="left",
            background="#fffbe6",  # Pale yellow, easy to distinguish
            relief="solid",
            borderwidth=1,
            font=("Arial", 9),
            wraplength=540,
            padx=8,
            pady=6,
        ).pack()

    def _hide(self, event=None):
        """Destroy the tooltip window when the cursor leaves the widget."""
        if self.tip_win:
            self.tip_win.destroy()
            self.tip_win = None


# ---------------------------------------------------------------------------
# GUI helper functions
# ---------------------------------------------------------------------------


def check_install(package_id: str):
    """
    Checks if a package is installed and installs it if missing.

    Tries pip first; if pip is unavailable or fails, falls back to conda.
    Prints status messages at each step so the user can see what is happening
    in the terminal.

    Args:
        package_id (str): The name of the package to check/install.
    """
    try:
        __import__(package_id)
        print(f"{package_id} is already installed.")
    except ModuleNotFoundError:
        print(f"{package_id} is not installed.")
        try:
            print(f"Trying to install {package_id} using pip...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_id])
        except subprocess.CalledProcessError as pip_error:
            print(f"Pip installation failed: {pip_error}")
            try:
                if shutil.which("conda"):
                    print(f"Trying to install {package_id} using conda...")
                    subprocess.check_call(["conda", "install", package_id, "-y"])
                else:
                    print("Conda is not available. Please install using pip.")
            except subprocess.CalledProcessError as conda_error:
                print(f"Conda installation failed: {conda_error}")
                print(f"Unable to install {package_id} with pip or conda.")


def _step_label(parent, number, title):
    """
    Render a bold numbered-step heading inside *parent*.

    Placed at grid row 0, spanning all three columns so it acts as a
    section header for the ttk.LabelFrame that wraps each step.

    Args:
        parent: The parent widget (usually a ttk.LabelFrame).
        number (int): The step number shown in the label text.
        title (str): The short description of what this step involves.
    """
    tk.Label(
        parent,
        text=f"  Step {number}:  {title}",
        font=STEP_FONT,
        fg=STEP_COLOR,
        anchor="w",
    ).grid(row=0, column=0, columnspan=3, sticky="ew", pady=(6, 2))


def _hint(parent, row, text):
    """
    Render small grey hint text at the given grid row inside *parent*.

    Uses HINT_FONT and HINT_COLOR so all hint labels look consistent.
    Text wraps at 700 px so it stays within the dialog width.

    Args:
        parent: The parent widget.
        row (int): Grid row at which the label is placed.
        text (str): The hint text to display (may contain newlines).
    """
    tk.Label(
        parent,
        text=text,
        font=HINT_FONT,
        fg=HINT_COLOR,
        justify="left",
        wraplength=700,
        anchor="w",
    ).grid(row=row, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))


def _section(parent, row, title, tooltip=None, pady=(10, 4)):
    """
    Create a ttk.LabelFrame that acts as a visual section divider.

    If *tooltip* is provided, a small ``? Help`` label is added in the
    top-right corner of the frame.  Hovering over it (or anywhere inside
    the frame) shows the full help text in a floating tooltip.

    Args:
        parent: The parent widget (usually the outer scrollable frame).
        row (int): Grid row at which the LabelFrame is placed.
        title (str): Text shown in the frame's border.
        tooltip (str | None): Multi-line help text shown on hover.
            Pass ``None`` to omit the help indicator.
        pady (tuple): Vertical padding above and below the frame.

    Returns:
        ttk.LabelFrame: The newly created frame (caller adds child widgets).
    """
    lf = ttk.LabelFrame(parent, text=f"  {title}  ", padding=(10, 6))
    lf.grid(row=row, column=0, columnspan=3, sticky="ew", pady=pady, padx=4)
    lf.columnconfigure(1, weight=1)

    if tooltip:
        # Small "? Help" indicator in the top-right corner of the section.
        # Hovering over it reveals the full instructional text.
        info = tk.Label(
            lf,
            text=" ? Help ",
            font=("Arial", 8, "italic"),
            fg="#2471a3",
            cursor="question_arrow",
        )
        info.grid(row=0, column=2, sticky="ne", pady=(0, 2), padx=(4, 0))
        # Bind tooltip to the indicator label so the user can find it easily.
        ToolTip(info, tooltip)
        # Also bind to the frame itself for discoverability.
        ToolTip(lf, tooltip)

    return lf


# ---------------------------------------------------------------------------
# Main application class
# ---------------------------------------------------------------------------


class App:
    """
    Shot Peening ML Predictor GUI.

    Workflow (from README):
      1. Train a new CNN on shot-peening simulation data (checkerboard input ->
         deformation output), or
      2. Load a previously trained model and evaluate it on new peen intensity
         patterns to predict and visualise deformation.

    The class owns the root Tk window and all dialogs are created as
    tk.Toplevel children so they stay associated with the main window.
    """

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def __init__(self, root_tk):
        """
        Initialise the application.

        Stores a reference to the root Tk window, configures the title and
        geometry, resets all per-session state variables, then draws the
        main menu.

        Args:
            root_tk (tk.Tk): The top-level Tk window created by the caller.
        """
        self.root = root_tk
        self.root.title("Model GUI")  # Window title shown in the OS taskbar
        self.root.geometry("1100x820")  # Main window size (tests check this value)
        self.window_size = "1100x820"  # Keep a copy for reference
        self.dialog_size = "1100x860"  # Larger size used for Train/Load dialogs
        self.test_train_data_path = ""  # Set when the user browses for data
        self.parent_process = None  # Reserved for any spawned sub-processes
        self.main_menu()

    # ------------------------------------------------------------------
    # Main menu
    # ------------------------------------------------------------------

    def main_menu(self):
        """
        Draw (or re-draw) the main menu.

        Clears all existing child widgets first so that clicking
        'Back to Main Menu' from a dialog always produces a clean screen.
        Presents two bordered card frames side-by-side — one for the
        'Train Model' workflow and one for the 'Load & Evaluate' workflow —
        matching the overview section of the README.
        """
        # Destroy any previously rendered widgets before rebuilding the menu.
        for widget in self.root.winfo_children():
            widget.destroy()

        # Two equally-weighted columns so the cards sit side-by-side.
        main_frame = tk.Frame(self.root, padx=20, pady=10)
        main_frame.pack(expand=True, fill=tk.BOTH)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)

        # -- Splash image --
        try:
            bullet_bill_path = os.path.join(os.path.dirname(__file__), "src", "peen-ml", "bullet_bill.png")
            image = Image.open(bullet_bill_path)
            image = image.resize((360, 220), Image.Resampling.LANCZOS)
            self.splash_image = ImageTk.PhotoImage(image)
            # Store the reference on self to prevent garbage collection
            ttk.Label(main_frame, image=self.splash_image).grid(row=0, column=0, columnspan=2, pady=(10, 4))
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not locate Bullet Bill logo: {e}") from e

        # -- Title --
        tk.Label(
            main_frame,
            text="Shot Peening ML Predictor",
            font=("Arial", 20, "bold"),
            fg=STEP_COLOR,
        ).grid(row=1, column=0, columnspan=2, pady=(4, 2))

        # -- One-line README summary --
        tk.Label(
            main_frame,
            text=(
                "Predict surface deformation from shot peening recipes using a\n"
                "CNN trained on simulation data — without running FEA each time."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="center",
        ).grid(row=2, column=0, columnspan=2, pady=(0, 10))

        # -- Workflow separator --
        ttk.Separator(main_frame, orient="horizontal").grid(row=3, column=0, columnspan=2, sticky="ew", pady=4)
        tk.Label(
            main_frame,
            text="What would you like to do?",
            font=("Arial", 11),
            fg=HINT_COLOR,
        ).grid(row=4, column=0, columnspan=2, pady=(4, 6))

        # -- Train card (left column) --
        train_frame = tk.Frame(main_frame, bd=2, relief="groove", padx=12, pady=10)
        train_frame.grid(row=5, column=0, padx=16, pady=6, sticky="nsew")
        tk.Label(train_frame, text="Train Model", font=("Arial", 13, "bold"), fg=STEP_COLOR).pack(anchor="w")
        tk.Label(
            train_frame,
            text=(
                "Have a dataset of shot-peening simulations?\n"
                "Train a new CNN model on checkerboard -> deformation pairs.\n\n"
                "You will need:\n"
                "  - A folder of Simulation_0/, Simulation_1/, ... subfolders\n"
                "  - Each containing checkerboard.npy + displacements.npy\n\n"
                "The trained model is saved automatically for later use."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        tk.Button(
            train_frame,
            text="Train Model  ->",
            command=self.train_model_dialog,
            width=18,
            height=2,
            bg="#2471a3",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(anchor="e")

        # -- Load card (right column) --
        load_frame = tk.Frame(main_frame, bd=2, relief="groove", padx=12, pady=10)
        load_frame.grid(row=5, column=1, padx=16, pady=6, sticky="nsew")
        tk.Label(load_frame, text="Load & Evaluate Model", font=("Arial", 13, "bold"), fg=STEP_COLOR).pack(anchor="w")
        tk.Label(
            load_frame,
            text=(
                "Already have a trained model?\n"
                "Load it to predict deformation from a new shot pattern.\n\n"
                "You will need:\n"
                "  - The trained .pth model file\n"
                "  - A peen-intensity folder (checkerboard.npy + mesh files)\n"
                "  - An output folder for the predicted displacements\n\n"
                "Visualise input patterns and predicted deformation."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        tk.Button(
            load_frame,
            text="Load Model  ->",
            command=self.load_model_dialog,
            width=18,
            height=2,
            bg="#1a7a4a",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(anchor="e")

        # -- Separator --
        ttk.Separator(main_frame, orient="horizontal").grid(row=6, column=0, columnspan=2, sticky="ew", pady=(10, 0))

        # -- Generate Dataset card (full width) --
        gen_frame = tk.Frame(main_frame, bd=2, relief="groove", padx=12, pady=10)
        gen_frame.grid(row=7, column=0, columnspan=2, padx=16, pady=(4, 10), sticky="ew")
        gen_frame.columnconfigure(0, weight=1)
        tk.Label(
            gen_frame,
            text="Generate Dataset",
            font=("Arial", 13, "bold"),
            fg="#6c3483",
        ).pack(anchor="w")
        tk.Label(
            gen_frame,
            text=(
                "No simulation data yet?  Generate a shot-peening dataset directly from the GUI.\n"
                "  Single Impact  —  uniform random shots, ~2 s/sim, CPU.\n"
                "  Gaussian Nozzle  —  realistic nozzle geometry, GPU-accelerated (RTX 4090)."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        tk.Button(
            gen_frame,
            text="Generate Dataset  ->",
            command=self.generate_dataset_dialog,
            width=22,
            height=2,
            bg="#6c3483",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(anchor="e")

        # -- Analytical Compare card (full width, row 8) --
        ac_frame = tk.Frame(main_frame, bd=2, relief="groove", padx=12, pady=10)
        ac_frame.grid(row=8, column=0, columnspan=2, padx=16, pady=(4, 10), sticky="ew")
        ac_frame.columnconfigure(0, weight=1)
        tk.Label(
            ac_frame,
            text="Analytical Compare",
            font=("Arial", 13, "bold"),
            fg="#117a65",
        ).pack(anchor="w")
        tk.Label(
            ac_frame,
            text=(
                "Compare Shen & Atluri (elastic-plastic) vs Sherafatnia (elastic) analytical models.\n"
                "Given shot positions from a dataset, compute the analytical displacement field\n"
                "and internal stress, then compare against the simulator ground truth.\n"
                "Shows deformation + stress heat maps side-by-side for both models."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))
        tk.Button(
            ac_frame,
            text="Analytical Compare  ->",
            command=self.analytical_dialog,
            width=22,
            height=2,
            bg="#117a65",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(anchor="e")

    # ------------------------------------------------------------------
    # Analytical Compare dialog
    # ------------------------------------------------------------------

    def analytical_dialog(self):
        """Open the Analytical Compare dialog (Shen-Atluri vs Sherafatnia)."""
        import analytical_mode as _am  # lazy import to avoid circular imports at startup
        import tempfile
        from tkinter import scrolledtext

        dialog = tk.Toplevel(self.root)
        dialog.title("Analytical Compare")
        dialog.geometry("1100x860")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.lift()
        dialog.focus_force()

        # ---- Scrollable canvas ----
        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        outer = tk.Frame(canvas, padx=18, pady=12)
        canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_frame_configure(event):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            canvas.itemconfig(canvas_window, width=event.width)

        outer.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        outer.columnconfigure(0, weight=1)

        # ---- Page title ----
        tk.Label(
            outer,
            text="Analytical Model Comparison",
            font=("Arial", 15, "bold"),
            fg="#117a65",
        ).grid(row=0, column=0, sticky="w", pady=(4, 2))
        tk.Label(
            outer,
            text=(
                "Compare Shen & Atluri (elastic-plastic) vs Sherafatnia (elastic) analytical models "
                "against the simulator ground truth."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
            wraplength=900,
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 8))

        # ---- Section 1: Dataset folder ----
        sec1 = _section(outer, row=3, title="Dataset")
        dataset_var = tk.StringVar()
        ds_row = tk.Frame(sec1)
        ds_row.grid(row=1, column=0, columnspan=3, sticky="ew", pady=4)
        ds_row.columnconfigure(0, weight=1)
        tk.Entry(ds_row, textvariable=dataset_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            ds_row,
            text="Browse...",
            width=10,
            command=lambda: self.browse_directory(dataset_var, parent=dialog),
        ).grid(row=0, column=1)

        # ---- Section 2: Simulation selector ----
        sec2 = _section(outer, row=4, title="Simulation")
        mode_var = tk.StringVar(value="single")
        sim_idx_var = tk.StringVar(value="0")
        batch_n_var = tk.StringVar(value="10")

        mode_frame = tk.Frame(sec2)
        mode_frame.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))
        tk.Radiobutton(mode_frame, text="Single simulation", variable=mode_var, value="single", font=BODY_FONT).pack(
            side="left", padx=(0, 16)
        )
        tk.Radiobutton(mode_frame, text="Batch (N random sims)", variable=mode_var, value="batch", font=BODY_FONT).pack(
            side="left"
        )

        param_frame = tk.Frame(sec2)
        param_frame.grid(row=2, column=0, columnspan=3, sticky="w", pady=4)
        tk.Label(param_frame, text="Sim index:", font=BODY_FONT).pack(side="left", padx=(0, 4))
        tk.Spinbox(param_frame, textvariable=sim_idx_var, from_=0, to=9999, width=6, font=BODY_FONT).pack(
            side="left", padx=(0, 20)
        )
        tk.Label(param_frame, text="N (batch):", font=BODY_FONT).pack(side="left", padx=(0, 4))
        tk.Spinbox(param_frame, textvariable=batch_n_var, from_=1, to=9999, width=6, font=BODY_FONT).pack(side="left")

        # ---- Section 3: Options ----
        sec3 = _section(outer, row=5, title="Options")
        sequential_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            sec3,
            text="Sequential work-hardening mode",
            variable=sequential_var,
            font=BODY_FONT,
        ).grid(row=1, column=0, sticky="w", pady=4)

        # ---- Section 4: Run + Results ----
        sec4 = _section(outer, row=6, title="Run + Results")

        run_btn = tk.Button(
            sec4,
            text="Run Analysis",
            font=("Arial", 11, "bold"),
            bg="#117a65",
            fg="white",
            relief="flat",
            width=18,
            height=2,
        )
        run_btn.grid(row=1, column=0, sticky="w", pady=(6, 4))

        progress_lbl = tk.Label(sec4, text="", font=BODY_FONT, fg=INFO_COLOR)
        progress_lbl.grid(row=1, column=1, sticky="w", padx=(12, 0))

        # Scrollable results frame for images
        results_outer = tk.Frame(sec4, bd=1, relief="sunken")
        results_outer.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(6, 4))
        results_outer.columnconfigure(0, weight=1)

        results_canvas = tk.Canvas(results_outer, height=620, highlightthickness=0)
        results_sb = ttk.Scrollbar(results_outer, orient="vertical", command=results_canvas.yview)
        results_canvas.configure(yscrollcommand=results_sb.set)
        results_sb.pack(side="right", fill="y")
        results_canvas.pack(side="left", fill="both", expand=True)

        results_frame = tk.Frame(results_canvas)
        results_win = results_canvas.create_window((0, 0), window=results_frame, anchor="nw")

        def _on_results_resize(event):
            results_canvas.configure(scrollregion=results_canvas.bbox("all"))

        def _on_results_canvas_resize(event):
            results_canvas.itemconfig(results_win, width=event.width)

        results_frame.bind("<Configure>", _on_results_resize)
        results_canvas.bind("<Configure>", _on_results_canvas_resize)

        # Metrics text box
        metrics_box = scrolledtext.ScrolledText(sec4, height=8, font=("Courier", 9), state="disabled", bg="#f8f8f8")
        metrics_box.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(4, 4))

        def _show_image(fig_path):
            """Display a PNG inside results_frame."""
            try:
                img = Image.open(fig_path)
                img.thumbnail((1050, 600))
                photo = ImageTk.PhotoImage(img)
                lbl = tk.Label(results_frame, image=photo)
                lbl.image = photo  # keep reference
                lbl.pack(pady=4)
                outer.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception as exc:  # pylint: disable=broad-except
                tk.Label(results_frame, text=f"Could not load figure: {exc}", fg=ERR_COLOR).pack()

        def _write_metrics(text):
            metrics_box.config(state="normal")
            metrics_box.delete("1.0", tk.END)
            metrics_box.insert(tk.END, text)
            metrics_box.config(state="disabled")

        def _do_run():
            dataset_dir = dataset_var.get().strip()
            if not dataset_dir or not os.path.isdir(dataset_dir):
                messagebox.showerror("Invalid folder", "Please browse to a valid dataset folder.", parent=dialog)
                return

            sim_dirs = sorted(
                [
                    d
                    for d in os.listdir(dataset_dir)
                    if d.startswith("Simulation_") and d[len("Simulation_") :].isdigit()
                ]
            )
            if not sim_dirs:
                messagebox.showerror(
                    "No simulations found",
                    "The selected folder contains no Simulation_N/ subdirectories.",
                    parent=dialog,
                )
                return

            run_btn.config(state="disabled", text="Running...")
            progress_lbl.config(text="Running...", fg=INFO_COLOR)

            # Clear previous results
            for widget in results_frame.winfo_children():
                widget.destroy()
            _write_metrics("")

            sequential = sequential_var.get()
            mode = mode_var.get()

            def _run():
                try:
                    if mode == "single":
                        sim_idx = int(sim_idx_var.get())
                        sim_dir = os.path.join(dataset_dir, f"Simulation_{sim_idx}")
                        if not os.path.isdir(sim_dir):
                            dialog.after(
                                0,
                                lambda: messagebox.showerror(
                                    "Not found",
                                    f"Simulation_{sim_idx} not found in {dataset_dir}",
                                    parent=dialog,
                                ),
                            )
                            return

                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                        fig_path = tmp.name
                        tmp.close()

                        results = _am.compare_to_dataset(sim_dir, out_path=fig_path, sequential=sequential)

                        # Build metrics text
                        lines = [f"{'Model':<14s}  {'comp':>4s}  {'r':>7s}  {'RMSE µm':>8s}  {'n nodes':>8s}"]
                        lines.append("-" * 52)
                        for model in ("shen_atluri", "sherafatnia"):
                            for comp in ("ux", "uy", "uz"):
                                m = results[model][comp]
                                r_s = f"{m['r']:>+7.4f}" if not (m["r"] != m["r"]) else "     nan"
                                rmse_s = f"{m['rmse_um']:>8.2f}" if not (m["rmse_um"] != m["rmse_um"]) else "     nan"
                                lines.append(f"  {model:<14s}  {comp:>4s}  {r_s}  {rmse_s}  {m['n']:>8d}")
                        metrics_text = "\n".join(lines)

                        dialog.after(0, lambda: _write_metrics(metrics_text))
                        dialog.after(0, lambda: _show_image(fig_path))

                    else:
                        n_sims = int(batch_n_var.get())
                        tmp_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
                        tmp_csv_path = tmp_csv.name
                        tmp_csv.close()

                        rows = _am.compare_dataset(
                            dataset_dir,
                            n_sims=n_sims,
                            sequential=sequential,
                            out_csv=tmp_csv_path,
                            verbose=True,
                        )

                        # Show summary text in metrics box
                        import io
                        import sys as _sys

                        buf = io.StringIO()
                        old_stdout = _sys.stdout
                        _sys.stdout = buf
                        try:
                            _am._print_summary(rows)
                        finally:
                            _sys.stdout = old_stdout
                        metrics_text = buf.getvalue()
                        dialog.after(0, lambda: _write_metrics(metrics_text))

                    dialog.after(0, lambda: progress_lbl.config(text="Done", fg=OK_COLOR))
                except Exception as exc:  # pylint: disable=broad-except
                    err_msg = str(exc)
                    dialog.after(0, lambda: progress_lbl.config(text=f"Error: {err_msg[:60]}", fg=ERR_COLOR))
                finally:
                    dialog.after(0, lambda: run_btn.config(state="normal", text="Run Analysis"))

            threading.Thread(target=_run, daemon=True).start()

        run_btn.config(command=_do_run)

        tk.Button(sec4, text="Close", command=dialog.destroy, width=12).grid(row=4, column=0, sticky="w", pady=(4, 0))

    # ------------------------------------------------------------------
    # Generate Dataset dialog
    # ------------------------------------------------------------------

    def generate_dataset_dialog(self):
        """
        Open the dataset-generation dialog.

        Two tabs let the user configure and launch either the Single Impact
        (native_dataset_gen.py) or Gaussian Nozzle (gaussian_nozzle_dataset_gen.py)
        generator.  Output from the subprocess is streamed live to the log widget.
        A Stop button terminates the process at any time.
        """
        import queue as _q

        dialog = tk.Toplevel(self.root)
        dialog.title("Generate Dataset")
        dialog.geometry("1100x920")
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.lift()
        dialog.focus_force()

        # Header
        hdr = tk.Frame(dialog, padx=18, pady=8)
        hdr.pack(fill="x")
        tk.Label(
            hdr,
            text="Generate a Shot Peening Dataset",
            font=("Arial", 15, "bold"),
            fg="#6c3483",
        ).pack(anchor="w")
        tk.Label(
            hdr,
            text=(
                "Configure the generator, set an output folder, then click Generate.  "
                "Progress streams live into the log below."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
        ).pack(anchor="w", pady=(0, 4))
        ttk.Separator(dialog, orient="horizontal").pack(fill="x", padx=18)

        # Notebook
        nb = ttk.Notebook(dialog)
        nb.pack(fill="both", expand=True, padx=10, pady=8)

        t1 = tk.Frame(nb, padx=14, pady=10)
        nb.add(t1, text="   Single Impact (Native)   ")
        t1.columnconfigure(0, weight=1)
        t1.rowconfigure(4, weight=1)
        self._build_native_gen_tab(t1, dialog, _q)

        t2 = tk.Frame(nb, padx=14, pady=10)
        nb.add(t2, text="   Gaussian Nozzle   ")
        t2.columnconfigure(0, weight=1)
        t2.rowconfigure(4, weight=1)
        self._build_gaussian_gen_tab(t2, dialog, _q)

        tk.Button(dialog, text="Close", command=dialog.destroy, width=14).pack(pady=(0, 8))

    # ------------------------------------------------------------------

    def _build_native_gen_tab(self, parent, dialog, _q):
        """Build the Single Impact (native_dataset_gen.py) tab."""

        tk.Label(
            parent,
            text="Single Impact Dataset Generator",
            font=("Arial", 12, "bold"),
            fg=STEP_COLOR,
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))
        tk.Label(
            parent,
            text=(
                "Each simulation fires N shots at uniform random positions on the plate.  "
                "~2 s/sim on CPU.  Produces Simulation_N/ folders compatible with model.py."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            wraplength=900,
            justify="left",
        ).grid(row=1, column=0, sticky="w", pady=(0, 8))

        # ---- Parameter grid ----
        pf = ttk.LabelFrame(parent, text="  Parameters  ", padding=(10, 6))
        pf.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for c in (1, 4):
            pf.columnconfigure(c, weight=1)

        def _lbl(text, row, col):
            tk.Label(pf, text=text, font=BODY_FONT, anchor="e").grid(
                row=row, column=col, sticky="e", padx=(6, 4), pady=2
            )

        def _ent(var, row, col, width=9):
            tk.Entry(pf, textvariable=var, width=width, font=BODY_FONT).grid(row=row, column=col, sticky="w", pady=2)

        def _unit(text, row, col):
            tk.Label(pf, text=text, font=HINT_FONT, fg=HINT_COLOR).grid(row=row, column=col, sticky="w", padx=(2, 10))

        # Output folder (full width)
        out_var = tk.StringVar(value="./Dataset_Native")
        _lbl("Output folder", 0, 0)
        out_ef = tk.Frame(pf)
        out_ef.grid(row=0, column=1, columnspan=6, sticky="ew", pady=2)
        out_ef.columnconfigure(0, weight=1)
        tk.Entry(out_ef, textvariable=out_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            out_ef, text="Browse...", width=9, command=lambda: self.browse_directory(out_var, parent=dialog)
        ).grid(row=0, column=1)

        # Left column
        n_sims_var = tk.StringVar(value="100")
        workers_var = tk.StringVar(value="1")
        seed_var = tk.StringVar(value="0")
        Nx_var = tk.StringVar(value="50")
        Ny_var = tk.StringVar(value="50")
        grid_var = tk.StringVar(value="5")
        shots_min_var = tk.StringVar(value="20")
        shots_max_var = tk.StringVar(value="120")

        _lbl("Simulations", 1, 0)
        _ent(n_sims_var, 1, 1)
        _lbl("Workers", 2, 0)
        _ent(workers_var, 2, 1)
        _unit("(1=sequential)", 2, 2)
        _lbl("Seed", 3, 0)
        _ent(seed_var, 3, 1)
        _lbl("Mesh Nx", 4, 0)
        _ent(Nx_var, 4, 1)
        _unit("elements", 4, 2)
        _lbl("Mesh Ny", 5, 0)
        _ent(Ny_var, 5, 1)
        _unit("elements", 5, 2)
        _lbl("Grid G", 6, 0)
        _ent(grid_var, 6, 1)
        _unit("G x G checkerboard", 6, 2)
        _lbl("Shots min", 7, 0)
        _ent(shots_min_var, 7, 1)
        _lbl("Shots max", 8, 0)
        _ent(shots_max_var, 8, 1)

        # Right column
        Lx_var = tk.StringVar(value="10")
        Ly_var = tk.StringVar(value="10")
        D_min_var = tk.StringVar(value="0.3")
        D_max_var = tk.StringVar(value="1.0")
        V_min_var = tk.StringVar(value="25")
        V_max_var = tk.StringVar(value="60")

        _lbl("Plate Lx", 1, 3)
        _ent(Lx_var, 1, 4)
        _unit("mm", 1, 5)
        _lbl("Plate Ly", 2, 3)
        _ent(Ly_var, 2, 4)
        _unit("mm", 2, 5)
        _lbl("Shot D min", 4, 3)
        _ent(D_min_var, 4, 4)
        _unit("mm", 4, 5)
        _lbl("Shot D max", 5, 3)
        _ent(D_max_var, 5, 4)
        _unit("mm", 5, 5)
        _lbl("Velocity min", 7, 3)
        _ent(V_min_var, 7, 4)
        _unit("m/s", 7, 5)
        _lbl("Velocity max", 8, 3)
        _ent(V_max_var, 8, 4)
        _unit("m/s", 8, 5)

        # ---- Material selectors ----
        try:
            from materials import WORKPIECE_MATERIALS, SHOT_MATERIALS as _SM

            _wp_names = [""] + sorted(WORKPIECE_MATERIALS.keys())
            _sp_names = [""] + sorted(_SM.keys())
        except ImportError:
            _wp_names = [""]
            _sp_names = [""]

        mat_frame = ttk.LabelFrame(pf, text="  Material (optional — leave blank for defaults)  ", padding=(6, 4))
        mat_frame.grid(row=9, column=0, columnspan=6, sticky="ew", pady=(6, 2))

        wp_mat_var = tk.StringVar(value="")
        sp_mat_var = tk.StringVar(value="")

        tk.Label(mat_frame, text="Workpiece", font=BODY_FONT, anchor="e").grid(row=0, column=0, sticky="e", padx=(6, 4))
        ttk.Combobox(mat_frame, textvariable=wp_mat_var, values=_wp_names, width=18, state="readonly").grid(
            row=0, column=1, sticky="w"
        )
        tk.Label(mat_frame, text="Shot", font=BODY_FONT, anchor="e").grid(row=0, column=2, sticky="e", padx=(16, 4))
        ttk.Combobox(mat_frame, textvariable=sp_mat_var, values=_sp_names, width=18, state="readonly").grid(
            row=0, column=3, sticky="w"
        )
        tk.Label(
            mat_frame,
            text="(logged to simulation_params.txt; enables material-conditioned training)",
            font=HINT_FONT,
            fg=HINT_COLOR,
        ).grid(row=1, column=0, columnspan=4, sticky="w", padx=4)

        # ---- Buttons + progress ----
        btn_frame = tk.Frame(parent)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        gen_btn = tk.Button(
            btn_frame,
            text="Generate",
            font=("Arial", 11, "bold"),
            bg="#6c3483",
            fg="white",
            relief="flat",
            width=14,
            height=2,
        )
        gen_btn.pack(side="left", padx=(0, 10))

        stop_btn = tk.Button(btn_frame, text="Stop", width=10, height=2, state="disabled")
        stop_btn.pack(side="left", padx=(0, 10))

        progress = ttk.Progressbar(btn_frame, orient=tk.HORIZONTAL, length=480, mode="indeterminate")
        progress.pack(side="left", fill="x", expand=True)

        # ---- Log ----
        log_frame = tk.Frame(parent)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 4))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log = tk.Text(log_frame, height=10, font=("Courier", 9), state="disabled", bg="#f8f8f8")
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        log.configure(yscrollcommand=log_sb.set)
        log.grid(row=0, column=0, sticky="nsew")
        log_sb.grid(row=0, column=1, sticky="ns")

        # ---- Wire up the generator ----
        script = os.path.join(os.path.dirname(__file__), "src", "peen-ml", "native_dataset_gen.py")

        def _collect_args():
            Lx = float(Lx_var.get()) / 1000.0  # mm -> m
            Ly = float(Ly_var.get()) / 1000.0
            D_min = float(D_min_var.get()) / 1000.0
            D_max = float(D_max_var.get()) / 1000.0
            args = [
                "--output",
                out_var.get(),
                "--n_sims",
                n_sims_var.get(),
                "--workers",
                workers_var.get(),
                "--seed",
                seed_var.get(),
                "--Lx",
                str(Lx),
                "--Ly",
                str(Ly),
                "--Nx",
                Nx_var.get(),
                "--Ny",
                Ny_var.get(),
                "--grid_size",
                grid_var.get(),
                "--n_shots_min",
                shots_min_var.get(),
                "--n_shots_max",
                shots_max_var.get(),
                "--D_min",
                str(D_min),
                "--D_max",
                str(D_max),
                "--V_min",
                V_min_var.get(),
                "--V_max",
                V_max_var.get(),
            ]
            if wp_mat_var.get():
                args += ["--workpiece_material", wp_mat_var.get()]
            if sp_mat_var.get():
                args += ["--shot_material", sp_mat_var.get()]
            return args

        self._wire_generator(
            dialog,
            script,
            _collect_args,
            out_var,
            log,
            progress,
            gen_btn,
            stop_btn,
            _q,
        )

    # ------------------------------------------------------------------

    def _build_gaussian_gen_tab(self, parent, dialog, _q):
        """Build the Gaussian Nozzle (gaussian_nozzle_dataset_gen.py) tab."""

        # CUDA indicator
        cuda_ok = torch.cuda.is_available()
        cuda_txt = f"GPU: {torch.cuda.get_device_name(0)}" if cuda_ok else "No CUDA GPU detected — will run on CPU"
        cuda_color = OK_COLOR if cuda_ok else ERR_COLOR

        tk.Label(
            parent,
            text="Gaussian Nozzle Dataset Generator",
            font=("Arial", 12, "bold"),
            fg=STEP_COLOR,
        ).grid(row=0, column=0, sticky="w", pady=(0, 2))

        hdr2 = tk.Frame(parent)
        hdr2.grid(row=1, column=0, sticky="ew", pady=(0, 8))
        tk.Label(
            hdr2,
            text=(
                "Shots emerge from a nozzle at height h, with positions Gaussian-distributed "
                "below it.  Exit velocities are also Gaussian.  GPU-accelerated CUDA kernel."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            wraplength=780,
            justify="left",
        ).pack(side="left")
        tk.Label(
            hdr2,
            text=f"  {cuda_txt}",
            font=("Arial", 9, "bold"),
            fg=cuda_color,
        ).pack(side="right", padx=(10, 0))

        # ---- Parameter grid ----
        pf = ttk.LabelFrame(parent, text="  Parameters  ", padding=(10, 6))
        pf.grid(row=2, column=0, sticky="ew", pady=(0, 8))
        for c in (1, 4):
            pf.columnconfigure(c, weight=1)

        def _lbl(text, row, col):
            tk.Label(pf, text=text, font=BODY_FONT, anchor="e").grid(
                row=row, column=col, sticky="e", padx=(6, 4), pady=2
            )

        def _ent(var, row, col, width=9):
            tk.Entry(pf, textvariable=var, width=width, font=BODY_FONT).grid(row=row, column=col, sticky="w", pady=2)

        def _unit(text, row, col):
            tk.Label(pf, text=text, font=HINT_FONT, fg=HINT_COLOR).grid(row=row, column=col, sticky="w", padx=(2, 10))

        # Output folder
        out_var = tk.StringVar(value="./Dataset_Gaussian")
        _lbl("Output folder", 0, 0)
        out_ef = tk.Frame(pf)
        out_ef.grid(row=0, column=1, columnspan=6, sticky="ew", pady=2)
        out_ef.columnconfigure(0, weight=1)
        tk.Entry(out_ef, textvariable=out_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            out_ef, text="Browse...", width=9, command=lambda: self.browse_directory(out_var, parent=dialog)
        ).grid(row=0, column=1)

        # Left column parameters
        n_sims_var = tk.StringVar(value="200")
        workers_var = tk.StringVar(value="1")
        seed_var = tk.StringVar(value="0")
        Nx_var = tk.StringVar(value="100")
        Ny_var = tk.StringVar(value="100")
        grid_var = tk.StringVar(value="20")
        shots_min_var = tk.StringVar(value="500")
        shots_max_var = tk.StringVar(value="2000")
        h_min_var = tk.StringVar(value="50")
        h_max_var = tk.StringVar(value="400")

        _lbl("Simulations", 1, 0)
        _ent(n_sims_var, 1, 1)
        _lbl("Workers", 2, 0)
        _ent(workers_var, 2, 1)
        _unit("(1=CUDA)", 2, 2)
        _lbl("Seed", 3, 0)
        _ent(seed_var, 3, 1)
        _lbl("Mesh Nx", 4, 0)
        _ent(Nx_var, 4, 1)
        _unit("elements", 4, 2)
        _lbl("Mesh Ny", 5, 0)
        _ent(Ny_var, 5, 1)
        _unit("elements", 5, 2)
        _lbl("Grid G", 6, 0)
        _ent(grid_var, 6, 1)
        _unit("G x G checkerboard", 6, 2)
        _lbl("Shots min", 7, 0)
        _ent(shots_min_var, 7, 1)
        _lbl("Shots max", 8, 0)
        _ent(shots_max_var, 8, 1)
        _lbl("Nozzle h min", 9, 0)
        _ent(h_min_var, 9, 1)
        _unit("mm standoff", 9, 2)
        _lbl("Nozzle h max", 10, 0)
        _ent(h_max_var, 10, 1)
        _unit("mm standoff", 10, 2)

        # Right column parameters
        Lx_var = tk.StringVar(value="1000")
        Ly_var = tk.StringVar(value="1000")
        D_min_var = tk.StringVar(value="4.0")
        D_max_var = tk.StringVar(value="10.0")
        V_min_var = tk.StringVar(value="25")
        V_max_var = tk.StringVar(value="80")
        div_min_var = tk.StringVar(value="5")
        div_max_var = tk.StringVar(value="30")

        _lbl("Plate Lx", 1, 3)
        _ent(Lx_var, 1, 4)
        _unit("mm", 1, 5)
        _lbl("Plate Ly", 2, 3)
        _ent(Ly_var, 2, 4)
        _unit("mm", 2, 5)
        _lbl("Shot D min", 4, 3)
        _ent(D_min_var, 4, 4)
        _unit("mm", 4, 5)
        _lbl("Shot D max", 5, 3)
        _ent(D_max_var, 5, 4)
        _unit("mm", 5, 5)
        _lbl("Velocity min", 7, 3)
        _ent(V_min_var, 7, 4)
        _unit("m/s", 7, 5)
        _lbl("Velocity max", 8, 3)
        _ent(V_max_var, 8, 4)
        _unit("m/s", 8, 5)
        _lbl("Div angle min", 9, 3)
        _ent(div_min_var, 9, 4)
        _unit("deg", 9, 5)
        _lbl("Div angle max", 10, 3)
        _ent(div_max_var, 10, 4)
        _unit("deg", 10, 5)

        # ---- Buttons + progress ----
        btn_frame = tk.Frame(parent)
        btn_frame.grid(row=3, column=0, sticky="ew", pady=(0, 6))

        gen_btn = tk.Button(
            btn_frame,
            text="Generate",
            font=("Arial", 11, "bold"),
            bg="#6c3483",
            fg="white",
            relief="flat",
            width=14,
            height=2,
        )
        gen_btn.pack(side="left", padx=(0, 10))

        stop_btn = tk.Button(btn_frame, text="Stop", width=10, height=2, state="disabled")
        stop_btn.pack(side="left", padx=(0, 10))

        progress = ttk.Progressbar(btn_frame, orient=tk.HORIZONTAL, length=480, mode="indeterminate")
        progress.pack(side="left", fill="x", expand=True)

        # ---- Log ----
        log_frame = tk.Frame(parent)
        log_frame.grid(row=4, column=0, sticky="nsew", pady=(0, 4))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        log = tk.Text(log_frame, height=10, font=("Courier", 9), state="disabled", bg="#f8f8f8")
        log_sb = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        log.configure(yscrollcommand=log_sb.set)
        log.grid(row=0, column=0, sticky="nsew")
        log_sb.grid(row=0, column=1, sticky="ns")

        # ---- Wire up the generator ----
        script = os.path.join(os.path.dirname(__file__), "src", "peen-ml", "gaussian_nozzle_dataset_gen.py")

        def _collect_args():
            Lx = float(Lx_var.get()) / 1000.0  # mm -> m
            Ly = float(Ly_var.get()) / 1000.0
            D_min = float(D_min_var.get()) / 1000.0
            D_max = float(D_max_var.get()) / 1000.0
            h_min = float(h_min_var.get()) / 1000.0
            h_max = float(h_max_var.get()) / 1000.0
            return [
                "--output",
                out_var.get(),
                "--n_sims",
                n_sims_var.get(),
                "--workers",
                workers_var.get(),
                "--seed",
                seed_var.get(),
                "--Lx",
                str(Lx),
                "--Ly",
                str(Ly),
                "--Nx",
                Nx_var.get(),
                "--Ny",
                Ny_var.get(),
                "--grid_size",
                grid_var.get(),
                "--n_shots_min",
                shots_min_var.get(),
                "--n_shots_max",
                shots_max_var.get(),
                "--D_min",
                str(D_min),
                "--D_max",
                str(D_max),
                "--V_min",
                V_min_var.get(),
                "--V_max",
                V_max_var.get(),
                "--h_min",
                str(h_min),
                "--h_max",
                str(h_max),
                "--div_min_deg",
                div_min_var.get(),
                "--div_max_deg",
                div_max_var.get(),
            ]

        self._wire_generator(
            dialog,
            script,
            _collect_args,
            out_var,
            log,
            progress,
            gen_btn,
            stop_btn,
            _q,
        )

    # ------------------------------------------------------------------

    def _wire_generator(self, dialog, script, collect_args_fn, out_var, log, progress, gen_btn, stop_btn, _q):
        """
        Attach Generate / Stop behaviour to a generator tab.

        Runs the given *script* as a subprocess (PYTHONUNBUFFERED=1 so that
        per-simulation progress lines appear immediately).  Output is routed
        through a thread-safe queue and drained into *log* every 150 ms via
        ``dialog.after``.

        Args:
            dialog       : The parent Toplevel window (used for after() calls).
            script       : Absolute path to the generator .py file.
            collect_args_fn : Callable() -> list[str] — builds CLI args from
                             the current parameter StringVars.  Called only
                             when the user clicks Generate so errors are
                             reported then, not during construction.
            out_var      : StringVar holding the output folder path.
            log          : tk.Text widget (read-only; enabled briefly to append).
            progress     : ttk.Progressbar (indeterminate mode).
            gen_btn      : The Generate button (disabled while running).
            stop_btn     : The Stop button (enabled while running).
            _q           : The ``queue`` module (passed in to avoid re-import).
        """
        log_q = _q.Queue()
        proc_ref = [None]
        running = [False]

        def _log_write(msg):
            log.config(state="normal")
            log.insert(tk.END, msg + "\n")
            log.see(tk.END)
            log.config(state="disabled")

        def _poll():
            try:
                while True:
                    _log_write(log_q.get_nowait())
            except _q.Empty:
                pass
            if running[0]:
                dialog.after(150, _poll)

        def _do_stop():
            p = proc_ref[0]
            if p and p.poll() is None:
                p.terminate()
                log_q.put("[Stopped by user]")

        def _do_generate():
            if not out_var.get().strip():
                messagebox.showerror(
                    "Missing output folder",
                    "Please enter or browse to an output folder before generating.",
                    parent=dialog,
                )
                return
            try:
                args = collect_args_fn()
            except ValueError as exc:
                messagebox.showerror("Invalid parameter", str(exc), parent=dialog)
                return

            running[0] = True
            gen_btn.config(state="disabled", text="Running...")
            stop_btn.config(state="normal")
            progress.start(15)
            _log_write(f"python {os.path.basename(script)}")
            _log_write("Output: " + out_var.get() + "\n")
            dialog.after(150, _poll)

            def _run():
                env = os.environ.copy()
                env["PYTHONUNBUFFERED"] = "1"
                try:
                    proc = subprocess.Popen(
                        [sys.executable, script] + args,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=env,
                    )
                    proc_ref[0] = proc
                    for line in iter(proc.stdout.readline, ""):
                        log_q.put(line.rstrip())
                    proc.wait()
                    if proc.returncode == 0:
                        log_q.put(f"\n[Done -- output written to: {out_var.get()}]")
                    else:
                        log_q.put(f"\n[Process exited with code {proc.returncode}]")
                except Exception as exc:  # pylint: disable=broad-except
                    log_q.put(f"ERROR: {exc}")
                finally:
                    running[0] = False
                    dialog.after(0, progress.stop)
                    dialog.after(0, lambda: gen_btn.config(state="normal", text="Generate"))
                    dialog.after(0, lambda: stop_btn.config(state="disabled"))

            threading.Thread(target=_run, daemon=True).start()

        gen_btn.config(command=_do_generate)
        stop_btn.config(command=_do_stop)

    # ------------------------------------------------------------------
    # Train Model dialog
    # ------------------------------------------------------------------

    def train_model_dialog(self):
        """
        Open the step-by-step training dialog.

        Based on the README 'Training and Evaluating the ML Model' section.
        The dialog contains four numbered steps inside ttk.LabelFrame widgets.
        Each section has a '? Help' indicator in its top-right corner —
        hovering over it shows the full instructional text as a tooltip,
        keeping the dialog compact by default.

        The scrollable canvas pattern (Canvas + inner Frame + Scrollbar) ensures
        all content is accessible even on small screens.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Train Model")
        dialog.geometry(self.dialog_size)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.lift()
        dialog.focus_force()

        # ---- Scrollable canvas so content never clips on small screens ----
        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        # Place the real content inside a Frame embedded in the canvas window.
        outer = tk.Frame(canvas, padx=18, pady=12)
        canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_frame_configure(event):
            """Recalculate the canvas scroll region whenever the inner frame resizes."""
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            """Keep the inner frame exactly as wide as the canvas (minus scrollbar)."""
            canvas.itemconfig(canvas_window, width=event.width)

        # Bind resize callbacks so the scrollable area always fits the content.
        outer.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        outer.columnconfigure(0, weight=1)

        # ---- Page title ----
        tk.Label(
            outer,
            text="Train a New Deformation Prediction Model",
            font=("Arial", 15, "bold"),
            fg=STEP_COLOR,
        ).grid(row=0, column=0, sticky="w", pady=(4, 2))
        tk.Label(
            outer,
            text=("Follow the four steps below.  Hover over '? Help' in any section " "for detailed instructions."),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
            wraplength=880,
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 8))

        # ======================================================
        # STEP 1 — Select training data
        # ======================================================
        sec1 = _section(
            outer,
            row=3,
            title="Step 1  —  Select Training Data Folder",
            tooltip=(
                "Browse to the PARENT folder that contains your Simulation_0/,\n"
                "Simulation_1/, ... subfolders.  The app will find them automatically.\n\n"
                "Each Simulation_N/ subfolder must contain:\n"
                "  checkerboard.npy  — the G x G shot-intensity grid (model input)\n"
                "  displacements.npy — the (N_nodes, 3) deformation array (ground truth)\n\n"
                "Don't have data yet?\n"
                "Run  src/peen-ml/native_dataset_gen.py  to generate a Python-native\n"
                "dataset — no Abaqus or FEA licence required (~2 s per simulation)."
            ),
        )

        tk.Label(
            sec1,
            text="Training & Testing Data Folder",
            font=BODY_FONT,
        ).grid(row=1, column=0, sticky="w", pady=(4, 2))

        # StringVar holds the chosen folder path; trace_add watches it for changes.
        data_folder_var = tk.StringVar()
        entry_frame1 = tk.Frame(sec1)
        entry_frame1.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        entry_frame1.columnconfigure(0, weight=1)

        tk.Entry(entry_frame1, textvariable=data_folder_var, font=BODY_FONT).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        tk.Button(
            entry_frame1,
            text="Browse...",
            command=lambda: self.browse_directory(data_folder_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # Keep a reference so tests (and other methods) can read the chosen path.
        self.test_train_data_path = data_folder_var

        # ======================================================
        # STEP 2 — Verify detected shapes
        # ======================================================
        sec2 = _section(
            outer,
            row=4,
            title="Step 2  —  Verify Detected Dataset Shape",
            tooltip=(
                "After browsing, the app reads the first Simulation_N/ folder and\n"
                "automatically detects:\n"
                "  num_nodes       — number of FEA mesh nodes\n"
                "  checkerboard_size — G for the G x G shot-intensity grid\n\n"
                "These values configure the CNN architecture so no manual entry is needed.\n"
                "Check that the numbers shown match your dataset before training.\n\n"
                "FC layer input = 128 x G x G  (three same-padding conv layers\n"
                "preserve the G x G spatial size throughout)."
            ),
        )

        # shape_var is updated automatically whenever data_folder_var changes.
        shape_var = tk.StringVar(value="(waiting — browse a folder in Step 1 first)")
        shape_lbl = tk.Label(sec2, textvariable=shape_var, font=HINT_FONT, fg=INFO_COLOR, justify="left")
        shape_lbl.grid(row=1, column=0, columnspan=3, sticky="w", padx=6, pady=(4, 6))

        def _update_shape(*_):
            """
            Callback fired whenever data_folder_var changes.

            Calls infer_dataset_shape() on the chosen folder to read
            num_nodes and checkerboard_size from the first Simulation_N/
            sub-folder, then updates shape_lbl with a human-readable summary.
            Turns green on success and red on failure.
            """
            folder = data_folder_var.get()
            if not folder or not os.path.isdir(folder):
                return  # Nothing to inspect yet — wait for a valid path
            try:
                n_nodes, cb_size = infer_dataset_shape(folder)
                # Count matching sub-folders to report how many training cases exist
                n_sims = len(
                    [
                        d
                        for d in os.listdir(folder)
                        if os.path.isdir(os.path.join(folder, d))
                        and d.startswith("Simulation_")
                        and d[len("Simulation_") :].isdigit()
                    ]
                )
                shape_var.set(
                    f"OK  |  {n_sims} simulation(s)   "
                    f"num_nodes={n_nodes}   "
                    f"checkerboard={cb_size}x{cb_size}   "
                    f"FC input=128x{cb_size}x{cb_size}={128*cb_size*cb_size}"
                )
                shape_lbl.config(fg=OK_COLOR)
            except Exception as exc:  # pylint: disable=broad-except
                shape_var.set(f"Could not read dataset: {exc}")
                shape_lbl.config(fg=ERR_COLOR)

        # Wire the callback to run on every write to data_folder_var.
        data_folder_var.trace_add("write", _update_shape)

        # ======================================================
        # STEP 2b — Choose Architecture
        # ======================================================
        sec_arch = _section(
            outer,
            row=5,
            title="Step 2b  —  Choose Model Architecture",
            tooltip=(
                "FC (Legacy): original model, Linear(512, N×3) output.  OOM at N > 100K.\n\n"
                "Conv Decoder: 170K params, convolutional decoder, node-count-agnostic.\n"
                "  AMP + gradient accumulation auto-enabled for grids > 256×256.\n\n"
                "SIREN / INR: implicit neural field — O(K) GPU memory regardless of mesh\n"
                "  size.  Trains on K=512 randomly-sampled nodes per step, evaluates at\n"
                "  any resolution including 1001×1001 (1M nodes) without OOM."
            ),
        )
        model_type_var = tk.StringVar(value="conv")
        for _val, _lbl in [
            ("fc", "FC — Legacy  (OOM at N > 100K)"),
            ("conv", "Convolutional Decoder  [recommended]"),
            ("siren", "SIREN / INR  [large meshes, memory-safe]"),
        ]:
            tk.Radiobutton(
                sec_arch,
                text=_lbl,
                variable=model_type_var,
                value=_val,
                font=HINT_FONT,
            ).grid(sticky="w", padx=12, pady=2)

        # ======================================================
        # STEP 3 — Train
        # ======================================================
        sec3 = _section(
            outer,
            row=6,
            title="Step 3  —  Train the Model",
            tooltip=(
                "Click 'Train' to start.  Training runs in the background so the\n"
                "window stays responsive while epochs complete.\n\n"
                "Settings depend on the chosen architecture (see Step 2b).\n\n"
                "Saved to:\n"
                "  FC:    <dataset_folder>/saved_model/\n"
                "  Conv:  <dataset_folder>/saved_model_conv/\n"
                "  SIREN: <dataset_folder>/saved_model_siren/\n\n"
                "Keep the .pth file — you will need it in the 'Load Model' screen."
            ),
        )

        btn_row = tk.Frame(sec3)
        btn_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 4))

        # Create the Train button first without a command; we assign it below
        # once _do_train is defined (it needs to close over train_btn itself).
        train_btn = tk.Button(
            btn_row,
            text="Train",
            font=("Arial", 11, "bold"),
            bg="#2471a3",
            fg="white",
            relief="flat",
            width=14,
            height=2,
        )
        train_btn.pack(side="left", padx=(0, 16))

        tk.Button(
            btn_row,
            text="Back to Main Menu",
            command=dialog.destroy,
            width=18,
        ).pack(side="left")

        # ======================================================
        # STEP 4 — Monitor progress
        # ======================================================
        sec4 = _section(
            outer,
            row=7,
            title="Step 4  —  Monitor Training Progress",
            tooltip=(
                "The log below updates as each epoch completes, showing:\n"
                "  Training Loss  — MSE on the training split\n"
                "  Validation Loss — MSE on the held-out 15% validation split\n\n"
                "The progress bar pulses left-right while training is running\n"
                "and stops automatically when training finishes or early-stops.\n\n"
                "If you see an error in the log, check that:\n"
                "  - Your dataset has at least 10 simulations (needed for a\n"
                "    non-empty validation split at 15%).\n"
                "  - All Simulation_N/ folders contain both .npy files."
            ),
        )

        # Indeterminate progress bar: pulses left/right during training.
        progress = ttk.Progressbar(sec4, orient=tk.HORIZONTAL, length=700, mode="indeterminate")
        progress.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(4, 6))

        # Scrollable log Text widget — read-only for the user.
        log_frame = tk.Frame(sec4)
        log_frame.grid(row=2, column=0, columnspan=3, sticky="ew")
        log_frame.columnconfigure(0, weight=1)

        log = tk.Text(log_frame, height=12, font=("Courier", 9), state="disabled", bg="#f8f8f8")
        log_scroll = ttk.Scrollbar(log_frame, orient="vertical", command=log.yview)
        log.configure(yscrollcommand=log_scroll.set)
        log.grid(row=0, column=0, sticky="ew")
        log_scroll.grid(row=0, column=1, sticky="ns")

        # ---- Loss curve (populated after training) ----
        plot_lf = ttk.LabelFrame(sec4, text="  Training Loss Curve  ", padding=(6, 4))
        plot_lf.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(8, 4))
        plot_lbl = tk.Label(
            plot_lf,
            text="(Loss curve will appear here after training completes.)",
            font=HINT_FONT,
            fg=HINT_COLOR,
        )
        plot_lbl.pack(pady=8)

        def _show_plot(path):
            try:
                img = Image.open(path)
                img = img.resize((860, 320), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                plot_lbl.config(image=photo, text="")
                plot_lbl._photo = photo  # prevent GC
                outer.update_idletasks()
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception as exc:  # pylint: disable=broad-except
                plot_lbl.config(text=f"Could not load loss curve: {exc}")

        def _log_write(msg, color="black"):
            """
            Append *msg* to the log Text widget and scroll to the bottom.

            Briefly enables the widget (which is otherwise read-only) to insert
            the text, then disables it again.

            Args:
                msg (str): Line to append.
                color (str): Reserved for future coloured log lines.
            """
            log.config(state="normal")
            log.insert(tk.END, msg + "\n")
            log.see(tk.END)  # Auto-scroll to the latest line
            log.config(state="disabled")

        def _do_train():
            """
            Validate the chosen folder and launch training in a background thread.

            Guards:
              - Shows an error dialog if no folder has been selected.
              - Disables the Train button while training is in progress so the
                user cannot accidentally launch a second training run.

            The inner ``_run()`` closure captures ``folder`` and calls
            ``train_save_gui()``, which handles dataset loading, model
            creation, training, and saving automatically.
            """
            folder = data_folder_var.get()
            if not folder or not os.path.isdir(folder):
                messagebox.showerror(
                    "No folder selected", "Please browse to your training data folder in Step 1 first."
                )
                return

            # Start animating before the thread starts so the UI feels immediate.
            progress.start(15)
            train_btn.config(state="disabled", text="Training...")
            _log_write(f"Starting training on: {folder}")
            _log_write("(This may take several minutes depending on dataset size.)")

            def _run():
                """
                Worker function executed in the background daemon thread.

                Dispatches to the selected architecture's train_save_*_gui()
                function.  The try/except/finally ensures the progress bar
                always stops and the Train button is re-enabled.
                """
                arch = model_type_var.get()
                try:
                    if arch == "fc":
                        train_save_gui(folder)
                        save_subdir = "saved_model"
                        model_file = "trained_displacement_predictor_full_model.pth"
                    elif arch == "conv":
                        train_save_conv_gui(folder)
                        save_subdir = "saved_model_conv"
                        model_file = "trained_conv_decoder_full_model.pth"
                    else:  # siren
                        train_save_siren_gui(folder)
                        save_subdir = "saved_model_siren"
                        model_file = "trained_siren_full_model.pth"

                    save_path = os.path.join(folder, save_subdir, model_file)
                    _log_write("\nTraining complete!")
                    _log_write(f"Model saved to:\n  {save_path}")
                    plot_path = os.path.join(folder, save_subdir, "training_loss_curve.png")
                    if os.path.exists(plot_path):
                        _log_write(f"Loss curve: {plot_path}")
                        dialog.after(0, lambda: _show_plot(plot_path))
                    _log_write("\nYou can now use 'Load Model' from the main menu to evaluate it.")
                except Exception as exc:  # pylint: disable=broad-except
                    _log_write(f"\nERROR during training:\n  {exc}")
                finally:
                    # Always clean up the UI regardless of success or failure.
                    progress.stop()
                    train_btn.config(state="normal", text="Train")

            # daemon=True so the thread is killed automatically if the window is closed.
            threading.Thread(target=_run, daemon=True).start()

        # Assign the command now that _do_train is fully defined.
        train_btn.config(command=_do_train)

    # ------------------------------------------------------------------
    # Load Model dialog
    # ------------------------------------------------------------------

    def load_model_dialog(self):
        """
        Open the step-by-step model-evaluation dialog.

        Based on the README 'Load Model' and 'Predict and Visualise
        Deformation' sections.  Each section has a '? Help' tooltip so the
        dialog stays compact.  Four numbered steps:

        1. Browse to the trained .pth model file.
        2. Browse to the peen-intensity folder containing checkerboard.npy.
        3. Set an output folder for predicted displacement files.
        4. Evaluate then visualise.
        """
        dialog = tk.Toplevel(self.root)
        dialog.title("Load Model & Evaluate")
        dialog.geometry(self.dialog_size)
        dialog.resizable(True, True)
        dialog.transient(self.root)
        dialog.lift()
        dialog.focus_force()

        # ---- Scrollable canvas ----
        canvas = tk.Canvas(dialog, highlightthickness=0)
        scrollbar = ttk.Scrollbar(dialog, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        outer = tk.Frame(canvas, padx=18, pady=12)
        canvas_window = canvas.create_window((0, 0), window=outer, anchor="nw")

        def _on_frame_configure(event):
            """Recalculate the canvas scroll region whenever the inner frame resizes."""
            canvas.configure(scrollregion=canvas.bbox("all"))

        def _on_canvas_configure(event):
            """Keep the inner frame exactly as wide as the canvas (minus scrollbar)."""
            canvas.itemconfig(canvas_window, width=event.width)

        # Bind resize callbacks so the scrollable area always fits the content.
        outer.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        outer.columnconfigure(0, weight=1)

        # ---- Page title ----
        tk.Label(
            outer,
            text="Load a Trained Model & Predict Deformation",
            font=("Arial", 15, "bold"),
            fg=STEP_COLOR,
        ).grid(row=0, column=0, sticky="w", pady=(4, 2))
        tk.Label(
            outer,
            text=(
                "Use a trained model to instantly predict surface deformation from a "
                "shot peening recipe.  Hover '? Help' for details on each step."
            ),
            font=HINT_FONT,
            fg=HINT_COLOR,
            justify="left",
            wraplength=880,
        ).grid(row=1, column=0, sticky="w", pady=(0, 6))
        ttk.Separator(outer, orient="horizontal").grid(row=2, column=0, sticky="ew", pady=(0, 8))

        # ======================================================
        # STEP 1 — Select the trained model file
        # ======================================================
        sec1 = _section(
            outer,
            row=3,
            title="Step 1  —  Select the Trained Model File",
            tooltip=(
                "Browse to the PyTorch model file (.pth) saved during training.\n\n"
                "Default location:\n"
                "  <dataset_folder>/saved_model/\n"
                "      trained_displacement_predictor_full_model.pth\n\n"
                "If you have not trained a model yet, close this dialog and use\n"
                "'Train Model' from the main menu first."
            ),
        )

        tk.Label(sec1, text="Model File  (.pth)", font=BODY_FONT).grid(row=1, column=0, sticky="w", pady=(4, 2))

        # StringVar stores the .pth file path chosen by the user.
        model_file_var = tk.StringVar()
        ef1 = tk.Frame(sec1)
        ef1.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ef1.columnconfigure(0, weight=1)
        tk.Entry(ef1, textvariable=model_file_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            ef1,
            text="Browse...",
            command=lambda: self.browse_file(model_file_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # ======================================================
        # STEP 2 — Select the peen intensity folder to evaluate
        # ======================================================
        sec2 = _section(
            outer,
            row=4,
            title="Step 2  —  Select Peen Intensity to Evaluate",
            tooltip=(
                "Browse to the simulation folder that contains the shot peening\n"
                "recipe you want to evaluate.\n\n"
                "You can select EITHER:\n"
                "  - A single Simulation_N/ folder (e.g. Simulation_497/)\n"
                "  - A parent folder containing multiple Simulation_N/ sub-folders\n\n"
                "Required files in the folder:\n"
                "  checkerboard.npy      — G x G shot-intensity grid (model input)\n"
                "  displacements.npy     — ground-truth displacements (for MSE metric)\n"
                "  node_coords.npy       — mesh node coordinates (deformation preview)\n"
                "  node_labels.npy       — mesh node labels\n"
                "  disp_node_labels.npy  — displacement node label mapping\n\n"
                "Tip: Click 'Preview Input Pattern' to confirm the shot distribution\n"
                "before running the model."
            ),
        )

        tk.Label(sec2, text="Peen Intensity Folder", font=BODY_FONT).grid(row=1, column=0, sticky="w", pady=(4, 2))

        checkerboard_folder_var = tk.StringVar()
        ef2 = tk.Frame(sec2)
        ef2.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ef2.columnconfigure(0, weight=1)
        tk.Entry(ef2, textvariable=checkerboard_folder_var, font=BODY_FONT).grid(
            row=0, column=0, sticky="ew", padx=(0, 6)
        )
        tk.Button(
            ef2,
            text="Browse...",
            command=lambda: self.browse_directory(checkerboard_folder_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # Inline 'Preview' button — shows the checkerboard colour-map before evaluation.
        preview_row = tk.Frame(sec2)
        preview_row.grid(row=3, column=0, columnspan=3, sticky="w", pady=(2, 4))
        tk.Button(
            preview_row,
            text="Preview Input Pattern",
            command=lambda: self.preview_file(checkerboard_folder_var.get()),
            width=22,
        ).pack(side="left")
        tk.Label(
            preview_row,
            text="  Opens a colour-map of checkerboard.npy",
            font=HINT_FONT,
            fg=HINT_COLOR,
        ).pack(side="left")

        # ======================================================
        # STEP 3 — Set the output path
        # ======================================================
        sec3 = _section(
            outer,
            row=5,
            title="Step 3  —  Set Output Path for Predictions",
            tooltip=(
                "Choose the folder where predicted displacement files will be written.\n\n"
                "The model creates one sub-folder per input simulation:\n"
                "  <output>/Simulation_0/pred_displacements.npy — (N_nodes, 3) array\n"
                "  <output>/Simulation_0/pred_displacements.csv — same data as CSV\n\n"
                "These files have the same shape as displacements.npy from your\n"
                "simulation data, so data_viz.py works on them directly.\n\n"
                "Tip: Use an empty folder to avoid mixing old and new predictions."
            ),
        )

        tk.Label(sec3, text="Output Path (folder)", font=BODY_FONT).grid(row=1, column=0, sticky="w", pady=(4, 2))

        output_path_var = tk.StringVar()
        ef3 = tk.Frame(sec3)
        ef3.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ef3.columnconfigure(0, weight=1)
        tk.Entry(ef3, textvariable=output_path_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            ef3,
            text="Browse...",
            command=lambda: self.browse_directory(output_path_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # ======================================================
        # OPTIONAL — Material Selection (inference conditioning)
        # ======================================================
        sec_mat = _section(
            outer,
            row=6,
            title="Optional  —  Material Properties for Inference",
            tooltip=(
                "Select the workpiece and shot materials used during shot peening.\n\n"
                "Only applies to material-conditioned models (trained with\n"
                "use_material=True / mat_dim=7).  Standard models ignore these\n"
                "selections — leave blank and they will be silently skipped.\n\n"
                "Workpiece: the part being peened (e.g. 316L-SS, Ti-6Al-4V).\n"
                "Shot:      the peening media (e.g. ceramic, steel).\n\n"
                "Leave both blank to use the model's built-in defaults."
            ),
        )

        try:
            from materials import WORKPIECE_MATERIALS as _WPM, SHOT_MATERIALS as _SPM

            _infer_wp_names = [""] + sorted(_WPM.keys())
            _infer_sp_names = [""] + sorted(_SPM.keys())
        except ImportError:
            _WPM = {}
            _SPM = {}
            _infer_wp_names = [""]
            _infer_sp_names = [""]

        infer_wp_var = tk.StringVar(value="")
        infer_sp_var = tk.StringVar(value="")

        mat_row = tk.Frame(sec_mat)
        mat_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 2))

        tk.Label(mat_row, text="Workpiece:", font=BODY_FONT, anchor="e").pack(side="left", padx=(6, 4))
        ttk.Combobox(mat_row, textvariable=infer_wp_var, values=_infer_wp_names, width=18, state="readonly").pack(
            side="left"
        )
        tk.Label(mat_row, text="  Shot:", font=BODY_FONT, anchor="e").pack(side="left", padx=(16, 4))
        ttk.Combobox(mat_row, textvariable=infer_sp_var, values=_infer_sp_names, width=18, state="readonly").pack(
            side="left"
        )

        tk.Label(
            sec_mat,
            text="(only used by material-conditioned models; ignored otherwise)",
            font=HINT_FONT,
            fg=HINT_COLOR,
        ).grid(row=2, column=0, columnspan=3, sticky="w", padx=6, pady=(0, 4))

        def _build_mat_features():
            """Return a normalised (7,) ndarray, or None if no materials selected."""
            wp_name = infer_wp_var.get().strip()
            sp_name = infer_sp_var.get().strip()
            if not wp_name and not sp_name:
                return None
            try:
                import numpy as _np
                from materials import get_workpiece, get_shot
                from model import normalize_mat_features as _norm

                wp = (
                    get_workpiece(wp_name) if wp_name else {"E": 113.8e9, "nu": 0.342, "sigma_yield": 880e6, "c": 3.0e9}
                )
                sp = get_shot(sp_name) if sp_name else {"rho_s": 7800.0, "E_s": 210e9, "nu_s": 0.30}
                raw = _np.array(
                    [
                        wp["E"],
                        wp["nu"],
                        wp["sigma_yield"],
                        wp["c"],
                        sp["E_s"],
                        sp["nu_s"],
                        sp["rho_s"],
                    ],
                    dtype=_np.float32,
                )
                return _norm(raw)
            except Exception as _e:
                print(f"[Material] Could not build feature vector: {_e}")
                return None

        # ======================================================
        # OPTIONAL — Curved Surface + Nozzle Trajectory
        # ======================================================
        sec_curved = _section(
            outer,
            row=7,
            title="Optional  —  Curved Surface & Nozzle Trajectory",
            tooltip=(
                "Supply an STL file to predict deformation on a 3D curved surface\n"
                "instead of the standard flat-plate mode.\n\n"
                "STL file (optional):\n"
                "  ASCII or binary triangle-mesh .stl file (e.g. aircraft panel,\n"
                "  turbine blade).  Requires 'trimesh' (pip install trimesh).\n\n"
                "Nozzle trajectory (optional):\n"
                "  Choose a built-in parametric scan pattern (raster / spiral /\n"
                "  zigzag) or load an arbitrary waypoint file (CSV with x,y,z\n"
                "  columns, or .npy with shape (T,3) or (T,4)).\n\n"
                "Leave both blank to use the standard flat-plate evaluation."
            ),
        )

        # STL file picker
        tk.Label(sec_curved, text="STL Surface File  (optional)", font=BODY_FONT).grid(
            row=1, column=0, sticky="w", pady=(4, 2)
        )
        stl_file_var = tk.StringVar()
        ef_stl = tk.Frame(sec_curved)
        ef_stl.grid(row=2, column=0, columnspan=3, sticky="ew", pady=4)
        ef_stl.columnconfigure(0, weight=1)
        tk.Entry(ef_stl, textvariable=stl_file_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            ef_stl,
            text="Browse...",
            command=lambda: self.browse_file(stl_file_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # Trajectory sub-section
        tk.Label(sec_curved, text="Nozzle Trajectory  (optional)", font=BODY_FONT).grid(
            row=3, column=0, sticky="w", pady=(8, 2)
        )

        traj_mode_var = tk.StringVar(value="none")
        mode_frame = tk.Frame(sec_curved)
        mode_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 4))

        tk.Radiobutton(
            mode_frame, text="None (static nozzle)", variable=traj_mode_var, value="none", font=BODY_FONT
        ).pack(side="left", padx=(0, 12))
        tk.Radiobutton(
            mode_frame, text="Parametric scan", variable=traj_mode_var, value="parametric", font=BODY_FONT
        ).pack(side="left", padx=(0, 12))
        tk.Radiobutton(mode_frame, text="Waypoint file", variable=traj_mode_var, value="file", font=BODY_FONT).pack(
            side="left"
        )

        # Parametric scan parameters
        param_frame = tk.Frame(sec_curved)
        param_frame.grid(row=5, column=0, columnspan=3, sticky="ew", pady=4)
        param_frame.columnconfigure(1, weight=0)

        _scan_pattern_var = tk.StringVar(value="raster")
        _scan_speed_var = tk.StringVar(value="0.05")
        _line_spacing_var = tk.StringVar(value="0.005")
        _z_standoff_var = tk.StringVar(value="0.15")
        _shots_step_var = tk.StringVar(value="10")
        _cb_size_var = tk.StringVar(value="20")

        def _prow(label, var, row, hint=""):
            tk.Label(param_frame, text=label, font=HINT_FONT, fg=HINT_COLOR, anchor="w").grid(
                row=row, column=0, sticky="w", padx=(8, 4), pady=1
            )
            tk.Entry(param_frame, textvariable=var, width=10, font=BODY_FONT).grid(
                row=row, column=1, sticky="w", pady=1
            )
            if hint:
                tk.Label(param_frame, text=hint, font=HINT_FONT, fg=HINT_COLOR).grid(
                    row=row, column=2, sticky="w", padx=4
                )

        tk.Label(param_frame, text="Scan pattern:", font=HINT_FONT, fg=HINT_COLOR).grid(
            row=0, column=0, sticky="w", padx=(8, 4), pady=1
        )
        ttk.Combobox(
            param_frame,
            textvariable=_scan_pattern_var,
            values=["raster", "spiral", "zigzag"],
            state="readonly",
            width=10,
        ).grid(row=0, column=1, sticky="w", pady=1)
        _prow("Scan speed (m/s):", _scan_speed_var, 1)
        _prow("Line spacing (m):", _line_spacing_var, 2)
        _prow("Z standoff (m):", _z_standoff_var, 3)
        _prow("Shots per step:", _shots_step_var, 4)
        _prow("Checkerboard size G:", _cb_size_var, 5)

        # Waypoint file picker
        waypoint_frame = tk.Frame(sec_curved)
        waypoint_frame.grid(row=6, column=0, columnspan=3, sticky="ew", pady=4)
        waypoint_frame.columnconfigure(0, weight=1)
        tk.Label(waypoint_frame, text="Waypoint file (.csv or .npy):", font=HINT_FONT, fg=HINT_COLOR).grid(
            row=0, column=0, sticky="w", padx=(8, 4)
        )
        waypoint_file_var = tk.StringVar()
        ef_wp = tk.Frame(waypoint_frame)
        ef_wp.grid(row=1, column=0, columnspan=2, sticky="ew", padx=8)
        ef_wp.columnconfigure(0, weight=1)
        tk.Entry(ef_wp, textvariable=waypoint_file_var, font=BODY_FONT).grid(row=0, column=0, sticky="ew", padx=(0, 6))
        tk.Button(
            ef_wp,
            text="Browse...",
            command=lambda: self.browse_file(waypoint_file_var, parent=dialog),
            width=10,
        ).grid(row=0, column=1)

        # ======================================================
        # STEP 4 — Run evaluation and visualise results
        # ======================================================
        sec4 = _section(
            outer,
            row=8,
            title="Step 4  —  Evaluate the Model & Visualise Results",
            tooltip=(
                "Run the buttons in order:\n\n"
                "1. Evaluate Model\n"
                "   Feeds checkerboard.npy through the trained CNN and writes\n"
                "   pred_displacements.npy/.csv to your output folder.\n"
                "   Also prints MSE and sMAPE vs. the ground-truth displacements.\n\n"
                "2. Preview Deformation\n"
                "   Opens a matplotlib window showing the predicted deformation\n"
                "   overlaid on the mesh.  Requires 'Evaluate Model' to have run\n"
                "   first (needs pred_displacements.npy in the output folder).\n\n"
                "Both windows are non-blocking — the dialog stays open while\n"
                "you view the plots."
            ),
        )

        btn_row = tk.Frame(sec4)
        btn_row.grid(row=1, column=0, columnspan=3, sticky="w", pady=(6, 4))

        # Button 1: run the CNN and save predicted displacements
        def _do_evaluate():
            stl = stl_file_var.get().strip()
            mode = traj_mode_var.get()
            if stl:
                # Curved-surface inference path
                try:
                    from nozzle_trajectory import (
                        ScanParams,
                        raster_scan,
                        spiral_scan,
                        zigzag_scan,
                        from_csv,
                        from_npy,
                    )

                    G = int(_cb_size_var.get())
                    traj = None
                    if mode == "parametric":
                        sp = ScanParams(
                            pattern=_scan_pattern_var.get(),
                            Lx=0.04,
                            Ly=0.04,  # sensible defaults; STL bounds override in code
                            z_standoff=float(_z_standoff_var.get()),
                            scan_speed=float(_scan_speed_var.get()),
                            line_spacing=float(_line_spacing_var.get()),
                        )
                        builders = {"raster": raster_scan, "spiral": spiral_scan, "zigzag": zigzag_scan}
                        traj = builders.get(sp.pattern, raster_scan)(sp)
                    elif mode == "file":
                        wp = waypoint_file_var.get().strip()
                        if wp:
                            if wp.lower().endswith(".csv"):
                                traj = from_csv(wp, z_standoff=float(_z_standoff_var.get()))
                            else:
                                traj = from_npy(wp, z_standoff=float(_z_standoff_var.get()))

                    kwargs = dict(
                        n_shots_per_step=int(_shots_step_var.get()),
                        h_nozzle=float(_z_standoff_var.get()),
                    )
                    # For "static nozzle" (no trajectory), use the eval folder's
                    # existing checkerboard as the peening-intensity input.
                    # Passing zeros would mean "no peening" which is not useful.
                    if traj is None:
                        _eval_cb_path = os.path.join(checkerboard_folder_var.get(), "checkerboard.npy")
                        if os.path.exists(_eval_cb_path):
                            import numpy as _np

                            traj_input = _np.load(_eval_cb_path)
                        else:
                            import numpy as _np

                            traj_input = _np.zeros((G, G), dtype="float32")
                    else:
                        traj_input = traj
                    curved_surface_inference(
                        model_path=model_file_var.get(),
                        stl_path=stl,
                        trajectory_or_checkerboard=traj_input,
                        G=G,
                        pred_save_dir=output_path_var.get(),
                        **kwargs,
                    )
                    print("Curved-surface inference complete.")
                except Exception as e:  # pylint: disable=broad-except
                    import traceback

                    print(f"Curved-surface inference failed: {e}")
                    traceback.print_exc()
            else:
                # Standard flat-plate path
                load_and_evaluate_model_gui(
                    model_file_var.get(),
                    checkerboard_folder_var.get(),
                    output_path_var.get(),
                    mat_features=_build_mat_features(),
                )

        tk.Button(
            btn_row,
            text="1. Evaluate Model",
            command=_do_evaluate,
            width=22,
            height=2,
            bg="#1a7a4a",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(side="left", padx=(0, 12))

        # Button 2: open the deformation visualiser (needs Evaluate to have run first)
        tk.Button(
            btn_row,
            text="2. Preview Deformation",
            command=lambda: self.preview_deformation(
                checkerboard_folder_var.get(),
                output_path_var.get(),
            ),
            width=22,
            height=2,
            bg="#7d6608",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(side="left", padx=(0, 12))

        # Button 3: 3D STL deformation preview (curved-surface path only)
        def _preview_stl():
            import os as _os
            import numpy as _np

            out_dir = output_path_var.get().strip()
            stl = stl_file_var.get().strip()
            pred_npy = _os.path.join(out_dir, "pred_displacements_on_stl.npy")
            if not stl:
                tk.messagebox.showwarning(
                    "No STL",
                    "Load an STL file before previewing STL deformation.",
                    parent=dialog,
                )
                return
            if not _os.path.exists(pred_npy):
                tk.messagebox.showwarning(
                    "Run Evaluate First",
                    "pred_displacements_on_stl.npy not found.\n" "Run '1. Evaluate Model' with an STL file first.",
                    parent=dialog,
                )
                return
            try:
                from stl_surface import STLSurface

                surface = STLSurface(stl)
                disp = _np.load(pred_npy)
                if len(disp) != surface.n_vertices:
                    tk.messagebox.showerror(
                        "Vertex count mismatch",
                        f"pred_displacements_on_stl.npy has {len(disp)} rows "
                        f"but the STL has {surface.n_vertices} vertices.\n\n"
                        "The model needs reference_node_coords.npy in its "
                        "saved_model/ directory so that predictions can be "
                        "spatially interpolated onto the STL mesh (Layer 2).\n\n"
                        "To fix: retrain with a dataset that includes "
                        "node_coords.npy in each Simulation_N/ folder.",
                        parent=dialog,
                    )
                    return
                visualize_stl_deformation(surface, disp, show=True)
            except Exception as _e:  # pylint: disable=broad-except
                tk.messagebox.showerror("Preview Error", str(_e), parent=dialog)

        tk.Button(
            btn_row,
            text="3. Preview STL Deformation",
            command=_preview_stl,
            width=24,
            height=2,
            bg="#1a5276",
            fg="white",
            font=("Arial", 10, "bold"),
            relief="flat",
        ).pack(side="left", padx=(0, 12))

        tk.Button(
            btn_row,
            text="Back to Main Menu",
            command=dialog.destroy,
            width=18,
            height=2,
        ).pack(side="left")

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_file_path(self, relative_path):
        """
        Resolve a relative path to an absolute one, handling both normal
        execution and PyInstaller-bundled (.exe) execution.

        When running as a PyInstaller bundle, ``sys._MEIPASS`` is the
        temporary directory where bundled assets are extracted.  In dev
        mode the current working directory is used instead.

        Args:
            relative_path (str): Path relative to the application root.
                Forward slashes are normalised to the OS separator so
                that ``"data/file.npy"`` works on Windows too.

        Returns:
            str: The corresponding absolute path.
        """
        base_path = getattr(sys, "_MEIPASS", os.path.abspath("."))
        # Normalise only the relative portion so that forward-slash inputs
        # are converted to the OS separator (backslash on Windows) before
        # joining, while leaving the base_path's separator style untouched.
        return os.path.join(base_path, os.path.normpath(relative_path))

    def browse_file(self, variable, parent=None):
        """
        Open a file-picker dialog and store the chosen path in *variable*.

        Does nothing if the user cancels (returns an empty string), which
        preserves whatever value was in the StringVar before.

        Args:
            variable (tk.StringVar): The variable to update with the chosen path.
            parent: Parent window so the picker stays in front of it.
        """
        filepath = filedialog.askopenfilename(parent=parent or self.root)
        if filepath:
            variable.set(filepath)
        if parent:
            parent.lift()
            parent.focus_force()

    def browse_directory(self, variable, parent=None):
        """
        Open a directory-picker dialog and store the chosen path in *variable*.

        Does nothing if the user cancels (returns an empty string), which
        preserves whatever value was in the StringVar before.

        Args:
            variable (tk.StringVar): The variable to update with the chosen path.
            parent: Parent window so the picker stays in front of it.
        """
        dirpath = filedialog.askdirectory(parent=parent or self.root)
        if dirpath:
            variable.set(dirpath)
        if parent:
            parent.lift()
            parent.focus_force()

    def preview_file(self, folder_path):
        """
        Visualise the checkerboard pattern (shot intensity grid) for a
        simulation folder.

        Matches the README 'Data Visualization' section.  Validates the path
        before scheduling the preview so that the user sees a clear error
        message rather than a crash.

        Validation order:
          1. Path must exist (non-empty string pointing to something on disk).
          2. Path must be a directory, not a file.
          3. If the directory is empty, warn the user.
          4. Must contain checkerboard.npy.

        Args:
            folder_path (str): Path to the simulation folder to preview.
        """
        # Guard: path must exist on disk
        if not folder_path or not os.path.exists(folder_path):
            messagebox.showerror("Error", f"The Folder path does not exist: {folder_path}")
            return

        # Guard: path must be a directory, not a file
        if not os.path.isdir(folder_path):
            messagebox.showerror("Error", f"The path is not a directory: {folder_path}")
            return

        # Guard: directory must not be empty
        if not os.listdir(folder_path):
            messagebox.showwarning("Warning", "The directory is empty.")
            return

        # Guard: checkerboard.npy must be present inside the folder
        cb_path = os.path.join(folder_path, "checkerboard.npy")
        if not os.path.exists(cb_path):
            messagebox.showerror(
                "checkerboard.npy not found",
                f"Could not find checkerboard.npy in:\n{folder_path}\n\n"
                "Make sure you selected a single Simulation_N/ subfolder "
                "(not the parent dataset folder).",
            )
            return

        # All checks passed — schedule the visualiser on the main thread.
        # Matplotlib requires the main thread for its GUI event loop; calling it
        # from a background thread produces warnings and can crash.
        # root.after(0, ...) posts the call onto Tkinter's event queue so it
        # executes on the main thread immediately after this callback returns.
        self.root.after(0, self.run_preview, folder_path)

    def run_preview(self, geometry_folder_path):
        """
        Open the checkerboard colour-map visualiser.

        Delegates to ``visualize_checkerboard`` from data_viz.py.
        Must be called on the main thread (Tkinter's event loop) so that
        matplotlib can create its GUI window.  Scheduled via
        ``root.after(0, ...)`` from ``preview_file``.

        Args:
            geometry_folder_path (str): Path to the folder containing
                checkerboard.npy (and optionally other mesh files).
        """
        try:
            visualize_checkerboard(geometry_folder_path)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Preview Error", f"Could not display checkerboard:\n{exc}")

    def preview_deformation(self, test_folder_path, deformation_folder_path):
        """
        Copy mesh files from the input folder to the output folder, then
        visualise the predicted deformation.

        Matches the README 'Predict and Visualise Deformation' section.
        The mesh files (node_coords.npy, node_labels.npy, disp_node_labels.npy)
        must live in the peen-intensity folder (Step 2); the predicted
        displacements live in the output folder (Step 3).  This method copies
        the mesh files across so that visualize_all() can find everything in
        one place.

        Validation order:
          1. Both folder paths must exist.
          2. All three required mesh files must be present in the input folder.
          3. Predicted displacements.npy must exist in the output folder
             (i.e. 'Evaluate Model' must have been run first).

        Args:
            test_folder_path (str): Peen intensity / input folder.
            deformation_folder_path (str): Output folder where predictions live.
        """
        # Guard: both folders must be valid directories
        if not test_folder_path or not os.path.isdir(test_folder_path):
            messagebox.showerror("Error", "Please browse to a valid peen intensity folder in Step 2 first.")
            return
        if not deformation_folder_path or not os.path.isdir(deformation_folder_path):
            messagebox.showerror("Error", "Please browse to a valid output folder in Step 3 first.")
            return

        # --- Curved-surface path ---------------------------------------------------
        # curved_surface_inference() saves pred_displacements_on_stl.npy and all
        # mesh arrays (node_coords, node_labels, etc.) directly into the output dir.
        # We must NOT overwrite those STL-derived mesh files with the flat-plate
        # ones from test_folder_path, so we handle this case first and return early.
        stl_pred = os.path.join(deformation_folder_path, "pred_displacements_on_stl.npy")
        if os.path.exists(stl_pred):
            try:
                shutil.copy2(stl_pred, os.path.join(deformation_folder_path, "displacements.npy"))
            except Exception as exc:  # pylint: disable=broad-except
                messagebox.showerror("Error", f"Could not stage STL predicted displacements: {exc}")
                return
            self.root.after(0, lambda: visualize_all(deformation_folder_path, scale_factor=1))
            return

        # --- Flat-plate path -------------------------------------------------------
        # Guard: mesh files must be present in the input folder
        required = ["node_coords.npy", "node_labels.npy", "disp_node_labels.npy"]
        missing = [f for f in required if not os.path.exists(os.path.join(test_folder_path, f))]
        if missing:
            messagebox.showerror(
                "Missing mesh files",
                "The peen intensity folder is missing:\n  " + "\n  ".join(missing) + "\n\n"
                "Make sure you selected a Simulation_N/ folder that contains these mesh files.",
            )
            return

        # Copy mesh files to the output folder so visualize_all() can find them.
        try:
            for fname in required:
                shutil.copy2(
                    os.path.join(test_folder_path, fname),
                    deformation_folder_path,
                )
            # Also copy optional files used by visualize_all; skip silently if absent.
            for opt in ["element_connectivity.npy", "checkerboard.npy"]:
                src = os.path.join(test_folder_path, opt)
                if os.path.exists(src):
                    shutil.copy2(src, deformation_folder_path)
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Could not copy mesh files: {exc}")
            return

        # Guard: predicted displacements must exist (i.e. Evaluate Model must have run).
        # load_and_evaluate_model_gui() saves predictions as:
        #   <output>/Simulation_0/pred_displacements.npy
        # (not 'displacements.npy' in the root — that's ground-truth from the input folder).
        pred_npy = os.path.join(deformation_folder_path, "Simulation_0", "pred_displacements.npy")
        if not os.path.exists(pred_npy):
            messagebox.showerror(
                "No prediction found",
                "pred_displacements.npy was not found in the output folder.\n\n"
                "Please click 'Evaluate Model' first to generate predictions, "
                "then try Preview again.",
            )
            return

        # visualize_all() → compute_deformed_mesh() reads 'displacements.npy' by that
        # exact name.  Copy the predicted file into the output root under that name so
        # data_viz.py can find it without any changes to that module.
        try:
            shutil.copy2(pred_npy, os.path.join(deformation_folder_path, "displacements.npy"))
        except Exception as exc:  # pylint: disable=broad-except
            messagebox.showerror("Error", f"Could not stage predicted displacements: {exc}")
            return

        # All checks passed — schedule the visualiser on the main thread.
        self.root.after(0, lambda: visualize_all(deformation_folder_path, scale_factor=1))

    def check_file_in_folder(self, folder_path, file_name):
        """
        Check whether *file_name* exists inside *folder_path*.

        A thin wrapper around os.path.exists that keeps callers readable.
        Returns False for non-existent folders without raising an exception.

        Args:
            folder_path (str): The directory to search in.
            file_name (str): The file name to look for.

        Returns:
            bool: True if the file exists, False otherwise.
        """
        return os.path.exists(os.path.join(folder_path, file_name))

    def train_model(self, data_folder):
        """
        Programmatically train a model using the data in *data_folder*.

        This method is the programmatic counterpart to the GUI Train button in
        ``train_model_dialog``.  It is used by tests and by callers that do not
        want the full dialog UX.

        ``num_nodes`` and ``checkerboard_size`` are auto-detected from the
        dataset via ``infer_dataset_shape``; no hard-coded values are needed.

        Args:
            data_folder (str): Path to the parent folder containing
                Simulation_N/ sub-folders with checkerboard.npy and
                displacements.npy.
        """
        # Guard: make sure the path exists before going further
        if not os.path.exists(data_folder):
            messagebox.showerror("Error", f"The folder path does not exist: {data_folder}")
            return
        try:
            # Detect dataset dimensions automatically so the CNN is built correctly.
            num_nodes, checkerboard_size = infer_dataset_shape(data_folder)
        except FileNotFoundError as exc:
            messagebox.showerror("Error", str(exc))
            return

        # Build data loaders, model, loss, and optimizer then run training.
        train_loader, val_loader, _, _ = create_data_loaders(data_folder)
        model = create_model(input_channels=1, num_nodes=num_nodes, checkerboard_size=checkerboard_size)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        train_model(model, train_loader, val_loader, criterion, optimizer, None)
        messagebox.showinfo("Training Complete", "Model training completed successfully!")

    def start_training(self, log_widget, progress_bar):
        """
        Legacy helper: insert a 'Training started' message and start the bar.

        Kept for test compatibility.  The real training now runs through
        ``train_model_dialog``'s ``_do_train`` closure which calls
        ``train_save_gui`` in a daemon thread.

        Args:
            log_widget (tk.Text): The log widget to write into.
            progress_bar (ttk.Progressbar): The bar to animate.
        """
        log_widget.config(state="normal")
        log_widget.insert(tk.END, "Training started...\n")
        log_widget.see(tk.END)
        log_widget.config(state="disabled")
        progress_bar.start(10)
        # Schedule finish_training to be called after a short delay (demo only)
        self.root.after(3000, lambda: self.finish_training(log_widget, progress_bar))

    def finish_training(self, log_widget, progress_bar):
        """
        Legacy helper: stop the progress bar and log a completion message.

        Kept for test compatibility.  Pair with ``start_training``.

        Args:
            log_widget (tk.Text): The log widget to write into.
            progress_bar (ttk.Progressbar): The bar to stop.
        """
        progress_bar.stop()
        log_widget.config(state="normal")
        log_widget.insert(tk.END, "Training completed!\n")
        log_widget.see(tk.END)
        log_widget.config(state="disabled")

    def num_of_simulations(self, folder_path):
        """
        Count the number of Simulation_N/ sub-folders inside *folder_path*.

        A sub-folder is counted only if its name matches the exact pattern
        'Simulation_' followed by one or more decimal digits (e.g.
        'Simulation_0', 'Simulation_42').  Folders like 'Simulation_abc' or
        'Simulation_' are ignored.

        Args:
            folder_path (str): The parent directory to inspect.

        Returns:
            int: The number of valid Simulation_N/ sub-folders found.
        """
        simulation_folders = [
            os.path.join(folder_path, folder)
            for folder in os.listdir(folder_path)
            if folder.startswith("Simulation_") and folder[len("Simulation_") :].isdigit()
        ]
        print(f"# Simulation folders: {len(simulation_folders)}")
        return len(simulation_folders)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
