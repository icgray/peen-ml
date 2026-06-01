"""
Headless screenshot helper for peen-ml GUI documentation.
Run from the project root:  python capture_screenshots.py
Saves PNG files to images/.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src', 'peen-ml'))

import tkinter as tk
from PIL import ImageGrab

OUT = os.path.join(os.path.dirname(__file__), "images")
os.makedirs(OUT, exist_ok=True)


def _win32_foreground(hwnd):
    """Use ctypes to force a Win32 HWND to the foreground."""
    try:
        import ctypes
        ctypes.windll.user32.ShowWindow(hwnd, 9)      # SW_RESTORE
        ctypes.windll.user32.SetForegroundWindow(hwnd)
    except Exception:
        pass


def grab_widget(widget, name):
    # Move window to a screen position clear of the terminal (right side)
    widget.geometry(f"+200+40")
    widget.lift()
    widget.focus_force()
    widget.attributes('-topmost', True)
    widget.update_idletasks()
    widget.update()
    time.sleep(0.6)   # let the OS paint on top
    try:
        _win32_foreground(widget.winfo_id())
        time.sleep(0.2)
    except Exception:
        pass
    x = widget.winfo_rootx()
    y = widget.winfo_rooty()
    w = widget.winfo_width()
    h = widget.winfo_height()
    img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
    path = os.path.join(OUT, f"{name}.png")
    img.save(path)
    print(f"  Saved {path}  ({w}×{h})")


def _minimize_all_except(hwnd_keep=None):
    """Minimise every visible top-level window except hwnd_keep.
    Returns list of HWNDs that were minimised so they can be restored."""
    minimised = []
    try:
        import ctypes
        EnumWindows     = ctypes.windll.user32.EnumWindows
        ShowWindow      = ctypes.windll.user32.ShowWindow
        IsWindowVisible = ctypes.windll.user32.IsWindowVisible
        IsIconic        = ctypes.windll.user32.IsIconic
        EnumProc = ctypes.WINFUNCTYPE(ctypes.c_bool,
                                      ctypes.POINTER(ctypes.c_int),
                                      ctypes.POINTER(ctypes.c_int))

        def _cb(hwnd, _lp):
            if hwnd != hwnd_keep and IsWindowVisible(hwnd) and not IsIconic(hwnd):
                ShowWindow(hwnd, 6)   # SW_MINIMIZE
                minimised.append(hwnd)
            return True

        EnumWindows(EnumProc(_cb), 0)
    except Exception:
        pass
    return minimised


def _restore_windows(hwnds):
    try:
        import ctypes
        for hwnd in hwnds:
            ctypes.windll.user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    except Exception:
        pass


def run():
    from shotpeen_gui import App

    root = tk.Tk()
    app = App(root)
    root.update_idletasks()
    root.update()

    # 1 — Main menu: minimise everything else first
    root.geometry("+200+40")
    root.lift(); root.focus_force(); root.attributes('-topmost', True)
    root.update_idletasks(); root.update(); time.sleep(0.3)
    minimised = _minimize_all_except(hwnd_keep=root.winfo_id())
    root.lift(); root.focus_force()
    time.sleep(0.5)
    grab_widget(root, "gui_main_menu")
    # Restore other windows so dialogs can open cleanly
    _restore_windows(minimised)
    root.attributes('-topmost', False)
    time.sleep(0.3)

    # 2 — Generate Dataset dialog
    app.generate_dataset_dialog()
    root.update_idletasks(); root.update(); time.sleep(0.5)
    gen_dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    if gen_dlg:
        dlg = gen_dlg[-1]
        m2 = _minimize_all_except(hwnd_keep=dlg.winfo_id())
        dlg.lift(); dlg.focus_force(); time.sleep(0.4)
        grab_widget(dlg, "gui_generate_dataset")
        _restore_windows(m2)
        dlg.destroy()

    # 3 — Train Model dialog
    app.train_model_dialog()
    root.update_idletasks(); root.update(); time.sleep(0.5)
    train_dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    if train_dlg:
        dlg = train_dlg[-1]
        m3 = _minimize_all_except(hwnd_keep=dlg.winfo_id())
        dlg.lift(); dlg.focus_force(); time.sleep(0.4)
        grab_widget(dlg, "train_model_page")
        _restore_windows(m3)
        dlg.destroy()

    # 4 — Load & Evaluate dialog
    app.load_model_dialog()
    root.update_idletasks(); root.update(); time.sleep(0.5)
    eval_dlg = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
    if eval_dlg:
        dlg = eval_dlg[-1]
        m4 = _minimize_all_except(hwnd_keep=dlg.winfo_id())
        dlg.lift(); dlg.focus_force(); time.sleep(0.4)
        grab_widget(dlg, "load_model_page")
        _restore_windows(m4)
        dlg.destroy()

    root.destroy()
    print("Done.")


if __name__ == "__main__":
    run()
