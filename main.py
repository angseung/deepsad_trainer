"""Entry point — launches the DeepSAD Gradio web UI.

Architecture:
  - Main thread  : QApplication event loop (required for Qt file dialogs)
  - Daemon thread: Gradio web server

Gradio button handlers dispatch file dialog requests back to the main thread
via QTimer.singleShot, which is the only thread-safe Qt call from non-Qt
threads.
"""

import signal
import sys
import threading

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication

from app import build_ui


def _run_gradio() -> None:
    demo = build_ui()
    demo.launch()


if __name__ == "__main__":
    qt_app = QApplication(sys.argv)
    qt_app.setQuitOnLastWindowClosed(False)

    import app as _app_module
    _app_module._dialog_runner = _app_module._DialogRunner()

    sigint_timer = QTimer()
    sigint_timer.timeout.connect(lambda: None)
    sigint_timer.start(200)

    signal.signal(signal.SIGINT, lambda *_: qt_app.quit())

    gradio_thread = threading.Thread(target=_run_gradio, daemon=True)
    gradio_thread.start()

    sys.exit(qt_app.exec())
