"""
Application entry point for the ErrP Visualizer.

Run with::

    python -m src.main [--input FILE] [--log-level LEVEL]

The module creates a ``QApplication``, applies the default light theme,
opens a :class:`~src.gui.file_window.FileWindow`, and enters the Qt
event loop.  An optional ``--input`` flag pre-loads a single ``.set``
file on launch.
"""

import logging
import argparse
import os

from PyQt5.QtWidgets import QApplication

from src.gui.file_window import FileWindow
from src.gui.themes.light_theme import apply_light_theme

LOG_LEVELS = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
}

def setup_logging(level: int):
    """Configure logging: WARNING for third-party libs, *level* for ``src.*``."""
    logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            force=True,
        )

    # log level for our codebase
    logging.getLogger("src").setLevel(level)

def main():
    """Parse CLI args, configure logging, and launch the Qt GUI."""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        required=False,
        help="Path to the input file"
    )

    parser.add_argument(
        "--log-level",
        type=str.upper,
        choices=LOG_LEVELS.keys(),
        default="INFO",
        help="Log level, choices are: [CRITICAL, ERROR, WARNING, INFO, DEBUG] (case insensitive)"
    )

    args = parser.parse_args()

    # may want to have this case just load the visualizer without file pre-loaded
    if args.input is not None:
        if not os.path.exists(args.input):
            parser.error(f"Input file does not exist: {args.input}")
        if not os.path.isfile(args.input):
            parser.error(f"Input path is not a file: {args.input}")

    setup_logging(LOG_LEVELS[args.log_level])

    app = QApplication([])
    apply_light_theme(app)
    w = FileWindow(args.input)
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
