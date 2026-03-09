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
    # log level for everything but our codebase
    logging.basicConfig(
            level=logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            force=True,
        )

    # log level for our codebase
    logging.getLogger("src").setLevel(level)

def main():
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
