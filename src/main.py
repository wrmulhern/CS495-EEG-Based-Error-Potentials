from PyQt5.QtWidgets import QApplication

from src.gui.file_window import FileWindow
from src.gui.themes.light_theme import apply_light_theme

def main():
    app = QApplication([])
    apply_light_theme(app)
    w = FileWindow()
    w.show()
    app.exec_()


if __name__ == "__main__":
    main()
