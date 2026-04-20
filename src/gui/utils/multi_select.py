"""Reusable multi-select dropdown widget for EEG channel selection."""

from typing import List

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFrame,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


class MultiSelectItemDelegate:
    """Utility for styling ``QListWidgetItem`` rows in the channel picker.

    Selected items get a blue (#1a73e8) background with white text;
    deselected items revert to the default white/black colours.  Used
    exclusively by :class:`MultiSelectDropdown`.
    """
    @staticmethod
    def update_item_style(item: QListWidgetItem, is_selected: bool):
        """Set the foreground and background colours of *item*."""
        if is_selected:
            item.setBackground(QColor("#1a73e8"))
            item.setForeground(QColor("#ffffff"))
        else:
            item.setBackground(QColor(Qt.white))
            item.setForeground(QColor(Qt.black))


class MultiSelectDropdown(QWidget):
    """Custom multi-select dropdown for EEG channel selection.

    Presents a ``QPushButton`` that, when clicked, opens a popup
    ``QListWidget``.  Items are toggled individually with a single
    click; the first item (``"All Channels"``) acts as a select-all /
    deselect-all toggle.  The popup closes on *Enter*, *Escape*, or
    focus loss.

    Signals:
        selectionChanged(list[str]): Emitted whenever the selection
            set changes.
        confirmed(): Emitted when the popup is dismissed (used to mark
            the Visualize button as needing re-run).
    """
    selectionChanged = pyqtSignal(list)
    confirmed = pyqtSignal()

    def __init__(self, items: List[str], parent=None):
        super().__init__(parent)
        self.items = items
        self.selected = set()
        self.is_open = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Button to show/hide dropdown
        self.button = QPushButton("All Channels")
        self.button.setStyleSheet("text-align: left; padding-left: 8px;")
        self.button.clicked.connect(self.toggle_dropdown)
        layout.addWidget(self.button)

        # Dropdown frame (initially hidden)
        self.dropdown_frame = QFrame()
        self.dropdown_frame.setFrameShape(QFrame.StyledPanel)
        self.dropdown_frame.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)

        dropdown_layout = QVBoxLayout(self.dropdown_frame)
        dropdown_layout.setContentsMargins(0, 0, 0, 0)
        dropdown_layout.setSpacing(0)

        # List widget with items
        self.list_widget = QListWidget()
        self.list_widget.setMaximumHeight(250)
        self.list_widget.setSelectionMode(QListWidget.NoSelection)
        self.list_widget.itemClicked.connect(self._on_item_clicked)

        self._original_list_keypress = self.list_widget.keyPressEvent
        self.list_widget.keyPressEvent = self._on_list_key_press

        for i, item_text in enumerate(items):
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

        dropdown_layout.addWidget(self.list_widget)
        self.dropdown_frame.setLayout(dropdown_layout)
        self.dropdown_frame.hide()

        self.dropdown_frame.installEventFilter(self)

    def toggle_dropdown(self):
        """Show the popup if hidden, or hide it if visible."""
        if self.dropdown_frame.isVisible():
            self.close_dropdown()
        else:
            self.open_dropdown()

    def open_dropdown(self):
        """Show the dropdown below the button."""
        pos = self.button.mapToGlobal(self.button.rect().bottomLeft())
        self.dropdown_frame.move(pos)
        self.dropdown_frame.resize(self.button.width(), 250)
        self.dropdown_frame.show()
        self.list_widget.setFocus()
        self.is_open = True

    def close_dropdown(self):
        """Hide the dropdown and confirm the selection."""
        self.dropdown_frame.hide()
        self.is_open = False
        self.confirmed.emit()

    def _on_item_clicked(self, item: QListWidgetItem):
        """Toggle the clicked item and synchronise the "All Channels" state."""
        idx = self.list_widget.row(item)
        item_text = self.items[idx]

        if item_text == "All Channels":
            if item_text in self.selected:
                self.selected.clear()
                for i in range(self.list_widget.count()):
                    list_item = self.list_widget.item(i)
                    MultiSelectItemDelegate.update_item_style(list_item, False)
            else:
                self.selected = set(self.items)
                for i in range(self.list_widget.count()):
                    list_item = self.list_widget.item(i)
                    MultiSelectItemDelegate.update_item_style(list_item, True)
        else:
            is_currently_selected = item_text in self.selected

            if is_currently_selected:
                self.selected.discard(item_text)
            else:
                self.selected.add(item_text)

            MultiSelectItemDelegate.update_item_style(item, not is_currently_selected)

            all_items = set(self.items[1:])
            individual_selected = self.selected.copy()
            individual_selected.discard("All Channels")

            all_checkbox = self.list_widget.item(0)
            if individual_selected == all_items:
                self.selected.add("All Channels")
                MultiSelectItemDelegate.update_item_style(all_checkbox, True)
            else:
                self.selected.discard("All Channels")
                MultiSelectItemDelegate.update_item_style(all_checkbox, False)

        self.selectionChanged.emit(list(self.selected))
        self.update_button_text()

    def _on_list_key_press(self, event):
        """Close the popup on Enter / Escape; delegate other keys."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.close_dropdown()
        elif event.key() == Qt.Key_Escape:
            self.close_dropdown()
        else:
            self._original_list_keypress(event)

    def eventFilter(self, obj, event):
        """Auto-close the popup when the dropdown frame loses focus."""
        if obj == self.dropdown_frame:
            from PyQt5.QtCore import QEvent
            if event.type() == QEvent.FocusOut:
                if self.dropdown_frame.isVisible():
                    self.close_dropdown()
                    return True
        return super().eventFilter(obj, event)

    def update_button_text(self):
        """Summarise the current selection on the dropdown button face."""
        if not self.selected:
            text = "No Selection"
        elif "All Channels" in self.selected and len(self.selected) == len(self.items):
            text = "All Channels"
        else:
            if len(self.selected) == 1:
                text = list(self.selected)[0]
            else:
                text = f"{len(self.selected)} selected"
        self.button.setText(text)

    def get_selected(self) -> List[str]:
        """Return the currently selected channel names."""
        return list(self.selected)

    def set_items(self, items: List[str]):
        """Replace the available items and clear the selection."""
        self.items = items
        self.list_widget.clear()
        self.selected.clear()

        for item_text in items:
            item = QListWidgetItem(item_text)
            self.list_widget.addItem(item)

        self.update_button_text()

    def keyPressEvent(self, event):
        """Handle Escape key to close dropdown."""
        if event.key() == Qt.Key_Escape and self.dropdown_frame.isVisible():
            self.close_dropdown()
        else:
            super().keyPressEvent(event)

    def focusOutEvent(self, event):
        """Close dropdown when parent widget loses focus."""
        if self.dropdown_frame.isVisible():
            self.close_dropdown()
        super().focusOutEvent(event)
