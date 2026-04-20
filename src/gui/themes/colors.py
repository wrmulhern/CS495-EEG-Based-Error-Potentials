"""Centralized color palette for the ErrP Visualizer.

Every hex color used in Qt stylesheets and Matplotlib theming is
defined here.  Widgets import ``LIGHT`` and ``DARK`` (or
:func:`get_palette`) instead of hardcoding hex strings, so the entire
visual identity can be updated in one place.

Usage::

    from src.gui.themes.colors import get_palette

    p = get_palette(is_dark=True)
    widget.setStyleSheet(f"background: {p.surface}; color: {p.text};")
"""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Palette:
    """Complete set of semantic color tokens for one theme variant."""

    # ── Surfaces / backgrounds ───────────────────────────────────────
    window: str          # main window background
    surface: str         # card / panel / dialog background
    surface_alt: str     # alternate rows, input fields, frames
    surface_dim: str     # disabled / greyed-out input background
    surface_elevated: str  # raised elements (buttons, elevated cards)
    surface_hover: str   # hover state on elevated elements
    surface_pressed: str # pressed state on elevated elements

    # ── Text ─────────────────────────────────────────────────────────
    text: str            # primary text
    text_secondary: str  # secondary / muted text
    text_disabled: str   # disabled text / placeholders

    # ── Borders & grid ───────────────────────────────────────────────
    border: str          # standard borders (inputs, cards)
    border_strong: str   # higher-contrast borders (spines, dividers)
    grid: str            # grid lines / subtle separators

    # ── Accent (blue) ────────────────────────────────────────────────
    accent: str          # primary accent (buttons, links, selection)
    accent_hover: str    # accent hover state
    accent_pressed: str  # accent pressed state
    accent_tint: str     # very light tint of the accent (hover bg)

    # ── Danger (red) ─────────────────────────────────────────────────
    danger: str          # destructive text / icons
    danger_border: str   # danger button borders
    danger_bg: str       # danger background tint
    danger_hover: str    # danger hover background

    # ── Semantic feedback ────────────────────────────────────────────
    success: str         # correct / success (#34a853)
    warning: str         # too-slow / caution (#f4a400)
    error: str           # wrong / error (#ea4335)

    # ── Toggle switch ────────────────────────────────────────────────
    toggle_track_off: str
    toggle_track_off_pressed: str
    toggle_track_on: str
    toggle_track_on_pressed: str

    # ── Icon / close buttons ─────────────────────────────────────────
    icon_hover_bg: str   # translucent hover backdrop for icon-only buttons

    # ── Drop zone ────────────────────────────────────────────────────
    drop_border: str     # idle dashed border
    drop_hover_border: str  # hover dashed border
    drop_hover_bg: str   # hover background


LIGHT = Palette(
    # Surfaces
    window="#ffffff",
    surface="#ffffff",
    surface_alt="#f1f3f4",
    surface_dim="#f1f3f4",
    surface_elevated="#ffffff",
    surface_hover="#f1f3f4",
    surface_pressed="#e8eaed",
    # Text
    text="#202124",
    text_secondary="#5f6368",
    text_disabled="#9aa0a6",
    # Borders
    border="#dadce0",
    border_strong="#dadce0",
    grid="#dadce0",
    # Accent
    accent="#1a73e8",
    accent_hover="#1666c1",
    accent_pressed="#1450b1",
    accent_tint="#e8f0fe",
    # Danger
    danger="#c5221f",
    danger_border="#d93025",
    danger_bg="#fce8e6",
    danger_hover="#fce8e6",
    # Feedback
    success="#34a853",
    warning="#f4a400",
    error="#ea4335",
    # Toggle
    toggle_track_off="#dadce0",
    toggle_track_off_pressed="#c7c9cc",
    toggle_track_on="#1a73e8",
    toggle_track_on_pressed="#1666c1",
    # Icon buttons
    icon_hover_bg="#dfe1e3",
    # Drop zone
    drop_border="#9aa0a6",
    drop_hover_border="#1a73e8",
    drop_hover_bg="#e8f0fe",
)


DARK = Palette(
    # Surfaces
    window="#121212",
    surface="#1e1e1e",
    surface_alt="#202124",
    surface_dim="#2d2d2d",
    surface_elevated="#303134",
    surface_hover="#3c4043",
    surface_pressed="#4a4e51",
    # Text
    text="#e8eaed",
    text_secondary="#9aa0a6",
    text_disabled="#5f6368",
    # Borders
    border="#5f6368",
    border_strong="#3c4043",
    grid="#3c4043",
    # Accent
    accent="#8ab4f8",
    accent_hover="#669df6",
    accent_pressed="#4a8af5",
    accent_tint="#1e1e1e",
    # Danger
    danger="#f28b82",
    danger_border="#f28b82",
    danger_bg="#2d2d2d",
    danger_hover="#3c4043",
    # Feedback
    success="#34a853",
    warning="#f4a400",
    error="#ea4335",
    # Toggle
    toggle_track_off="#5f6368",
    toggle_track_off_pressed="#80868b",
    toggle_track_on="#8ab4f8",
    toggle_track_on_pressed="#669df6",
    # Icon buttons
    icon_hover_bg="#4f5153",
    # Drop zone
    drop_border="#5f6368",
    drop_hover_border="#8ab4f8",
    drop_hover_bg="#1e1e1e",
)


def get_palette(is_dark: bool = False) -> Palette:
    """Return the :class:`Palette` for the requested theme."""
    return DARK if is_dark else LIGHT
