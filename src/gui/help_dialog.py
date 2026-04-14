"""Modal help dialog that renders the project README from GitHub."""

import threading
import urllib.request

from PyQt5.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QTextBrowser,
    QVBoxLayout,
)

from src.gui.themes.colors import get_palette

README_URL = "https://raw.githubusercontent.com/wrmulhern/CS495-EEG-Based-Error-Potentials/main/README.md"


class HelpDialog(QDialog):
    """Modal help dialog accessible from the top-bar "? Help" button.

    On open, a background thread fetches the project README from
    GitHub (:data:`README_URL`) and renders it as HTML via the
    ``markdown`` package (if installed) or a regex-based fallback.
    While the fetch is in flight the dialog shows a "Loading..." message
    so it opens instantly.  If the network request fails, the dialog
    falls back to :attr:`_CONTENT`, an embedded HTML quick-start guide.

    The dialog respects the current light / dark theme.
    """

    _CONTENT = """
<h2 style="margin-top:0;">ErrP Visualizer &mdash; Quick Guide</h2>

<h3>What is an ErrP?</h3>
<p>An <b>Error-Related Potential (ErrP)</b> is a brain signal that appears in EEG when
a person perceives or makes an error. Two main components:</p>
<ul>
  <li><b>ERN / Ne</b> (50&ndash;150 ms) &mdash; negative deflection shortly after the error,
      generated in the anterior cingulate cortex.</li>
  <li><b>Pe</b> (200&ndash;400 ms) &mdash; positive deflection reflecting conscious error awareness.</li>
</ul>

<h3>End-to-end workflow</h3>
<ol>
  <li>Click <b>Record EEG</b> in the top bar. This opens a web application where you can run
      a <b>Flanker Task</b> &mdash; a standard cognitive paradigm that reliably elicits ErrP
      signals using a connected BCI headset.</li>
  <li>Complete the task. The web app exports your session as a <b>.set</b> or <b>.csv</b> file.</li>
  <li>Drop that file into this app to visualize your ErrP.</li>
</ol>

<h3>Loading files</h3>
<ul>
  <li>Drag and drop one or more files onto the drop zone, or click <b>Browse (&hellip;)</b>.</li>
  <li><b>.set</b> files are loaded directly. <b>.csv</b> files (e.g. from OpenBCI Ganglion)
      are <b>automatically converted</b> to .set format &mdash; no manual steps required.
      A converted file is saved alongside the original CSV.</li>
  <li>Each file opens in its own <b>tab</b>. Tabs are fully independent.</li>
  <li>Files load <b>lazily</b>: data is only read when you first click <b>Visualize</b> on that tab.</li>
  <li>Close a single tab with its <b>&times;</b> button, or remove all tabs with <b>Clear All</b>.</li>
</ul>

<h3>Graph types</h3>
<ul>
  <li><b>ErrP Time Series</b> &mdash; averaged ERP waveform across all (or selected) channels.
      Best for inspecting the ERN and Pe components over time.</li>
  <li><b>Topographic Map</b> &mdash; scalp voltage map at up to three time points.
      Requires &ge;19 channels. The epoch window is fixed to the full range.</li>
  <li><b>Joint Maps</b> &mdash; time series and topomaps combined in one figure.
      Topomap times outside the epoch window show as <i>Out of range</i> placeholders.</li>
</ul>

<h3>Graph options</h3>
<ul>
  <li><b>Epoch (ms)</b> &mdash; crop the time axis. Leave blank for the full epoch.
      Disabled automatically for Topographic Map.</li>
  <li><b>Sensor</b> &mdash; plot a single channel instead of all channels (Time Series only).</li>
  <li><b>Topomap times (s)</b> &mdash; three time points (in seconds) for the scalp maps.</li>
  <li><b>Display Events and Responses</b> &mdash; overlays the ERN window (blue, 50&ndash;150 ms)
      and Pe window (green, 200&ndash;400 ms) with hover-activated labels.</li>
</ul>

<h3>Downloading a graph</h3>
<p>Click <b>Download Graph</b> in the bottom bar to save the currently displayed figure
as a high-resolution PNG (300&thinsp;dpi).</p>

<h3>Dark mode</h3>
<p>Toggle <b>Dark mode</b> in the top bar. The theme applies to both the Qt UI and the
embedded Matplotlib figures.</p>

<h3>Supported file formats</h3>
<ul>
  <li><b>EEGLAB .set</b> &mdash; epoched data with &ge;2 trials. Companion <b>.fdt</b> files
      are handled automatically.</li>
  <li><b>.csv</b> &mdash; OpenBCI Ganglion format. Automatically converted to .set on load.</li>
</ul>
"""

    def __init__(self, is_dark: bool = False, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Help — ErrP Visualizer")
        self.setMinimumSize(620, 520)
        self.resize(660, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(12)

        self.browser = QTextBrowser()
        self.browser.setOpenExternalLinks(True)
        self.browser.setFrameShape(QFrame.NoFrame)
        self.browser.setHtml("<p style='color:gray;'>Loading README from GitHub…</p>")
        layout.addWidget(self.browser, stretch=1)

        btn_box = QDialogButtonBox(QDialogButtonBox.Close)
        btn_box.rejected.connect(self.accept)
        layout.addWidget(btn_box)

        self._apply_theme(is_dark)

        self._is_dark = is_dark
        t = threading.Thread(target=self._load_readme, daemon=True)
        t.start()

    def _load_readme(self):
        """Background worker: fetch, convert, and inject the README HTML."""
        try:
            req = urllib.request.Request(
                README_URL,
                headers={"User-Agent": "ErrP-Visualizer"}
            )
            with urllib.request.urlopen(req, timeout=6) as resp:
                md_text = resp.read().decode("utf-8")
            html = self._md_to_html(md_text)
        except Exception:
            html = self._CONTENT

        from PyQt5.QtCore import QMetaObject, Qt, Q_ARG
        QMetaObject.invokeMethod(
            self.browser, "setHtml",
            Qt.QueuedConnection,
            Q_ARG(str, html)
        )

    @staticmethod
    def _md_to_html(md: str) -> str:
        """Convert Markdown text to HTML, with a regex fallback if ``markdown`` is not installed."""
        try:
            import markdown
            return markdown.markdown(md, extensions=["fenced_code", "tables"])
        except ImportError:
            pass

        import re
        html = md
        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$',  r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$',   r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', html)
        html = re.sub(r'`(.+?)`', r'<code>\1</code>', html)
        html = re.sub(r'\[(.+?)\]\((.+?)\)', r'<a href="\2">\1</a>', html)
        html = re.sub(r'^\s*[-*] (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'(<li>.*</li>\n?)+', r'<ul>\g<0></ul>', html)
        html = re.sub(r'\n\n+', '</p><p>', html)
        return f"<p>{html}</p>"

    def _apply_theme(self, is_dark: bool):
        p = get_palette(is_dark)
        self.setStyleSheet(
            f"QDialog {{ background: {p.surface}; }}"
            f"QTextBrowser {{ background: {p.surface}; color: {p.text}; border: none; font-size: 13px; }}"
            f"QPushButton {{ background: {p.surface_elevated}; color: {p.text}; border: 1px solid {p.border};"
            f" border-radius: 4px; padding: 4px 16px; }}"
            f"QPushButton:hover {{ background: {p.surface_hover}; }}"
        )
