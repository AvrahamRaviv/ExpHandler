"""Channels screen: per-network 2D channel-score heatmap.

Each layer is a row, each channel a colored cell (color = score). Supports
per-layer vs per-network normalization, descending sort within layers,
side-by-side comparison of several files, and a per-channel diff (A−B) when
exactly two architecture-matched files are selected.
"""

import os

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QAbstractItemView, QGroupBox,
    QPushButton, QLineEdit, QComboBox, QCheckBox, QFileDialog,
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIntValidator, QDoubleValidator

import numpy as np
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

from scanners.channel_scores import discover_channel_scores, load_channel_scores

_SEQ_CMAPS = ["viridis", "magma", "plasma", "cividis"]
_DIFF_CMAP = "coolwarm"
_MAX_YTICKS = 60          # avoid unreadable axis on very deep nets
_LABEL_MAXLEN = 28


class ChannelsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._project: str = ""
        # Per-record cache keyed by abs path: parsed channel-score dict.
        # Auto-discovered records are reset on every load(); manual ones
        # (rec["_manual"]) persist across project/arch switches.
        self._records: dict[str, dict] = {}
        # In-session selection cache keyed by the scanned dir.
        self._selections: dict[str, set] = {}
        self._cur_dir: str = ""

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: file selector + controls ────────────────────────────
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(4, 4, 4, 4)
        side_layout.setSpacing(4)

        title = QLabel("Score files")
        title.setStyleSheet("font-weight: bold;")
        side_layout.addWidget(title)

        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter…")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._apply_filter)
        side_layout.addWidget(self.filter_input)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_all = QPushButton("All")
        self.btn_all.setFixedHeight(24)
        self.btn_all.setStyleSheet("font-size: 11px;")
        self.btn_all.clicked.connect(self._select_all)
        self.btn_none = QPushButton("None")
        self.btn_none.setFixedHeight(24)
        self.btn_none.setStyleSheet("font-size: 11px;")
        self.btn_none.clicked.connect(self._select_none)
        btn_row.addWidget(self.btn_all)
        btn_row.addWidget(self.btn_none)
        side_layout.addLayout(btn_row)

        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.file_list.itemSelectionChanged.connect(self._on_selection_changed)
        side_layout.addWidget(self.file_list)

        self.btn_load = QPushButton("📂 Load file…")
        self.btn_load.setFixedHeight(26)
        self.btn_load.setToolTip("Add channel-score JSON files from disk")
        self.btn_load.clicked.connect(self._load_files_dialog)
        side_layout.addWidget(self.btn_load)

        # Controls
        ctrl = QGroupBox("Display")
        ctrl_layout = QVBoxLayout(ctrl)
        ctrl_layout.setSpacing(4)

        ctrl_layout.addWidget(QLabel("Plot type"))
        self.plot_box = QComboBox()
        self.plot_box.addItem("Heatmap", "heatmap")
        self.plot_box.addItem("Ridgeline (per-layer dist)", "ridgeline")
        self.plot_box.addItem("Rank curve (sorted scores)", "rank")
        self.plot_box.addItem("Kept vs pruned", "kept")
        self.plot_box.setToolTip(
            "Visualization mode. Heatmap is the per-channel map; the others "
            "summarize the per-layer score distributions for paper figures."
        )
        self.plot_box.currentIndexChanged.connect(self._on_plot_type_changed)
        ctrl_layout.addWidget(self.plot_box)

        ctrl_layout.addWidget(QLabel("Scope"))
        self.norm_box = QComboBox()
        self.norm_box.addItem("Per-layer", "layer")
        self.norm_box.addItem("Global", "net")
        self.norm_box.setToolTip("Reference used when Normalized is on.")
        self.norm_box.currentIndexChanged.connect(self._render)
        ctrl_layout.addWidget(self.norm_box)

        self.normalized_chk = QCheckBox("Normalized (→ 0–1 by scope)")
        self.normalized_chk.setChecked(True)
        self.normalized_chk.setToolTip(
            "On: min-max each layer (Per-layer) or the whole net (Global) to "
            "0–1.\nOff: raw score units on a shared color bar (Color scale "
            "applies)."
        )
        self.normalized_chk.stateChanged.connect(self._on_normalized_changed)
        ctrl_layout.addWidget(self.normalized_chk)

        ctrl_layout.addWidget(QLabel("Color scale (raw only)"))
        self.scale_box = QComboBox()
        self.scale_box.addItem("Linear", "linear")
        self.scale_box.addItem("Log", "log")
        self.scale_box.addItem("Robust (2–98%)", "robust")
        self.scale_box.setToolTip(
            "Raw mode only. Heavy-tailed scores (e.g. weight norms) crush a "
            "linear scale; Log or Robust spread the low end."
        )
        self.scale_box.currentIndexChanged.connect(self._render)
        self.scale_box.setEnabled(False)   # Normalized on by default
        ctrl_layout.addWidget(self.scale_box)

        self.sort_chk = QCheckBox("Sort channels by score (per layer)")
        self.sort_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.sort_chk)

        topn_row = QHBoxLayout()
        topn_row.setSpacing(4)
        topn_row.addWidget(QLabel("Top-N / layer"))
        self.topn_input = QLineEdit()
        self.topn_input.setPlaceholderText("all")
        self.topn_input.setFixedWidth(70)
        self.topn_input.setValidator(QIntValidator(1, 10_000_000, self))
        self.topn_input.setToolTip(
            "Keep only the N highest-scoring channels per layer "
            "(empty = all). Side-by-side only."
        )
        self.topn_input.editingFinished.connect(self._render)
        topn_row.addWidget(self.topn_input)
        topn_row.addStretch()
        ctrl_layout.addLayout(topn_row)

        self.visible_chk = QCheckBox("Color range: top-N only")
        self.visible_chk.setToolTip(
            "When Top-N is set, span the color bar over only the shown top-N "
            "channels (pooled across layers) instead of all channels — more "
            "contrast among the kept ones."
        )
        self.visible_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.visible_chk)

        ctrl_layout.addWidget(QLabel("Layout"))
        self.portrait_chk = QCheckBox("Portrait (layers as columns)")
        self.portrait_chk.setToolTip(
            "Flip the map: layers run along the x-axis, channels up the y-axis."
        )
        self.portrait_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.portrait_chk)

        self.stretch_chk = QCheckBox("Stretch layers to equal width")
        self.stretch_chk.setToolTip(
            "Extend every layer across the full width, so a narrow layer shows "
            "as wide cells instead of a short row."
        )
        self.stretch_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.stretch_chk)

        ctrl_layout.addWidget(QLabel("View"))
        self.view_box = QComboBox()
        self.view_box.addItem("Side-by-side", "side")
        self.view_box.addItem("Diff (A−B)", "diff")
        self.view_box.currentIndexChanged.connect(self._render)
        ctrl_layout.addWidget(self.view_box)

        ctrl_layout.addWidget(QLabel("Colormap"))
        self.cmap_box = QComboBox()
        for c in _SEQ_CMAPS:
            self.cmap_box.addItem(c, c)
        self.cmap_box.currentIndexChanged.connect(self._render)
        ctrl_layout.addWidget(self.cmap_box)

        # ── Distribution-plot controls (ridgeline / rank / kept) ──────
        thr_row = QHBoxLayout()
        thr_row.setSpacing(4)
        thr_row.addWidget(QLabel("Threshold (score)"))
        self.threshold_input = QLineEdit()
        self.threshold_input.setPlaceholderText("none")
        self.threshold_input.setFixedWidth(70)
        self.threshold_input.setValidator(QDoubleValidator(self))
        self.threshold_input.setToolTip(
            "Draw a score cut-off line on the Ridgeline / Rank-curve plots."
        )
        self.threshold_input.editingFinished.connect(self._render)
        thr_row.addWidget(self.threshold_input)
        thr_row.addStretch()
        ctrl_layout.addLayout(thr_row)

        self.rank_norm_chk = QCheckBox("Normalized rank (0–1 x-axis)")
        self.rank_norm_chk.setToolTip(
            "Rank curve: scale each layer's rank axis to 0–1 so layers of "
            "different widths overlay."
        )
        self.rank_norm_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.rank_norm_chk)

        self.logx_chk = QCheckBox("Log axis (heavy tails)")
        self.logx_chk.setToolTip(
            "Spread heavy-tailed scores: log-x on Ridgeline, log-y on the "
            "Rank curve. Ignored if scores are not strictly positive."
        )
        self.logx_chk.stateChanged.connect(self._render)
        ctrl_layout.addWidget(self.logx_chk)

        sep_row = QHBoxLayout()
        sep_row.setSpacing(4)
        sep_row.addWidget(QLabel("Separation metric"))
        self.sep_metric_box = QComboBox()
        self.sep_metric_box.addItem("Cohen's d", "cohend")
        self.sep_metric_box.addItem("Gini", "gini")
        self.sep_metric_box.setToolTip(
            "Kept-vs-pruned: per-layer separability. Cohen's d = standardized "
            "mean gap between kept and pruned; Gini = importance concentration."
        )
        self.sep_metric_box.currentIndexChanged.connect(self._render)
        sep_row.addWidget(self.sep_metric_box)
        sep_row.addStretch()
        ctrl_layout.addLayout(sep_row)

        side_layout.addWidget(ctrl)

        self.hint = QLabel("")
        self.hint.setWordWrap(True)
        self.hint.setStyleSheet("color: gray; font-size: 11px;")
        side_layout.addWidget(self.hint)

        side.setMinimumWidth(210)
        side.setMaximumWidth(320)
        splitter.addWidget(side)

        # ── Right: matplotlib canvas ──────────────────────────────────
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        splitter.addWidget(plot_widget)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

        self._sync_controls()           # initial (Heatmap) enable state

    # ── Public API ────────────────────────────────────────────────────
    def load(self, project: str, root_dir: str):
        """Show only score files under ``root_dir`` (plus manual ones)."""
        if self._cur_dir:                       # snapshot the dir we leave
            self._selections[self._cur_dir] = set(self._selected_paths())
        self._project = project
        self._cur_dir = root_dir

        # Reset auto-discovered records to this dir only; keep manual files.
        self._records = {p: r for p, r in self._records.items()
                         if r.get("_manual")}
        for p in discover_channel_scores(root_dir):
            if p not in self._records:
                rec = load_channel_scores(p)
                if rec:
                    rec["label"] = self._path_label(p, root_dir)
                    self._records[p] = rec
        self._rebuild_list(self._selections.get(root_dir, set()))
        self._render()

    @staticmethod
    def _path_label(path: str, root_dir: str) -> str:
        """Label = file path relative to the scanned dir, sans suffix.

        Keeps the per-arch sub-dir (e.g. ``loc_mean/run_l1``) so files that
        share a scorer but live in different suffix dirs stay distinct.
        """
        try:
            rel = os.path.relpath(path, root_dir)
        except ValueError:
            rel = os.path.basename(path)
        for suf in ("_channel_scores.json", ".json"):
            if rel.endswith(suf):
                rel = rel[: -len(suf)]
                break
        return rel or os.path.basename(path)

    # ── File list management ──────────────────────────────────────────
    def _rebuild_list(self, selected: set):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for path in sorted(self._records, key=lambda p: self._records[p]["label"]):
            rec = self._records[path]
            item = QListWidgetItem(rec["label"])
            item.setData(Qt.UserRole, path)
            item.setToolTip(path)
            self.file_list.addItem(item)
            item.setSelected(path in selected)
        self.file_list.blockSignals(False)

    def _load_files_dialog(self):
        start = os.path.expanduser("~")
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Load channel-score files", start, "JSON files (*.json)"
        )
        added = 0
        for p in paths:
            ap = os.path.abspath(p)
            if ap in self._records:
                continue
            rec = load_channel_scores(ap)
            if rec:
                rec["_manual"] = True          # survives project/arch switches
                self._records[ap] = rec
                added += 1
        if added:
            keep = set(self._selected_paths())
            self._rebuild_list(keep)
            self._render()

    # ── Selection helpers ─────────────────────────────────────────────
    def _selected_paths(self) -> list:
        return [it.data(Qt.UserRole) for it in self.file_list.selectedItems()
                if not it.isHidden()]

    def _on_selection_changed(self):
        if self._cur_dir:
            self._selections[self._cur_dir] = set(self._selected_paths())
        self._render()

    def _apply_filter(self, text: str):
        text = text.strip().lower()
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            it.setHidden(text != "" and text not in it.text().lower())
        self.file_list.blockSignals(False)
        self._render()

    def _select_all(self):
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            it = self.file_list.item(i)
            if not it.isHidden():
                it.setSelected(True)
        self.file_list.blockSignals(False)
        self._on_selection_changed()

    def _select_none(self):
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            self.file_list.item(i).setSelected(False)
        self.file_list.blockSignals(False)
        self._on_selection_changed()

    # ── Rendering ─────────────────────────────────────────────────────
    def _build_matrix(self, rec: dict, sort: bool, top_n: int | None):
        """Ragged layer scores → (L, Wmax) array with NaN padding.

        ``top_n`` keeps only the N best channels per layer (best = highest
        score, or lowest when ``higher_is_better`` is False). When ``sort`` is
        off the kept channels stay in original index order; when on they are
        ordered best-first.
        """
        hib = rec.get("higher_is_better", True)
        layers = rec["layers"]
        rows = []
        for l in layers:
            s = l["scores"]
            rank = s if hib else -s          # larger rank == "better"
            if top_n and s.size > top_n:
                idx = np.argpartition(rank, -top_n)[-top_n:]
                idx = idx[np.argsort(rank[idx])[::-1]] if sort else np.sort(idx)
                s = s[idx]
            elif sort:
                s = s[np.argsort(rank)[::-1]]
            rows.append(s)
        wmax = max(r.size for r in rows)
        m = np.full((len(layers), wmax), np.nan)
        for i, r in enumerate(rows):
            m[i, : r.size] = r
        return m

    def _top_n(self) -> int | None:
        txt = self.topn_input.text().strip()
        return int(txt) if txt.isdigit() and int(txt) > 0 else None

    @staticmethod
    def _norm_per_layer(m: np.ndarray) -> np.ndarray:
        """Per-row min-max to [0,1], ignoring NaN padding."""
        out = np.full_like(m, np.nan)
        for i in range(m.shape[0]):
            valid = np.isfinite(m[i])
            if not valid.any():
                continue
            v = m[i, valid]
            lo, hi = float(v.min()), float(v.max())
            out[i, valid] = 0.5 if hi <= lo else (v - lo) / (hi - lo)
        return out

    @staticmethod
    def _norm_global(m: np.ndarray, gmin: float, gmax: float) -> np.ndarray:
        """Whole-net min-max to [0,1], ignoring NaN padding."""
        if gmax <= gmin:
            return np.where(np.isfinite(m), 0.5, np.nan)
        return (m - gmin) / (gmax - gmin)

    @staticmethod
    def _flat(rec: dict) -> np.ndarray:
        """Cached flat array of all finite scores across the network."""
        f = rec.get("_flat")
        if f is None:
            allv = np.concatenate([l["scores"].ravel() for l in rec["layers"]])
            f = allv[np.isfinite(allv)]
            rec["_flat"] = f
        return f

    @staticmethod
    def _range_norm(vals: np.ndarray, scale: str):
        """Shared color norm (score units) over the given value pool.

        ``vals`` is whatever set the bar should span — the whole net, or only
        the visible top-N channels.
        """
        if vals.size == 0:
            return Normalize(0.0, 1.0)
        gmin, gmax = float(vals.min()), float(vals.max())
        if scale == "log":
            if gmin > 0:
                return LogNorm(vmin=gmin, vmax=gmax)
            pos = np.abs(vals[vals != 0])
            lt = float(np.percentile(pos, 10)) if pos.size else 1.0
            return SymLogNorm(linthresh=max(lt, 1e-12), vmin=gmin, vmax=gmax)
        if scale == "robust":
            lo, hi = np.percentile(vals, [2, 98])
            lo, hi = float(lo), float(hi)
            if hi <= lo:
                hi = lo + 1.0
            return Normalize(lo, hi)
        return Normalize(gmin, gmax)

    def _set_axes(self, ax, rec: dict, portrait: bool, wmax: int):
        """Label the layer axis with names + the channel axis with 'channel'.

        Landscape: layers on y (0 at top), channels on x. Portrait: layers on x
        (labels rotated), channels on y (0 at bottom). ``wmax`` bounds the
        channel axis explicitly — needed for the stretched pcolormesh path,
        harmless for imshow.
        """
        layers = rec["layers"]
        n = len(layers)
        names = [self._trunc(l["name"]) for l in layers]
        ticks = np.arange(n) + 0.5
        ax.set_aspect("auto")
        if portrait:
            ax.set_xlim(0, n)
            ax.set_ylim(0, wmax)
            if n <= _MAX_YTICKS:
                ax.set_xticks(ticks)
                ax.set_xticklabels(names, fontsize=6, rotation=90)
            else:
                ax.set_xticks([])
            ax.set_ylabel("channel", fontsize=8)
            ax.tick_params(axis="y", labelsize=7)
        else:
            ax.set_xlim(0, wmax)
            ax.set_ylim(n, 0)
            if n <= _MAX_YTICKS:
                ax.set_yticks(ticks)
                ax.set_yticklabels(names, fontsize=6)
            else:
                ax.set_yticks([])
            ax.set_xlabel("channel", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)

    def _draw_matrix(self, ax, dmat, rec, cmap, norm, stretch, portrait):
        """Render one (L, Wmax) data matrix and return a mappable for the bar.

        ``stretch`` off → single imshow (NaN padding stays transparent, narrow
        layers are left-aligned). ``stretch`` on → one pcolormesh per layer with
        channel boundaries spread across the full width, so every layer fills
        the axis regardless of its channel count. ``portrait`` transposes the
        layer/channel axes (channel 0 at the bottom in portrait, left in
        landscape).
        """
        L, wmax = dmat.shape
        if not stretch:
            data = np.ma.masked_invalid(dmat.T if portrait else dmat)
            if portrait:
                extent, origin = [0, L, 0, wmax], "lower"
            else:
                extent, origin = [0, wmax, L, 0], "upper"
            im = ax.imshow(data, aspect="auto", interpolation="nearest",
                           cmap=cmap, norm=norm, extent=extent, origin=origin)
            self._set_axes(ax, rec, portrait, wmax)
            return im
        for i in range(L):
            valid = np.isfinite(dmat[i])
            w = int(valid.sum())
            if w == 0:
                continue
            s = dmat[i, valid][np.newaxis, :]          # (1, w)
            edges = np.linspace(0.0, wmax, w + 1)      # stretched channel grid
            lay = np.array([i, i + 1], dtype=float)
            if portrait:
                X, Y = np.meshgrid(lay, edges)
                C = s.T
            else:
                X, Y = np.meshgrid(edges, lay)
                C = s
            ax.pcolormesh(X, Y, np.ma.masked_invalid(C), cmap=cmap, norm=norm,
                          antialiased=False)
        self._set_axes(ax, rec, portrait, wmax)
        return ScalarMappable(norm=norm, cmap=cmap)

    @staticmethod
    def _trunc(s: str) -> str:
        return s if len(s) <= _LABEL_MAXLEN else "…" + s[-(_LABEL_MAXLEN - 1):]

    def _render(self):
        self.figure.clear()
        paths = self._selected_paths()
        recs = [self._records[p] for p in paths if p in self._records]

        if not recs:
            self.hint.setText("Select one or more score files to view.")
            self.canvas.draw()
            return

        # Distribution plots branch off before the heatmap diff/side logic;
        # each is side-by-side only (one subplot column per selected file).
        plot_type = self.plot_box.currentData()
        if plot_type == "ridgeline":
            self._render_ridgeline(recs)
            self.figure.canvas.draw()
            return
        if plot_type == "rank":
            self._render_rank(recs)
            self.figure.canvas.draw()
            return
        if plot_type == "kept":
            self._render_kept_pruned(recs)
            self.figure.canvas.draw()
            return

        # ── Heatmap (default) ──────────────────────────────────────────
        sort = self.sort_chk.isChecked()
        top_n = self._top_n()
        scope = self.norm_box.currentData()           # "layer" | "net"
        normalized = self.normalized_chk.isChecked()
        scale = self.scale_box.currentData()
        view = self.view_box.currentData()
        cmap_name = self.cmap_box.currentData()

        # Diff availability: exactly 2, matching architecture.
        diff_ok = (len(recs) == 2
                   and recs[0]["arch_key"] == recs[1]["arch_key"])

        if view == "diff" and diff_ok:
            self._render_diff(recs)
        else:
            if view == "diff" and not diff_ok:
                self.hint.setText("Diff needs exactly 2 files with identical "
                                  "architecture. Showing side-by-side.")
            else:
                msgs = []
                if top_n:
                    msgs.append(f"Top {top_n} channels/layer (x = rank).")
                if diff_ok:
                    msgs.append("2 matched files — View→Diff for A−B.")
                self.hint.setText("  ".join(msgs))
            self._render_side(recs, sort, top_n, scope, normalized, scale, cmap_name)
        self.figure.canvas.draw()

    def _side_hint(self, diff_ok: bool) -> str:
        if diff_ok:
            return "2 matched files — switch View to Diff for A−B map."
        return ""

    def _on_normalized_changed(self):
        # Color scale (Log/Robust) only bites on raw heatmap values.
        self._sync_controls()
        self._render()

    def _on_plot_type_changed(self):
        self._sync_controls()
        self._render()

    def _sync_controls(self):
        """Enable only the controls each plot type honors (single source of
        truth for the enable matrix). Called on plot-type / normalized change
        and once at construction."""
        pt = self.plot_box.currentData()
        is_heat = pt == "heatmap"
        is_line = pt in ("ridgeline", "rank")
        for w in (self.portrait_chk, self.stretch_chk, self.visible_chk,
                  self.sort_chk, self.view_box):
            w.setEnabled(is_heat)
        self.norm_box.setEnabled(pt in ("heatmap", "ridgeline"))
        self.normalized_chk.setEnabled(pt in ("heatmap", "ridgeline", "rank"))
        self.scale_box.setEnabled(is_heat and not self.normalized_chk.isChecked())
        self.topn_input.setEnabled(pt != "kept")
        self.cmap_box.setEnabled(is_heat or is_line)
        self.threshold_input.setEnabled(is_line)
        self.rank_norm_chk.setEnabled(pt == "rank")
        self.logx_chk.setEnabled(is_line)
        self.sep_metric_box.setEnabled(pt == "kept")

    def _render_side(self, recs, sort, top_n, scope, normalized, scale, cmap_name):
        portrait = self.portrait_chk.isChecked()
        stretch = self.stretch_chk.isChecked()
        # Each file owns its channel axis once stretched/portrait, so share the
        # layer axis only in the plain landscape case.
        axes = self.figure.subplots(1, len(recs),
                                    sharey=not (portrait or stretch),
                                    squeeze=False)[0]
        cmap = colormaps[cmap_name].copy()
        cmap.set_bad(alpha=0.0)
        # Visible-only range: derive the color span from the shown cells (the
        # pooled top-N across layers) instead of every channel. No-op without
        # top-N (m then holds all channels anyway).
        vis_only = self.visible_chk.isChecked() and top_n is not None
        for ax, rec in zip(axes, recs):
            m = self._build_matrix(rec, sort, top_n)
            finite = m[np.isfinite(m)]
            pool = finite if vis_only else self._flat(rec)
            if not normalized:
                # Raw score units, shared bar; scope only affects normalization
                # so it is irrelevant here — sort still applies per layer.
                norm = self._range_norm(pool, scale)
                dmat = m
                clabel = "score (top-N)" if vis_only else "score"
            elif scope == "layer":
                dmat = self._norm_per_layer(m)
                norm = Normalize(0.0, 1.0)
                clabel = "per-layer 0–1"
            else:   # global normalized
                lo, hi = ((float(pool.min()), float(pool.max()))
                          if pool.size else (rec["gmin"], rec["gmax"]))
                dmat = self._norm_global(m, lo, hi)
                norm = Normalize(0.0, 1.0)
                clabel = "top-N 0–1" if vis_only else "global 0–1"
            im = self._draw_matrix(ax, dmat, rec, cmap, norm, stretch, portrait)
            ax.set_title(self._trunc(rec["label"]), fontsize=8)
            cbar = self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.set_label(clabel, fontsize=7)
            cbar.ax.tick_params(labelsize=6)

    def _render_diff(self, recs):
        a, b = recs
        # Channel-aligned diff per layer (sort disabled so indices correspond).
        layers = a["layers"]
        wmax = max(l["scores"].size for l in layers)
        m = np.full((len(layers), wmax), np.nan)
        for i, (la, lb) in enumerate(zip(layers, b["layers"])):
            d = la["scores"] - lb["scores"]
            m[i, : d.size] = d
        cmap = colormaps[_DIFF_CMAP].copy()
        cmap.set_bad(alpha=0.0)
        finite = m[np.isfinite(m)]
        vmax = float(np.abs(finite).max()) if finite.size else 1.0
        vmax = vmax or 1.0
        norm = Normalize(-vmax, vmax)
        ax = self.figure.subplots(1, 1)
        im = self._draw_matrix(ax, m, a, cmap, norm,
                               self.stretch_chk.isChecked(),
                               self.portrait_chk.isChecked())
        ax.set_title(f"{self._trunc(a['label'])}  −  {self._trunc(b['label'])}",
                     fontsize=8)
        self.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        self.hint.setText("Diff = A − B (channel-aligned). Red > 0, blue < 0.")

    # ── Distribution-plot helpers ─────────────────────────────────────
    def _threshold(self) -> float | None:
        txt = self.threshold_input.text().strip()
        try:
            return float(txt) if txt else None
        except ValueError:
            return None

    def _pool_range(self, rec: dict) -> tuple[float, float]:
        """(min, max) over the file's finite scores, for global normalization."""
        f = self._flat(rec)
        return ((float(f.min()), float(f.max())) if f.size
                else (rec["gmin"], rec["gmax"]))

    def _depth_colors(self, n: int):
        """n RGBA colors sampled along the chosen colormap (shallow → deep)."""
        cmap = colormaps[self.cmap_box.currentData()]
        if n <= 1:
            return [cmap(0.5)]
        return [cmap(i / (n - 1)) for i in range(n)]

    def _depth_colorbar(self, ax, n: int):
        """Thin 'layer depth' bar used when there are too many layers for a
        per-line legend."""
        sm = ScalarMappable(norm=Normalize(0, max(n - 1, 1)),
                            cmap=colormaps[self.cmap_box.currentData()])
        cbar = self.figure.colorbar(sm, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("layer depth", fontsize=7)
        cbar.ax.tick_params(labelsize=6)

    @staticmethod
    def _message_ax(ax, msg: str):
        """Blank axes carrying only a centered gray message."""
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    @staticmethod
    def _hist_support(finite: np.ndarray, logx: bool) -> tuple[float, float]:
        lo, hi = float(finite.min()), float(finite.max())
        if logx and lo <= 0:                 # log needs a positive lower bound
            pos = finite[finite > 0]
            lo = float(pos.min()) if pos.size else 1e-12
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi

    @staticmethod
    def _density(v: np.ndarray, lo: float, hi: float, logx: bool):
        """numpy-histogram density (counts, peak-normed) → (dens, centers).

        A degenerate layer (<2 points or zero spread) returns a narrow unit
        spike at the value rather than calling histogram with one bin.
        """
        v = v[np.isfinite(v)]
        if v.size < 2 or float(v.max()) <= float(v.min()):
            c = float(v.mean()) if v.size else 0.0
            return np.array([0.0, 1.0, 0.0]), np.array([c - 1e-6, c, c + 1e-6])
        if logx and lo > 0:
            bins = np.logspace(np.log10(lo), np.log10(hi), 41)
        else:
            bins = np.linspace(lo, hi, 41)
        counts, edges = np.histogram(v, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        peak = counts.max() or 1
        return counts / peak, centers

    @staticmethod
    def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
        """Standardized mean difference between kept (a) and pruned (b)."""
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            return 0.0
        na, nb = a.size, b.size
        sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
        return 0.0 if sp2 <= 0 else float((a.mean() - b.mean()) / np.sqrt(sp2))

    @staticmethod
    def _gini(v: np.ndarray) -> float:
        """Gini coefficient of |scores| — concentration of channel importance."""
        v = np.sort(np.abs(v[np.isfinite(v)]))
        n = v.size
        if n == 0 or v.sum() == 0:
            return 0.0
        idx = np.arange(1, n + 1)
        return float((2 * (idx * v).sum()) / (n * v.sum()) - (n + 1) / n)

    @staticmethod
    def _paired_hist(ax, kept: np.ndarray, pruned: np.ndarray):
        """Overlaid kept (green) / pruned (red) score histograms on a shared
        bin edge set."""
        parts = [a for a in (kept, pruned) if a.size]
        if not parts:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            return
        pool = np.concatenate(parts)
        bins = np.linspace(float(pool.min()), float(pool.max()), 40)
        if kept.size:
            ax.hist(kept, bins=bins, color="#2ca02c", alpha=0.6, label="kept")
        if pruned.size:
            ax.hist(pruned, bins=bins, color="#d62728", alpha=0.6, label="pruned")
        ax.legend(fontsize=6, loc="best")
        ax.set_xlabel("score", fontsize=8)
        ax.tick_params(labelsize=7)

    # ── Distribution-plot renderers ───────────────────────────────────
    def _render_ridgeline(self, recs):
        """Stacked per-layer score densities (one subplot column per file)."""
        axes = self.figure.subplots(1, len(recs), squeeze=False)[0]
        scope = self.norm_box.currentData()
        normalized = self.normalized_chk.isChecked()
        logx = self.logx_chk.isChecked()
        thr = self._threshold()
        top_n = self._top_n()
        note = []
        for ax, rec in zip(axes, recs):
            m = self._build_matrix(rec, sort=False, top_n=top_n)
            if normalized:
                if scope == "layer":
                    m = self._norm_per_layer(m)
                else:
                    m = self._norm_global(m, *self._pool_range(rec))
            L = m.shape[0]
            colors = self._depth_colors(L)
            finite = m[np.isfinite(m)]
            if finite.size == 0:
                self._message_ax(ax, "no finite scores")
                continue
            use_logx = logx and float(finite.min()) > 0
            if logx and not use_logx:
                note.append("log-x off (non-positive scores)")
            lo, hi = self._hist_support(finite, use_logx)
            ticks = []
            for i in range(L):
                v = m[i][np.isfinite(m[i])]
                if v.size == 0:
                    continue
                base = float(L - 1 - i)          # deepest layer at the bottom
                dens, centers = self._density(v, lo, hi, use_logx)
                ax.fill_between(centers, base, base + 0.9 * dens,
                                color=colors[i], alpha=0.8, linewidth=0.4,
                                edgecolor="white")
                ticks.append((base + 0.3, self._trunc(rec["layers"][i]["name"])))
            if thr is not None:
                ax.axvline(thr, color="k", lw=0.8, ls="--")
            if use_logx:
                ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            ax.set_ylim(-0.5, L)
            if L <= _MAX_YTICKS and ticks:
                ax.set_yticks([t for t, _ in ticks])
                ax.set_yticklabels([nm for _, nm in ticks], fontsize=6)
            else:
                ax.set_yticks([])
            ax.set_xlabel("score", fontsize=8)
            ax.tick_params(axis="x", labelsize=7)
            ax.set_title(self._trunc(rec["label"]), fontsize=8)
        hint = ("Ridgeline: per-layer score density (peak-normed); deeper "
                "layers lower. Flat/wide = redundant, peaked = concentrated.")
        if note:
            hint += "  " + "; ".join(sorted(set(note)))
        self.hint.setText(hint)

    def _render_rank(self, recs):
        """Per-layer scores sorted best-first vs rank (scree-style)."""
        axes = self.figure.subplots(1, len(recs), squeeze=False)[0]
        norm_rank = self.rank_norm_chk.isChecked()
        logy = self.logx_chk.isChecked()
        normalized = self.normalized_chk.isChecked()
        thr = self._threshold()
        top_n = self._top_n()
        for ax, rec in zip(axes, recs):
            hib = rec.get("higher_is_better", True)
            layers = rec["layers"]
            L = len(layers)
            colors = self._depth_colors(L)
            show_leg = L <= 10
            for i, l in enumerate(layers):
                s = l["scores"]
                s = s[np.isfinite(s)]
                if s.size == 0:
                    continue
                order = np.argsort(s)[::-1] if hib else np.argsort(s)
                ys = s[order]
                if top_n:
                    ys = ys[:top_n]
                if normalized and ys.size:
                    lo, hi = float(ys.min()), float(ys.max())
                    ys = (np.full_like(ys, 0.5) if hi <= lo
                          else (ys - lo) / (hi - lo))
                xs = ((np.arange(ys.size) / max(ys.size - 1, 1)) if norm_rank
                      else np.arange(ys.size))
                lbl = self._trunc(l["name"]) if show_leg else None
                ax.plot(xs, ys, color=colors[i], lw=0.8, label=lbl)
            if logy:
                ax.set_yscale("log")            # masks non-positive y, no crash
            if thr is not None:
                ax.axhline(thr, color="k", lw=0.8, ls="--")
            ax.set_xlabel("rank (0–1)" if norm_rank else "rank", fontsize=8)
            ax.set_ylabel("score (per-layer 0–1)" if normalized else "score",
                          fontsize=8)
            ax.tick_params(labelsize=7)
            if show_leg:
                ax.legend(fontsize=6, loc="best")
            else:
                self._depth_colorbar(ax, L)
            ax.set_title(self._trunc(rec["label"]), fontsize=8)
        self.hint.setText("Rank curve: per-layer scores sorted best-first; "
                          "color = layer depth. A sharp knee = concentrated "
                          "importance (safe to prune).")

    def _render_kept_pruned(self, recs):
        """Kept vs pruned score distributions + per-layer separation metric."""
        usable = [r for r in recs
                  if any(l["kept"] is not None for l in r["layers"])]
        if not usable:
            ax = self.figure.subplots(1, 1)
            self._message_ax(ax, "No kept/pruned mask in the selected file(s).\n"
                                 "This view needs layers with a 'kept' field.")
            self.hint.setText("Kept vs pruned needs a prune mask; none present.")
            return
        metric = self.sep_metric_box.currentData()      # "cohend" | "gini"
        gs = self.figure.add_gridspec(2, len(usable), height_ratios=[3, 1])
        for j, rec in enumerate(usable):
            ax_d = self.figure.add_subplot(gs[0, j])
            ax_m = self.figure.add_subplot(gs[1, j])
            kept_vals, pruned_vals, seps, names = [], [], [], []
            for l in rec["layers"]:
                if l["kept"] is None:
                    continue
                s, k = l["scores"], l["kept"]
                ok = np.isfinite(s)
                kv, pv = s[ok & k], s[ok & ~k]
                kept_vals.append(kv)
                pruned_vals.append(pv)
                seps.append(self._cohens_d(kv, pv) if metric == "cohend"
                            else self._gini(s[ok]))
                names.append(self._trunc(l["name"]))
            allk = np.concatenate(kept_vals) if kept_vals else np.array([])
            allp = np.concatenate(pruned_vals) if pruned_vals else np.array([])
            self._paired_hist(ax_d, allk, allp)
            ax_d.set_title(self._trunc(rec["label"]), fontsize=8)
            x = np.arange(len(seps))
            ax_m.bar(x, seps, color="#4878a8")
            ax_m.axhline(0, color="gray", lw=0.5)
            ax_m.set_ylabel("Cohen's d" if metric == "cohend" else "Gini",
                            fontsize=7)
            if len(names) <= _MAX_YTICKS:
                ax_m.set_xticks(x)
                ax_m.set_xticklabels(names, fontsize=5, rotation=90)
            else:
                ax_m.set_xticks([])
            ax_m.tick_params(axis="y", labelsize=6)
        self.hint.setText("Kept (green) vs pruned (red) scores; bottom bars = "
                          "per-layer separation. Clean split = reliable scorer.")
