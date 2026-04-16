"""Plots screen: experiment selector + embedded matplotlib figure."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QAbstractItemView, QGroupBox,
    QPushButton, QLineEdit,
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure


class PlotsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._project: str = ""

        splitter = QSplitter(Qt.Horizontal)

        # ── Left: selector panel ──────────────────────────────────────
        selector_widget = QWidget()
        selector_layout = QVBoxLayout(selector_widget)
        selector_layout.setContentsMargins(4, 4, 4, 4)
        selector_layout.setSpacing(4)

        self.exp_label = QLabel("Experiments")
        self.exp_label.setStyleSheet("font-weight: bold;")
        selector_layout.addWidget(self.exp_label)

        # Filter box
        self.filter_input = QLineEdit()
        self.filter_input.setPlaceholderText("Filter…")
        self.filter_input.setClearButtonEnabled(True)
        self.filter_input.textChanged.connect(self._apply_filter)
        selector_layout.addWidget(self.filter_input)

        # Select All / Clear All buttons
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
        selector_layout.addLayout(btn_row)

        # Experiment list
        self.exp_list = QListWidget()
        self.exp_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.exp_list.itemSelectionChanged.connect(self._update_plot)
        selector_layout.addWidget(self.exp_list)

        # Loss key selector (DVNR only)
        self.loss_box = QGroupBox("Loss keys")
        self.loss_box.setVisible(False)
        loss_layout = QVBoxLayout(self.loss_box)

        loss_btn_row = QHBoxLayout()
        loss_btn_row.setSpacing(4)
        self.btn_loss_all = QPushButton("All")
        self.btn_loss_all.setFixedHeight(24)
        self.btn_loss_all.setStyleSheet("font-size: 11px;")
        self.btn_loss_all.clicked.connect(lambda: self._select_all_list(self.loss_list))
        self.btn_loss_none = QPushButton("None")
        self.btn_loss_none.setFixedHeight(24)
        self.btn_loss_none.setStyleSheet("font-size: 11px;")
        self.btn_loss_none.clicked.connect(lambda: self._select_none_list(self.loss_list))
        loss_btn_row.addWidget(self.btn_loss_all)
        loss_btn_row.addWidget(self.btn_loss_none)
        loss_layout.addLayout(loss_btn_row)

        self.loss_list = QListWidget()
        self.loss_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.loss_list.itemSelectionChanged.connect(self._update_plot)
        loss_layout.addWidget(self.loss_list)
        selector_layout.addWidget(self.loss_box)

        selector_widget.setMinimumWidth(190)
        selector_widget.setMaximumWidth(300)
        splitter.addWidget(selector_widget)

        # ── Right: matplotlib ─────────────────────────────────────────
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

    # ── Load ─────────────────────────────────────────────────────────
    def load(self, project: str, data: list):
        self._project = project
        self._data = data

        self.exp_list.blockSignals(True)
        self.loss_list.blockSignals(True)
        self.filter_input.blockSignals(True)

        self.exp_list.clear()
        self.loss_list.clear()
        self.filter_input.clear()

        if project == "DVNR":
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(True)
            if data:
                for k in data[0]["losses"].keys():
                    item = QListWidgetItem(k)
                    self.loss_list.addItem(item)
                    item.setSelected(True)
            self.loss_box.setVisible(True)

        elif project == "ODT":
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(True)
            self.loss_box.setVisible(False)

        elif project == "VBP":
            for exp in data:
                label = f"{exp['setup']} / {exp['kr_folder']}"
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, f"{exp['setup']}/{exp['kr_folder']}")
                self.exp_list.addItem(item)
                item.setSelected(True)
            self.loss_box.setVisible(False)

        self.exp_list.blockSignals(False)
        self.loss_list.blockSignals(False)
        self.filter_input.blockSignals(False)
        self._update_plot()

    # ── Filter ───────────────────────────────────────────────────────
    def _apply_filter(self, text: str):
        """Show/hide items matching the filter; preserve selection state."""
        text = text.strip().lower()
        self.exp_list.blockSignals(True)
        for i in range(self.exp_list.count()):
            item = self.exp_list.item(i)
            match = text == "" or text in item.text().lower()
            item.setHidden(not match)
        self.exp_list.blockSignals(False)
        self._update_plot()

    # ── Select All / None ─────────────────────────────────────────────
    def _select_all(self):
        self._select_all_list(self.exp_list)

    def _select_none(self):
        self._select_none_list(self.exp_list)

    def _select_all_list(self, lst: QListWidget):
        lst.blockSignals(True)
        for i in range(lst.count()):
            item = lst.item(i)
            if not item.isHidden():
                item.setSelected(True)
        lst.blockSignals(False)
        self._update_plot()

    def _select_none_list(self, lst: QListWidget):
        lst.blockSignals(True)
        for i in range(lst.count()):
            lst.item(i).setSelected(False)
        lst.blockSignals(False)
        self._update_plot()

    # ── Selection helpers ─────────────────────────────────────────────
    def _selected_exp_names(self) -> list:
        return [item.text() for item in self.exp_list.selectedItems()
                if not item.isHidden()]

    def _selected_exp_keys(self) -> list:
        return [item.data(Qt.UserRole) or item.text()
                for item in self.exp_list.selectedItems()
                if not item.isHidden()]

    def _selected_losses(self) -> list:
        return [item.text() for item in self.loss_list.selectedItems()]

    # ── Plot update ───────────────────────────────────────────────────
    def _update_plot(self):
        self.figure.clear()
        if self._project == "DVNR":
            self._plot_dvnr()
        elif self._project == "ODT":
            self._plot_odt()
        elif self._project == "VBP":
            self._plot_vbp()
        self.canvas.draw()

    # ── DVNR ──────────────────────────────────────────────────────────
    def _plot_dvnr(self):
        selected_exps = self._selected_exp_names()
        selected_losses = self._selected_losses()
        if not selected_exps or not selected_losses:
            self._empty("Select experiments and loss keys")
            return
        exp_map = {e["exp_name"]: e for e in self._data}
        ax = self.figure.add_subplot(111)
        for exp_name in selected_exps:
            exp = exp_map.get(exp_name)
            if not exp:
                continue
            for loss_key in selected_losses:
                values = exp["losses"].get(loss_key, [])
                ax.plot(range(1, len(values) + 1), values,
                        marker="o", markersize=3,
                        label=f"{exp_name} / {loss_key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("DVNR — Loss curves")
        ax.grid(True, alpha=0.3)
        if len(selected_exps) * len(selected_losses) <= 10:
            ax.legend(fontsize=8)

    # ── ODT ───────────────────────────────────────────────────────────
    def _plot_odt(self):
        selected_exps = self._selected_exp_names()
        if not selected_exps:
            self._empty("Select experiments")
            return
        exp_map = {e["exp_name"]: e for e in self._data}
        ax = self.figure.add_subplot(111)
        all_keys = []
        for name in selected_exps:
            exp = exp_map.get(name)
            if exp:
                all_keys.extend(k for k in exp["metrics"]
                                 if exp["metrics"][k] is not None)
        keys = list(dict.fromkeys(all_keys))

        import numpy as np
        x = np.arange(len(keys))
        width = 0.8 / max(len(selected_exps), 1)
        for i, name in enumerate(selected_exps):
            exp = exp_map.get(name)
            if not exp:
                continue
            vals = [v if isinstance(v, (int, float)) else 0
                    for v in (exp["metrics"].get(k) for k in keys)]
            ax.bar(x + i * width, vals, width, label=name)
        ax.set_xticks(x + width * (len(selected_exps) - 1) / 2)
        ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Value")
        ax.set_title("ODT — Metrics comparison")
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    # ── VBP ───────────────────────────────────────────────────────────
    def _plot_vbp(self):
        selected_keys = self._selected_exp_keys()
        if not selected_keys:
            self._empty("Select experiments")
            return
        exp_map = {f"{e['setup']}/{e['kr_folder']}": e for e in self._data}
        ax = self.figure.add_subplot(111)
        for key in selected_keys:
            exp = exp_map.get(key)
            if not exp:
                continue
            ft_epochs = [e for e in exp["epochs"] if e["phase"] in ("FT", "PAT")]
            if not ft_epochs:
                continue
            x = [e["epoch"] for e in ft_epochs]
            y = [e["val_acc"] for e in ft_epochs]
            ax.plot(x, y, marker=".", markersize=4, label=key)
        ax.set_xlabel("FT Epoch")
        ax.set_ylabel("Val Accuracy")
        ax.set_title("VBP — Validation accuracy")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    def _empty(self, msg: str = ""):
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=13)
        ax.axis("off")
