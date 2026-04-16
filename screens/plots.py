"""Plots screen: experiment selector + embedded matplotlib figure."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QAbstractItemView, QGroupBox,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget,
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

        # Metric selector (ODT only)
        self.metric_box = QGroupBox("Metrics")
        self.metric_box.setVisible(False)
        metric_layout = QVBoxLayout(self.metric_box)

        metric_filter = QLineEdit()
        metric_filter.setPlaceholderText("Filter metrics…")
        metric_filter.setClearButtonEnabled(True)
        metric_filter.textChanged.connect(self._apply_metric_filter)
        metric_layout.addWidget(metric_filter)
        self.metric_filter_input = metric_filter

        metric_btn_row = QHBoxLayout()
        metric_btn_row.setSpacing(4)
        self.btn_metric_all = QPushButton("All")
        self.btn_metric_all.setFixedHeight(24)
        self.btn_metric_all.setStyleSheet("font-size: 11px;")
        self.btn_metric_all.clicked.connect(lambda: self._select_all_list(self.metric_list))
        self.btn_metric_none = QPushButton("None")
        self.btn_metric_none.setFixedHeight(24)
        self.btn_metric_none.setStyleSheet("font-size: 11px;")
        self.btn_metric_none.clicked.connect(lambda: self._select_none_list(self.metric_list))
        metric_btn_row.addWidget(self.btn_metric_all)
        metric_btn_row.addWidget(self.btn_metric_none)
        metric_layout.addLayout(metric_btn_row)

        self.metric_list = QListWidget()
        self.metric_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.metric_list.itemSelectionChanged.connect(self._update_plot)
        metric_layout.addWidget(self.metric_list)
        selector_layout.addWidget(self.metric_box)

        # Keep Ratio selector (VBP only)
        self.kr_box = QGroupBox("Keep Ratios")
        self.kr_box.setVisible(False)
        kr_layout = QVBoxLayout(self.kr_box)

        kr_btn_row = QHBoxLayout()
        kr_btn_row.setSpacing(4)
        self.btn_kr_all = QPushButton("All")
        self.btn_kr_all.setFixedHeight(24)
        self.btn_kr_all.setStyleSheet("font-size: 11px;")
        self.btn_kr_all.clicked.connect(lambda: self._select_all_list(self.kr_list))
        self.btn_kr_none = QPushButton("None")
        self.btn_kr_none.setFixedHeight(24)
        self.btn_kr_none.setStyleSheet("font-size: 11px;")
        self.btn_kr_none.clicked.connect(lambda: self._select_none_list(self.kr_list))
        kr_btn_row.addWidget(self.btn_kr_all)
        kr_btn_row.addWidget(self.btn_kr_none)
        kr_layout.addLayout(kr_btn_row)

        self.kr_list = QListWidget()
        self.kr_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.kr_list.itemSelectionChanged.connect(self._update_plot)
        kr_layout.addWidget(self.kr_list)
        selector_layout.addWidget(self.kr_box)

        selector_widget.setMinimumWidth(190)
        selector_widget.setMaximumWidth(300)
        splitter.addWidget(selector_widget)

        # ── Right: stacked widget (matplotlib | ODT table) ────────────
        self.right_stack = QStackedWidget()

        # Page 0: matplotlib (DVNR / VBP)
        plot_widget = QWidget()
        plot_layout = QVBoxLayout(plot_widget)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, plot_widget)
        plot_layout.addWidget(self.toolbar)
        plot_layout.addWidget(self.canvas)
        self.right_stack.addWidget(plot_widget)   # index 0

        # Page 1: ODT table
        self.odt_table = QTableWidget()
        self.odt_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.odt_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.odt_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.odt_table.setSortingEnabled(True)
        self.right_stack.addWidget(self.odt_table)  # index 1

        splitter.addWidget(self.right_stack)

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
        self.metric_list.blockSignals(True)
        self.kr_list.blockSignals(True)
        self.filter_input.blockSignals(True)

        self.exp_list.clear()
        self.loss_list.clear()
        self.metric_list.clear()
        self.kr_list.clear()
        self.filter_input.clear()

        if project == "DVNR":
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(False)
            if data:
                for k in data[0]["losses"].keys():
                    item = QListWidgetItem(k)
                    self.loss_list.addItem(item)
                    item.setSelected(k == "loss_nr")
            self.loss_box.setVisible(True)
            self.right_stack.setCurrentIndex(0)

        elif project == "ODT":
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(False)
            self.loss_box.setVisible(False)
            # Populate metric list; select only total_metric by default
            all_keys = []
            for exp in data:
                all_keys.extend(k for k in exp["metrics"] if k not in all_keys)
            for k in all_keys:
                item = QListWidgetItem(k)
                self.metric_list.addItem(item)
                item.setSelected(k == "total_metric")
            self.metric_box.setVisible(True)
            self.right_stack.setCurrentIndex(1)

        elif project == "VBP":
            _DEFAULT_SETUPS = {
                "global_gv_vnr10ft200",
                "global_g_ft200",
                "global_dvp_10vnr_ft200",
                "global_dp_ft200",
            }
            # Unique setups
            seen_setups = []
            for exp in data:
                if exp["setup"] not in seen_setups:
                    seen_setups.append(exp["setup"])
            for s in seen_setups:
                item = QListWidgetItem(s)
                self.exp_list.addItem(item)
                item.setSelected(s in _DEFAULT_SETUPS)
            # Unique keep ratios, sorted
            seen_krs = []
            for exp in data:
                kr = exp.get("keep_ratio")
                if kr is not None and kr not in seen_krs:
                    seen_krs.append(kr)
            seen_krs.sort()
            for kr in seen_krs:
                label = f"{kr:.2f}" if isinstance(kr, float) else str(kr)
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, kr)
                self.kr_list.addItem(item)
                item.setSelected(False)
            self.loss_box.setVisible(False)
            self.metric_box.setVisible(False)
            self.kr_box.setVisible(True)
            self.right_stack.setCurrentIndex(0)

        self.exp_list.blockSignals(False)
        self.loss_list.blockSignals(False)
        self.metric_list.blockSignals(False)
        self.kr_list.blockSignals(False)
        self.filter_input.blockSignals(False)
        self.exp_label.setText("Setups" if project == "VBP" else "Experiments")
        self.kr_box.setVisible(project == "VBP")
        self._update_plot()

    # ── Filters ──────────────────────────────────────────────────────
    def _apply_metric_filter(self, text: str):
        text = text.strip().lower()
        self.metric_list.blockSignals(True)
        for i in range(self.metric_list.count()):
            item = self.metric_list.item(i)
            item.setHidden(text != "" and text not in item.text().lower())
        self.metric_list.blockSignals(False)
        self._update_plot()

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
        if self._project == "ODT":
            self.right_stack.setCurrentIndex(1)
            self._show_odt_table()
        else:
            self.right_stack.setCurrentIndex(0)
            self.figure.clear()
            if self._project == "DVNR":
                self._plot_dvnr()
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
    def _show_odt_table(self):
        selected_exps = self._selected_exp_names()
        keys = [self.metric_list.item(i).text()
                for i in range(self.metric_list.count())
                if self.metric_list.item(i).isSelected()
                and not self.metric_list.item(i).isHidden()]

        self.odt_table.setSortingEnabled(False)
        self.odt_table.clearContents()

        if not selected_exps or not keys:
            self.odt_table.setRowCount(0)
            self.odt_table.setColumnCount(0)
            return

        exp_map = {e["exp_name"]: e for e in self._data}

        # Build rows with sort key
        sort_col = "total_metric" if "total_metric" in keys else keys[0]
        rows = []
        for name in selected_exps:
            exp = exp_map.get(name)
            vals = {}
            for k in keys:
                v = exp["metrics"].get(k) if exp else None
                vals[k] = v if isinstance(v, (int, float)) else None
            sort_val = vals.get(sort_col)
            rows.append((name, vals, sort_val))

        # Sort descending by sort_col (None → bottom)
        rows.sort(key=lambda r: (r[2] is None, -(r[2] or 0)))

        cols = ["Experiment"] + keys
        self.odt_table.setColumnCount(len(cols))
        self.odt_table.setHorizontalHeaderLabels(cols)
        self.odt_table.setRowCount(len(rows))

        for r, (name, vals, _) in enumerate(rows):
            name_item = QTableWidgetItem(name)
            name_item.setFlags(name_item.flags() & ~Qt.ItemIsEditable)
            self.odt_table.setItem(r, 0, name_item)
            for c, k in enumerate(keys, start=1):
                v = vals.get(k)
                txt = f"{v:.4f}" if v is not None else "—"
                cell = QTableWidgetItem(txt)
                cell.setFlags(cell.flags() & ~Qt.ItemIsEditable)
                cell.setTextAlignment(Qt.AlignCenter)
                self.odt_table.setItem(r, c, cell)

        self.odt_table.setSortingEnabled(True)

    # ── VBP ───────────────────────────────────────────────────────────
    def _plot_vbp(self):
        import matplotlib.cm as mcm
        from matplotlib.lines import Line2D

        selected_setups = set(self._selected_exp_names())
        selected_krs = {self.kr_list.item(i).data(Qt.UserRole)
                        for i in range(self.kr_list.count())
                        if self.kr_list.item(i).isSelected()}
        if not selected_setups or not selected_krs:
            self._empty("Select setups and keep ratios")
            return
        matching = [e for e in self._data
                    if e["setup"] in selected_setups
                    and e.get("keep_ratio") in selected_krs]
        if not matching:
            self._empty("No experiments match selection")
            return

        ax = self.figure.add_subplot(111)

        setups = sorted(set(e["setup"] for e in matching))
        cmap = mcm.get_cmap("tab10")
        setup_color = {s: cmap(i % 10) for i, s in enumerate(setups)}

        plotted = False
        for exp in matching:
            x = exp.get("pruned_macs_G")
            y = exp.get("best_acc")
            if x is None or y is None:
                continue
            color = setup_color[exp["setup"]]
            ax.scatter(x, y, color=color, s=70, zorder=3)
            kr = exp.get("keep_ratio", "")
            ann = f"{kr:.2f}" if isinstance(kr, float) else str(kr)
            ax.annotate(ann, (x, y), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
            plotted = True

        if not plotted:
            self._empty("No valid data (missing MACs or accuracy)")
            return

        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=setup_color[s], markersize=8, label=s)
                   for s in setups]
        ax.legend(handles=handles, fontsize=8)
        ax.set_xlabel("Pruned MACs (G)")
        ax.set_ylabel("Best Accuracy")
        ax.set_title("VBP — Best Accuracy vs. Pruned MACs")
        ax.grid(True, alpha=0.3)

    def _empty(self, msg: str = ""):
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=13)
        ax.axis("off")
