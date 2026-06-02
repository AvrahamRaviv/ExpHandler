"""Plots screen: experiment selector + embedded matplotlib figure."""

from PyQt5.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QSplitter,
    QListWidget, QListWidgetItem, QLabel, QAbstractItemView, QGroupBox,
    QPushButton, QLineEdit, QTableWidget, QTableWidgetItem, QHeaderView,
    QStackedWidget, QComboBox,
)
from PyQt5.QtCore import Qt, QTimer

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from config import get_plots_default, save_plots_default


class PlotsScreen(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._data: list = []
        self._project: str = ""
        # In-session per-project selection cache. Survives project switches
        # while the app is running. Persisted defaults live in config json.
        self._selections: dict[str, dict] = {}
        # NORMNET metric selection, cached per view (curves vs vnorm use
        # different stat sets, so they can't share one "losses" slot).
        self._nn_metric_sel: dict[str, set] = {}

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

        # View selector (NORMNET only): curves / pair compare / V-norm health
        self.nn_plot_type_box = QGroupBox("View")
        self.nn_plot_type_box.setVisible(False)
        nn_pt_layout = QVBoxLayout(self.nn_plot_type_box)
        self.nn_plot_type = QComboBox()
        self.nn_plot_type.addItem("Curves (per run)", "curves")
        self.nn_plot_type.addItem("Pair compare (val_acc + Δ)", "pairs")
        self.nn_plot_type.addItem("V-norm health", "vnorm")
        self.nn_plot_type.currentIndexChanged.connect(self._on_nn_plot_type_changed)
        nn_pt_layout.addWidget(self.nn_plot_type)
        selector_layout.addWidget(self.nn_plot_type_box)

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

        # Plot type selector (VBP only)
        self.vbp_plot_type_box = QGroupBox("Plot type")
        self.vbp_plot_type_box.setVisible(False)
        vbp_pt_layout = QVBoxLayout(self.vbp_plot_type_box)
        self.vbp_plot_type = QComboBox()
        self.vbp_plot_type.addItem("Acc vs MACs", "acc_vs_macs")
        self.vbp_plot_type.addItem("FT Curves", "ft_curves")
        self.vbp_plot_type.currentIndexChanged.connect(self._on_vbp_plot_type_changed)
        vbp_pt_layout.addWidget(self.vbp_plot_type)
        max_epoch_row = QHBoxLayout()
        max_epoch_row.setSpacing(4)
        self.vbp_max_epoch_label = QLabel("Max epoch:")
        self.vbp_max_epoch = QLineEdit()
        self.vbp_max_epoch.setPlaceholderText("all")
        self.vbp_max_epoch.setFixedWidth(70)
        self.vbp_max_epoch.editingFinished.connect(self._update_plot)
        max_epoch_row.addWidget(self.vbp_max_epoch_label)
        max_epoch_row.addWidget(self.vbp_max_epoch)
        max_epoch_row.addStretch()
        vbp_pt_layout.addLayout(max_epoch_row)
        selector_layout.addWidget(self.vbp_plot_type_box)

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

        # Save current selection as the persisted default for this project.
        self.save_default_btn = QPushButton("⭐ Save as default")
        self.save_default_btn.setFixedHeight(26)
        self.save_default_btn.setToolTip(
            "Save current selection as the default for this project (persists across launches)"
        )
        self.save_default_btn.clicked.connect(self._save_current_as_default)
        selector_layout.addWidget(self.save_default_btn)

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

    # ── Selection cache helpers ───────────────────────────────────────
    def _snapshot_current_selection(self) -> dict:
        return {
            "exps": [self.exp_list.item(i).text()
                     for i in range(self.exp_list.count())
                     if self.exp_list.item(i).isSelected()],
            "losses": [self.loss_list.item(i).text()
                       for i in range(self.loss_list.count())
                       if self.loss_list.item(i).isSelected()],
            "metrics": [self.metric_list.item(i).text()
                        for i in range(self.metric_list.count())
                        if self.metric_list.item(i).isSelected()],
            "krs": [self.kr_list.item(i).data(Qt.UserRole)
                    for i in range(self.kr_list.count())
                    if self.kr_list.item(i).isSelected()],
        }

    def _resolve(self, project: str, kind: str, hardcoded: set | None) -> set:
        """Pick selection set for `kind`: in-session cache > persisted > hardcoded."""
        sess = self._selections.get(project, {}).get(kind)
        if sess is not None:
            return set(sess)
        persisted = get_plots_default(project)
        if kind in persisted:
            return set(persisted[kind])
        return set(hardcoded) if hardcoded else set()

    def _save_current_as_default(self):
        if not self._project:
            return
        save_plots_default(self._project, self._snapshot_current_selection())
        original = self.save_default_btn.text()
        self.save_default_btn.setText("✓ Saved")
        QTimer.singleShot(1500, lambda: self.save_default_btn.setText(original))

    # ── Load ─────────────────────────────────────────────────────────
    def load(self, project: str, data: list):
        # Snapshot outgoing project's selection so switching back restores it.
        if self._project and self.exp_list.count() > 0:
            self._selections[self._project] = self._snapshot_current_selection()

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
            _DEFAULT_EXPS = {
                "debug_v350",
                "debug_MX_rdb_attn_dwds_out_int8",
                "debug_MX_rdb_attn_dwds_out_int8_lff_int12",
            }
            exp_select = self._resolve(project, "exps", _DEFAULT_EXPS)
            loss_select = self._resolve(project, "losses", {"loss_nr"})
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(exp["exp_name"] in exp_select)
            if data:
                for k in data[0]["losses"].keys():
                    item = QListWidgetItem(k)
                    self.loss_list.addItem(item)
                    item.setSelected(k in loss_select)
            self.loss_box.setVisible(True)
            self.right_stack.setCurrentIndex(0)

        elif project == "DOF":
            exp_select = self._resolve(project, "exps", None)
            loss_select = self._resolve(project, "losses", {"total_loss"})
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(exp["exp_name"] in exp_select)
            if data:
                for k in data[0]["losses"].keys():
                    item = QListWidgetItem(k)
                    self.loss_list.addItem(item)
                    item.setSelected(k in loss_select)
            self.loss_box.setVisible(True)
            self.right_stack.setCurrentIndex(0)

        elif project == "ODT":
            _DEFAULT_EXPS = {"infer_float", "infer_float_mx_wa_afs_int8"}
            exp_select = self._resolve(project, "exps", _DEFAULT_EXPS)
            metric_select = self._resolve(project, "metrics", {"total_metric"})
            for exp in data:
                item = QListWidgetItem(exp["exp_name"])
                self.exp_list.addItem(item)
                item.setSelected(exp["exp_name"] in exp_select)
            self.loss_box.setVisible(False)
            all_keys = []
            for exp in data:
                all_keys.extend(k for k in exp["metrics"] if k not in all_keys)
            for k in all_keys:
                item = QListWidgetItem(k)
                self.metric_list.addItem(item)
                item.setSelected(k in metric_select)
            self.metric_box.setVisible(True)
            self.right_stack.setCurrentIndex(1)

        elif project == "VBP":
            _DEFAULT_SETUPS = {
                "global_gv_vnr10ft200",
                "global_g_ft200",
                "global_dvp_10vnr_ft200",
                "global_dp_ft200",
            }
            setup_select = self._resolve(project, "exps", _DEFAULT_SETUPS)
            kr_select = self._resolve(project, "krs", None)
            seen_setups = []
            for exp in data:
                if exp["setup"] not in seen_setups:
                    seen_setups.append(exp["setup"])
            for s in seen_setups:
                item = QListWidgetItem(s)
                self.exp_list.addItem(item)
                item.setSelected(s in setup_select)
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
                item.setSelected(kr in kr_select)
            self.loss_box.setVisible(False)
            self.metric_box.setVisible(False)
            self.kr_box.setVisible(True)
            self.right_stack.setCurrentIndex(0)

        elif project == "NORMNET":
            exp_select = self._resolve(project, "exps", None)
            for exp in data:
                item = QListWidgetItem(exp["name"])
                self.exp_list.addItem(item)
                item.setSelected(exp["name"] in exp_select)
            self.metric_box.setVisible(False)
            self.kr_box.setVisible(False)
            self._populate_nn_metric_list()   # sets loss_list items + loss_box
            self.right_stack.setCurrentIndex(0)

        self.exp_list.blockSignals(False)
        self.loss_list.blockSignals(False)
        self.metric_list.blockSignals(False)
        self.kr_list.blockSignals(False)
        self.filter_input.blockSignals(False)
        _labels = {"VBP": "Setups", "NORMNET": "Runs"}
        self.exp_label.setText(_labels.get(project, "Experiments"))
        self.kr_box.setVisible(project == "VBP")
        self.vbp_plot_type_box.setVisible(project == "VBP")
        self.nn_plot_type_box.setVisible(project == "NORMNET")
        self._update_vbp_max_epoch_visibility()
        self._update_plot()

    def _on_vbp_plot_type_changed(self):
        self._update_vbp_max_epoch_visibility()
        self._update_plot()

    def _update_vbp_max_epoch_visibility(self):
        ft = (self._project == "VBP"
              and self.vbp_plot_type.currentData() == "ft_curves")
        self.vbp_max_epoch_label.setVisible(ft)
        self.vbp_max_epoch.setVisible(ft)

    # ── NORMNET view handling ─────────────────────────────────────────
    _NN_CURVE_METRICS = ["train_loss", "val_loss", "val_acc", "lr"]
    _NN_VNORM_STATS = ["mean", "median", "std",
                       "frac_below_0_01", "frac_below_0_1", "frac_below_1_0"]

    def _on_nn_plot_type_changed(self):
        if self._project != "NORMNET":
            return
        self._populate_nn_metric_list()
        self._update_plot()

    def _populate_nn_metric_list(self):
        """Fill the metric list for the current NORMNET view; pairs has none."""
        pt = self.nn_plot_type.currentData()
        if pt == "vnorm":
            items, default = self._NN_VNORM_STATS, {"mean", "median"}
            title, show = "V-norm stats", True
        elif pt == "pairs":
            items, default, title, show = [], set(), "Metrics", False
        else:  # curves
            items, default = self._NN_CURVE_METRICS, {"val_acc"}
            title, show = "Metrics", True
        sel = self._nn_metric_sel.get(pt, default)

        self.loss_list.blockSignals(True)
        self.loss_list.clear()
        for k in items:
            it = QListWidgetItem(k)
            self.loss_list.addItem(it)
            it.setSelected(k in sel)
        self.loss_list.blockSignals(False)
        self.loss_box.setTitle(title)
        self.loss_box.setVisible(show)

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
            elif self._project == "DOF":
                self._plot_dof()
            elif self._project == "VBP":
                self._plot_vbp()
            elif self._project == "NORMNET":
                self._plot_normnet()
            self.canvas.draw()

    # ── Legend helpers ────────────────────────────────────────────────
    @staticmethod
    def _legend_kwargs(n: int, base: float = 8.0) -> dict:
        if n <= 0:
            return {"fontsize": base}
        if n <= 10:
            return {"fontsize": base}
        # Reduce font for each extra entry past 10. Floor at 4pt.
        fs = max(4.0, base - 0.3 * (n - 10))
        ncol = 1 if n <= 16 else (2 if n <= 32 else 3)
        return {"fontsize": fs, "ncol": ncol}

    # ── DVNR ──────────────────────────────────────────────────────────
    def _plot_dvnr(self):
        self._plot_loss_curves(title="DVNR — Loss curves", strip_prefix="debug_MX_")

    # ── DOF ───────────────────────────────────────────────────────────
    def _plot_dof(self):
        self._plot_loss_curves(title="DOF — Loss curves")

    def _plot_loss_curves(self, title: str, strip_prefix: str = ""):
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
                legend_name = exp_name.replace(strip_prefix, "") if strip_prefix else exp_name
                ax.plot(range(1, len(values) + 1), values,
                        marker="o", markersize=3,
                        label=f"{legend_name} / {loss_key}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        n = len(selected_exps) * len(selected_losses)
        if n > 0:
            ax.legend(**self._legend_kwargs(n, base=8.0))

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
        plot_type = self.vbp_plot_type.currentData()
        if plot_type == "ft_curves":
            self._plot_vbp_ft_curves()
        else:
            self._plot_vbp_acc_vs_macs()

    def _plot_vbp_acc_vs_macs(self):
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
            y_pct = y * 100
            ax.scatter(x, y_pct, color=color, s=70, zorder=3)
            kr = exp.get("keep_ratio", "")
            ax.annotate(f"{y_pct:.2f}", (x, y_pct), textcoords="offset points",
                        xytext=(5, 4), fontsize=7)
            plotted = True

        if not plotted:
            self._empty("No valid data (missing MACs or accuracy)")
            return

        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=setup_color[s], markersize=8, label=s)
                   for s in setups]
        ax.legend(handles=handles, **self._legend_kwargs(len(setups), base=8.0))
        ax.set_xlabel("Pruned MACs (G)")
        ax.set_ylabel("Best Accuracy (%)")
        ax.set_title("VBP — Best Accuracy vs. Pruned MACs")
        ax.grid(True, alpha=0.3)

    def _plot_vbp_ft_curves(self):
        import matplotlib.cm as mcm

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

        max_epoch_text = self.vbp_max_epoch.text().strip()
        try:
            max_epoch = int(max_epoch_text) if max_epoch_text else None
        except ValueError:
            max_epoch = None

        plotted = False
        for exp in matching:
            # Preserve log order, then split: anything not FT = pre-FT
            # (covers PAT, SPARSE, or any other tag). FT epochs sorted
            # so the max-epoch filter behaves predictably.
            epochs = exp.get("epochs", [])
            pre_eps = [e for e in epochs if e["phase"].upper() != "FT"]
            ft_eps = sorted(
                (e for e in epochs if e["phase"].upper() == "FT"),
                key=lambda e: e["epoch"],
            )
            if max_epoch is not None:
                ft_eps = [e for e in ft_eps if e["epoch"] <= max_epoch]

            ret_acc = None
            retentions = exp.get("step_retentions") or []
            if retentions:
                ret_acc = retentions[-1].get("acc")

            n_pre = len(pre_eps)
            xs_pre = list(range(1, n_pre + 1))
            ys_pre = [e["val_acc"] * 100 for e in pre_eps]

            ret_x = ret_y = None
            if ret_acc is not None:
                ret_x = n_pre + 1
                ret_y = ret_acc * 100

            ft_offset = (n_pre + 2) if ret_acc is not None else (n_pre + 1)
            xs_ft = list(range(ft_offset, ft_offset + len(ft_eps)))
            ys_ft = [e["val_acc"] * 100 for e in ft_eps]

            if not xs_pre and not xs_ft and ret_x is None:
                continue

            color = setup_color[exp["setup"]]
            kr = exp.get("keep_ratio", "")
            kr_label = f"{kr:.2f}" if isinstance(kr, float) else str(kr)
            label = f"{exp['setup']} / kr={kr_label}"

            # Pre-FT and FT are drawn as separate segments so the line
            # never connects through the retention point — keeps the
            # post-prune drop visible without a misleading slope.
            if xs_pre:
                ax.plot(xs_pre, ys_pre, marker="o", markersize=3,
                        color=color, label=label)
                label = None
            if xs_ft:
                ax.plot(xs_ft, ys_ft, marker="o", markersize=3,
                        color=color, label=label)
                label = None
            if ret_x is not None:
                ax.scatter([ret_x], [ret_y], color=color, marker="x",
                           s=60, zorder=5, label=label)
                label = None
            plotted = True

        if not plotted:
            self._empty("No epoch data in selection")
            return

        ax.set_xlabel("Cumulative epoch (PAT → prune → FT)")
        ax.set_ylabel("Val Accuracy (%)")
        ax.set_title("VBP — Training Trajectory (X = retention after prune)")
        ax.grid(True, alpha=0.3)
        if matching:
            ax.legend(**self._legend_kwargs(len(matching), base=7.0))

    # ── NORMNET ───────────────────────────────────────────────────────
    def _selected_nn_runs(self) -> list:
        names = set(self._selected_exp_names())
        return [e for e in self._data if e.get("name") in names]

    def _plot_normnet(self):
        pt = self.nn_plot_type.currentData()
        if pt == "pairs":
            self._plot_normnet_pairs()
        elif pt == "vnorm":
            self._plot_normnet_vnorm()
        else:
            self._plot_normnet_curves()

    def _plot_normnet_curves(self):
        runs = self._selected_nn_runs()
        metrics = self._selected_losses()
        self._nn_metric_sel["curves"] = set(metrics)
        if not runs or not metrics:
            self._empty("Select runs and metrics")
            return
        n = len(metrics)
        for mi, metric in enumerate(metrics):
            ax = self.figure.add_subplot(n, 1, mi + 1)
            for exp in runs:
                xs, ys = [], []
                for e in exp.get("epochs", []):
                    ep, v = e.get("epoch"), e.get(metric)
                    if ep is None or v is None:
                        continue
                    xs.append(ep)
                    ys.append(v)
                if not xs:
                    continue
                arm = exp.get("arm", "")
                ls = "-" if arm == "normalized" else "--"
                ax.plot(xs, ys, marker="o", markersize=3, linestyle=ls,
                        label=f"{exp['name']} [{arm[:4]}]")
            if metric == "lr":
                ax.set_yscale("log")
            ax.set_ylabel(metric)
            ax.grid(True, alpha=0.3)
            if mi == n - 1:
                ax.set_xlabel("Epoch")
            if mi == 0:
                ax.set_title("NORMNET — per-run curves (— normalized, -- baseline)")
                ax.legend(**self._legend_kwargs(len(runs)))

    def _plot_normnet_pairs(self):
        import matplotlib.cm as mcm
        from scanners.normnet import pair_runs

        runs = self._selected_nn_runs()
        if not runs:
            self._empty("Select runs (normalized + baseline)")
            return
        pairs = [p for p in pair_runs(runs)
                 if p["normalized"] and p["baseline"]]
        if not pairs:
            self._empty("No complete normalized+baseline pair in selection")
            return

        ax = self.figure.add_subplot(2, 1, 1)
        ax2 = self.figure.add_subplot(2, 1, 2)
        cmap = mcm.get_cmap("tab10")
        def _xy(epochs, metric="val_acc"):
            xs, ys = [], []
            for e in epochs:
                ep, v = e.get("epoch"), e.get(metric)
                if ep is None or v is None:
                    continue
                xs.append(ep)
                ys.append(v)
            return xs, ys

        for i, p in enumerate(pairs):
            color = cmap(i % 10)
            norm, base = p["normalized"], p["baseline"]
            lbl = p["label"]
            nxs, nys = _xy(norm["epochs"])
            bxs, bys = _xy(base["epochs"])
            ax.plot(nxs, [y * 100 for y in nys], marker="o", markersize=3,
                    color=color, label=f"{lbl} norm")
            ax.plot(bxs, [y * 100 for y in bys], marker="s", markersize=3,
                    linestyle="--", color=color, label=f"{lbl} base")
            bmap = dict(zip(bxs, bys))
            dxs, dys = [], []
            for ep, y in zip(nxs, nys):
                if ep in bmap:
                    dxs.append(ep)
                    dys.append((y - bmap[ep]) * 100)
            ax2.plot(dxs, dys, marker="o", markersize=3, color=color, label=lbl)

        ax.set_ylabel("Val acc (%)")
        ax.set_title("NORMNET — val_acc: normalized vs baseline")
        ax.grid(True, alpha=0.3)
        ax.legend(**self._legend_kwargs(len(pairs) * 2, base=7.0))

        ax2.axhline(0, color="gray", linewidth=0.8)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Δ val_acc (pp)")
        ax2.grid(True, alpha=0.3)
        dbest = ", ".join(
            f"{p['label']}: {p['delta_best'] * 100:+.2f}pp"
            for p in pairs if p["delta_best"] is not None
        )
        ax2.set_title(f"Δ best_val_acc — {dbest}" if dbest
                      else "Δ val_acc per epoch (norm − base)")

    def _plot_normnet_vnorm(self):
        runs = [e for e in self._selected_nn_runs() if e.get("vnorm")]
        stats = self._selected_losses()
        self._nn_metric_sel["vnorm"] = set(stats)
        if not runs or not stats:
            self._empty("Select normalized run(s) with V-norm data and stat(s)")
            return
        ax = self.figure.add_subplot(111)
        for exp in runs:
            vn = exp["vnorm"]
            xs = [v["idx"] for v in vn]
            for stat in stats:
                ys = [v.get(stat) for v in vn]
                ax.plot(xs, ys, marker="o", markersize=3,
                        label=f"{exp['name']} / {stat}")
        ax.set_xlabel("V-norm snapshot (0 = post-reparam, then per epoch)")
        ax.set_ylabel("value")
        ax.set_title("NORMNET — per-layer V-norm aggregate over training")
        ax.grid(True, alpha=0.3)
        ax.legend(**self._legend_kwargs(len(runs) * len(stats), base=7.0))

    def _empty(self, msg: str = ""):
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, msg, ha="center", va="center",
                transform=ax.transAxes, color="gray", fontsize=13)
        ax.axis("off")
