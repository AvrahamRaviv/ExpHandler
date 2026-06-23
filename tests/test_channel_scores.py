"""Tests for channel-score loading + headless heatmap render.

Run: QT_QPA_PLATFORM=offscreen python tests/test_channel_scores.py
"""

import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scanners.channel_scores import discover_channel_scores, load_channel_scores


def _write_net(path, scorer, n_layers=30, seed=0):
    rng = np.random.default_rng(seed)
    widths = [64, 64, 128, 128, 256, 256, 512, 512]
    layers = []
    for i in range(n_layers):
        w = widths[i % len(widths)]
        scores = rng.random(w).tolist()
        ent = {"name": f"layer{i // 3}.{i % 3}.conv", "scores": scores}
        if i % 2 == 0:
            ent["kept"] = [bool(s > 0.3) for s in scores]
        layers.append(ent)
    json.dump({"schema": "channel_scores/v1", "model": "resnet50",
               "scorer": scorer, "stage": "pre_prune", "layers": layers},
              open(path, "w"))


def test_loader():
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "run_a_channel_scores.json")
        _write_net(p, "l1norm")
        rec = load_channel_scores(p)
        assert rec is not None
        assert rec["model"] == "resnet50" and rec["scorer"] == "l1norm"
        assert rec["higher_is_better"] is True
        assert len(rec["layers"]) == 30
        assert isinstance(rec["layers"][0]["scores"], np.ndarray)
        assert rec["layers"][0]["kept"] is not None       # even layer
        assert rec["layers"][1]["kept"] is None            # odd layer, no mask
        assert rec["gmin"] >= 0.0 and rec["gmax"] <= 1.0
        assert len(rec["arch_key"]) == 30
        print("test_loader OK")


def test_tolerant():
    with tempfile.TemporaryDirectory() as d:
        # Missing optional keys + one bad layer (skipped) + empty scores.
        data = {"layers": [
            {"name": "good", "scores": [1.0, 2.0, 3.0]},
            {"name": "empty", "scores": []},
            {"scores": [1.0]},                       # no name
            {"name": "bad", "scores": ["x", "y"]},   # non-numeric -> skipped
        ]}
        p = os.path.join(d, "x_channel_scores.json")
        json.dump(data, open(p, "w"))
        rec = load_channel_scores(p)
        assert rec is not None
        assert len(rec["layers"]) == 1 and rec["layers"][0]["name"] == "good"
        assert rec["model"] is None
        assert rec["label"] == "x"                   # filename-stem fallback
        print("test_tolerant OK")

    # Unreadable / non-dict returns None.
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "junk_channel_scores.json")
        open(p, "w").write("not json")
        assert load_channel_scores(p) is None
        print("test_tolerant(none) OK")


def test_discover_and_render():
    with tempfile.TemporaryDirectory() as d:
        fam = os.path.join(d, "resnet50_family")
        os.makedirs(fam)
        _write_net(os.path.join(fam, "run_l1_channel_scores.json"), "l1norm", seed=1)
        _write_net(os.path.join(fam, "run_vn_channel_scores.json"), "vnorm", seed=2)
        found = discover_channel_scores(d)
        assert len(found) == 2, found
        print("test_discover OK")

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PyQt5.QtWidgets import QApplication
        from screens.channels import ChannelsScreen
        app = QApplication.instance() or QApplication([])
        scr = ChannelsScreen()
        scr.load("NORMNET", d)
        assert scr.file_list.count() == 2
        # Select both, exercise every render path.
        for i in range(2):
            scr.file_list.item(i).setSelected(True)
        # Top-N: width is clamped to N per layer (here all layers >= 10).
        scr.topn_input.setText("10")
        scr.view_box.setCurrentIndex(0)
        scr.normalized_chk.setChecked(True)
        scr._render()
        m = scr._build_matrix(scr._records[found[0]], sort=True, top_n=10)
        assert m.shape[1] == 10, m.shape
        scr.topn_input.setText("")                # back to all

        for scope in range(2):                    # per-layer / global
            scr.norm_box.setCurrentIndex(scope)
            for sort in (False, True):
                scr.sort_chk.setChecked(sort)
                scr.view_box.setCurrentIndex(0)   # side-by-side
                # Normalized (scale ignored / disabled)
                scr.normalized_chk.setChecked(True)
                assert not scr.scale_box.isEnabled()
                scr._render()
                # Raw with each color scale
                scr.normalized_chk.setChecked(False)
                assert scr.scale_box.isEnabled()
                for scale in range(3):            # linear / log / robust
                    scr.scale_box.setCurrentIndex(scale)
                    scr._render()
        scr.view_box.setCurrentIndex(1)           # diff (arch matches)
        scr._render()
        # Save a render so we can eyeball it.
        out = os.path.join(tempfile.gettempdir(), "channels_render.png")
        scr.figure.savefig(out, dpi=110)
        print(f"test_render OK -> {out}")
        return out


def test_dir_isolation_and_labels():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from screens.channels import ChannelsScreen
    app = QApplication.instance() or QApplication([])

    with tempfile.TemporaryDirectory() as root:
        a = os.path.join(root, "archA")
        b = os.path.join(root, "archB")
        for sub in ("loc_mean", "glob_none"):
            os.makedirs(os.path.join(a, sub))
            _write_net(os.path.join(a, sub, "run_l1_channel_scores.json"), "l1")
        os.makedirs(b)
        _write_net(os.path.join(b, "run_l1_channel_scores.json"), "l1")

        scr = ChannelsScreen()
        scr.load("NORMNET", a)
        labels = {scr.file_list.item(i).text() for i in range(scr.file_list.count())}
        # Only archA files, and labels carry the suffix sub-dir.
        assert labels == {"loc_mean/run_l1", "glob_none/run_l1"}, labels

        # Switching arch must NOT accumulate archA's files.
        scr.load("NORMNET", b)
        labels_b = {scr.file_list.item(i).text() for i in range(scr.file_list.count())}
        assert labels_b == {"run_l1"}, labels_b
        assert scr.file_list.count() == 1
        print("test_dir_isolation_and_labels OK")


def test_visible_range():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from screens.channels import ChannelsScreen
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as d:
        _write_net(os.path.join(d, "run_channel_scores.json"), "l1", seed=3)
        scr = ChannelsScreen()
        scr.load("VBP", d)
        scr.file_list.item(0).setSelected(True)
        rec = scr._records[scr.file_list.item(0).data(_role())]

        m = scr._build_matrix(rec, sort=True, top_n=5)
        visible = m[np.isfinite(m)]
        allv = scr._flat(rec)
        # Top-5 pool excludes the network's low tail, so its min sits well
        # above the global min -> a tighter color range.
        n_all = scr._range_norm(allv, "linear")
        n_vis = scr._range_norm(visible, "linear")
        assert n_vis.vmin > n_all.vmin, (n_vis.vmin, n_all.vmin)
        assert n_vis.vmin == visible.min()
        print("test_visible_range OK")


def _role():
    from PyQt5.QtCore import Qt
    return Qt.UserRole


def _new_screen_with(d, files):
    """Build a ChannelsScreen over dir ``d`` with all discovered files selected.

    ``files`` is a list of (filename, writer_fn) tuples already written to ``d``.
    """
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt5.QtWidgets import QApplication
    from screens.channels import ChannelsScreen
    QApplication.instance() or QApplication([])
    scr = ChannelsScreen()
    scr.load("NORMNET", d)
    for i in range(scr.file_list.count()):
        scr.file_list.item(i).setSelected(True)
    return scr


def _plot_idx(scr, key):
    """Index of a plot_box entry by its userData key."""
    return next(i for i in range(scr.plot_box.count())
               if scr.plot_box.itemData(i) == key)


def test_render_new_plot_types():
    import itertools
    with tempfile.TemporaryDirectory() as d:
        _write_net(os.path.join(d, "run_l1_channel_scores.json"), "l1", seed=1)
        _write_net(os.path.join(d, "run_vn_channel_scores.json"), "vn", seed=2)
        scr = _new_screen_with(d, None)
        assert scr.file_list.count() == 2

        # Ridgeline: enable matrix + control sweep.
        scr.plot_box.setCurrentIndex(_plot_idx(scr, "ridgeline"))
        assert not scr.portrait_chk.isEnabled()
        assert scr.threshold_input.isEnabled() and scr.logx_chk.isEnabled()
        assert not scr.rank_norm_chk.isEnabled()
        for nm, lx, th in itertools.product((False, True), (False, True),
                                            ("", "0.5")):
            scr.normalized_chk.setChecked(nm)
            scr.logx_chk.setChecked(lx)
            scr.threshold_input.setText(th)
            scr._render()

        # Rank curve.
        scr.plot_box.setCurrentIndex(_plot_idx(scr, "rank"))
        assert scr.rank_norm_chk.isEnabled() and not scr.portrait_chk.isEnabled()
        for nr, ly, tn in itertools.product((False, True), (False, True),
                                            ("", "10")):
            scr.rank_norm_chk.setChecked(nr)
            scr.logx_chk.setChecked(ly)
            scr.topn_input.setText(tn)
            scr._render()
        scr.topn_input.setText("")

        # Kept vs pruned (fixtures carry masks on even layers).
        scr.plot_box.setCurrentIndex(_plot_idx(scr, "kept"))
        assert scr.sep_metric_box.isEnabled()
        assert not scr.topn_input.isEnabled()
        for mi in range(scr.sep_metric_box.count()):
            scr.sep_metric_box.setCurrentIndex(mi)
            scr._render()

        out = os.path.join(tempfile.gettempdir(), "channels_dist_render.png")
        scr.figure.savefig(out, dpi=110)
        print(f"test_render_new_plot_types OK -> {out}")


def test_kept_absent_message():
    with tempfile.TemporaryDirectory() as d:
        # A file whose layers all omit the kept mask.
        data = {"schema": "channel_scores/v1", "model": "m", "scorer": "l1",
                "layers": [{"name": f"l{i}", "scores": list(range(1, 9))}
                           for i in range(4)]}
        json.dump(data, open(os.path.join(d, "nomask_channel_scores.json"), "w"))
        scr = _new_screen_with(d, None)
        scr.plot_box.setCurrentIndex(_plot_idx(scr, "kept"))
        scr._render()                                  # must not raise
        assert "mask" in scr.hint.text().lower(), scr.hint.text()
        print("test_kept_absent_message OK")


def test_tiny_layer_no_crash():
    with tempfile.TemporaryDirectory() as d:
        # Single-channel layer + constant-score layer, both degenerate.
        data = {"schema": "channel_scores/v1", "model": "m", "scorer": "l1",
                "layers": [{"name": "single", "scores": [0.5]},
                           {"name": "const", "scores": [0.3] * 20}]}
        json.dump(data, open(os.path.join(d, "tiny_channel_scores.json"), "w"))
        scr = _new_screen_with(d, None)
        for key in ("ridgeline", "rank"):
            scr.plot_box.setCurrentIndex(_plot_idx(scr, key))
            scr.logx_chk.setChecked(True)              # hit positivity guards
            scr._render()                              # must not raise
        print("test_tiny_layer_no_crash OK")


if __name__ == "__main__":
    test_loader()
    test_tolerant()
    test_dir_isolation_and_labels()
    test_visible_range()
    test_render_new_plot_types()
    test_kept_absent_message()
    test_tiny_layer_no_crash()
    out = test_discover_and_render()
    print("ALL PASS")
