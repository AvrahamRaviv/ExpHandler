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


if __name__ == "__main__":
    test_loader()
    test_tolerant()
    test_dir_isolation_and_labels()
    out = test_discover_and_render()
    print("ALL PASS")
