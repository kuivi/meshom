"""B105: the ORCA size layer builds a preset's grid once and keeps its cell sizes in the process and on disk."""
import numpy as np

from meshom.orca import sizes


def test_orca_size_builds_the_grid_once(tmp_path, monkeypatch):
    calls = []
    t0 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    h0 = np.array([10.0, 20.0, 30.0, 2.0, 40.0])

    def fake_build(name, extra=0):
        calls.append(name)
        return "grid"

    monkeypatch.setattr("meshom.orca.build", fake_build)
    monkeypatch.setattr(sizes, "cell_sizes", lambda g: (t0, h0))
    monkeypatch.setattr(sizes, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(sizes, "_SIZES", {})
    monkeypatch.setattr(sizes, "_FIELDS", {})
    a = sizes.orca_size("ORCA2", 0.0)
    assert sizes.orca_size("ORCA2", 0.0) is a and calls == ["ORCA2"]          # in-process
    assert len(list(tmp_path.glob("ORCA2_*.npz"))) == 1
    monkeypatch.setattr(sizes, "_SIZES", {})                                   # a new process: from disk
    monkeypatch.setattr(sizes, "_FIELDS", {})
    b = sizes.orca_size("ORCA2", 5.0)
    assert calls == ["ORCA2"] and np.array_equal(b.h, h0)
    assert np.allclose(b(t0), np.maximum(h0, 5.0))
