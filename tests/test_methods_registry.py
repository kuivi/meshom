"""meshom.methods registry: every registered name has its OVERRIDES entry (the options a variant fixes over its
module's DEFAULTS, {} for a base method or a variant with another entry path), defaults(name) is the module's DEFAULTS
with them on top, and declaring them changed no mesh: on uniform at scale 4 four variants give the same arrays with no
options, with defaults(name) submitted unchanged (what the UI form sends), and as their base method
with OVERRIDES[name]."""
import sys
import numpy as np
import pytest

from meshom.cases import CASES
from meshom.methods import METHODS, OVERRIDES, defaults
from meshom.size import scaled

FIXED = {
    "m02_linf_frontal_tri": {"n": 6},
    "m05_quasi_structured_loose": {"tol_angle": 45.0, "tol_size": 0.4},
    "m06_qmorph_field": {"field": True},
    "m07_instant_meshes_im": {"orientation": "im"},
    "m08_quadriflow_im": {"orientation": "im"},
    "m08_quadriflow_keep": {"orientation": "im", "keep": 12.0},
    "m12_cubed_ot_polish": {"polish": 1.0},
    "m13_squadgen_spring": {"size_iters": 0},
    "m13_squadgen_raw": {"size_iters": 0, "spring_iters": 0},
    "m14_blocks_igm": {"igm": True},
    "m14_blocks_tri": {"fill_method": "tri"},
    "m14_blocks_mix": {"fill_method": "mix", "front": False},
    "m14_blocks_cut": {"cut": True},
}


def test_every_method_has_overrides():
    assert set(OVERRIDES) == set(METHODS)
    for name, o in FIXED.items():
        assert OVERRIDES[name] == o, name
    for name, o in OVERRIDES.items():
        base = getattr(sys.modules[METHODS[name].__module__], "DEFAULTS", None) or {}
        assert set(o) <= set(base), name                      # a fixed option is one of the module's options
        assert defaults(name) == {**base, **o}
    d = defaults("m15_tri")
    d["packing"]["n"] = 5                                     # a copy: the module's DEFAULTS stay as they are
    assert defaults("m15_tri")["packing"] == {}


def _arrays(m) -> tuple:
    return m.points, m.tris, m.quads, m.node_flag


@pytest.mark.parametrize("name, base", [("m14_blocks_tri", "m14_blocks"), ("m08_quadriflow_keep", "m08_quadriflow"),
                                        ("m02_linf_frontal_tri", None), ("m13_squadgen_raw", "m13_squadgen")])
def test_variant_meshes_unchanged(name, base):
    domain, h0 = CASES["uniform"]
    h = scaled(h0, 4.0)
    ref = _arrays(METHODS[name](domain, h, {}))
    others = {"defaults": _arrays(METHODS[name](domain, h, defaults(name)))}
    if base:
        others["base"] = _arrays(METHODS[base](domain, h, dict(OVERRIDES[name])))
    for k, arr in others.items():
        assert all(np.array_equal(a, b) for a, b in zip(ref, arr)), (name, k)
    print(f"{name}: {len(ref[0])} nodes, identical to {sorted(others)}")
