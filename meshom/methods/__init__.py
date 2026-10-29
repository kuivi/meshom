"""Fill methods: one module per method, meshom/methods/mNN_<name>.py, entry point generate(domain, h, opts) -> Mesh.

A method module registers its entry point (and any variants) with the `register` decorator; the name defaults to the
module name:

    from meshom.methods import register

    @register()                         # -> METHODS["m01_isotropic"] in meshom/methods/m01_isotropic.py
    def generate(domain, h, opts): ...

    @register("m01_isotropic_tri")      # a variant under its own name
    def generate_tri(domain, h, opts): ...

    @register("m14_blocks_tri", overrides={"fill_method": "tri"})     # a variant that fixes options
    def generate_tri(domain, h, opts):
        return run(domain, h, dict(OVERRIDES["m14_blocks_tri"], **(opts or {})))[0]

A variant's fixed options live in OVERRIDES[name] (a base method: {}), declared at registration and applied in the
body (the caller's opts win over them), so a method form can show them: defaults(name) is the module's DEFAULTS with
OVERRIDES[name] on top. A variant that differs by its entry path rather than by options (m01_triangles: no
recombination; m15's *_fit and the rep_* variants: another method followed by a stage) has OVERRIDES {} and its
module's DEFAULTS, which for the composed variants are not the options they take (their docstrings say which).
effective_tolerance(name, opts) is the size tolerance a run uses (decisions G38; a <base>_rep variant: its base
method's): the method and the size gate both take it. own_optimize(name) is the default of the method's own
optimisation stage (option "optimize", decisions Z1: on for the m14 methods), None for a method without one.

Importing meshom.methods imports every mNN_* module of this package, so METHODS holds all of them; also every ref_*
module (an external generator run as a reference baseline, e.g. ref_jigsaw: not a meshom method), and every rep_*
module (variants built from other methods, e.g. rep_variants: a method followed by meshom.repair).
"""
import copy
import importlib
import pkgutil
import re
import sys
from typing import Callable

from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.size import SizeField

METHODS: dict[str, Callable[[Domain, SizeField, dict], Mesh]] = {}
OVERRIDES: dict[str, dict] = {}       # name -> the options the variant fixes over its module's DEFAULTS ({}: none)


def register(name: str | None = None, overrides: dict | None = None) -> Callable:
    def deco(fn: Callable) -> Callable:
        key = name or fn.__module__.rsplit(".", 1)[-1]
        METHODS[key] = fn
        OVERRIDES[key] = dict(overrides or {})
        return fn
    return deco


def defaults(name: str) -> dict:
    """The options METHODS[name] runs with when given none: its module's DEFAULTS (if any) with OVERRIDES[name] on
    top (a copy: what a method form shows)."""
    base = getattr(sys.modules[METHODS[name].__module__], "DEFAULTS", None) or {}
    return copy.deepcopy({**base, **OVERRIDES[name]})


def effective_tolerance(name: str, opts: dict | None) -> float | None:
    """The size tolerance of a run of METHODS[name] (decisions G38): opts["size_tolerance"] when given, else the
    method's default (defaults(name), of the base method for a <base>_rep variant: 0.20 for the m14 methods, none for
    the others); 0 or None = off (the method fits
    s = h, the size gate stays at G9's 0.15). The method builds at it and the size gate judges at it."""
    o = opts or {}
    if name.endswith("_rep") and name[:-4] in METHODS:  # rep_variants: the base method, then the repair
        name = name[:-4]
    t = o["size_tolerance"] if "size_tolerance" in o else (
        defaults(name).get("size_tolerance") if name in METHODS else None)
    return float(t) if t else None


def own_optimize(name: str) -> bool | None:
    """The default of a method's own optimisation stage (meshom.optimize as its last stage, decisions Z1): its
    defaults(name)["optimize"] (of the base method for a <base>_rep variant: True for the m14 methods), None when the
    method has no such option (then only [mesh] optimize runs the stage, after the method, in meshom run)."""
    if name.endswith("_rep") and name[:-4] in METHODS:
        name = name[:-4]
    d = defaults(name) if name in METHODS else {}
    return bool(d["optimize"]) if "optimize" in d else None


def matching(name: str) -> list[str]:
    """Registered names equal to `name` or starting with `name` + "_" ("m01" -> m01_isotropic, m01_isotropic_tri);
    [name] itself when nothing matches, so the lookup fails with a KeyError naming it."""
    return [k for k in METHODS if k == name or k.startswith(name + "_")] or [name]


for _m in pkgutil.iter_modules(__path__):
    if re.fullmatch(r"(m\d\d|ref|rep)_\w+", _m.name):
        importlib.import_module(f"{__name__}.{_m.name}")
