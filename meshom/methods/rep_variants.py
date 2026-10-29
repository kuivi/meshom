"""Repair variants: an existing method, then meshom.repair (local cavity remeshing of its defect clusters).

`<method>_rep` runs the base method with the options other than "repair", then repair(mesh, h, domain,
opts["repair"]); for m14 the block quads (the first n_block_quads quads) are passed as is_block, so cavities take
only the first block rows; m14_blocks_rep adds the repair's coast stage (coast_row "after", decisions B65: m14's coast
gaps hold one dislocation per column mismatch, which only the coast row can take) unless the repair options set
coast_row. With the size tolerance t (opts["size_tolerance"], else m14's default 0.20; 0 = off;
decisions G38, R22) m14 runs without it (no smooth fit
inside the stitch), the repair's bad edge is an edge beyond max(t, the repair's default 0.15) (`bad_edge`, unless the
repair options give one), and meshom.size_fit.smooth_fit at t is the last stage, on the repaired mesh (coast and the
remaining block nodes fixed; its record in <run dir>/tol_fit_final.json). (The fit before the repair, with the bad
edge at t, left triangle clusters of 3 the repair no longer removed: uniform and disc s1, 51 operations accepted
against 175 at t = 0.10, disc failing tri_cluster_max at 0.10, 0.20 and 0.30.) The optimisation stage (m14's option
`optimize`, on by default; decisions Z1) runs last, after the repair, the coast stage and the fit (m14 itself runs
without it; its record in <run dir>/optimize.json). The repair record goes to <run dir>/repair.json when meshom
compete runs it (MESHOM_RUN_DIR). The method time of the record includes the repair and the stage. The m02 / m03
_rep variants have no stage (their base methods have no `optimize` option). With m14's `gap_adapt` and
`gap_judge = "post"` (decisions B63) these later stages are also m14's judge of every gap trim (m14_blocks.run(post=)),
the corridor size gate then at the tolerance. m14's fixed-block record for meshom run's land cut (m14_blocks.CUT_FIXED)
is carried onto the mesh these stages return (`carry_cut_fixed`, decisions B60). The cells of the fixed blocks whose
kind is in the option `freeze_blocks` (default FREEZE_KINDS: the hand-placed blocks) are frozen in all these stages
(decisions R24): never a defect, in a cluster or a cavity of the repair and its coast stage, never split by the coast
fix, never in an operation of the optimisation stage; their nodes fixed in each stage and in the fit; the cells next
to them are still worked on. `()` = none frozen (the behaviour before R24).
"""
import json
import os
from pathlib import Path

import numpy as np

from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.methods import METHODS, effective_tolerance, register
from meshom.size import SizeField

COAST_FIT_MARGIN = 0.02   # B64: the final fit's wall at tol - 0.02 on a real (conforming) coast; size_fit's own is 0.01


def _split(opts: dict | None) -> tuple[dict, dict | None]:
    o = dict(opts or {})
    return o, o.pop("repair", None)


def _repair(mesh: Mesh, domain: Domain, h: SizeField, ropts: dict | None, is_block=None, keep_block: list | None = None,
            extra: dict | None = None, frozen=None) -> Mesh:
    """repair, its record to <run dir>/repair.json; keep_block (a list) receives the block mask after the repair;
    extra is merged into the record; frozen: the frozen cells (R24)."""
    from meshom.repair import repair

    out, info = repair(mesh, h, domain, ropts, is_block=is_block, frozen=frozen)
    blk = info.pop("is_block_after", None)
    info.pop("frozen_after", None)
    if keep_block is not None:
        keep_block.append(blk)
    if extra:
        info.update(extra)
    d = os.environ.get("MESHOM_RUN_DIR")
    if d:
        (Path(d) / "repair.json").write_text(json.dumps(info, indent=1, default=float))
    return out


FREEZE_KINDS = ("hand",)   # R24: the fixed blocks whose cells the later stages leave alone (option freeze_blocks)
REP_MAX_BLOCKS = None   # B8: m14_blocks_rep places blocks without a cap unless its options give max_blocks (the
#                         other m14 variants: 32; without the repair the extra seams cost irregular nodes on band)


@register("m14_blocks_rep", overrides={})           # composed: m14_blocks, then repair
def m14_blocks_rep(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    from meshom.methods.m14.coastfix import coast_fix
    from meshom.methods.m14.gaps import GATES
    from meshom.methods.m14_blocks import run

    from meshom.repair import DEFAULTS as RD, repair

    from meshom.methods.m14_blocks import DEFAULTS as MD, optimize_stage

    base, ropts = _split(opts)
    base.setdefault("max_blocks", REP_MAX_BLOCKS)       # B8: no block cap
    cfix = bool(base.pop("coast_fix", True)) and bool(domain.loops)   # B64: conforming only (cut: no loops)
    kinds = tuple(base.pop("freeze_blocks", FREEZE_KINDS))             # R24: fixed blocks frozen in the stages below
    opt = bool(base.get("optimize", MD["optimize"]))    # Z1: the stage after the repair / coast stage / fit, not in m14
    oopts = base.get("optimize_opts", MD["optimize_opts"])
    base["optimize"] = False
    ropts = {"coast_row": "after", **(ropts or {})}     # B65: the coast stage after the repair (m14's coast gaps)
    tol = effective_tolerance("m14_blocks_rep", base)   # G38, R22: m14 and the repair as without it, the fit after them
    base["size_tolerance"] = None
    if tol is not None:                                 # a bad edge of the repair: beyond the tolerance, at least G9's
        ropts = {"bad_edge": max(float(tol), RD["bad_edge"]), **(ropts or {})}

    def post(mesh: Mesh, n_block_quads: int, rec: dict | None = None, final: bool = True) -> Mesh:
        """The stages after m14: repair, coast stage, fit, the optimisation stage (final: their records written);
        rec: m14's fixed-block record of `mesh` (fixed_id, fixed_kind): the cells of the blocks of a kind in
        freeze_blocks are frozen in every stage (R24)."""
        blk = np.zeros(mesh.n_elems, dtype=bool)
        blk[mesh.n_tris:mesh.n_tris + int(n_block_quads)] = True
        fz0 = frozen_cells(rec, kinds)

        def fz(m: Mesh):                                    # the frozen cells on a later mesh (same node positions)
            return None if fz0 is None else cells_on(mesh, np.flatnonzero(fz0), m)

        rep = (lambda m, b, after=None: _repair(m, domain, h, ropts, b, after, frozen=fz0)) if final else \
            (lambda m, b, after=None: _repair_quiet(repair, m, domain, h, ropts, b, after, frozen=fz0))
        if tol is None:
            out = rep(mesh, blk)
            out = coast_fix(out, out.node_flag != 0, frozen=fz(out))[0] if cfix else out
            return optimize_stage(out, h, None, oopts, fz(out))[0] if opt else out
        from meshom.size_fit import smooth_fit

        after = []
        out = rep(mesh, blk, after)
        keep = out.node_flag != 0                           # R22: the smooth fit again as the last stage, blocks fixed
        b = after[0]
        keep[out.tris[b[:out.n_tris]].ravel()] = True
        keep[out.quads[b[out.n_tris:]].ravel()] = True
        crec = None
        if cfix:                                            # B64: coast-row corners split, triangle clusters broken
            out, keep, crec = coast_fix(out, keep, frozen=fz(out))
        f = fz(out)
        if f is not None:                                   # R24: the frozen nodes stay (block nodes: kept above)
            keep[out.quads[f[out.n_tris:]].ravel()] = True
        zone = out.tri_zone
        fit_kw = base.get("tol_fit", {})
        if cfix:                                            # B64: the wall further inside on a fixed real coast
            fit_kw = {"margin": COAST_FIT_MARGIN, **fit_kw}
        out, r = smooth_fit(out, h, float(tol), keep, **fit_kw)
        out.tri_zone = zone
        if cfix:                                            # B64: again on the fitted nodes (the fit moves them), no node
            out, _, c2 = coast_fix(out, np.ones(out.n_nodes, dtype=bool), frozen=fz(out))   # removed or moved
            crec = {"before_fit": crec, "after_fit": c2}
        d = os.environ.get("MESHOM_RUN_DIR")
        if d and final:                                     # the final fit's record next to the repair's
            (Path(d) / "tol_fit_final.json").write_text(json.dumps(
                {**{k: r[k] for k in ("tol", "free_nodes", "before", "after", "total_s")}, "coast_fix": crec},
                indent=1, default=float))
        return optimize_stage(out, h, float(tol), oopts, fz(out))[0] if opt else out     # Z1: the last stage

    judge = None
    if base.get("gap_adapt") and base.get("gap_judge") == "post":     # B63: the trims judged on the final mesh
        judge = lambda m, st: post(m, st["info"]["n_block_quads"], st, final=False)      # noqa: E731
        if tol is not None and base.get("gap_gates") is None:         # its size gate at the tolerance (G38)
            base["gap_gates"] = {**GATES, "ln_sh_p99": float(tol)}
    from meshom.methods import m14_blocks

    mesh, info = run(domain, h, base, post=judge)
    out = post(mesh, info["stitch"]["n_block_quads"], m14_blocks.CUT_FIXED)
    m14_blocks.CUT_FIXED = carry_cut_fixed(m14_blocks.CUT_FIXED, mesh, out)    # B60: meshom run's land cut takes it
    return out


def carry_cut_fixed(rec: dict | None, old: Mesh, new: Mesh) -> dict | None:
    """m14's fixed-block record (m14_blocks.CUT_FIXED, ids of m14's mesh `old`) carried onto `new`, the mesh the
    repair, the coast stage and the fit made of it (they renumber nodes and elements; decisions B60): nodes by
    position (the fixed blocks' nodes those stages keep where they are), a protected cell stays protected where `new`
    has an element on the same nodes; a cell the repair remeshed loses its protection (the mask decides there), bank
    edges and coast nodes the repair removed are dropped."""
    if rec is None:
        return None
    nmap = _node_map(old, np.r_[rec["coast_nodes"], rec["bank_edges"].ravel()], new)
    protect = cells_on(old, np.flatnonzero(rec["protect"]), new)
    cn, be = nmap[rec["coast_nodes"]], nmap[rec["bank_edges"]]
    return {"protect": protect, "coast_nodes": cn[cn >= 0], "bank_edges": be[(be >= 0).all(axis=1)]}


def _elems(m: Mesh) -> np.ndarray:
    """(T+Q, 4) element node ids, triangles padded with -1."""
    return np.r_[np.c_[m.tris, np.full(m.n_tris, -1)], m.quads].astype(np.int64)


def _node_map(old: Mesh, nodes: np.ndarray, new: Mesh) -> np.ndarray:
    """(old.n_nodes,): the node of `new` at the same position as each of the old `nodes` (-1: none; others -1)."""
    from scipy.spatial import cKDTree

    used = np.unique(np.asarray(nodes, dtype=np.int64))
    nmap = np.full(old.n_nodes, -1, dtype=np.int64)
    if len(used):
        dist, j = cKDTree(new.points).query(old.points[used], distance_upper_bound=1e-9)
        nmap[used[np.isfinite(dist)]] = j[np.isfinite(dist)]
    return nmap


def cells_on(old: Mesh, cells: np.ndarray, new: Mesh) -> np.ndarray:
    """Element mask of `new`: the cells (element ids of `old`) found in `new` on nodes at the same positions (the
    stages after m14 renumber nodes and elements; a cell they remeshed or whose node moved is not found)."""
    eo, en = _elems(old), _elems(new)
    pe = eo[cells]
    nmap = _node_map(old, pe[pe >= 0], new)
    pm = np.where(pe >= 0, nmap[np.maximum(pe, 0)], -1)
    pm = pm[((pm >= 0) | (pe < 0)).all(axis=1)]       # the cells whose nodes are all still there
    cand = np.flatnonzero(np.isin(en, np.r_[pm[pm >= 0], -1]).all(axis=1))
    at = {tuple(np.sort(en[i])): i for i in cand}
    hit = [at.get(tuple(np.sort(r)), -1) for r in pm]
    out = np.zeros(new.n_elems, dtype=bool)
    out[[i for i in hit if i >= 0]] = True
    return out


def frozen_cells(rec: dict | None, kinds: tuple) -> np.ndarray | None:
    """The element mask of the fixed blocks of a kind in `kinds` (m14's record: fixed_id per element, fixed_kind per
    block; decisions R24); None when there is none."""
    if rec is None or not kinds or "fixed_id" not in rec:
        return None
    fid = np.asarray(rec["fixed_id"])
    on = np.array([k in kinds for k in rec["fixed_kind"]] + [False])
    m = on[np.where(fid >= 0, fid, len(on) - 1)]
    return m if m.any() else None


def _repair_quiet(repair, mesh: Mesh, domain: Domain, h: SizeField, ropts: dict | None, is_block, keep_block=None,
                  frozen=None):
    """repair without its record (gap_adapt's judge, B63)."""
    out, info = repair(mesh, h, domain, ropts, is_block=is_block, frozen=frozen)
    if keep_block is not None:
        keep_block.append(info.pop("is_block_after", None))
    return out


@register("m02_linf_frontal_rep", overrides={})     # composed: m02_linf_frontal, then repair
def m02_linf_frontal_rep(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    base, ropts = _split(opts)
    return _repair(METHODS["m02_linf_frontal"](domain, h, base), domain, h, ropts)


@register("m03_parallelograms_rep", overrides={})   # composed: m03_parallelograms, then repair
def m03_parallelograms_rep(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    base, ropts = _split(opts)
    return _repair(METHODS["m03_parallelograms"](domain, h, base), domain, h, ropts)
