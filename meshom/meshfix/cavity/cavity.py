"""Cavity search (decisions B134): grow a disk of elements around a defect until its boundary admits a pattern.

The fill of a cavity is decided by its boundary. For a boundary node v (an interior node of the mesh) the outside
elements take the ideal angle out(v) = 90 deg per quad + 60 deg per triangle; the fill must supply 360 - out(v) for
v to end regular. A fill made of grid patterns supplies 90 (a corner: one quad) or 180 (a side: two elements). So:

- convexify: every boundary node whose fill angle is not 90 or 180 (concave 270, an outside triangle, a node with
  >= 4 outside quads) takes all its outside elements into the cavity; repeated until none is left. Holes and pinches
  close the same way (their nodes are concave). A cavity that reaches the mesh boundary (coast, artificial) is a
  coast cavity (`coast=True`: its mesh-boundary nodes are allowed on the loop; module patterns decides) or rejected.
- a grid cavity has exactly 4 corners (fill angle 90; discrete Gauss-Bonnet: the defects inside have net charge 0)
  and 4 sides with edge counts (e0, e1, e2, e3) between them.
- growth: one element ring at a time (every element sharing a node with the cavity), then convexify again.
"""
import numpy as np

from meshom.optimize import boundary_loop

QUAD_DEG, TRI_DEG = 90, 60


def loop_of(W, S) -> tuple | None:
    return boundary_loop([W.cells(e) for e in S])


def out_angle(W, v: int, S: set) -> tuple[int, list]:
    """(ideal angle of the elements at v outside S, those elements)."""
    out = [e for e in W.at(v) if e not in S]
    return sum(QUAD_DEG if W.nv[e] == 4 else TRI_DEG for e in out), out


def convexify(W, S: set, max_elems: int, coast: bool = False, reads: set | None = None) -> set | None:
    """The smallest superset of S (by the rule of the module docstring) whose boundary nodes all have fill angle 90
    or 180; None when it exceeds max_elems or reaches a fixed / mesh-boundary node (coast=False) or a fixed node
    (coast=True). reads (optional): the elements looked at are added to it (the read set of the fast engine, B140).
    The Numba kernel ncavity.convexify_kernel (B139); convexify_py is the same rule in Python."""
    from meshom.meshfix.cavity.ncavity import convexify_kernel, scratch

    c = scratch(W)
    art = getattr(W, "art", None)
    if art is None or len(art) < len(W.P):
        a = np.zeros(len(W.P), dtype=bool)
        if art is not None:
            a[: len(art)] = art
        art = a
    seed = np.fromiter(S, dtype=np.int64, count=len(S))
    status, out, c[2] = convexify_kernel(W.el, W.nv, W.inc, W.ninc, W.bnode, art, seed, int(max_elems), bool(coast),
                                         c[0], c[1], c[2])
    if reads is not None:
        reads.update(out.tolist())
    if status == 1:
        W.last_stop = "stop_size"
        return None
    if status == 2:
        W.last_stop = "stop_boundary"
        return None
    S = set(out.tolist())
    if loop_of(W, S) is None:                               # several loops left (a hole the rule cannot see)
        W.last_stop = "stop_loop"
        return None
    return S


def convexify_py(W, S: set, max_elems: int, coast: bool = False, reads: set | None = None) -> set | None:
    """convexify in Python (the reference of the kernel)."""
    S = set(S)
    for _ in range(10_000):
        if reads is not None:
            reads.update(S)
        if len(S) > max_elems:
            W.last_stop = "stop_size"
            return None
        nodes = {v for e in S for v in W.cells(e)}
        add = set()
        for v in nodes:
            if W.bnode[v]:
                if not coast or W.art[v]:
                    W.last_stop = "stop_boundary"
                    return None
                continue                                   # coast node: any fill angle (patterns decide)
            ang, out = out_angle(W, v, S)
            if not out:
                continue
            fill = 360 - ang
            if fill in (90, 180):
                continue
            add.update(out)
        if not add:
            lp = loop_of(W, S)
            if lp is None:                                  # several loops left (a hole the rule cannot see)
                W.last_stop = "stop_loop"
                return None
            return S
        S |= add
    return None


COAST_CORNER_DEG = 135.0   # B134: a coast node of a loop is a corner below this cavity-side angle, a side below
COAST_CONCAVE_DEG = 225.0  # this one, and concave (no grid cavity) above


def loop_angle(P: np.ndarray, a: int, v: int, b: int) -> float:
    """Angle (deg, 0-360) at v on the left of the path a -> v -> b (the cavity side of a CCW loop)."""
    x = P[v]
    u, w = P[b] - x, P[a] - x
    u = u - (u @ x) * x
    w = w - (w @ x) * x
    return float(np.degrees(np.arctan2(np.cross(u, w) @ x, u @ w)) % 360.0)


def corners(W, S: set, lp: tuple, coast: bool = False) -> list[int] | None:
    """Loop indices of the corners. An interior node is a corner when its fill angle is 90. With coast=True a run of
    coast (mesh-boundary) edges of the loop is part of one side however jagged it is (no corner inside a run), and a
    coast node where the loop leaves the coast, or touches it, is a corner when its cavity-side angle is below
    COAST_CORNER_DEG; None when a loop node is on the mesh boundary and coast=False, or such a coast node is concave
    (>= COAST_CONCAVE_DEG)."""
    out = []
    n = len(lp)
    for i, v in enumerate(lp):
        if W.bnode[v]:
            if not coast:
                return None
            a, b = lp[i - 1], lp[(i + 1) % n]
            if W.is_bedge(a, v) and W.is_bedge(v, b):
                continue                                   # inside a coast run: the run is one side, however jagged
            th = loop_angle(W.P, a, v, b)
            if th >= COAST_CONCAVE_DEG:
                return None
            if th < COAST_CORNER_DEG:
                out.append(i)
            continue
        ang, _ = out_angle(W, v, S)
        if 360 - ang == 90:
            out.append(i)
    return out


def ring(W, S: set) -> set:
    nodes = {v for e in S for v in W.cells(e)}
    return S | {e for v in nodes for e in W.at(v)}


def grow(W, seed: set, max_elems: int, max_steps: int, why: dict | None = None, coast: bool = False,
         reads: set | None = None):
    """Yield (S, loop, corner indices) for every grid cavity (4 corners) met while growing from seed; why counts the
    reasons growth steps gave no cavity (stop_boundary, stop_size, loop, corners_<n>); reads (optional) collects the
    elements looked at (convexify)."""
    why = {} if why is None else why
    S = set(seed)
    seen = set()
    for _ in range(max_steps):
        C = convexify(W, S, max_elems, coast, reads)
        if C is None:
            k = getattr(W, "last_stop", "stop")
            why[k] = why.get(k, 0) + 1
            return
        key = frozenset(C)
        if key not in seen:
            seen.add(key)
            lp = loop_of(W, C)
            if lp is None:
                why["loop"] = why.get("loop", 0) + 1
            else:
                cr = corners(W, C, lp, coast)
                if cr is not None and len(cr) == 4:
                    yield C, lp, cr
                else:
                    k = f"corners_{len(cr) if cr is not None else 'b'}"
                    why[k] = why.get(k, 0) + 1
        S = ring(W, C)
