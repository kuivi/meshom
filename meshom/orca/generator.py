# SPDX-License-Identifier: GPL-3.0-or-later
"""A close port of G. Madec's ORCA grid generator (NEMO, OPA8 / LODYC; tropical stretch by A. Sellar, Met Office),
CeCILL licence, read from github.com/NCAR/tx1_4; Madec and Imbard (1996), doi:10.1007/BF00211684; see NOTICE.

NEMO's tripolar ORCA grid in Python. The Fortran is `supergrid/ORCA_gridgen` of tx1_4 at commit 557efafc1b5a (OPA8 /
LODYC 1999, the stretch Met Office 2005), with the user guide `tripole-user-guide.pdf`; the paper is in Clim. Dyn. 12,
381-388. Each law below names the Fortran file and routine it repeats. Nothing is fitted here: the parameters come
from a preset (`meshom.orca.presets`).

The construction (tripole.f90, program order):

1. `set_coefficients` (functions.f90): the three base functions of the north half-grid's ellipses, A (half-axis along
   the pole axis), B (the other half-axis) and Y0 (the centre on the pole axis), as functions of the continuous
   meridional index j of the north part (j = 1 on the grid equator, the join; j = rjpnth on the pole line). RA2 is found
   by bisection so that the ellipses end on the segment between the two poles (rphi1 on rlampo - 180, 180 - rphi2 on
   rlampo).
2. `hgr_ctl` (hgrctl.f90): the checks of the base functions (boundary values, A and B decreasing, |dY0| < |dA|).
3. `hgr_nth` / `nth_car` (hgrnth.f90, nthcar.f90): each meridian of the north half-grid (180 degrees of longitude, the
   other half by symmetry) is the orthogonal trajectory of the ellipses from the grid equator (a circle of radius REQ
   in the north polar stereographic plane), integrated in x by leapfrog with an Asselin filter; its points at the
   requested j are placed on ellipse j where the ellipse's normal has the meridian's slope.
4. `msh_sth` (mshsth.f90): the south part, Mercator rows (isotropic, latitude asin(tanh(rdx (j - jpeqt)))) and
   meridians every rdx degrees.
5. `msh_nth` (mshnth.f90): the half-grid placed into the global arrays, mirrored about the U-F column of the second
   pole, the north fold row, the two east-west wrap columns, the rotation by rlampo.
6. `msh_trp` (mshtrp.f90, trop.f90, shoots.f90), when the preset has a stretch: the tropical rows replaced by the rows
   of dphi/dj = R cos(phi) - (R - Rmin) (exp(-(c phi)^n / a^n) + exp(-(c phi)^n / b^n)) / 2, c found by Newton so that
   the stretched rows meet the Mercator row nearest pphi_join.

Two extensions, both used by the presets and both stated where they act:

- the pivot: the tx1_4 code has the pole line on a V-F row (F-pivot, rjpnth = jpnord + 1/2; ORCA1, ORCA05). The user
  guide (section 5.2.4) says ORCA2 came from an older version with the fold on a T-U row (T-pivot): the same code with
  rjpnth = jpnord.
- refinement (the user guide, 5.2.4: "ORCA025 is derived from ORCA05 by quadrature"): the generator's grid is a map of
  the continuous indices (I, J); a grid refined q times takes its T points at the base grid's (I, J) on the 1/q lattice
  and its U, V, F points half a fine step further. `grid` evaluates the map at any such lattice.

Continuous indices (Fortran's, 1-based): I is the meridian index, U and F points at integer I, T and V at I - 1/2;
longitude rdx (I - 1) + rlampo - 180 east of the rotation. J is the row index, T and U at integer J, V and F at
J + 1/2; rows J <= jpeq are the south part, J >= jpeq the north one (north j = J - jpeq + 1).
"""
from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange

RPI = 3.141592653589793          # param.f90 rpi
RAD = RPI / 180.0                # param.f90 rad
RZERO = 1e-14                    # param.f90 rzero (and trop.f90's)
GAMMA = 0.1                      # nthcar.f90: the Asselin filter of the leapfrog


@dataclass(frozen=True)
class Stretch:
    """The tropical stretch of trop.f90 (module header and `dphidj`)."""
    ppres: float                 # R: the baseline (Mercator) resolution at the equator, degrees
    ppresmin: float              # Rmin: the smallest meridional spacing, degrees
    jpexp: int = 4               # n: the exponent of the "bi-gaussian"
    ppa: float = 12.0            # a
    ppb: float = 8.0             # b
    pphi_join: float = 20.0      # the stretched rows meet the Mercator row nearest this latitude
    jptint: int = 100            # high-resolution points per grid box (the integration of dj/dphi)


@dataclass(frozen=True)
class GenParams:
    """The generator's parameters (param.f90, functions.f90 `set_coefficients`, trop.f90)."""
    jpi: int                     # zonal points + 1 (360 / rdx + 1)
    jpeqt: int                   # rows from the south end to the equator, both included (unstretched)
    jpeqn: int                   # rows from the equator (excluded) to the join (included)
    jpnord: int                  # rows of the north part, the join included
    rlampo: float = 253.0        # 73 + 180: the second pole's meridian is rlampo - 180 (+ 180), the grid starts there
    rphi1: float = 50.0          # the first pole's latitude (on rlampo - 180)
    rphi2: float = 114.0         # 180 - the second pole's latitude (66 N, on rlampo)
    ra1: float = 1e-5            # functions.f90: the first base function's cubic coefficient
    rb2: float = 1.0             # functions.f90: the second base function's tanh scale
    pivot: str = "F"             # "F": the pole line is a V-F row (this code); "T": a T-U row (the older version)
    jpfac: int = 5               # param.f90: north fine grid / north half-grid
    jpint: int = 10              # param.f90: integration steps / north fine grid
    stretch: Stretch | None = None

    @property
    def rdx(self) -> float:
        return 360.0 / float(self.jpi - 1)

    @property
    def jpeq(self) -> int:
        return self.jpeqt + self.jpeqn

    @property
    def jpj(self) -> int:
        return self.jpeq + self.jpnord - 1


@dataclass
class Coefficients:
    """functions.f90 `set_coefficients`: the real parameters (cf packs them for the Numba kernels)."""
    rjpeqt: float
    rjpeq: float
    rjpnth: float                # the pole line's north index (jpnord + 1/2 for an F pivot, jpnord for a T pivot)
    rjpnthm1: float
    rjpj: float                  # the pole line's global row index
    rdx: float
    rphisud: float
    rphieq: float
    req: float
    ra1: float
    ra2: float
    rb1: float
    rb2: float
    ry0: float
    ry1: float
    ry2: float
    rphifa: float
    ra2_iterations: int
    cf: np.ndarray = field(repr=False, default=None)


# ---- functions.f90: the base functions (scalar, Numba) -------------------------------------------------------
# cf = [rjpeqt, rjpeq, rjpnth, rjpnthm1, rdx, req, ra1, ra2, rb1, rb2, ry0]

@njit(cache=True)
def _fsy0p(pj, cf):
    """functions.f90 FSY0P: Mercator latitude (deg) of row pj, THETA = rdx rad (pj - rjpeqt)."""
    return np.arcsin(np.tanh(cf[4] * RAD * (pj - cf[0]))) / RAD


@njit(cache=True)
def _fsphiy(pphi):
    """functions.f90 FSPHIY: north polar stereographic radius of latitude pphi (deg)."""
    return np.tan(RPI / 4.0 - RAD * pphi / 2.0)


@njit(cache=True)
def _fsfa(pj, cf):
    """functions.f90 FSFA: the first base function's index map (atan saturation with a cubic correction)."""
    u = cf[7] * (pj - 1.0) / cf[3]
    return (np.arctan(u) + u ** 3 / (3.0 * (u ** 3 / (cf[6] * cf[7]) + 1.0))) * cf[3] / cf[7]


@njit(cache=True)
def _fsa(pj, cf):
    """functions.f90 FSA = FSPHIY(FSPHIA), FSPHIA = FSY0P(FSFA + rjpeq): the half-axis along the pole axis."""
    return _fsphiy(_fsy0p(_fsfa(pj, cf) + cf[1], cf))


@njit(cache=True)
def _fsb(pj, cf):
    """functions.f90 FSB = FSPHIY(FSPHIB), FSPHIB = FSFB FSY0P(pj + rjpeq - 1), FSFB = 1 + rb1 tanh(rb2 x^3)."""
    fb = 1.0 + cf[8] * np.tanh(cf[9] * ((pj - 1.0) / cf[3]) ** 3)
    return _fsphiy(fb * _fsy0p(pj + cf[1] - 1.0, cf))


@njit(cache=True)
def _fsy0(pj, cf):
    """functions.f90 FSY0: the ellipse's centre on the pole axis."""
    return (_fsphiy(_fsy0p(pj + cf[1] - 1.0, cf)) - _fsa(pj, cf)) * cf[10]


@njit(cache=True)
def _fsfun(pj, px, py, cf):
    """functions.f90 FSFUN: (A x)^2 + (B (y - Y0))^2 - (A B)^2, zero on ellipse pj."""
    a = _fsa(pj, cf)
    b = _fsb(pj, cf)
    return (a * px) ** 2 + (b * (py - _fsy0(pj, cf))) ** 2 - (a * b) ** 2


@njit(cache=True)
def _fsder(pj, px, py, cf):
    """functions.f90 FSDER: dy/dx of the orthogonal trajectory, (B / A)^2 (y - Y0) / x."""
    a = _fsa(pj, cf)
    b = _fsb(pj, cf)
    return (b / a) ** 2 * (py - _fsy0(pj, cf)) / px


@njit(cache=True)
def _meridian(zi, zj, cf, jph, ncompt, master):
    """nthcar.f90 `nth_car` for one meridian (jc = 0) of the north half-grid at meridian index zi, its points at the
    north indices zj (any, in [1, rjpnth]). Returns lam, phi (rad; lam in [-pi, 0], Fortran's half-grid longitude).

    I. the orthogonal trajectory from the grid equator: x from REQ cos(lam0) to 0 in jph - 1 equal steps; y by a
       Matsuno start then leapfrog with an Asselin filter (GAMMA); at each step the index of the ellipse through the
       point by ncompt bisections of FSFUN. II. at each zj the slope, interpolated linearly in the index between the
       two integration points around it, gives the point of ellipse zj with that normal slope; the last point (zj =
       rjpnth) is on the pole line, x = 0 and y clamped between the two poles. III. the stereographic plane to
       longitude / latitude (the master meridians, on the y axis, by the sign of y)."""
    rjpnth, req = cf[2], cf[5]
    zwj = np.empty(jph)
    zwx = np.empty(jph)
    zwy = np.empty(jph)
    zwp = np.empty(jph)
    zlam0 = RAD * (90.0 - cf[4] * (zi - 1.0))                      # rad * fsx0p(zi)
    zwj[0] = 1.0
    zwj[jph - 1] = rjpnth
    zwx[0] = req * np.cos(zlam0)
    zwy[0] = req * np.sin(zlam0)
    zdx = zwx[0] / float(jph - 1)
    for jh in range(1, jph):
        zwx[jh] = zwx[0] - zdx * float(jh)
    zwy[1] = zwy[0] - zdx * _fsder(zwj[0], zwx[0], zwy[0], cf)
    for jh in range(1, jph - 1):
        zj1 = 1.0
        zj2 = rjpnth
        for _ in range(ncompt):
            zj3 = (zj1 + zj2) * 0.5
            zfs = _fsfun(zj3, zwx[jh], zwy[jh], cf)
            if abs(zfs) < RZERO:
                zj2 = zj3
                zj1 = zj3
            if zfs < 0.0:
                zj1 = zj3
            if zfs > 0.0:
                zj2 = zj3
        zwj[jh] = (zj1 + zj2) * 0.5
        zwy[jh + 1] = zwy[jh - 1] - 2.0 * zdx * _fsder(zwj[jh], zwx[jh], zwy[jh], cf)
        zwy[jh] = zwy[jh] + GAMMA * (zwy[jh - 1] - 2.0 * zwy[jh] + zwy[jh + 1])
    zwp[jph - 1] = 0.0
    for jh in range(jph - 1):
        zwp[jh] = _fsder(zwj[jh], zwx[jh], zwy[jh], cf)
    mono = True
    for jh in range(jph - 1):
        if not zwj[jh + 1] > zwj[jh]:
            mono = False
            break
    yp1 = _fsy0(rjpnth, cf) - _fsa(rjpnth, cf)
    yp2 = _fsy0(rjpnth, cf) + _fsa(rjpnth, cf)
    ypmin, ypmax = min(yp1, yp2), max(yp1, yp2)
    n = zj.shape[0]
    lam = np.empty(n)
    phi = np.empty(n)
    for k in range(n):
        z = zj[k]
        if z >= rjpnth:                                            # the pole line
            x = 0.0
            y = max(min(zwy[jph - 1], ypmax), ypmin)
            last = True
        else:
            last = False
            ij = -1
            if mono:                                               # itab: zwj(ij) <= z < zwj(ij + 1)
                lo, hi = 0, jph - 1
                while hi - lo > 1:
                    mid = (lo + hi) // 2
                    if zwj[mid] <= z:
                        lo = mid
                    else:
                        hi = mid
                if zwj[lo] <= z < zwj[lo + 1]:
                    ij = lo
            else:
                for jh in range(jph - 1):
                    if z >= zwj[jh] and z < zwj[jh + 1]:
                        ij = jh
            if ij < 0:
                lam[k] = np.nan
                phi[k] = np.nan
                continue
            zpente = zwp[ij] + (z - zwj[ij]) * (zwp[ij + 1] - zwp[ij]) / (zwj[ij + 1] - zwj[ij])
            za = _fsa(z, cf)
            zb = _fsb(z, cf)
            ztheta = np.arctan(zpente * za / zb)
            x = zb * np.cos(ztheta)
            y = za * np.sin(ztheta) + _fsy0(z, cf)
        if master:
            lam[k] = 0.0 if y > 0.0 else -RPI
            phi[k] = RPI / 2.0 - 2.0 * np.arctan(abs(y))
        elif last:
            lam[k] = 0.0 if y >= 0.0 else -RPI
            phi[k] = RPI / 2.0 - 2.0 * np.arctan(abs(y))
        else:
            lam[k] = -RPI / 2.0 + np.arctan(y / x)
            phi[k] = RPI / 2.0 - 2.0 * np.arctan(np.sqrt(x * x + y * y))
    return lam, phi


@njit(parallel=True, cache=True)
def _meridians(zis, zj, cf, jph, ncompt, zi_master2):
    """`_meridian` for every zi (parallel over meridians): lam, phi (len(zis), len(zj))."""
    m = zis.shape[0]
    lam = np.empty((m, zj.shape[0]))
    phi = np.empty((m, zj.shape[0]))
    for k in prange(m):
        master = zis[k] == 1.0 or zis[k] == zi_master2
        a, b = _meridian(zis[k], zj, cf, jph, ncompt, master)
        lam[k] = a
        phi[k] = b
    return lam, phi


# ---- functions.f90 set_coefficients, hgrctl.f90 hgr_ctl ------------------------------------------------------

def _pack(c: Coefficients) -> np.ndarray:
    return np.array([c.rjpeqt, c.rjpeq, c.rjpnth, c.rjpnthm1, c.rdx, c.req, c.ra1, c.ra2, c.rb1, c.rb2, c.ry0],
                    np.float64)


def set_coefficients(p: GenParams) -> Coefficients:
    """functions.f90 `set_coefficients`. The pole line is at north index rjpnth = jpnord + 1/2 (F pivot, as in the
    code) or jpnord (T pivot, the older version: module docstring); rjpj = jpj + rjpnth - jpnord. RA2 by the code's
    bisection on [0.1, 10] (stops when |FSPHIA(rjpnth) - RPHIFA| < 1e-14, the bracket narrower than 1e-14 or after
    2000 steps); RB1 from FSB(rjpnth) = 0; RY0 from FSY0(rjpnth) = the middle of the two poles."""
    half = 0.5 if p.pivot == "F" else 0.0
    rjpnth = float(p.jpnord) + half
    c = Coefficients(rjpeqt=float(p.jpeqt), rjpeq=float(p.jpeq), rjpnth=rjpnth, rjpnthm1=rjpnth - 1.0,
                     rjpj=float(p.jpj) + half, rdx=p.rdx, rphisud=0.0, rphieq=0.0, req=0.0, ra1=p.ra1, ra2=0.0,
                     rb1=0.0, rb2=p.rb2, ry0=0.0, ry1=0.0, ry2=0.0, rphifa=0.0, ra2_iterations=0)
    cf = _pack(c)
    c.rphisud = float(_fsy0p(1.0, cf))
    c.rphieq = float(_fsy0p(c.rjpeq, cf))
    c.req = float(np.tan(RPI / 4.0 - RAD * c.rphieq / 2.0))
    c.ry1 = float(_fsphiy(p.rphi1))
    c.ry2 = float(_fsphiy(p.rphi2))
    c.rphifa = 90.0 - 2.0 * np.arctan(0.5 * abs(c.ry1 - c.ry2)) / RAD          # fsyphi
    cf = _pack(c)

    def fsphia(ra2):
        cf[7] = ra2
        return float(_fsy0p(_fsfa(rjpnth, cf) + c.rjpeq, cf))

    zra0, zra1, icompt, zf2 = 1e-1, 10.0, 0, np.inf
    while abs(zf2 - c.rphifa) >= RZERO:
        zf0 = fsphia(zra0)
        fsphia(zra1)
        ra2 = (zra0 + zra1) / 2.0
        zf2 = fsphia(ra2)
        icompt += 1
        if abs(zra0 - zra1) < RZERO or icompt >= 2000:
            break
        if (zf0 - c.rphifa) * (zf2 - c.rphifa) > 0:
            zra0 = ra2
        else:
            zra1 = ra2
    c.ra2, c.ra2_iterations = ra2, icompt
    zy0pjn = float(_fsy0p(c.rjpj, cf))
    rbmax = (90.0 - zy0pjn) / zy0pjn
    c.rb1 = rbmax / np.tanh(p.rb2)
    cf = _pack(c)
    rphify0 = 90.0 - 2.0 * np.arctan(0.5 * (c.ry1 + c.ry2)) / RAD
    c.ry0 = float(_fsphiy(rphify0) / (_fsphiy(zy0pjn) - _fsa(rjpnth, cf)))
    c.cf = _pack(c)
    return c


def hgr_ctl(p: GenParams, c: Coefficients) -> dict:
    """hgrctl.f90 `hgr_ctl`: FSA(1) = FSB(1) = REQ, FSB(rjpnth) = 0, FSY0(1) = 0 (to 1e-14), and at every integer
    north index FSA and FSB decreasing and |dFSY0| < |dFSA| (here by central differences of step 1e-4 instead of the
    code's analytic derivatives, whose FSDFA does not match FSFA: the signs are what the check needs). Raises when a
    check fails, as the code stops."""
    cf = c.cf
    za1, zb1, zy1 = _fsa(1.0, cf), _fsb(1.0, cf), _fsy0(1.0, cf)
    zb2 = _fsb(c.rjpnth, cf)
    bc = max(abs(za1 - c.req), abs(zb1 - c.req), abs(zb2), abs(zy1))
    jj = np.arange(1, p.jpnord + 1, dtype=np.float64)
    e = 1e-4
    lo, hi = np.maximum(jj - e, 1.0), np.minimum(jj + e, c.rjpnth)
    da = np.array([_fsa(b, cf) - _fsa(a, cf) for a, b in zip(lo, hi)])
    db = np.array([_fsb(b, cf) - _fsb(a, cf) for a, b in zip(lo, hi)])
    dy = np.array([_fsy0(b, cf) - _fsy0(a, cf) for a, b in zip(lo, hi)])
    out = {"boundary_max": float(bc), "a_decreasing": bool((da < 0).all()), "b_decreasing": bool((db < 0).all()),
           "no_overlap": bool((np.abs(dy) <= np.abs(da)).all())}
    if bc > 1e-14 or not (out["a_decreasing"] and out["b_decreasing"] and out["no_overlap"]):
        raise ValueError(f"hgr_ctl: the base functions fail the checks: {out}")
    return out


def _ncompt(rjpnth: float) -> int:
    """hgrnth.f90: the bisection count, the first n with 0.5^n (jpjn - 1) < 1e-14, jpjn - 1 = 2 (rjpnth - 1)."""
    span = 2.0 * (rjpnth - 1.0)
    for jn in range(1, 101):
        if 0.5 ** jn * span < RZERO:
            return jn
    return 100


# ---- trop.f90, shoots.f90, mshtrp.f90: the tropical stretch ---------------------------------------------------

def _dphidj(phi, pc, s: Stretch):
    """trop.f90 `dphidj`: R cos(phi) - (R - Rmin) (exp(-(c phi)^n / a^n) + exp(-(c phi)^n / b^n)) / 2."""
    zcphi = (pc * phi) ** s.jpexp
    return s.ppres * np.cos(phi * RAD) - (s.ppres - s.ppresmin) * 0.5 * (np.exp(-zcphi / s.ppa ** s.jpexp)
                                                                           + np.exp(-zcphi / s.ppb ** s.jpexp))


def _integ(hphi, pc, s):
    """trop.f90 `integ`: j(phi) by the trapezium rule of dj/dphi = 1 / dphidj from j = 1 on the equator."""
    zdj = 1.0 / _dphidj(hphi, pc, s)
    steps = np.diff(hphi) * 0.5 * (zdj[:-1] + zdj[1:])
    hj = np.empty_like(hphi)
    hj[0] = 1.0
    acc = 1.0
    for k in range(len(steps)):                    # sequential, as the Fortran loop (the same rounding)
        acc = acc + steps[k]
        hj[k + 1] = acc
    return hj


@dataclass
class TropRows:
    """mshtrp.f90 `msh_trp`'s result: c, the high-resolution j(phi) table it last integrated (hphi, hj), njc (the
    stretched index of the join row, the equator is 1), phic and the join's unstretched row."""
    c: float
    newton_iterations: int
    njc: int
    phic: float
    ijgcanc: int                 # the unstretched T row nearest pphi_join (1-based)
    hphi: np.ndarray = field(repr=False)
    hj: np.ndarray = field(repr=False)

    def lat(self, t: np.ndarray) -> np.ndarray:
        """Latitude (deg) of stretched north index t (1 on the equator, njc on the join; mshtrp.f90: linear
        interpolation of the table between its two points around t, phi(njc) = phic exactly)."""
        t = np.asarray(t, np.float64)
        k = np.searchsorted(self.hj, t, side="right")             # hj(k - 1) <= t < hj(k): the code's first hj > t
        k = np.clip(k, 1, len(self.hj) - 1)
        w = (t - self.hj[k - 1]) / (self.hj[k] - self.hj[k - 1])
        out = (1.0 - w) * self.hphi[k - 1] + w * self.hphi[k]
        out = np.where(t <= 1.0, 0.0, out)
        return np.where(t >= self.njc, self.phic, out)


def trop_rows(p: GenParams, c: Coefficients, lat_col1_t: np.ndarray) -> TropRows:
    """mshtrp.f90 `msh_trp` up to the tropical latitudes: the join row (the unstretched T row of column 1 nearest
    pphi_join, searched north from the equator), the table hphi (nhp = NINT(jptint phic / Rmin) points from 0 to phic),
    c by shoots.f90 `newton` (start 1, derivative by the increment 1e-6 c, stop when |j(phic) - njc| < 1e-6 or the
    correction is below 1e-6 c, at most 100 steps), njc = NINT(j(phic)) at c = 1. As in the code the table kept is the
    last one integrated (at the last derivative step) while c is the corrected one."""
    s = p.stretch
    if np.max(np.abs(lat_col1_t[p.jpeqt - 1])) > RZERO:
        raise ValueError("msh_trp: the equator's latitude is not zero")
    zlat, ijgcanc, zphic = 0.0, None, None
    for jj in range(p.jpeqt, len(lat_col1_t) + 1):
        zlatp, zlat = zlat, float(lat_col1_t[jj - 1])
        if zlat >= s.pphi_join:
            if zlat - s.pphi_join < s.pphi_join - zlatp:
                zphic, ijgcanc = zlat, jj
            else:
                zphic, ijgcanc = zlatp, jj - 1
            break
    if ijgcanc is None:
        raise ValueError("msh_trp: pphi_join not found")
    nhp = int(np.rint(s.jptint * zphic / s.ppresmin))
    hphi = zphic * np.arange(nhp, dtype=np.float64) / float(nhp - 1)
    hj = _integ(hphi, 1.0, s)
    njc = int(np.rint(hj[-1]))
    x, con, it = 1.0, 0, 0
    for it in range(1, 101):
        y = njc - _integ(hphi, x, s)[-1]
        d = 1e-6 * x
        hj = _integ(hphi, x + d, s)
        dy = ((njc - hj[-1]) - y) / d
        xcorr = -y / dy
        x = x + xcorr
        if abs(y) < 1e-6:
            con = 1
            break
        if abs(xcorr / x) < 1e-6:
            con = 2
            break
    if con == 0:
        raise ValueError("msh_trp: Newton did not converge")
    return TropRows(float(x), it, njc, zphic, ijgcanc, hphi, hj)


# ---- the grid --------------------------------------------------------------------------------------------------

@dataclass
class OrcaGrid:
    """A generated ORCA grid in NEMO's layout: lon, lat (deg) of the T, U, V and F points, arrays (rows, columns) with
    the two east-west wrap columns (the last two repeat the first two) and the north fold row (the last row repeats a
    row below it, mirrored), as `coordinates.nc` has them (gridio.f90 `write_glo`)."""
    params: GenParams
    coef: Coefficients
    lon: dict                    # point -> (nj, ni)
    lat: dict
    pivot: str                   # "F" or "T"
    q: int                       # refinement of the generator's grid (1: none)
    trop: TropRows | None
    rows_j: np.ndarray           # (nj,) the continuous unstretched-or-stretched row index J of the T rows
    j_equator: int               # 0-based T row on the equator
    j_pole: int                  # 0-based row of the pole line (F row for an F pivot, T row for a T pivot)
    info: dict = field(default_factory=dict)


def _lon_south(p: GenParams, zi: np.ndarray) -> np.ndarray:
    """mshsth.f90 FSLAM = 90 - FSX0P = rdx (I - 1): the half-grid's longitude convention before msh_nth's flip."""
    return -p.rdx * (np.asarray(zi, np.float64) - 1.0)


def _rotate(lam_deg: np.ndarray, rlampo: float) -> np.ndarray:
    """mshnth.f90: longitude x -1 (increasing eastward), + RLAMPO, MOD(... + 720, 360) - 180."""
    return np.mod(-lam_deg + rlampo + 720.0, 360.0) - 180.0


def north_points(p: GenParams, c: Coefficients, zi: np.ndarray, zj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The north part at meridian indices zi (any real; mshnth.f90's symmetries outside [1, 1 + 180 / rdx]: about
    I = 1, lam -> -lam; about I = 1 + 180 / rdx, lam -> -360 deg - lam) and north indices zj (in [1, rjpnth]; the fold
    row's mirror is the caller's): half-grid lam (deg, before the flip and rotation), phi (deg), (len(zi), len(zj))."""
    zi = np.asarray(zi, np.float64)
    zmax = 1.0 + 180.0 / p.rdx
    period = 2.0 * (zmax - 1.0)
    r = np.mod(zi - 1.0, period)                                   # 0 .. period
    flip = r > (zmax - 1.0)
    base = np.where(flip, period - r, r) + 1.0                     # in [1, zmax]
    ub, inv = np.unique(base, return_inverse=True)
    jph = p.jpint * p.jpfac * int(round(2.0 * (c.rjpnth - 1.0))) + 1
    ncompt = _ncompt(c.rjpnth)
    lam, phi = _meridians(ub, np.asarray(zj, np.float64), c.cf, jph, ncompt, zmax)
    lam, phi = np.degrees(lam[inv]), np.degrees(phi[inv])
    lam = np.where(flip[:, None], -lam, lam)
    # unwrap: the half-grid lam is -rdx (I - 1) near the equator; add the multiple of 360 that brings it there
    lam = lam + 360.0 * np.round((_lon_south(p, zi)[:, None] - lam) / 360.0)
    return lam, phi


def grid(p: GenParams, q: int = 1, i_offset: float | None = None, j_offset: float | None = None,
         check: bool = True) -> OrcaGrid:
    """The ORCA grid of p, NEMO's layout (T, U, V, F; two wrap columns; the fold row(s)).

    Without offsets: the code's own grid (tripole.f90, `_grid_q1`): T at (I, J) = (ji - 1/2, jj), U (ji, jj),
    V (ji - 1/2, jj + 1/2), F (ji, jj + 1/2), ji = 1 .. jpi + 1, jj = 1 .. jpj + 1, then the stretch. With offsets
    (`_grid_lattice`): T at I = i_offset + i / q, J = j_offset + j / q (J in the stretched rows when p has a stretch),
    U, V, F half a fine step east / north; the pivot follows from where the pole line falls."""
    c = set_coefficients(p)
    ctl = hgr_ctl(p, c) if check else {}
    if i_offset is None and j_offset is None and q == 1 and p.pivot == "F":
        return _grid_q1(p, c, ctl)
    return _grid_lattice(p, c, ctl, q, float(i_offset or 0.0), float(j_offset or 0.0))


def _points(p: GenParams, c: Coefficients, zi: np.ndarray, jj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """lon, lat (deg, rotated) of the unstretched map at meridian indices zi (columns) and global row indices jj
    (rows, any real <= rjpj): south (jj <= jpeq) Mercator, north by `north_points`. (len(jj), len(zi))."""
    jj = np.asarray(jj, np.float64)
    zi = np.asarray(zi, np.float64)
    lam = np.empty((len(jj), len(zi)))
    phi = np.empty((len(jj), len(zi)))
    south = jj <= c.rjpeq
    if south.any():
        lam[south] = _lon_south(p, zi)[None, :]
        phi[south] = np.degrees(np.arcsin(np.tanh(c.rdx * RAD * (jj[south] - c.rjpeqt))))[:, None]
    if (~south).any():
        a, b = north_points(p, c, zi, jj[~south] - c.rjpeq + 1.0)
        lam[~south] = a.T
        phi[~south] = b.T
    return _rotate(lam, p.rlampo), phi


def _grid_q1(p: GenParams, c: Coefficients, ctl: dict) -> OrcaGrid:
    """The code's grid: msh_sth + msh_nth (+ msh_trp). Columns ji = 1 .. jpim (jpim - 1, jpim: the wrap columns,
    mshnth.f90 'cyclic east-west'); rows jj = 1 .. jpj (the pole line: F row jpj for an F pivot, T row jpj for a T
    pivot) and the fold row jpjm = jpj + 1 (mshnth.f90 'repliement au nord': T-U from the row below mirrored, V-F from
    two rows below for an F pivot; for a T pivot V-F from the row below, T-U from two below)."""
    jpim = p.jpi + 1
    ji = np.arange(1, jpim + 1, dtype=np.float64)
    jj = np.arange(1, p.jpj + 1, dtype=np.float64)
    zi = {"t": ji - 0.5, "u": ji, "v": ji - 0.5, "f": ji}
    dj = {"t": 0.0, "u": 0.0, "v": 0.5, "f": 0.5}
    lon, lat = {}, {}
    rows = {}
    for k in "tuvf":
        jrow = jj + dj[k]
        if p.pivot == "T" and k in "vf":
            jrow = np.minimum(jrow, c.rjpj)        # V-F of the pivot row: placeholder, replaced by the fold below
        lo, la = _points(p, c, zi[k][:-2], jrow)
        lon[k] = np.concatenate([lo, lo[:, :2] + 360.0], axis=1)
        lon[k] = np.mod(lon[k] + 180.0, 360.0) - 180.0
        lat[k] = np.concatenate([la, la[:, :2]], axis=1)
        rows[k] = jrow
    # the fold: mirror about the column iis = (jpim - 1) / 2 + 1 (the second pole's U-F column)
    iis = (jpim - 1) // 2 + 1
    idx = np.arange(1, jpim + 1)
    ijist = (2 * iis - idx + 1 - 1) % (jpim - 2)                   # 0-based, modulo the 360 / rdx unique columns
    ijisu = (2 * iis - idx - 1) % (jpim - 2)
    for k in "tuvf":
        mir = ijist if k in "tv" else ijisu
        if p.pivot == "F":
            src = -1 if k in "tu" else -2                          # T-U from row jpj, V-F from row jpj - 1
            lon[k] = np.vstack([lon[k], lon[k][src, mir][None]])
            lat[k] = np.vstack([lat[k], lat[k][src, mir][None]])
        else:
            if k in "vf":                                          # V-F row jpj = V-F row jpj - 1 mirrored
                lon[k][-1] = lon[k][-2, mir]
                lat[k][-1] = lat[k][-2, mir]
            src = -2 if k in "tu" else -2
            lon[k] = np.vstack([lon[k], lon[k][src, mir][None]])
            lat[k] = np.vstack([lat[k], lat[k][src, mir][None]])
    trop = None
    j_eq = p.jpeqt - 1
    rows_j = np.r_[jj, p.jpj + 1.0]
    if p.stretch is not None:
        trop = trop_rows(p, c, lat["t"][:, 0])
        lon, lat, j_eq, rows_j = _stretch(p, trop, lon, lat)
    j_pole = len(rows_j) - 2
    info = {"hgr_ctl": ctl, "ra2": c.ra2, "ra2_iterations": c.ra2_iterations, "rb1": c.rb1, "ry0": c.ry0,
            "req": c.req, "rphieq": c.rphieq, "rphisud": c.rphisud}
    if trop is not None:
        info.update(trop_c=trop.c, trop_newton_iterations=trop.newton_iterations, trop_njc=trop.njc,
                    trop_phic=trop.phic, trop_ijgcanc=trop.ijgcanc)
    return OrcaGrid(p, c, lon, lat, p.pivot, 1, trop, rows_j, j_eq, j_pole, info)


def _stretch(p: GenParams, tr: TropRows, lon: dict, lat: dict):
    """mshtrp.f90 `msh_trp` and `filltrop`: the unstretched rows ijgcapr + 1 .. ijgcanc - 1 replaced by 2 (njc - 1) - 1
    stretched T rows (and the V-F rows ijgcapr .. ijgcanc - 1), the T latitudes at stretched index 1 .. njc - 1, the V-F
    at 1.5 .. njc - 1/2, mirrored south; longitudes from row ijscapr."""
    njc = tr.njc
    ijgtrop = tr.ijgcanc - p.jpeqt + 1
    ijgcapr = p.jpeqt - ijgtrop + 1
    ijgnth = lat["t"].shape[0]
    ijsnth = ijgnth - (ijgtrop - 2) * 2 + (njc - 2) * 2
    ijscapr = ijgcapr
    ijseqt = ijscapr + njc - 1
    ijscanc = ijseqt + njc - 1
    tlat = tr.lat(np.arange(1, njc, dtype=np.float64))                  # T: t = 1 .. njc - 1
    vlat = tr.lat(np.arange(1, njc, dtype=np.float64) + 0.5)            # V: t = 1.5 .. njc - 1/2
    out_lon, out_lat = {}, {}
    for k in "tuvf":
        ni = lat[k].shape[1]
        a = np.empty((ijsnth, ni))
        b = np.empty((ijsnth, ni))
        a[:ijscapr] = lon[k][:ijgcapr]
        b[:ijscapr] = lat[k][:ijgcapr]
        a[ijscanc - 1:] = lon[k][tr.ijgcanc - 1:]
        b[ijscanc - 1:] = lat[k][tr.ijgcanc - 1:]
        a[ijscapr:ijscanc - 1] = a[ijscapr - 1][None]
        if k in "tu":
            b[ijseqt - 1:ijscanc - 1] = tlat[:, None]                     # rows ijseqt .. ijscanc - 1
            b[ijscapr:ijseqt - 1] = -tlat[1:][::-1, None]                  # rows ijscapr + 1 .. ijseqt - 1
        else:
            b[ijseqt - 1:ijscanc - 1] = vlat[:, None]
            b[ijscapr - 1:ijseqt - 1] = -vlat[::-1, None]                  # rows ijscapr .. ijseqt - 1
        out_lon[k], out_lat[k] = a, b
    rows_j = np.arange(1, ijsnth + 1, dtype=np.float64)
    return out_lon, out_lat, ijseqt - 1, rows_j


@dataclass
class RowMap:
    """The stretched row index J_s against the generator's rows (mshtrp.f90 `msh_trp`): J_s <= ijscapr: J = J_s;
    J_s >= ijscanc: J = J_s - shift; between them the stretched tropical rows, latitude +- trop.lat(|J_s - ijseqt| + 1).
    Without a stretch J_s = J."""
    trop: TropRows | None
    ijscapr: float = 0.0
    ijseqt: float = 0.0
    ijscanc: float = 0.0
    shift: float = 0.0

    def to_base(self, js: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(J of the generator's rows, mask of the tropical rows) for stretched rows js."""
        js = np.asarray(js, np.float64)
        if self.trop is None:
            return js, np.zeros(js.shape, bool)
        trop = (js > self.ijscapr + 1e-12) & (js < self.ijscanc - 1e-12)
        return np.where(js >= self.ijscanc - 1e-12, js - self.shift, js), trop


def row_map(p: GenParams, c: Coefficients) -> RowMap:
    """The stretch's row map (identity without one)."""
    if p.stretch is None:
        return RowMap(None)
    _, lat1 = _points(p, c, np.array([0.5]), np.arange(1, p.jpj + 1, dtype=np.float64))
    tr = trop_rows(p, c, lat1[:, 0])
    ijgtrop = tr.ijgcanc - p.jpeqt + 1
    ijscapr = p.jpeqt - ijgtrop + 1
    ijseqt = ijscapr + tr.njc - 1
    ijscanc = ijseqt + tr.njc - 1
    return RowMap(tr, float(ijscapr), float(ijseqt), float(ijscanc), float(ijscanc - tr.ijgcanc))


def lattice(p: GenParams, c: Coefficients, rm: RowMap, zi: dict, jrows: dict) -> tuple[dict, dict]:
    """lon, lat of the grid at the meridian indices zi[k] and the (stretched) row indices jrows[k] of each point k.
    Rows north of the pole line J_p are the fold's mirror, P(I, J) = P(2 - I, 2 J_p - J) (mshnth.f90 'repliement au
    nord': the grid is symmetric about the pole axis, so a row beyond the pole line is a row below it seen from the
    other side)."""
    jp = c.rjpj + rm.shift
    lon, lat = {}, {}
    for k in zi:
        js = np.asarray(jrows[k], np.float64)
        zk = np.asarray(zi[k], np.float64)
        beyond = js > jp + 1e-12
        jm = np.where(beyond, 2.0 * jp - js, js)
        jb, trop = rm.to_base(jm)
        lo = np.empty((len(js), len(zk)))
        la = np.empty_like(lo)
        for sel, z in ((~beyond, zk), (beyond, 2.0 - zk)):
            plain = sel & ~trop
            if plain.any():
                lo[plain], la[plain] = _points(p, c, z, jb[plain])
            tsel = sel & trop
            if tsel.any():
                lo[tsel] = _rotate(_lon_south(p, z), p.rlampo)[None, :]
                t = jm[tsel] - rm.ijseqt
                la[tsel] = (np.sign(t) * rm.trop.lat(np.abs(t) + 1.0))[:, None]
        lon[k], lat[k] = lo, la
    return lon, lat


def _grid_lattice(p, c, ctl, q, i_offset, j_offset):
    """The grid on a lattice (a refined one for q > 1, the user guide's 'quadrature'): T at I = i_offset + i / q,
    J_s = j_offset + j / q (i, j = 0, 1, ...), U at I + 1/(2q), V at J_s + 1/(2q), F both; 360 q / rdx columns plus
    the two wrap columns (repeating the first two, the code's layout); rows from j = 0 to one row beyond the pole
    line's row j_p (a T row: T pivot; an F row: F pivot)."""
    rm = row_map(p, c)
    s = (c.rjpj + rm.shift - j_offset) * q                       # the pole line in fine rows
    if abs(s - round(s)) < 1e-9:
        pivot, j_p = "T", int(round(s))
    elif abs(s - 0.5 - round(s - 0.5)) < 1e-9:
        pivot, j_p = "F", int(round(s - 0.5))
    else:
        raise ValueError(f"the pole line J = {c.rjpj + rm.shift} is neither a T nor an F row of the lattice")
    nu = int(round((p.jpi - 1) * q))
    ci = np.arange(nu + 2, dtype=np.float64)
    jr = np.arange(j_p + 2, dtype=np.float64)
    h = 0.5 / q
    zi = {"t": i_offset + ci / q, "u": i_offset + ci / q + h, "v": i_offset + ci / q, "f": i_offset + ci / q + h}
    jj = {"t": j_offset + jr / q, "u": j_offset + jr / q, "v": j_offset + jr / q + h, "f": j_offset + jr / q + h}
    lon, lat = lattice(p, c, rm, zi, jj)
    for k in "tuvf":                                               # the wrap columns repeat the first two exactly
        lon[k][:, -2:] = lon[k][:, :2]
        lat[k][:, -2:] = lat[k][:, :2]
    j_eq = int(round(((rm.ijseqt if rm.trop is not None else c.rjpeqt) - j_offset) * q))
    info = {"hgr_ctl": ctl, "ra2": c.ra2, "ra2_iterations": c.ra2_iterations, "rb1": c.rb1, "ry0": c.ry0,
            "req": c.req, "rphieq": c.rphieq, "q": q, "i_offset": i_offset, "j_offset": j_offset}
    if rm.trop is not None:
        info.update(trop_c=rm.trop.c, trop_newton_iterations=rm.trop.newton_iterations, trop_njc=rm.trop.njc,
                    trop_phic=rm.trop.phic, trop_ijgcanc=rm.trop.ijgcanc)
    return OrcaGrid(p, c, lon, lat, pivot, q, rm.trop, jj["t"], j_eq, j_p, info)
