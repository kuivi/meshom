# m05: quasi-structured quad meshing: m02 + cavity remeshing with quad patterns

Code: `meshom/methods/m05_quasi_structured.py` (variants `m05_quasi_structured` = strict acceptance,
`m05_quasi_structured_loose` = tol_angle 45°, tol_size 0.4), `meshom/methods/m05/` (`patterns.py` patterns and chord
matching, `cavity.py` Numba cavity growth, `remesh.py` fill/placement/acceptance/passes); test
`tests/test_m05_quasi_structured.py`. Results: the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`.

- **What.** Reberol 2021 §6 (Gmsh algo 11) on m02's mixed mesh, adapted to mixed output. Seeds = irregular interior
  nodes and triangles; a cavity (disk of quads and triangles, never touching the coast row) grows from the seed,
  is convexified and grown along its longest side (§6.2.1, ≤ 300 elements); its boundary sides are matched to a
  pattern by the chord system W s = N, least Σs² (§6.1.1, exact rational elimination, memoized). Patterns: regular
  quad, val-3 triangle, val-5 pentagon, and mixed ones for odd boundaries (a triangle with three quads around it, a
  pentagon with one triangle), plus 3-5 transition patterns. No midpoint split at all: odd loops are filled by the
  mixed patterns, so h never halves. Fill placed by an h-weighted Tutte map in the gnomonic plane, then local
  `smooth` "size" on cavity + 2 rings. Accepted only if the irregularity Σ(deg−4)² + #tri falls, nothing inverts,
  and the worst corner and worst |ln s/h| of the region stay within tolerance of the old ones. Passes as in §6.2.2
  (regular maximal, singular minimal, regular, transitions), 3 rounds, failed-seed regulator. Final `smooth` 30
  "size" + 3 "angle" with the coastal element row frozen (m02 already smoothed it).
- **Scale 1, raw, m02 → strict / loose** (n3/n5/T vs ρ+/ρ−; tri %; quad θmax max/p99.9; elem ratio p99; |ln s/h| p99):
  uniform 24/418/406 → 44/286/250 / 79/137/66 (ρ 8/0); 0.8 → 0.5 / 0.1 %; 138.3/119.8 → 135.0/122.6 / 129.9/124.8°;
  1.08 → 1.09 / 1.12; 0.173 → 0.168 / 0.210. Disc 80/1304/1264 → 107/781/684 / 201/365/172 (ρ 26/−18); 0.6 → 0.3 /
  0.1 %; 137.4/118.5 → 131.6/119.5 / 130.6/123.7°; 1.09 → 1.09 / 1.11; 0.206 → 0.193 / 0.202. Band 452/5176/4842 →
  496/2202/1726 / 605/1185/588 (ρ 94/−86); 1.1 → 0.4 / 0.1 %; 142.0/125.5 → 136.1/124.0 / 135.5/125.4°; 1.12 → 1.13 /
  1.14; 0.266 → 0.270 / 0.278. Continent 222/2098/1914 → 244/1002/766 / 323/557/234 (ρ 13/−5.5); 1.0 → 0.4 / 0.1 %;
  138.0/124.4 → 135.3/124.5 / 135.3/125.2°, coast 104.4° unchanged; 1.10 → 1.11 / 1.13; 0.253 → 0.249 / 0.263.
- **What works.** Dislocation pairs (triangle + val-5) go by the hundreds: triangles 1.6–2.8× (strict) / 6–8×
  (loose) fewer, worst quad angle 3–8° lower (p99.9 within ±3°), no inverted element, Euler right, coast row as in
  m02, gates as m02 at scale 1. Seam edges leave the worst 1 % (uniform loose: 76 % of the worst-1 % edges sit at a
  defect, 22 % one hop away; m02: 48 % at a defect). Runner `--smooth` (boundary fixed) lifts m05 coast quads to
  115–119° (m02's stay at 104°) and costs loose 2 gates on band and continent; scale-4 continent fails elem ratio p99.
- **What fails.** T stays 6–18× (loose) / 18–60× (strict) the net ρ_c; |ln s/h| p99 ≤ 0.15 is not reached. ~88 % of
  tries (continent, scale 4) are rejected on geometry: a regular patch over a dislocation must stretch rows, and m02's
  dislocations sit where a row has to end to keep s = h (the 8-singularity cube sphere m12 has p99 0.25). Fewer
  defects = worse size fidelity (loose: +0.03–0.04 on p99). Scale 4 (g = 0.4) misleads: a patch spans a whole
  transition there and size p99 rises 0.04–0.07.
- **Time.** Cavity stage at scale 1 (runner total − m02): uniform 21 / 6 s, disc 71 / 18 s, band 146 / 61 s,
  continent 69 / 32 s (strict / loose); m02 itself 2–23 s. Time goes to per-cavity Python work: submesh `smooth`
  calls, quality checks and the reverts of rejected tries; Numba threads set to 1 (prange overhead on 100-node
  submeshes). Peak memory ≤ 1.4 GB.
- **Production needs.** A local optimizer that moves rows rather than nodes (Winslow / untangler on cavity + rings)
  so regular patches pass the size test; pattern choice aware of the size field (let singularities stay where h
  bends rows); cavities that may include the coast row (coast-conforming patterns); the cavity loop in Numba and
  independent cavities in parallel. As is: a cleanup pass (2–8× fewer triangles), not a route to ρ_c.
