# m10: patch layout (separatrices) → MILP quantisation → Coons fill

Code: `meshom/methods/m10_patch_layout.py`, kernels `meshom/methods/m10/trace.py` (Numba: motorcycle tracing,
separatrix angles), `meshom/methods/m10/layout.py` (planar graph, faces, MILP, fill); test
`tests/test_m10_patch_layout.py`. Results: the competition runs (`meshom compete`), scales 4 and 1, raw and `--smooth`.

- **What.** QuadWild-style (Pietroni 2021) on the sphere: `cross_field` ("cones") on `background_mesh(h, 1.0)`; ± cone
  pairs closer than 1 h are cancelled; each cone of index k sends 4(1−k) separatrices (directions from a phase fit of
  the field on a circle of 1 h, first 1 h straight), coast corners with |A − 180°| > 45° send round(A/90°) − 1. All traces advance
  together (motorcycle graph, steps of 0.25 h, RK2 in the field) and stop at a hit (T-junction), a head-on meeting
  within 0.75 h (joined), or the coast (snapped to a coast node, last 2 h bent onto it). The layout is a planar graph
  (rotation system; Euler = 2 on every run); a face is a patch if its corners (quarter turns from the field
  directions; sector count at cones) are exactly 4 convex. Counts per arc: one MILP (scipy/HiGHS, not Bi-MDF):
  opposite patch sides equal (soft), cost |x − ℓ|/ℓ plus a convex 20× penalty beyond 0.25 ℓ + 1, ℓ = ∫ds/h scaled by
  σ_f = √(area/h² ÷ ℓ_u ℓ_v) per patch (a curved patch needs more cells than its sides say: bias +0.15 → +0.02);
  patch sides may be 0 (collapse, every trace path ≥ 1). Fill: nodes on arcs at equal ∫ds/h, a Coons grid per patch,
  zero arcs and collapsed patches merged by union-find. Faces that are not patches, patches with opposite sides off by
  > 30 % (and > 2 cells) or a convex corner outside 55–125°, and patches whose Coons grid has a quad angle > 145° go
  to the fallback: m02's L∞ packing grown from their boundaries, local sphere Delaunay, `recombine(1/3)`. Then 20
  `smooth` "size" sweeps (coast fixed).
- **Simplified:** isotropic cones only (no anisotropic or coast-aligned cone placement, no cone moving); no
  separatrix–separatrix crossing (motorcycle T-junctions instead of QuadWild's full tracing and patch selection);
  MILP instead of Bi-MDF; non-patch faces filled by m02 packing instead of being re-split.
- **Scale 1, raw** (uniform / disc / band / continent): 50.8k / 207k / 415k / 187k nodes; triangles 0 / 0.13 / 0.75 /
  0.93 %; irregular interior nodes 0.016 / 0.16 / 0.85 / 1.17 %; n3/n5/T 8/0/0, 32/283/280, 222/3250/3138,
  295/1801/1746 vs ρ+/ρ− 8/0, 26/−18, 94/−86, 13/−5.5; quad θmax max 121.9 / 144.2 / 148.7 / 149.4°, p99.9 116.9 /
  118.6 / 131.2 / 133.1° (continent coast 140.7 / 136.8°); elem ratio p99 1.01 / 1.06 / 1.10 / 1.10; |ln s/h| p99
  0.20 / 0.28 / 0.33 / 0.40. `--smooth`: quad max 121 / 131 / 141 / 145°, p99.9 116 / 118 / 127 / 126°. Gates 12/14,
  16/18, 14/18 (16/18 smoothed), 21/26 (22/26 smoothed). Time 2.6 / 16 / 39 / 16 s (band: bg 10.5 s, field 7.6 s,
  MILP 4.2 s, trace + layout 0.4 s), peak RSS ≤ 0.9 GB.
- **Scale 4:** uniform = the cube (9 patches, 8 cones = the only irregular nodes, all quads, 121.8°); disc 34 cones,
  187 arcs, 71/73 patches kept, 21/17/6 irregular (≈ the cones), 135°; band 126 cones, 684 arcs, 198/229 patches,
  64/405/402, 138°; continent 18 cones, 143 arcs, 35/54 patches, 18/157/169, 135°.
- **The h gradient is the failure mode.** At scale 4 most patches survive; at scale 1 the field has more cones and the
  traces run far across the size gradient, so opposite sides of a patch disagree in cells (ℓ in units of h differs by
  > 30 %): 31/108 (disc), 195/476 (band), 34/70 (continent) patches are demoted to the packing fallback, and the
  irregular count is then the fallback's (m02-like, n5 ≈ T). Long thin strips between near-parallel separatrices
  shear (opposite sides sized independently, T-junctions at different places): 145° check. Size fidelity is worse
  than m02/m03 (0.28–0.40 vs 0.17–0.31) because the MILP must trade ℓ against matching opposite sides.
- **Production would need:** a layout made in the metric of h (traces and cone placement that keep the gradient
  across patches, e.g. graded patch splits / transition patches with valence-3/5 pairs, as in Jezdimirovic 2021),
  Bi-MDF with the right cost, patch splitting instead of fallback, joint node placement on opposite sides (or a
  parametric smoothing per patch), and a cheaper background/field (2/3 of the time).
