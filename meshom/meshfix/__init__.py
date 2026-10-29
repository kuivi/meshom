"""Mesh fixing: improve a finished meshom mesh everywhere (the competition bench).

Contract for a competing method (A, B, C, D, ...):

1. Input: a case, `meshom.meshfix.cases.load(NAME)` -> Case, NAME in cases.CASE_SPECS (v4, v4_pacific, v4_atlantic,
   v5, core2). Case.mesh (meshom.mesh.Mesh), Case.h (the run's size field: h(xyz (M,3) unit vectors) -> (M,)
   metres; call it for new nodes), Case.h_node, Case.kind (0 interior, 1 coast, 2 artificial boundary),
   Case.fixed (= kind 2: the cut-out's artificial boundary, which every method keeps exactly in place, edges
   included; it is no sink for defects), Case.tol (the size tolerance), Case.gates, Case.dist_coast_km(xyz),
   Case.is_open(xyz). Every other node may move, coast nodes included (coast fidelity is scored), and nodes and
   elements may be added or removed. node_flag of the mesh is meshom's boundary flag (1 on boundary nodes), not a
   lock.

2. A method is a function  fix(case: Case, opts: dict | None = None) -> (Mesh, record: dict).  The record holds at
   least {"method": NAME, "case": case.name, "wall_s": seconds of the method alone}; anything else is free.

3. Save to <MESHFIX_DIR>/results/<method>/<case>.npz (meshom.compete._save, the Mesh npz format) and the record to
   <case>.json next to it. MESHFIX_DIR = <main checkout>/.tmp/meshfix (cases.meshfix_dir(); $MESHFIX_DIR
   overrides), shared by every worktree.

4. Score:  python -m meshom.meshfix.score --case NAME --mesh <MESHFIX_DIR>/results/<method>/<case>.npz
   (or meshom.meshfix.score.score_file(NAME, path)) -> <case>.score.json and the maps <case>.<window>.png next to
   the mesh. The baselines (input, `meshom optimize` defaults, size_fit.smooth_fit on all interior nodes) are in
   <MESHFIX_DIR>/baselines/<method>/<case>.*, their table in <MESHFIX_DIR>/baselines/table.md
   (meshom.meshfix.baselines).
"""
