"""Run competitor B on a compute node; score and save each completed case."""
import argparse
import json
from pathlib import Path

from meshom.compete import _save
from meshom.meshfix.cases import CASE_SPECS, load, meshfix_dir
from meshom.meshfix.score import score_file, table
from . import fix


def main():
    ap = argparse.ArgumentParser(__doc__)
    ap.add_argument("cases", nargs="+", choices=sorted(CASE_SPECS))
    ap.add_argument("--out", type=Path, default=meshfix_dir() / "results" / "b_transport")
    ap.add_argument("--opts", default="{}")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for name in a.cases:
        print("Loading", name, flush=True)
        case = load(name)
        mesh, record = fix(case, json.loads(a.opts))
        p = a.out / f"{name}.npz"
        _save(mesh, p)
        p.with_suffix(".json").write_text(json.dumps(record, indent=1))
        res = score_file(name, p, case=case)
        print(table([res]), flush=True)


if __name__ == "__main__":
    main()
