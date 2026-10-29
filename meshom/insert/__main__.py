"""python -m meshom.insert --run RUN_DIR (--insert SPEC | --config FILE.toml) [--name N] [--out DIR] [--mesh NAME]

SPEC: fields of an [[inserts]] entry (meshom.config.InsertCfg) as 'key=value; key=value' with TOML values, e.g.
'strait=Gibraltar; cells=10' or 'name=denmark; path=[[-31.0, 64.5], [-22.0, 67.5]]; cells=12'. --config takes every
[[inserts]] entry of a TOML file (a run config or a file of [[inserts]] alone). Output in --out (default the run
folder): insert_<name>.npz / .json / .score.json / .map.png / .profile.png and fesom-insert_<name>/ (decisions
B158-B163). Exit code 0 when every insert was built, 2 when one was refused (the reason in its .json and the log).
"""
import argparse
import sys
import tomllib
from pathlib import Path


def parse_inline(text: str) -> dict:
    out = {}
    for part in [p.strip() for p in text.split(";") if p.strip()]:
        k, v = part.split("=", 1)
        k, v = k.strip(), v.strip()
        try:
            out[k] = tomllib.loads(f"x = {v}")["x"]
        except tomllib.TOMLDecodeError:
            out[k] = v
    return out


def entries(args) -> list:
    from meshom.config import InsertCfg, _build

    if args.insert:
        d = parse_inline(args.insert)
        if args.name:
            d["name"] = args.name
        return [_build(InsertCfg, d, "insert")]
    with open(args.config, "rb") as f:
        t = tomllib.load(f)
    return [_build(InsertCfg, x, f"inserts[{i}]") for i, x in enumerate(t.get("inserts", []))]


def main(argv=None) -> int:
    from meshom.config import load as load_cfg
    from meshom.insert.core import run, spec_from

    ap = argparse.ArgumentParser(prog="python -m meshom.insert", description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--insert")
    g.add_argument("--config", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--mesh")
    ap.add_argument("--no-write", action="store_true", help="no model files")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.run / "config.resolved.toml")
    code = 0
    for e in entries(a):
        rec = run(a.run, spec_from(e, cfg), a.out, a.mesh, write=not a.no_write)
        if rec.get("status") != "ok":
            code = 2
    return code


if __name__ == "__main__":
    sys.exit(main())
