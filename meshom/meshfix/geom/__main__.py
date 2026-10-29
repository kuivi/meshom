import argparse
import json
import sys

from meshom.meshfix.geom import _parse, run


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m meshom.meshfix.geom")
    ap.add_argument("--case", nargs="+", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--no-score", action="store_true")
    a = ap.parse_args(argv)
    opts = _parse(a.set)
    for c in a.case:
        r = run(c, opts, a.tag, score=not a.no_score)
        if not a.no_score:
            keep = {k: r[k] for k in ("quad_angle", "triangles", "size", "coast", "gates") if k in r}
            keep["validity_ok"] = r["validity"]["ok"]
            keep["defect_units_open"] = r["irregular"]["defect_units_open"]
            print(json.dumps(keep, default=float), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
