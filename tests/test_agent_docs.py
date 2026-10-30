"""The guide for AI coding assistants (AGENTS.md and its topic files) stays tied to the code: front matter and index,
repository paths and Markdown links (with their anchors), example configs, config sections and keys, method names,
module names and decision ids it cites all exist."""
import importlib
import importlib.util
import re
import tomllib
from pathlib import Path

import pytest

from meshom import config

ROOT = Path(__file__).resolve().parents[1]
ENTRY = ROOT / "AGENTS.md"
GUIDE = ROOT / "docs" / "agents"
TOPICS = sorted(GUIDE.glob("*.md"))
DOCS = [ENTRY, *TOPICS]

SECTIONS = {"[mesh]": config.MeshCfg, "[coast]": config.CoastCfg, "[size]": config.SizeCfg,
            "[[straits]]": config.StraitCfg, "[[strait_regions]]": config.StraitRegionCfg,
            "[[blocks]]": config.BlockCfg, "[[inserts]]": config.InsertCfg, "[bathymetry]": config.BathymetryCfg,
            "[[bathymetry.regions]]": config.DepthRegionCfg, "[run]": config.RunCfg}
REP_OPTS = {"repair", "coast_fix", "freeze_blocks"}     # options the `_rep` variants take (methods/rep_variants.py)
# `[size] from` is read by config._size_tables, not a SizeCfg field
NOT_EXAMPLES = {"config.resolved.toml", "pyproject.toml", "FILE.toml", "CONFIG.toml", "h.toml"}
KEY_LINE = re.compile(r"^- `(\[\[?[\w.]+\]\]?)`(?: `(\w+)`)?: (.+)$")
REPO_PATH = re.compile(r"(?<![\w./-])((?:docs|meshom|scripts|tests|ui)/[\w./-]*)")
LINK = re.compile(r"\]\(([^)\s]+)\)")
MODULE = re.compile(r"(?<![\w./])meshom(?:\.[A-Za-z_]\w*)+")
METHOD = re.compile(r"(?<![\w/])(m\d\d_[a-z][a-z0-9_]*[a-z0-9]|ref_jigsaw(?:_[a-z]+)?)(?![\w*])")
DECISION = re.compile(r"\b([BCGMORSZ]\d{1,3})\b")
PROSE_KEY = re.compile(r"`(\[\[?[\w.]+\]\]?) (\w+)`")


def text(p: Path) -> str:
    return p.read_text(encoding="utf-8")


def front_matter(p: Path) -> dict:
    m = re.match(r"---\n(.*?)\n---\n", text(p), re.S)
    assert m, f"{p.name}: no front matter"
    return dict(line.split(": ", 1) for line in m.group(1).splitlines())


def slugs(p: Path) -> set:
    """GitHub's heading anchors of a Markdown file (fenced code skipped)."""
    out, fence = set(), False
    for line in text(p).splitlines():
        if line.startswith("```"):
            fence = not fence
        elif not fence and (m := re.match(r"#+\s+(.*)", line)):
            out.add(re.sub(r"[^\w\- ]", "", m.group(1).strip().lower()).replace(" ", "-"))
    return out


def key_lines(p: Path) -> list:
    body = text(p).split("## Config keys used", 1)[1].split("\n## ", 1)[0]
    return [m.groups() for line in body.splitlines() if (m := KEY_LINE.match(line))]


def allowed_keys(section: str, kind: str | None) -> set:
    from dataclasses import fields

    from meshom.methods import METHODS, defaults
    from meshom.optimize import DEFAULTS as OPT

    if section == "[gates]":
        return set(config.DEFAULT_GATES)
    if section == "[[size.layers]]":
        return {f.name for f in fields(config.LAYERS[kind])}
    if section == "[mesh.opts]":
        assert kind in METHODS, kind
        if kind.endswith("_rep") and kind[:-4] in METHODS:
            return set(defaults(kind[:-4])) | REP_OPTS
        return set(defaults(kind))
    if section == "[mesh.optimize_opts]":
        return set(OPT)
    return {f.name for f in fields(SECTIONS[section])} | ({"from"} if section == "[size]" else set())


def test_topics_have_front_matter():
    assert len(TOPICS) >= 10
    for p in TOPICS:
        fm = front_matter(p)
        assert fm.get("name") == f"meshom-{p.stem}", p.name
        assert len(fm.get("description", "")) > 40, p.name


def test_index_lists_every_topic_and_only_them():
    index = text(ENTRY).split("## Index", 1)[1]
    listed = set(re.findall(r"\(docs/agents/([\w-]+)\.md\)", index))
    assert listed == {p.stem for p in TOPICS}


def test_repository_paths_and_links_exist():
    bad = []
    for p in DOCS:
        t = text(p)
        for m in REPO_PATH.finditer(t):
            tok = m.group(1).rstrip(".,;:")
            if t[m.end():m.end() + 1] in ("*", "<"):
                tok = tok[:tok.rfind("/") + 1]
            if not (ROOT / tok).exists():
                bad.append(f"{p.name}: {tok}")
        for target in LINK.findall(t):
            if target.startswith(("http:", "https:", "mailto:")):
                continue
            path, _, anchor = target.partition("#")
            dest = (p.parent / path).resolve() if path else p
            if not dest.exists():
                bad.append(f"{p.name}: link {target}")
            elif anchor and dest.suffix == ".md" and anchor not in slugs(dest):
                bad.append(f"{p.name}: anchor {target}")
    assert not bad, bad


def test_named_example_configs_parse():
    names = {n for p in DOCS for n in re.findall(r"(?<![\w*.-])([\w.-]+\.toml)\b", text(p))} - NOT_EXAMPLES
    assert names
    for n in sorted(names):
        f = ROOT / "docs" / "examples" / n
        assert f.is_file(), n
        d = tomllib.loads(text(f))
        if "mesh" in d:
            config.load(f)
        else:
            assert d.get("inserts"), n
            for i, x in enumerate(d["inserts"]):
                config._build(config.InsertCfg, x, f"{n}: inserts[{i}]")


def test_config_keys_exist():
    bad = []
    for p in TOPICS:
        assert "## Config keys used" in text(p), p.name
        lines = key_lines(p)
        assert lines, p.name
        for section, kind, rest in lines:
            keys = re.findall(r"`(\w+)`", rest)
            assert keys, (p.name, section)
            bad += [f"{p.name}: {section} {kind or ''} {k}" for k in keys if k not in allowed_keys(section, kind)]
        for section, key in PROSE_KEY.findall(text(p)):
            if section in SECTIONS or section == "[gates]":
                if key not in allowed_keys(section, None):
                    bad.append(f"{p.name}: {section} {key}")
    assert not bad, bad


def test_method_names_are_registered():
    from meshom.methods import METHODS

    named = {m for p in DOCS for m in METHOD.findall(text(p))}
    assert {"m14_blocks_rep", "m15_tri", "m16_orca"} <= named
    assert named <= set(METHODS), sorted(named - set(METHODS))


def _resolves(dotted: str) -> bool:
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        name = ".".join(parts[:i])
        try:
            if importlib.util.find_spec(name) is None:
                continue
        except ModuleNotFoundError:
            continue
        obj = importlib.import_module(name)
        for a in parts[i:]:
            if not hasattr(obj, a):
                return False
            obj = getattr(obj, a)
        return True
    return False


def test_module_names_resolve():
    named = {m for p in DOCS for m in MODULE.findall(text(p))}
    assert named
    bad = sorted(n for n in named if not _resolves(n))
    assert not bad, bad


@pytest.fixture(scope="module")
def decision_ids() -> set:
    return set(re.findall(r"^\| ([A-Z]\d+[a-z]?) \|", text(ROOT / "docs" / "decisions.md"), re.M))


def test_decision_ids_exist(decision_ids):
    bad = sorted({f"{p.name}: {d}" for p in DOCS for d in DECISION.findall(text(p)) if d not in decision_ids})
    assert not bad, bad
