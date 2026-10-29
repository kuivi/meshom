import os
from pathlib import Path

import pytest


def _data_dir(env: str, default: str) -> Path:
    p = Path(os.environ.get(env, default)).expanduser()
    if not p.is_dir():
        pytest.skip(f"{env} not found: {p}")
    return p


@pytest.fixture
def medi_dir() -> Path:
    return _data_dir("MESHOM_MEDI_DIR", "~/work/projects/medi/mesh")


@pytest.fixture
def fesom2_dir() -> Path:
    return _data_dir("MESHOM_FESOM2_DIR", "~/work/projects/fesom2/fesom2")
