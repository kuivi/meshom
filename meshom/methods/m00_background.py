"""m00: stub method for the runner — the isotropic background triangle mesh at spacing h (no recombination).

Not a competitor: it checks the runner and gives the triangle-only baseline every method is compared with.
"""
from meshom.delaunay import background_mesh
from meshom.domain import Domain
from meshom.mesh import Mesh
from meshom.methods import register
from meshom.size import SizeField


@register()
def generate(domain: Domain, h: SizeField, opts: dict) -> Mesh:
    return background_mesh(domain, h, factor=1.0)
