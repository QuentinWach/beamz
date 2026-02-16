"""Optional parametric material placeholders.

Graphene is not part of the curated runtime material library, but this module keeps
an explicit placeholder class for future extension.
"""

from __future__ import annotations

from dataclasses import dataclass

from beamz.design.materials import DrudeMaterial, Material2D, PoleResidueMaterial


@dataclass
class Graphene:
    """Simplified parametric graphene placeholder."""

    mu_c: float = 0.0
    temp: float = 300.0
    gamma: float = 0.00041
    scaling: float = 1.0
    include_interband: bool = True

    @property
    def medium(self) -> Material2D:
        intraband = DrudeMaterial(
            coeffs=((1e14 * max(self.scaling, 1e-9), 1e12),),
            eps_inf=1.0,
            name="Graphene_intraband",
        )
        pr = intraband_to_pole_residue(intraband)
        return Material2D(ss=pr, tt=pr, name="Graphene")


GrapheneClass = Graphene


def intraband_to_pole_residue(intraband: DrudeMaterial) -> PoleResidueMaterial:
    if not intraband.coeffs:
        return PoleResidueMaterial(eps_inf=intraband.eps_inf)

    fp, delta = intraband.coeffs[0]
    omega_p = 2.0 * 3.141592653589793 * fp
    gamma = 2.0 * 3.141592653589793 * delta
    c = (omega_p**2) / (2.0 * max(gamma, 1e-30))
    return PoleResidueMaterial(eps_inf=intraband.eps_inf, poles=((-gamma + 0j, c + 0j),))


__all__ = ["Graphene", "GrapheneClass"]
