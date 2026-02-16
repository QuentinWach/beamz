"""Parametric material placeholders with names."""

from __future__ import annotations

from dataclasses import dataclass

from beamz.components.medium import Drude, Medium2D


@dataclass
class Graphene:
    """Simplified parametric graphene model."""

    mu_c: float = 0.0
    temp: float = 300.0
    gamma: float = 0.00041
    scaling: float = 1.0
    include_interband: bool = True

    @property
    def medium(self) -> Medium2D:
        # Lightweight compatibility model: intraband Drude-like conductivity only.
        # Parameters are placeholders to provide a stable interface.
        intraband = Drude(coeffs=((1e14 * max(self.scaling, 1e-9), 1e12),), eps_inf=1.0)
        pr = intraband_to_pole_residue(intraband)
        return Medium2D(ss=pr, tt=pr, name="Graphene")


# Kept for compatibility with API requests mentioning `GrapheneClass`.
GrapheneClass = Graphene


def intraband_to_pole_residue(intraband: Drude):
    from beamz.components.medium import PoleResidue

    # Simple one-pole surrogate from the first Drude term.
    if not intraband.coeffs:
        return PoleResidue(eps_inf=intraband.eps_inf)
    fp, delta = intraband.coeffs[0]
    omega_p = 2.0 * 3.141592653589793 * fp
    gamma = 2.0 * 3.141592653589793 * delta
    c = (omega_p**2) / (2.0 * max(gamma, 1e-30))
    return PoleResidue(eps_inf=intraband.eps_inf, poles=((-gamma + 0j, c + 0j),))


__all__ = ["Graphene", "GrapheneClass"]
