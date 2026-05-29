"""Field storage and update logic for FDTD simulations."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from beamz.shared_kernels import (
    advance_e_from_curl,
    advance_h_from_curl,
    build_cpml_3d_terms,
    build_tm_xy_cpml_terms,
)
from beamz.simulation import ops
from beamz.simulation.boundaries import (
    build_h_boundary_views_for_e_3d,
    cpml_curl_e_to_h_3d,
    cpml_curl_h_to_e_3d,
    full_pec_curl_e_to_h_2d_xy,
    full_pec_curl_h_to_e_2d_xy,
    full_pec_e_update_coefficients_3d,
    full_pec_h_update_coefficients_3d,
    full_pec_update_e_from_h_3d,
    full_pec_update_h_from_e_3d,
    has_full_pec_2d_xy,
    has_full_pec_3d,
    initialize_full_pec_3d_state,
    initialize_tm_2d_xy_state,
    resolve_metallic_edges,
    sync_compact_fields_from_full_pec_3d,
    tm_xy_cpml_curl_e_to_h_2d,
    tm_xy_cpml_curl_h_to_e_2d,
    tm_xy_curl_e_to_h_2d,
    tm_xy_curl_h_to_e_2d,
    xy_te_curl_e_to_h_2d,
    xy_te_curl_h_to_e_2d,
)
from beamz.simulation.yee import (
    sample_voxel_grid_at_component_2d,
    sample_voxel_grid_at_component_3d,
    sample_voxel_grid_at_tm_xy_full_component_2d,
)


def _component_source_tuple(source, components, targets):
    if not source:
        return (None, None, None)
    arrays = [jnp.zeros_like(target) for target in targets]
    for i, comp in enumerate(components):
        for val, indices in source.get(comp, ()):
            arrays[i] = arrays[i].at[indices].add(val)
    return tuple(arrays)


@dataclass
class CpmlTm2DxyState:
    psi_h_terms: jnp.ndarray
    psi_e_terms: jnp.ndarray
    sigma_h_terms: jnp.ndarray
    kappa_h_aux_terms: jnp.ndarray
    alpha_h_terms: jnp.ndarray
    kappa_h_direct_terms: jnp.ndarray
    sigma_e_terms: jnp.ndarray
    kappa_e_terms: jnp.ndarray
    alpha_e_terms: jnp.ndarray


@dataclass
class Cpml3DState:
    psi_h_terms: tuple[jnp.ndarray, ...]
    psi_e_terms: tuple[jnp.ndarray, ...]
    a_h_terms: tuple[jnp.ndarray, ...]
    b_h_terms: tuple[jnp.ndarray, ...]
    inv_kappa_h_terms: tuple[jnp.ndarray, ...]
    a_e_terms: tuple[jnp.ndarray, ...]
    b_e_terms: tuple[jnp.ndarray, ...]
    inv_kappa_e_terms: tuple[jnp.ndarray, ...]


class Fields:
    """Container for E/H field arrays on staggered Yee grid with FDTD update logic."""

    def __init__(
        self,
        permittivity,
        conductivity,
        permeability,
        resolution,
        plane_2d="xy",
        _init_materials=True,
    ):
        """Initialize field arrays on a Yee grid for 2D (all 6 components) or 3D (Ex, Ey, Ez, Hx, Hy, Hz) simulations."""
        self.resolution = resolution
        self.plane_2d = plane_2d
        # Store references to material grids owned by Design (convert to JAX arrays)
        self.permittivity = jnp.asarray(permittivity)
        self.conductivity = jnp.asarray(conductivity)
        self.permeability = jnp.asarray(permeability)
        self.metallic_masks = None
        self.full_pec_3d_state = None
        self.tm_xy_state = None
        self.cpml_tm_xy_state = None
        self.cpml_3d_state = None

        self.has_pml = False
        self.has_cpml = False

        # Infer dimensionality and shape from material arrays
        is_3d = self.permittivity.ndim == 3
        grid_shape = self.permittivity.shape

        if is_3d:
            nz, ny, nx = grid_shape
            self._init_fields_3d(nx, ny, nz)
        else:
            dim1, dim2 = grid_shape
            self._init_fields_2d(dim1, dim2)

        if _init_materials:
            self._init_material_parameters()

    def set_pml_conductivity(self, pml_data):
        """Set effective conductivity for PML regions."""
        self.has_pml = True
        self.has_cpml = str(pml_data.get("formulation", "sponge")).lower() == "cpml"
        # Convert PML data arrays to JAX
        self.pml_data = {
            k: jnp.asarray(v) if hasattr(v, "__array__") else v
            for k, v in pml_data.items()
        }
        self.cpml_tm_xy_state = None
        self.cpml_3d_state = None
        self._init_material_parameters()

    def _init_material_parameters(self):
        """Initialize material parameters including PML conductivity if present."""
        is_3d = self.permittivity.ndim == 3
        base_sigma = self.conductivity

        if self.has_pml and hasattr(self, "pml_data"):
            sigma_pml = jnp.zeros_like(base_sigma)
            if is_3d:
                pml_keys = ("sigma_x", "sigma_y", "sigma_z")
            else:
                pml_keys = {
                    "xy": ("sigma_x", "sigma_y"),
                    "yz": ("sigma_y", "sigma_z"),
                    "xz": ("sigma_x", "sigma_z"),
                }.get(self.plane_2d, ())
            for key in pml_keys:
                if key in self.pml_data:
                    sigma_pml = sigma_pml + self.pml_data[key]
            total_sigma = jnp.maximum(base_sigma, sigma_pml)
        else:
            total_sigma = base_sigma
        self.total_conductivity = total_sigma

        if is_3d:
            for comp in ("x", "y", "z"):
                eps, sig, region = ops.material_slice_for_e_3d(
                    self.permittivity, total_sigma, comp
                )
                setattr(self, f"eps_{comp}", eps)
                setattr(self, f"sig_{comp}", sig)
                setattr(self, f"region_{comp}", region)
            self.eps_ex = self.eps_x
            self.eps_ey = self.eps_y
            self.eps_ez = self.eps_z

            self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = (
                ops.magnetic_conductivity_terms_3d(
                    total_sigma,
                    self.permeability,
                    self.Hx.shape,
                    self.Hy.shape,
                    self.Hz.shape,
                )
            )
            self.mu_hx = sample_voxel_grid_at_component_3d(
                self.permeability,
                "Hx",
                stored_shape=tuple(self.Hx.shape),
            )
            self.mu_hy = sample_voxel_grid_at_component_3d(
                self.permeability,
                "Hy",
                stored_shape=tuple(self.Hy.shape),
            )
            self.mu_hz = sample_voxel_grid_at_component_3d(
                self.permeability,
                "Hz",
                stored_shape=tuple(self.Hz.shape),
            )
        else:
            for comp in ("x", "y", "z"):
                eps, sig, region = ops.material_slice_for_e_2d_component(
                    self.permittivity, total_sigma, comp, self.plane_2d
                )
                setattr(self, f"eps_{comp}", eps)
                setattr(self, f"sig_{comp}", sig)
                setattr(self, f"region_{comp}", region)
            self.eps_ex = self._sample_e_material_2d("Ex")
            self.eps_ey = self._sample_e_material_2d("Ey")
            self.eps_ez = self._sample_e_material_2d("Ez")

            self.sigma_m_hx, self.sigma_m_hy, self.sigma_m_hz = (
                ops.magnetic_conductivity_terms_2d_full(
                    total_sigma,
                    self.permeability,
                    self.Hx.shape,
                    self.Hy.shape,
                    self.Hz.shape,
                    self.plane_2d,
                )
            )
            self.mu_hx = self._sample_h_material_2d("Hx")
            self.mu_hy = self._sample_h_material_2d("Hy")
            self.mu_hz = self._sample_h_material_2d("Hz")
            if self.plane_2d == "xy":
                self.eps_tm_ez = sample_voxel_grid_at_tm_xy_full_component_2d(
                    self.permittivity,
                    "Ez",
                )
                self.mu_tm_hx = sample_voxel_grid_at_tm_xy_full_component_2d(
                    self.permeability,
                    "Hx",
                )
                self.mu_tm_hy = sample_voxel_grid_at_tm_xy_full_component_2d(
                    self.permeability,
                    "Hy",
                )

    def _sample_e_material_2d(self, component: str):
        if self.plane_2d == "xy" and component == "Ez":
            return sample_voxel_grid_at_tm_xy_full_component_2d(
                self.permittivity,
                component,
            )
        return sample_voxel_grid_at_component_2d(
            self.permittivity,
            component,
            self.plane_2d,
            stored_shape=tuple(getattr(self, component).shape),
        )

    def _sample_h_material_2d(self, component: str):
        if self.plane_2d == "xy" and component in {"Hx", "Hy"}:
            return sample_voxel_grid_at_tm_xy_full_component_2d(
                self.permeability,
                component,
            )
        return sample_voxel_grid_at_component_2d(
            self.permeability,
            component,
            self.plane_2d,
            stored_shape=tuple(getattr(self, component).shape),
        )

    def permittivity_for_component(self, component: str):
        """Return effective permittivity on an E-component Yee lattice."""
        mapping = {"Ex": "eps_ex", "Ey": "eps_ey", "Ez": "eps_ez"}
        try:
            return getattr(self, mapping[component])
        except KeyError as exc:
            raise ValueError(f"Unsupported E component {component!r}") from exc

    def permeability_for_component(self, component: str):
        """Return effective permeability on an H-component Yee lattice."""
        mapping = {"Hx": "mu_hx", "Hy": "mu_hy", "Hz": "mu_hz"}
        try:
            return getattr(self, mapping[component])
        except KeyError as exc:
            raise ValueError(f"Unsupported H component {component!r}") from exc

    def material_for_component(self, component: str):
        """Return the material array collocated with a Yee field component."""
        if component in {"Ex", "Ey", "Ez"}:
            return self.permittivity_for_component(component)
        if component in {"Hx", "Hy", "Hz"}:
            return self.permeability_for_component(component)
        raise ValueError(f"Unsupported field component {component!r}")

    def material_at_component(self, component: str, index):
        """Return material values collocated with a component support/index."""
        return self.material_for_component(component)[index]

    def _initialize_cpml_tm_xy_state(self) -> CpmlTm2DxyState | None:
        if not (self.has_cpml and self.plane_2d == "xy" and self.pml_data):
            return None
        if self.tm_xy_state is None:
            self.tm_xy_state = initialize_tm_2d_xy_state(self)
        state = self.tm_xy_state
        terms = build_tm_xy_cpml_terms(
            self.pml_data.get("tm_xy_cpml"),
            ez_shape=state.Ez.shape,
        )
        if terms is None:
            return None

        return CpmlTm2DxyState(
            psi_h_terms=jnp.zeros((2, *state.Ez.shape), dtype=state.Ez.dtype),
            psi_e_terms=jnp.stack(
                (jnp.zeros_like(state.Ez), jnp.zeros_like(state.Ez)), axis=0
            ),
            sigma_h_terms=terms.sigma_h_terms,
            kappa_h_aux_terms=terms.kappa_h_aux_terms,
            alpha_h_terms=terms.alpha_h_terms,
            kappa_h_direct_terms=terms.kappa_h_direct_terms,
            sigma_e_terms=terms.sigma_e_terms,
            kappa_e_terms=terms.kappa_e_terms,
            alpha_e_terms=terms.alpha_e_terms,
        )

    def _ensure_cpml_tm_xy_state(self, dt):
        self._cpml_dt = float(dt)
        if self.cpml_tm_xy_state is None:
            self.cpml_tm_xy_state = self._initialize_cpml_tm_xy_state()
        return self.cpml_tm_xy_state

    def _initialize_cpml_3d_state(self) -> Cpml3DState | None:
        if not (self.has_cpml and self.permittivity.ndim == 3 and self.pml_data):
            return None
        terms = build_cpml_3d_terms(self.pml_data, dt=self._cpml_dt)
        if terms is None:
            return None
        return Cpml3DState(
            psi_h_terms=tuple(jnp.zeros_like(term) for term in terms.b_h_terms),
            psi_e_terms=tuple(jnp.zeros_like(term) for term in terms.b_e_terms),
            a_h_terms=terms.a_h_terms,
            b_h_terms=terms.b_h_terms,
            inv_kappa_h_terms=terms.inv_kappa_h_terms,
            a_e_terms=terms.a_e_terms,
            b_e_terms=terms.b_e_terms,
            inv_kappa_e_terms=terms.inv_kappa_e_terms,
        )

    def _ensure_cpml_3d_state(self, dt):
        self._cpml_dt = float(dt)
        if self.cpml_3d_state is None:
            self.cpml_3d_state = self._initialize_cpml_3d_state()
        return self.cpml_3d_state

    def _init_fields_3d(self, nx, ny, nz):
        """Initialize 3D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) with proper Yee grid staggering."""
        self.Ex = jnp.zeros((nz, ny, nx - 1))
        self.Ey = jnp.zeros((nz, ny - 1, nx))
        self.Ez = jnp.zeros((nz - 1, ny, nx))
        self.Hx = jnp.zeros((nz - 1, ny - 1, nx))
        self.Hy = jnp.zeros((nz - 1, ny, nx - 1))
        self.Hz = jnp.zeros((nz, ny - 1, nx - 1))

    def _init_fields_2d(self, dim1, dim2):
        """Initialize 2D field arrays (Ex, Ey, Ez, Hx, Hy, Hz) on staggered Yee grid for the selected plane."""
        # dim1, dim2 correspond to the two active dimensions
        # xy: (y, x), yz: (z, y), xz: (z, x)

        if self.plane_2d == "xy":
            ny, nx = dim1, dim2
            # Native TMz set (Ez, Hx, Hy)
            self.Ez = jnp.zeros((ny + 1, nx + 1))
            self.Hx = jnp.zeros((ny, nx + 1))
            self.Hy = jnp.zeros((ny + 1, nx))
            # TE set (Hz, Ex, Ey)
            self.Hz = jnp.zeros((ny - 1, nx - 1))
            self.Ex = jnp.zeros((ny, nx - 1))
            self.Ey = jnp.zeros((ny - 1, nx))

        elif self.plane_2d == "yz":
            nz, ny = dim1, dim2
            # TE-like set (Ex, Hy, Hz)
            self.Ex = jnp.zeros((nz, ny))
            self.Hy = jnp.zeros((nz - 1, ny))
            self.Hz = jnp.zeros((nz, ny - 1))
            # TM-like set (Hx, Ey, Ez)
            self.Hx = jnp.zeros((nz - 1, ny - 1))
            self.Ey = jnp.zeros((nz, ny - 1))
            self.Ez = jnp.zeros((nz - 1, ny))

        elif self.plane_2d == "xz":
            nz, nx = dim1, dim2
            # TE-like set (Ey, Hx, Hz)
            self.Ey = jnp.zeros((nz, nx))
            self.Hx = jnp.zeros((nz - 1, nx))
            self.Hz = jnp.zeros((nz, nx - 1))
            # TM-like set (Hy, Ex, Ez)
            self.Hy = jnp.zeros((nz - 1, nx - 1))
            self.Ex = jnp.zeros((nz, nx - 1))
            self.Ez = jnp.zeros((nz - 1, nx))

    def available_components(self):
        """Return list of available field components."""
        return ["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"]

    def ensure_tm_xy_state(self):
        """Initialize native xy-plane TMz metadata on first use."""
        if self.plane_2d != "xy":
            return None
        if self.tm_xy_state is None:
            self.tm_xy_state = initialize_tm_2d_xy_state(self)
        return self.tm_xy_state

    def apply_tm_xy_pec_masks(self):
        """Zero constrained native xy-plane TMz samples on PEC boundaries."""
        if self.plane_2d != "xy":
            return
        state = self.ensure_tm_xy_state()
        if state is None:
            return
        self.Ez = jnp.where(state.ez_mask, 0.0, self.Ez)
        self.Hx = jnp.where(state.hx_mask, 0.0, self.Hx)
        self.Hy = jnp.where(state.hy_mask, 0.0, self.Hy)
        state.Ez = self.Ez
        state.Hx = self.Hx
        state.Hy = self.Hy

    def set_metallic_masks(self, masks):
        """Attach precomputed per-component metallic-wall masks."""
        self.metallic_masks = {
            name: jnp.asarray(mask, dtype=bool) for name, mask in (masks or {}).items()
        }

    def _apply_metallic_mask(self, component: str, field: jnp.ndarray) -> jnp.ndarray:
        if not self.metallic_masks:
            return field
        mask = self.metallic_masks.get(component)
        if mask is None:
            return field
        return jnp.where(mask, jnp.asarray(0.0, dtype=field.dtype), field)

    def apply_metallic_boundaries_h(self):
        """Zero H-field Yee samples that lie on metallic walls."""
        self.Hx = self._apply_metallic_mask("Hx", self.Hx)
        self.Hy = self._apply_metallic_mask("Hy", self.Hy)
        self.Hz = self._apply_metallic_mask("Hz", self.Hz)

    def apply_metallic_boundaries_e(self):
        """Zero E-field Yee samples that lie on metallic walls."""
        self.Ex = self._apply_metallic_mask("Ex", self.Ex)
        self.Ey = self._apply_metallic_mask("Ey", self.Ey)
        self.Ez = self._apply_metallic_mask("Ez", self.Ez)

    def update_h(self, dt, source_m=None):
        """Execute the H-field half of an FDTD time step."""
        is_3d = self.permittivity.ndim == 3

        if is_3d:
            if has_full_pec_3d(getattr(self, "boundaries", None)):
                if self.full_pec_3d_state is None:
                    self.full_pec_3d_state = initialize_full_pec_3d_state(self)
                state = self.full_pec_3d_state
                h_decay, h_source = full_pec_h_update_coefficients_3d(state, dt)
                state.Hx, state.Hy, state.Hz = full_pec_update_h_from_e_3d(
                    state.Ex,
                    state.Ey,
                    state.Ez,
                    state.Hx,
                    state.Hy,
                    state.Hz,
                    self.resolution,
                    h_decay=h_decay,
                    h_source=h_source,
                    h_mask=(
                        state.masks["Hx"],
                        state.masks["Hy"],
                        state.masks["Hz"],
                    ),
                    source_m=_component_source_tuple(
                        source_m,
                        ("Hx", "Hy", "Hz"),
                        (state.Hx, state.Hy, state.Hz),
                    ),
                )
                sync_compact_fields_from_full_pec_3d(self, state)
                return
            else:
                cpml = self._ensure_cpml_3d_state(dt)
                if cpml is not None:
                    curlE_x, curlE_y, curlE_z, cpml.psi_h_terms = cpml_curl_e_to_h_3d(
                        self.Ex,
                        self.Ey,
                        self.Ez,
                        self.resolution,
                        a_h_terms=cpml.a_h_terms,
                        b_h_terms=cpml.b_h_terms,
                        inv_kappa_h_terms=cpml.inv_kappa_h_terms,
                        psi_h_terms=cpml.psi_h_terms,
                    )
                else:
                    curlE_x, curlE_y, curlE_z = ops.curl_e_to_h_3d(
                        self.Ex, self.Ey, self.Ez, self.resolution
                    )
        else:
            if self.plane_2d == "xy":
                state = self.ensure_tm_xy_state()
                if has_full_pec_2d_xy(getattr(self, "boundaries", None), self.plane_2d):
                    curlE_x, curlE_y = full_pec_curl_e_to_h_2d_xy(
                        self.Ez,
                        self.resolution,
                        self.Hx.shape,
                        self.Hy.shape,
                    )
                else:
                    cpml = self._ensure_cpml_tm_xy_state(dt)
                    if cpml is not None:
                        curlE_x, curlE_y, cpml.psi_h_terms = tm_xy_cpml_curl_e_to_h_2d(
                            self.Ez,
                            self.resolution,
                            sigma_h_terms=cpml.sigma_h_terms,
                            kappa_h_aux_terms=cpml.kappa_h_aux_terms,
                            alpha_h_terms=cpml.alpha_h_terms,
                            kappa_h_direct_terms=cpml.kappa_h_direct_terms,
                            psi_h_terms=cpml.psi_h_terms,
                            dt=self._cpml_dt,
                        )
                    else:
                        curlE_x, curlE_y = tm_xy_curl_e_to_h_2d(
                            self.Ez,
                            self.resolution,
                            self.Hx.shape,
                            self.Hy.shape,
                            state.metallic_edges,
                        )
                curlE_z = xy_te_curl_e_to_h_2d(
                    self.Ex,
                    self.Ey,
                    self.resolution,
                    self.Hz.shape,
                )
            else:
                curlE_x, curlE_y, curlE_z = ops.curl_e_to_h_2d(
                    (self.Ex, self.Ey, self.Ez), self.resolution, plane=self.plane_2d
                )

        if source_m:
            for comp in ("Hx", "Hy", "Hz"):
                if comp in source_m:
                    for val, indices in source_m[comp]:
                        if comp == "Hx":
                            curlE_x = curlE_x.at[indices].add(val)
                        elif comp == "Hy":
                            curlE_y = curlE_y.at[indices].add(val)
                        else:
                            curlE_z = curlE_z.at[indices].add(val)

        if (not is_3d) and self.plane_2d == "xy":
            state = self.ensure_tm_xy_state()
            self.Hx = advance_h_from_curl(self.Hx, curlE_x, state.sigma_m_hx, dt)
            self.Hy = advance_h_from_curl(self.Hy, curlE_y, state.sigma_m_hy, dt)
            self.apply_tm_xy_pec_masks()
            self.Hz = advance_h_from_curl(self.Hz, curlE_z, self.sigma_m_hz, dt)
            self.apply_metallic_boundaries_h()
        else:
            self.Hx = advance_h_from_curl(self.Hx, curlE_x, self.sigma_m_hx, dt)
            self.Hy = advance_h_from_curl(self.Hy, curlE_y, self.sigma_m_hy, dt)
            self.Hz = advance_h_from_curl(self.Hz, curlE_z, self.sigma_m_hz, dt)
            self.apply_metallic_boundaries_h()

    def update_e(self, dt, source_j=None):
        """Execute the E-field half of an FDTD time step."""
        is_3d = self.permittivity.ndim == 3

        if is_3d:
            if has_full_pec_3d(getattr(self, "boundaries", None)):
                if self.full_pec_3d_state is None:
                    self.full_pec_3d_state = initialize_full_pec_3d_state(self)
                state = self.full_pec_3d_state
                e_decay, e_source = full_pec_e_update_coefficients_3d(state, dt)
                state.Ex, state.Ey, state.Ez = full_pec_update_e_from_h_3d(
                    state.Hx,
                    state.Hy,
                    state.Hz,
                    state.Ex,
                    state.Ey,
                    state.Ez,
                    self.resolution,
                    e_decay=e_decay,
                    e_source=e_source,
                    e_mask=(
                        state.masks["Ex"],
                        state.masks["Ey"],
                        state.masks["Ez"],
                    ),
                    source_j=_component_source_tuple(
                        source_j,
                        ("Ex", "Ey", "Ez"),
                        (state.Ex, state.Ey, state.Ez),
                    ),
                )
                sync_compact_fields_from_full_pec_3d(self, state)
                return
            else:
                cpml = self._ensure_cpml_3d_state(dt)
                if cpml is not None:
                    curlH_x, curlH_y, curlH_z, cpml.psi_e_terms = cpml_curl_h_to_e_3d(
                        self.Hx,
                        self.Hy,
                        self.Hz,
                        self.resolution,
                        a_e_terms=cpml.a_e_terms,
                        b_e_terms=cpml.b_e_terms,
                        inv_kappa_e_terms=cpml.inv_kappa_e_terms,
                        psi_e_terms=cpml.psi_e_terms,
                        metallic_edges=resolve_metallic_edges(
                            getattr(self, "boundaries", None), is_3d=True
                        ),
                    )
                else:
                    boundary_views = build_h_boundary_views_for_e_3d(
                        self.Hx, self.Hy, self.Hz, getattr(self, "boundaries", None)
                    )
                    curlH_x, curlH_y, curlH_z = ops.curl_h_to_e_3d(
                        self.Hx,
                        self.Hy,
                        self.Hz,
                        self.resolution,
                        ex_shape=self.Ex.shape,
                        ey_shape=self.Ey.shape,
                        ez_shape=self.Ez.shape,
                        boundary_views=boundary_views,
                    )
        else:
            if self.plane_2d == "xy":
                state = self.ensure_tm_xy_state()
                curlH_x, curlH_y = xy_te_curl_h_to_e_2d(
                    self.Hz,
                    self.resolution,
                    self.Ex.shape,
                    self.Ey.shape,
                    state.metallic_edges,
                )
                if has_full_pec_2d_xy(getattr(self, "boundaries", None), self.plane_2d):
                    curlH_z = full_pec_curl_h_to_e_2d_xy(
                        self.Hx,
                        self.Hy,
                        self.resolution,
                        self.Ez.shape,
                    )
                else:
                    cpml = self._ensure_cpml_tm_xy_state(dt)
                    if cpml is not None:
                        curlH_z, cpml.psi_e_terms = tm_xy_cpml_curl_h_to_e_2d(
                            self.Hx,
                            self.Hy,
                            self.resolution,
                            self.Ez.shape,
                            state.metallic_edges,
                            sigma_e_terms=cpml.sigma_e_terms,
                            kappa_e_terms=cpml.kappa_e_terms,
                            alpha_e_terms=cpml.alpha_e_terms,
                            psi_e_terms=cpml.psi_e_terms,
                            dt=self._cpml_dt,
                        )
                    else:
                        curlH_z = tm_xy_curl_h_to_e_2d(
                            self.Hx,
                            self.Hy,
                            self.resolution,
                            self.Ez.shape,
                            state.metallic_edges,
                        )
            else:
                curlH_x, curlH_y, curlH_z = ops.curl_h_to_e_2d(
                    (self.Hx, self.Hy, self.Hz),
                    self.resolution,
                    (self.Ex.shape, self.Ey.shape, self.Ez.shape),
                    plane=self.plane_2d,
                )

        if source_j:
            for comp in ("Ex", "Ey", "Ez"):
                if comp in source_j:
                    for val, indices in source_j[comp]:
                        if comp == "Ex":
                            curlH_x = curlH_x.at[indices].add(val)
                        elif comp == "Ey":
                            curlH_y = curlH_y.at[indices].add(val)
                        else:
                            curlH_z = curlH_z.at[indices].add(val)

        if (not is_3d) and self.plane_2d == "xy":
            state = self.ensure_tm_xy_state()
            self.Ex = advance_e_from_curl(
                self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x
            )
            self.Ey = advance_e_from_curl(
                self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y
            )
            self.Ez = advance_e_from_curl(
                self.Ez,
                curlH_z,
                state.sig_z_region,
                state.eps_z_region,
                dt,
                (slice(None), slice(None)),
            )
            self.apply_tm_xy_pec_masks()
            self.apply_metallic_boundaries_e()
        else:
            self.Ex = advance_e_from_curl(
                self.Ex, curlH_x, self.sig_x, self.eps_x, dt, self.region_x
            )
            self.Ey = advance_e_from_curl(
                self.Ey, curlH_y, self.sig_y, self.eps_y, dt, self.region_y
            )
            self.Ez = advance_e_from_curl(
                self.Ez, curlH_z, self.sig_z, self.eps_z, dt, self.region_z
            )
            self.apply_metallic_boundaries_e()

    def update(self, dt, source_j=None, source_m=None):
        """Execute one FDTD time step (2D or 3D) with optional source injection."""
        self.update_h(dt, source_m=source_m)
        self.update_e(dt, source_j=source_j)

    def update_materials(self, permittivity=None, conductivity=None, permeability=None):
        """Update material grids and recompute Yee parameters."""
        if permittivity is not None:
            self.permittivity = jnp.asarray(permittivity)
        if conductivity is not None:
            self.conductivity = jnp.asarray(conductivity)
        if permeability is not None:
            self.permeability = jnp.asarray(permeability)
        self._init_material_parameters()
