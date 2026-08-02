"""Intentional dynamic fields and public entry points for Vulture."""

projection_residual
condition_number
plot_mode_field_components
plot_result_field
view_simulation_3d
_.discretize
_.add_task
_.cache_spec
_.__path__
_.update_grid
_.auto
_.to_monitor
_.to_source
_.selected_mode
_.world_origin
gaussian_band_pulse
attach_material_coefficients
ForwardFieldChunkWriter
_.finalize
compute_overlap_gradient_memmap
_.with_objective
_.initial_state
_.density_for_step
_.apply_gradient
compute_overlap_gradient
create_optimization_mask
_.clear_compiled_cache
_.memory_estimate
_.show3d
total_conductivity
eps_ex
eps_ey
eps_ez
mu_hx
mu_hy
mu_hz
_.removed_degenerate_triangles
_.removed_duplicate_triangles
_.removed_unreferenced_vertices
_.winding_or_normals_fixed
_.remaining_issues
_.connected_components
_.is_uniform

# functools.singledispatch registrations are reached through lower_source.dispatch().
_lower_custom_source
_lower_gaussian_beam_source
_lower_gaussian_source
_lower_mode_source

# Public native mode-solver fields and methods are consumed by downstream users
# and optional serialization paths, not necessarily by BeamZ itself.
_.component_offsets
_.phase_reference_component
_.from_subpixel_diagonal
_.diagonal_eps
_.to_hdf5
_.from_hdf5
_.n_group
_.dispersion
_.n_eff
_.k_eff

# Public design metadata is consumed by callers and examples outside beamz/.
_.component_name
