import jax

from beamz.simulation.ops import advance_e_field, advance_h_field


def create_step(sim):
    """Create a JIT-compiled full FDTD step function."""
    resolution = sim.resolution
    dt = sim.dt
    plane_2d = sim.plane_2d

    eps_x, sig_x, region_x = sim.fields.eps_x, sim.fields.sig_x, sim.fields.region_x
    eps_y, sig_y, region_y = sim.fields.eps_y, sim.fields.sig_y, sim.fields.region_y
    eps_z, sig_z, region_z = sim.fields.eps_z, sim.fields.sig_z, sim.fields.region_z
    sigma_m_hx = sim.fields.sigma_m_hx
    sigma_m_hy = sim.fields.sigma_m_hy
    sigma_m_hz = sim.fields.sigma_m_hz

    from beamz.simulation.ops import (
        curl_e_to_h_2d,
        curl_e_to_h_3d,
        curl_h_to_e_2d,
        curl_h_to_e_3d,
    )

    if sim.is_3d:

        @jax.jit
        def step(ex, ey, ez, hx, hy, hz):
            curl_e_x, curl_e_y, curl_e_z = curl_e_to_h_3d(ex, ey, ez, resolution)
            hx_new = advance_h_field(hx, curl_e_x, sigma_m_hx, dt)
            hy_new = advance_h_field(hy, curl_e_y, sigma_m_hy, dt)
            hz_new = advance_h_field(hz, curl_e_z, sigma_m_hz, dt)
            curl_h_x, curl_h_y, curl_h_z = curl_h_to_e_3d(
                hx_new,
                hy_new,
                hz_new,
                resolution,
                ex_shape=ex.shape,
                ey_shape=ey.shape,
                ez_shape=ez.shape,
            )
            ex_new = advance_e_field(ex, curl_h_x, sig_x, eps_x, dt, region_x)
            ey_new = advance_e_field(ey, curl_h_y, sig_y, eps_y, dt, region_y)
            ez_new = advance_e_field(ez, curl_h_z, sig_z, eps_z, dt, region_z)
            return ex_new, ey_new, ez_new, hx_new, hy_new, hz_new

    else:

        @jax.jit
        def step(ex, ey, ez, hx, hy, hz):
            curl_e_x, curl_e_y, curl_e_z = curl_e_to_h_2d(
                (ex, ey, ez), resolution, plane=plane_2d
            )
            hx_new = advance_h_field(hx, curl_e_x, sigma_m_hx, dt)
            hy_new = advance_h_field(hy, curl_e_y, sigma_m_hy, dt)
            hz_new = advance_h_field(hz, curl_e_z, sigma_m_hz, dt)
            curl_h_x, curl_h_y, curl_h_z = curl_h_to_e_2d(
                (hx_new, hy_new, hz_new),
                resolution,
                (ex.shape, ey.shape, ez.shape),
                plane=plane_2d,
            )
            ex_new = advance_e_field(ex, curl_h_x, sig_x, eps_x, dt, region_x)
            ey_new = advance_e_field(ey, curl_h_y, sig_y, eps_y, dt, region_y)
            ez_new = advance_e_field(ez, curl_h_z, sig_z, eps_z, dt, region_z)
            return ex_new, ey_new, ez_new, hx_new, hy_new, hz_new

    return step


def create_step_h(sim):
    """Create a JIT-compiled H-update function."""
    resolution = sim.resolution
    dt = sim.dt
    plane_2d = sim.plane_2d
    sigma_m_hx = sim.fields.sigma_m_hx
    sigma_m_hy = sim.fields.sigma_m_hy
    sigma_m_hz = sim.fields.sigma_m_hz

    from beamz.simulation.ops import curl_e_to_h_2d, curl_e_to_h_3d

    if sim.is_3d:

        @jax.jit
        def step_h(ex, ey, ez, hx, hy, hz):
            curl_e_x, curl_e_y, curl_e_z = curl_e_to_h_3d(ex, ey, ez, resolution)
            hx_new = advance_h_field(hx, curl_e_x, sigma_m_hx, dt)
            hy_new = advance_h_field(hy, curl_e_y, sigma_m_hy, dt)
            hz_new = advance_h_field(hz, curl_e_z, sigma_m_hz, dt)
            return hx_new, hy_new, hz_new

    else:

        @jax.jit
        def step_h(ex, ey, ez, hx, hy, hz):
            curl_e_x, curl_e_y, curl_e_z = curl_e_to_h_2d(
                (ex, ey, ez), resolution, plane=plane_2d
            )
            hx_new = advance_h_field(hx, curl_e_x, sigma_m_hx, dt)
            hy_new = advance_h_field(hy, curl_e_y, sigma_m_hy, dt)
            hz_new = advance_h_field(hz, curl_e_z, sigma_m_hz, dt)
            return hx_new, hy_new, hz_new

    return step_h


def create_step_e(sim):
    """Create a JIT-compiled E-update function."""
    resolution = sim.resolution
    dt = sim.dt
    plane_2d = sim.plane_2d
    eps_x, sig_x, region_x = sim.fields.eps_x, sim.fields.sig_x, sim.fields.region_x
    eps_y, sig_y, region_y = sim.fields.eps_y, sim.fields.sig_y, sim.fields.region_y
    eps_z, sig_z, region_z = sim.fields.eps_z, sim.fields.sig_z, sim.fields.region_z

    from beamz.simulation.ops import curl_h_to_e_2d, curl_h_to_e_3d

    if sim.is_3d:

        @jax.jit
        def step_e(ex, ey, ez, hx, hy, hz):
            curl_h_x, curl_h_y, curl_h_z = curl_h_to_e_3d(
                hx,
                hy,
                hz,
                resolution,
                ex_shape=ex.shape,
                ey_shape=ey.shape,
                ez_shape=ez.shape,
            )
            ex_new = advance_e_field(ex, curl_h_x, sig_x, eps_x, dt, region_x)
            ey_new = advance_e_field(ey, curl_h_y, sig_y, eps_y, dt, region_y)
            ez_new = advance_e_field(ez, curl_h_z, sig_z, eps_z, dt, region_z)
            return ex_new, ey_new, ez_new

    else:

        @jax.jit
        def step_e(ex, ey, ez, hx, hy, hz):
            curl_h_x, curl_h_y, curl_h_z = curl_h_to_e_2d(
                (hx, hy, hz),
                resolution,
                (ex.shape, ey.shape, ez.shape),
                plane=plane_2d,
            )
            ex_new = advance_e_field(ex, curl_h_x, sig_x, eps_x, dt, region_x)
            ey_new = advance_e_field(ey, curl_h_y, sig_y, eps_y, dt, region_y)
            ez_new = advance_e_field(ez, curl_h_z, sig_z, eps_z, dt, region_z)
            return ex_new, ey_new, ez_new

    return step_e
