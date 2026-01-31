"""Convert a topology optimization .npz file to a .gds file.

Usage:
    python npz_to_gds.py topo_bend_data.npz topo_bend.gds
    python npz_to_gds.py topo_bend_data.npz  # outputs topo_bend_data.gds
"""
import argparse
import os

import gdspy
import matplotlib.pyplot as plt
import numpy as np


def npz_to_gds(npz_path, gds_path=None):
    data = np.load(npz_path)
    eps = data['permittivity']
    dx = float(data['dx'])
    n_core = float(data['n_core'])
    n_clad = float(data['n_clad'])

    print(f"Loaded {npz_path}: grid {eps.shape}, dx={dx*1e9:.1f} nm")
    print(f"  n_core={n_core}, n_clad={n_clad}")

    # Pad with cladding so contours close at grid boundaries
    eps_padded = np.full((eps.shape[0] + 2, eps.shape[1] + 2), n_clad**2)
    eps_padded[1:-1, 1:-1] = eps

    eps_threshold = (n_core**2 + n_clad**2) / 2

    fig_temp, ax_temp = plt.subplots()
    cs = ax_temp.contour(eps_padded.T, levels=[eps_threshold])
    plt.close(fig_temp)

    lib = gdspy.GdsLibrary(unit=1e-6, precision=1e-9)
    cell = lib.new_cell("main")

    for seg in cs.allsegs[0]:
        # Offset by -1 for padding, then convert grid coords to microns
        verts_um = [((x - 1) * dx * 1e6, (y - 1) * dx * 1e6) for x, y in seg]
        if len(verts_um) >= 3:
            cell.add(gdspy.Polygon(verts_um, layer=0))

    if gds_path is None:
        gds_path = os.path.splitext(npz_path)[0] + '.gds'

    lib.write_gds(gds_path)
    print(f"Wrote {gds_path} ({len(cell.polygons)} polygons)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert topology optimization .npz to .gds')
    parser.add_argument('npz', help='Input .npz file (from 4_topo.py)')
    parser.add_argument('gds', nargs='?', default=None, help='Output .gds file (default: same name as input)')
    args = parser.parse_args()
    npz_to_gds(args.npz, args.gds)
