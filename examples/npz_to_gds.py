"""Convert a topology optimization .npz file to a .gds file.

Usage:
    python npz_to_gds.py topo_bend_data.npz topo_bend.gds
    python npz_to_gds.py topo_bend_data.npz  # outputs topo_bend_data.gds
"""
import argparse
import os

import numpy as np

from beamz.design.io import export_grid_gds


def npz_to_gds(npz_path, gds_path=None):
    data = np.load(npz_path)
    eps = data['permittivity']
    dx = float(data['dx'])
    n_core = float(data['n_core'])
    n_clad = float(data['n_clad'])

    print(f"Loaded {npz_path}: grid {eps.shape}, dx={dx*1e9:.1f} nm")
    print(f"  n_core={n_core}, n_clad={n_clad}")

    if gds_path is None:
        gds_path = os.path.splitext(npz_path)[0] + '.gds'

    n_polys = export_grid_gds(eps, gds_path, n_core=n_core, n_clad=n_clad, dx=dx)
    print(f"Wrote {gds_path} ({n_polys} polygons)")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert topology optimization .npz to .gds')
    parser.add_argument('npz', help='Input .npz file (from 4_topo.py)')
    parser.add_argument('gds', nargs='?', default=None, help='Output .gds file (default: same name as input)')
    args = parser.parse_args()
    npz_to_gds(args.npz, args.gds)
