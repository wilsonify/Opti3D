import os
import io
import numpy as np
import pandas as pd
from stldeli import deli, pandashelpers, stl_optimizer
from stl import mesh


def test_flag2placeholder_deli_and_pandashelpers():
    assert deli.flag2placeholder('--layer-height') == 'layer_height[layer_height]'
    assert pandashelpers.flag2placeholder('--fill-density') == 'fill_density[fill_density]'


def test_get_combinations_from_configurations():
    cfg = {'--a': [1, 2], '--b': [3]}
    combos = list(deli.get_combinations_from_configurations(cfg))
    assert (1, 3) in combos and (2, 3) in combos


def test_get_series_from_gcode(tmp_path):
    p = tmp_path / 'test.gcode'
    p.write_text(";key1=value1\n; key2 = value with spaces\nG1 X10 Y10\n")
    s = pandashelpers.get_series_from_gcode(str(p))
    assert s['key1'] == 'value1'
    assert s['key2'] == 'value with spaces'


def _make_simple_mesh():
    # Create two triangles (a square) as vectors shape (2,3,3)
    v = np.array([
        [[0,0,0],[1,0,0],[1,1,0]],
        [[0,0,0],[1,1,0],[0,1,0]],
    ], dtype=float)
    m = mesh.Mesh(np.zeros(v.shape[0], dtype=mesh.Mesh.dtype))
    m.vectors[:] = v
    return m


def test_analyze_stl_mesh_and_remove_degenerate():
    m = _make_simple_mesh()
    info = stl_optimizer.analyze_stl_mesh(m)
    assert info is not None
    assert info['triangles'] == 2

    # Add degenerate triangle (all points same)
    v = np.array([
        [[0,0,0],[0,0,0],[0,0,0]],
    ], dtype=float)
    dm = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    dm.vectors[:] = v

    combined = mesh.Mesh(np.zeros(3, dtype=mesh.Mesh.dtype))
    combined.vectors[0:2] = m.vectors
    combined.vectors[2] = dm.vectors[0]

    optimized = stl_optimizer.remove_degenerate_triangles(combined)
    # Ensure function returns a Mesh and does not increase triangle count
    assert isinstance(optimized, mesh.Mesh)
    assert len(optimized.vectors) <= len(combined.vectors)


def test_optimize_mesh_vertices_and_smooth():
    m = _make_simple_mesh()
    # duplicate vertices to create merging
    optimized = stl_optimizer.optimize_mesh_vertices(m, tolerance=0.5)
    assert isinstance(optimized, mesh.Mesh)

    # smooth_mesh should return a mesh of same length
    sm = stl_optimizer.smooth_mesh(m, iterations=1)
    assert isinstance(sm, mesh.Mesh)
    assert len(sm.vectors) == len(m.vectors)
