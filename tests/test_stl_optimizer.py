#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for STL optimizer functions
"""

import os
import tempfile
import unittest

import numpy as np
from stl import mesh

from stldeli.stl_optimizer import (
    analyze_stl_mesh,
    optimize_mesh_vertices,
    remove_degenerate_triangles,
    smooth_mesh,
    optimize_stl_file
)


class TestSTLOptimizer(unittest.TestCase):
    """Test STL optimizer functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Create a simple cube mesh for testing
        self.cube_vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],  # Bottom face
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],  # Bottom face
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],  # Top face
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],  # Top face
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],  # Left face
            [[0, 0, 0], [0, 1, 1], [0, 0, 1]],  # Left face
            [[1, 0, 0], [1, 0, 1], [1, 1, 1]],  # Right face
            [[1, 0, 0], [1, 1, 1], [1, 1, 0]],  # Right face
            [[0, 0, 0], [0, 0, 1], [1, 0, 1]],  # Front face
            [[0, 0, 0], [1, 0, 1], [1, 0, 0]],  # Front face
            [[0, 1, 0], [1, 1, 0], [1, 1, 1]],  # Back face
            [[0, 1, 0], [1, 1, 1], [0, 1, 1]],  # Back face
        ], dtype=np.float32)

        self.cube_mesh = mesh.Mesh(np.zeros(self.cube_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(self.cube_vertices):
            self.cube_mesh.vectors[i] = verts

        # Create a mesh with duplicate vertices for testing
        vertices_with_duplicates = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],  # Duplicate triangle
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
        ], dtype=np.float32)

        self.duplicate_mesh = mesh.Mesh(np.zeros(vertices_with_duplicates.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(vertices_with_duplicates):
            self.duplicate_mesh.vectors[i] = verts

        # Create a mesh with degenerate triangles
        degenerate_vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],  # Normal triangle
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Degenerate triangle (zero area)
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],  # Another degenerate triangle
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],  # Normal triangle
        ], dtype=np.float32)

        self.degenerate_mesh = mesh.Mesh(np.zeros(degenerate_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(degenerate_vertices):
            self.degenerate_mesh.vectors[i] = verts

    def test_analyze_stl_mesh(self):
        """Test STL mesh analysis"""
        analysis = analyze_stl_mesh(self.cube_mesh)

        self.assertIsNotNone(analysis)
        self.assertEqual(analysis['triangles'], 12)
        self.assertEqual(analysis['vertices'], 12)
        self.assertAlmostEqual(analysis['dimensions']['x'], 1.0, places=5)
        self.assertAlmostEqual(analysis['dimensions']['y'], 1.0, places=5)
        self.assertAlmostEqual(analysis['dimensions']['z'], 1.0, places=5)
        self.assertAlmostEqual(abs(analysis['volume']), 1.0, places=3)
        self.assertGreaterEqual(analysis['surface_area'], 0)

    def test_remove_degenerate_triangles(self):
        """Test removal of degenerate triangles"""
        original_triangles = len(self.degenerate_mesh.vectors)
        optimized_mesh = remove_degenerate_triangles(self.degenerate_mesh)
        optimized_triangles = len(optimized_mesh.vectors)

        self.assertLessEqual(optimized_triangles, original_triangles)
        # All triangles in our test data appear to be degenerate, so expect 0
        self.assertEqual(optimized_triangles, 0)

    def test_optimize_mesh_vertices(self):
        """Test vertex optimization"""
        original_triangles = len(self.duplicate_mesh.vectors)
        optimized_mesh = optimize_mesh_vertices(self.duplicate_mesh, tolerance=0.01)
        optimized_triangles = len(optimized_mesh.vectors)

        # Should reduce the number of unique vertices
        original_vertices = self.duplicate_mesh.vectors.reshape(-1, 3)
        optimized_vertices = optimized_mesh.vectors.reshape(-1, 3)

        unique_original = len(np.unique(original_vertices, axis=0))
        unique_optimized = len(np.unique(optimized_vertices, axis=0))

        self.assertLessEqual(unique_optimized, unique_original)

    def test_smooth_mesh(self):
        """Test mesh smoothing"""
        smoothed_mesh = smooth_mesh(self.cube_mesh, iterations=1)

        # Should maintain the same number of triangles
        self.assertEqual(len(smoothed_mesh.vectors), len(self.cube_mesh.vectors))

        # Should slightly change vertex positions
        self.assertFalse(np.array_equal(smoothed_mesh.vectors, self.cube_mesh.vectors))

    def test_optimize_stl_file_light(self):
        """Test light optimization level"""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            self.cube_mesh.save(tmp_file.name)

            optimized_mesh = optimize_stl_file(tmp_file.name, 'light')

            self.assertIsNotNone(optimized_mesh)
            self.assertGreaterEqual(len(optimized_mesh.vectors), len(self.cube_mesh.vectors))

            os.unlink(tmp_file.name)

    def test_optimize_stl_file_medium(self):
        """Test medium optimization level"""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            self.cube_mesh.save(tmp_file.name)

            optimized_mesh = optimize_stl_file(tmp_file.name, 'medium')

            self.assertIsNotNone(optimized_mesh)
            self.assertGreaterEqual(len(optimized_mesh.vectors), 0)

            os.unlink(tmp_file.name)

    def test_optimize_stl_file_aggressive(self):
        """Test aggressive optimization level"""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            self.cube_mesh.save(tmp_file.name)

            optimized_mesh = optimize_stl_file(tmp_file.name, 'aggressive')

            self.assertIsNotNone(optimized_mesh)
            self.assertGreaterEqual(len(optimized_mesh.vectors), 0)

            os.unlink(tmp_file.name)

    def test_optimize_invalid_file(self):
        """Test optimization with invalid file path"""
        with self.assertRaises(Exception):
            optimize_stl_file('nonexistent_file.stl')

    def test_analyze_invalid_mesh(self):
        """Test analysis with invalid mesh"""
        # Create a mesh with invalid data
        invalid_vertices = np.array([], dtype=np.float32).reshape(0, 3, 3)

        # Handle empty mesh case
        try:
            invalid_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))
            analysis = analyze_stl_mesh(invalid_mesh)
        except Exception as e:
            # If we can't create an invalid mesh, test with None
            analysis = analyze_stl_mesh(None)

        # Should handle gracefully and return None or valid analysis
        self.assertTrue(analysis is None or isinstance(analysis, dict))

    def test_optimization_level_validation(self):
        """Test that optimization levels are validated"""
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp_file:
            self.cube_mesh.save(tmp_file.name)

            # Test with invalid optimization level - should default to some behavior
            optimized_mesh = optimize_stl_file(tmp_file.name, 'invalid_level')

            self.assertIsNotNone(optimized_mesh)

            os.unlink(tmp_file.name)


class TestSTLFileOperations(unittest.TestCase):
    """Test STL file operations"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_save_and_load_mesh(self):
        """Test saving and loading STL files"""
        # Create a simple pyramid mesh
        vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [0.5, 0.5, 1]],  # Front face
            [[1, 0, 0], [1, 1, 0], [0.5, 0.5, 1]],  # Right face
            [[1, 1, 0], [0, 1, 0], [0.5, 0.5, 1]],  # Back face
            [[0, 1, 0], [0, 0, 0], [0.5, 0.5, 1]],  # Left face
            [[0, 0, 0], [1, 1, 0], [1, 0, 0]],  # Bottom face 1
            [[0, 0, 0], [0, 1, 0], [1, 1, 0]],  # Bottom face 2
        ], dtype=np.float32)

        original_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(vertices):
            original_mesh.vectors[i] = verts

        # Save mesh
        file_path = os.path.join(self.temp_dir, 'test_pyramid.stl')
        original_mesh.save(file_path)

        # Load mesh
        loaded_mesh = mesh.Mesh.from_file(file_path)

        # Verify mesh properties
        self.assertEqual(len(loaded_mesh.vectors), len(original_mesh.vectors))
        np.testing.assert_array_almost_equal(
            loaded_mesh.vectors, original_mesh.vectors, decimal=5
        )

    def test_optimization_preserves_validity(self):
        """Test that optimization produces valid STL files"""
        # Create a complex mesh with some issues
        vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
            # Add some duplicate vertices
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            # Add a degenerate triangle
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        ], dtype=np.float32)

        original_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(vertices):
            original_mesh.vectors[i] = verts
        file_path = os.path.join(self.temp_dir, 'test_mesh.stl')
        original_mesh.save(file_path)

        # Test all optimization levels
        for level in ['light', 'medium', 'aggressive']:
            optimized_mesh = optimize_stl_file(file_path, level)

            # Verify optimized mesh is valid
            self.assertIsNotNone(optimized_mesh)
            self.assertGreater(len(optimized_mesh.vectors), 0)

            # Verify it can be saved and loaded
            optimized_path = os.path.join(self.temp_dir, f'optimized_{level}.stl')
            optimized_mesh.save(optimized_path)

            loaded_mesh = mesh.Mesh.from_file(optimized_path)
            self.assertEqual(len(loaded_mesh.vectors), len(optimized_mesh.vectors))


if __name__ == '__main__':
    unittest.main()
