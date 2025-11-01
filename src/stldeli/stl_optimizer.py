"""
STL file optimization utilities
"""

import logging
from typing import Dict, Any, Optional
import numpy as np
from stl import mesh

logger = logging.getLogger(__name__)

def analyze_stl_mesh(mesh_data: mesh.Mesh) -> Optional[Dict[str, Any]]:
    """
    Analyze STL mesh and return detailed information
    """
    try:
        vertices = len(mesh_data.vectors)
        triangles = len(mesh_data.vectors)

        # Calculate bounding box
        min_coords = mesh_data.min_
        max_coords = mesh_data.max_
        dimensions = max_coords - min_coords

        # Calculate volume
        volume = mesh_data.get_mass_properties()[0]

        # Calculate surface area
        surface_area = mesh_data.areas.sum()

        return {
            'vertices': vertices,
            'triangles': triangles,
            'dimensions': {
                'x': float(dimensions[0]),
                'y': float(dimensions[1]),
                'z': float(dimensions[2])
            },
            'min_coords': {
                'x': float(min_coords[0]),
                'y': float(min_coords[1]),
                'z': float(min_coords[2])
            },
            'max_coords': {
                'x': float(max_coords[0]),
                'y': float(max_coords[1]),
                'z': float(max_coords[2])
            },
            'volume': float(volume),
            'surface_area': float(surface_area)
        }
    except (ValueError, IndexError) as e:
        logger.error("Data error analyzing STL mesh: %s", str(e))
        return None
    except AttributeError as e:
        logger.error("Attribute error analyzing STL mesh: %s", str(e))
        return None
    except (TypeError, KeyError, RuntimeError) as e:
        logger.error("Error analyzing STL mesh: %s", str(e))
        return None

def optimize_mesh_vertices(mesh_data: mesh.Mesh, tolerance: float = 0.01) -> mesh.Mesh:
    """
    Optimize mesh by removing duplicate vertices within tolerance
    """
    try:
        vertices = mesh_data.vectors.reshape(-1, 3)

        # Round vertices to tolerance to find duplicates
        rounded_vertices = np.round(vertices / tolerance) * tolerance
        unique_vertices, indices = np.unique(rounded_vertices, axis=0, return_inverse=True)

        # Reconstruct triangles with new vertex indices
        triangles = indices.reshape(-1, 3)

        # Create new mesh
        optimized_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        for i, tri in enumerate(triangles):
            optimized_mesh.vectors[i] = unique_vertices[tri]

        return optimized_mesh
    except (ValueError, IndexError) as e:
        logger.error("Data error optimizing mesh vertices: %s", str(e))
        raise
    except MemoryError as e:
        logger.error("Memory error optimizing mesh vertices: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error optimizing mesh vertices: %s", str(e))
        raise

def remove_degenerate_triangles(mesh_data: mesh.Mesh) -> mesh.Mesh:
    """
    Remove degenerate triangles (zero area)
    """
    try:
        # Calculate triangle areas
        areas = mesh_data.areas

        # Keep only triangles with area > threshold
        valid_triangles = areas > 1e-10

        if np.all(valid_triangles):
            return mesh_data  # No degenerate triangles found

        # Create new mesh with only valid triangles
        valid_count = np.sum(valid_triangles)
        if valid_count == 0:
            # Return empty mesh if no valid triangles
            return mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

        optimized_mesh = mesh.Mesh(np.zeros(valid_count, dtype=mesh.Mesh.dtype))
        valid_indices = np.where(valid_triangles)[0]

        for i, idx in enumerate(valid_indices):
            optimized_mesh.vectors[i] = mesh_data.vectors[idx]

        return optimized_mesh
    except (ValueError, IndexError) as e:
        logger.error("Data error removing degenerate triangles: %s", str(e))
        raise
    except MemoryError as e:
        logger.error("Memory error removing degenerate triangles: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error removing degenerate triangles: %s", str(e))
        raise

def smooth_mesh(mesh_data: mesh.Mesh, iterations: int = 1) -> mesh.Mesh:
    """
    Apply simple Laplacian smoothing to mesh
    """
    try:
        # Create a copy of the mesh manually
        current_mesh = mesh.Mesh(np.zeros(mesh_data.vectors.shape[0], dtype=mesh.Mesh.dtype))
        current_mesh.vectors = mesh_data.vectors.copy()

        for _ in range(iterations):
            vertices = current_mesh.vectors.reshape(-1, 3)

            # Build vertex adjacency
            vertex_map = {}
            for i, vertex in enumerate(vertices):
                key = tuple(vertex)
                if key not in vertex_map:
                    vertex_map[key] = []
                vertex_map[key].append(i)

            # Calculate smoothed positions
            smoothed_vertices = vertices.copy()
            for i, vertex in enumerate(vertices):
                # Find adjacent vertices (simplified approach)
                adjacent_indices = []
                for tri_idx, tri in enumerate(current_mesh.vectors):
                    if any(np.array_equal(vertex, v) for v in tri):
                        for v in tri:
                            if not np.array_equal(vertex, v):
                                adjacent_indices.append(np.where((vertices == v).all(axis=1))[0][0])

                if adjacent_indices:
                    # Average with adjacent vertices
                    adjacent_vertices = vertices[adjacent_indices]
                    smoothed_vertices[i] = 0.5 * vertex + 0.5 * np.mean(adjacent_vertices, axis=0)

            # Update mesh
            current_mesh.vectors = smoothed_vertices.reshape(-1, 3, 3)

        return current_mesh
    except (ValueError, IndexError) as e:
        logger.error("Data error smoothing mesh: %s", str(e))
        raise
    except MemoryError as e:
        logger.error("Memory error smoothing mesh: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error smoothing mesh: %s", str(e))
        raise

def optimize_stl_file(file_path: str, optimization_level: str = 'medium') -> mesh.Mesh:
    """
    Optimize STL file with specified level
    Returns optimized mesh data
    """
    try:
        # Load the mesh
        mesh_data = mesh.Mesh.from_file(file_path)
        original_triangles = len(mesh_data.vectors)

        logger.info("Original mesh: %d triangles", original_triangles)

        if optimization_level == 'light':
            # Light optimization: remove degenerate triangles only
            optimized_mesh = remove_degenerate_triangles(mesh_data)

        elif optimization_level == 'medium':
            # Medium optimization: remove duplicates and degenerate triangles
            optimized_mesh = remove_degenerate_triangles(mesh_data)
            optimized_mesh = optimize_mesh_vertices(optimized_mesh, tolerance=0.01)

        elif optimization_level == 'aggressive':
            # Aggressive optimization: higher tolerance and smoothing
            optimized_mesh = remove_degenerate_triangles(mesh_data)
            optimized_mesh = optimize_mesh_vertices(optimized_mesh, tolerance=0.1)
            optimized_mesh = smooth_mesh(optimized_mesh, iterations=1)

        else:
            optimized_mesh = mesh_data

        optimized_triangles = len(optimized_mesh.vectors)
        reduction_percentage = ((original_triangles - optimized_triangles) / original_triangles) * 100

        logger.info("Optimized mesh: %d triangles (%.1f%% reduction)", optimized_triangles, reduction_percentage)

        return optimized_mesh

    except (ValueError, IndexError) as e:
        logger.error("Data error optimizing STL file: %s", str(e))
        raise
    except MemoryError as e:
        logger.error("Memory error optimizing STL file: %s", str(e))
        raise
    except Exception as e:
        logger.error("Error optimizing STL file: %s", str(e))
        raise
