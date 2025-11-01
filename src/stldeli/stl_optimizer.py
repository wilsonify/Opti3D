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
    Analyze STL mesh and return detailed information.
    """
    try:
        vertices = len(mesh_data.vectors)
        triangles = len(mesh_data.vectors)

        min_coords = mesh_data.min_
        max_coords = mesh_data.max_
        dimensions = max_coords - min_coords

        volume = mesh_data.get_mass_properties()[0]
        surface_area = mesh_data.areas.sum()

        return {
            "vertices": vertices,
            "triangles": triangles,
            "dimensions": {
                "x": float(dimensions[0]),
                "y": float(dimensions[1]),
                "z": float(dimensions[2]),
            },
            "min_coords": {
                "x": float(min_coords[0]),
                "y": float(min_coords[1]),
                "z": float(min_coords[2]),
            },
            "max_coords": {
                "x": float(max_coords[0]),
                "y": float(max_coords[1]),
                "z": float(max_coords[2]),
            },
            "volume": float(volume),
            "surface_area": float(surface_area),
        }

    except (ValueError, IndexError, AttributeError, TypeError, KeyError, RuntimeError) as e:
        logger.error("Error analyzing STL mesh: %s", e)
        return None


def optimize_mesh_vertices(mesh_data: mesh.Mesh, tolerance: float = 0.01) -> mesh.Mesh:
    """
    Optimize mesh by removing duplicate vertices within tolerance.
    """
    try:
        vertices = mesh_data.vectors.reshape(-1, 3)
        rounded_vertices = np.round(vertices / tolerance) * tolerance
        unique_vertices, indices = np.unique(rounded_vertices, axis=0, return_inverse=True)

        triangles = indices.reshape(-1, 3)
        optimized_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        for i, tri in enumerate(triangles):
            optimized_mesh.vectors[i] = unique_vertices[tri]

        return optimized_mesh

    except (ValueError, IndexError, MemoryError) as e:
        logger.error("Error optimizing mesh vertices: %s", e)
        raise


def remove_degenerate_triangles(mesh_data: mesh.Mesh) -> mesh.Mesh:
    """
    Remove degenerate triangles (zero area).
    """
    try:
        areas = mesh_data.areas
        valid_triangles = areas > 1e-10

        if np.all(valid_triangles):
            return mesh_data

        valid_indices = np.nonzero(valid_triangles)[0]
        if valid_indices.size == 0:
            return mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))

        optimized_mesh = mesh.Mesh(np.zeros(valid_indices.size, dtype=mesh.Mesh.dtype))
        optimized_mesh.vectors[:] = mesh_data.vectors[valid_indices]
        return optimized_mesh

    except (ValueError, IndexError, MemoryError) as e:
        logger.error("Error removing degenerate triangles: %s", e)
        raise


# --- Refactored smoothing logic (reduced cognitive complexity) --- #

def _build_adjacency(vertices: np.ndarray, triangles: np.ndarray) -> Dict[int, set]:
    adjacency = {i: set() for i in range(len(vertices))}
    for tri in triangles:
        for i in tri:
            adjacency[i].update(j for j in tri if j != i)
    return adjacency


def _laplacian_smooth(vertices: np.ndarray, adjacency: Dict[int, set]) -> np.ndarray:
    smoothed = vertices.copy()
    for i, neighbors in adjacency.items():
        if neighbors:
            smoothed[i] = 0.5 * vertices[i] + 0.5 * np.mean(vertices[list(neighbors)], axis=0)
    return smoothed


def smooth_mesh(mesh_data: mesh.Mesh, iterations: int = 1) -> mesh.Mesh:
    """
    Apply simple Laplacian smoothing to mesh.
    """
    try:
        current_mesh = mesh.Mesh(np.zeros(mesh_data.vectors.shape[0], dtype=mesh.Mesh.dtype))
        current_mesh.vectors[:] = mesh_data.vectors.copy()

        for _ in range(iterations):
            vertices = current_mesh.vectors.reshape(-1, 3)
            triangles = np.arange(len(vertices)).reshape(-1, 3)
            adjacency = _build_adjacency(vertices, triangles)
            smoothed_vertices = _laplacian_smooth(vertices, adjacency)
            current_mesh.vectors[:] = smoothed_vertices.reshape(-1, 3, 3)

        return current_mesh

    except (ValueError, IndexError, MemoryError) as e:
        logger.error("Error smoothing mesh: %s", e)
        raise


def optimize_stl_file(file_path: str, optimization_level: str = "medium") -> mesh.Mesh:
    """
    Optimize STL file with specified level.
    """
    try:
        mesh_data = mesh.Mesh.from_file(file_path)
        original_triangles = len(mesh_data.vectors)

        logger.info("Original mesh: %d triangles", original_triangles)

        optimized_mesh = mesh_data
        if optimization_level in {"light", "medium", "aggressive"}:
            optimized_mesh = remove_degenerate_triangles(mesh_data)

        if optimization_level in {"medium", "aggressive"}:
            optimized_mesh = optimize_mesh_vertices(optimized_mesh, tolerance=0.01)

        if optimization_level == "aggressive":
            optimized_mesh = optimize_mesh_vertices(optimized_mesh, tolerance=0.1)
            optimized_mesh = smooth_mesh(optimized_mesh, iterations=1)

        optimized_triangles = len(optimized_mesh.vectors)
        reduction_percentage = ((original_triangles - optimized_triangles) / original_triangles) * 100
        logger.info(
            "Optimized mesh: %d triangles (%.1f%% reduction)",
            optimized_triangles,
            reduction_percentage,
        )

        return optimized_mesh

    except (ValueError, IndexError, MemoryError) as e:
        logger.error("Error optimizing STL file: %s", e)
        raise
