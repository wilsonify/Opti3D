#!/usr/bin/env python
# coding: utf-8

"""
Script to create sample STL files for testing
"""

import numpy as np
from stl import mesh
import os


def create_cube(filename, size=10.0):
    """Create a simple cube STL file"""
    vertices = np.array([
        # Bottom face
        [[0, 0, 0], [size, 0, 0], [size, size, 0]],
        [[0, 0, 0], [size, size, 0], [0, size, 0]],
        # Top face
        [[0, 0, size], [0, size, size], [size, size, size]],
        [[0, 0, size], [size, size, size], [size, 0, size]],
        # Front face
        [[0, 0, 0], [0, size, 0], [0, size, size]],
        [[0, 0, 0], [0, size, size], [0, 0, size]],
        # Back face
        [[size, 0, 0], [size, 0, size], [size, size, size]],
        [[size, 0, 0], [size, size, size], [size, size, 0]],
        # Left face
        [[0, 0, 0], [0, 0, size], [size, 0, size]],
        [[0, 0, 0], [size, 0, size], [size, 0, 0]],
        # Right face
        [[0, size, 0], [size, size, 0], [size, size, size]],
        [[0, size, 0], [size, size, size], [0, size, size]],
    ], dtype=np.float32)
    
    # Create mesh data with proper structure
    cube = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(vertices):
        cube.vectors[i] = verts
    cube.save(filename)
    print(f"Created cube: {filename}")


def create_sphere(filename, radius=5.0, segments=16):
    """Create a sphere STL file"""
    vertices = []
    
    # Generate sphere vertices using spherical coordinates
    for i in range(segments):
        theta1 = (i / segments) * 2 * np.pi
        theta2 = ((i + 1) / segments) * 2 * np.pi
        
        for j in range(segments // 2):
            phi1 = (j / (segments // 2)) * np.pi
            phi2 = ((j + 1) / (segments // 2)) * np.pi
            
            # Calculate vertices
            v1 = [
                radius * np.sin(phi1) * np.cos(theta1),
                radius * np.sin(phi1) * np.sin(theta1),
                radius * np.cos(phi1)
            ]
            v2 = [
                radius * np.sin(phi1) * np.cos(theta2),
                radius * np.sin(phi1) * np.sin(theta2),
                radius * np.cos(phi1)
            ]
            v3 = [
                radius * np.sin(phi2) * np.cos(theta2),
                radius * np.sin(phi2) * np.sin(theta2),
                radius * np.cos(phi2)
            ]
            v4 = [
                radius * np.sin(phi2) * np.cos(theta1),
                radius * np.sin(phi2) * np.sin(theta1),
                radius * np.cos(phi2)
            ]
            
            # Create two triangles for each quad
            vertices.append([v1, v2, v3])
            vertices.append([v1, v3, v4])
    
    sphere = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(vertices):
        sphere.vectors[i] = verts
    sphere.save(filename)
    print(f"Created sphere: {filename}")


def create_complex_mesh(filename):
    """Create a complex mesh with duplicates and degenerate triangles"""
    # Start with a cube
    base_vertices = np.array([
        [[0, 0, 0], [10, 0, 0], [10, 10, 0]],
        [[0, 0, 0], [10, 10, 0], [0, 10, 0]],
        [[0, 0, 10], [0, 10, 10], [10, 10, 10]],
        [[0, 0, 10], [10, 10, 10], [10, 0, 10]],
        [[0, 0, 0], [0, 10, 0], [0, 10, 10]],
        [[0, 0, 0], [0, 10, 10], [0, 0, 10]],
        [[10, 0, 0], [10, 0, 10], [10, 10, 10]],
        [[10, 0, 0], [10, 10, 10], [10, 10, 0]],
        [[0, 0, 0], [0, 0, 10], [10, 0, 10]],
        [[0, 0, 0], [10, 0, 10], [10, 0, 0]],
        [[0, 10, 0], [10, 10, 0], [10, 10, 10]],
        [[0, 10, 0], [10, 10, 10], [0, 10, 10]],
    ], dtype=np.float32)
    
    # Add duplicate triangles
    duplicate_vertices = np.array([
        [[0, 0, 0], [10, 0, 0], [10, 10, 0]],  # Duplicate
        [[0, 0, 0], [10, 10, 0], [0, 10, 0]],  # Duplicate
    ], dtype=np.float32)
    
    # Add degenerate triangles
    degenerate_vertices = np.array([
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],  # Zero area
        [[5, 5, 5], [5, 5, 5], [5, 5, 5]],  # Zero area
    ], dtype=np.float32)
    
    # Add some random triangles
    random_vertices = np.random.rand(10, 3, 3).astype(np.float32) * 15
    
    # Combine all vertices
    all_vertices = np.vstack([base_vertices, duplicate_vertices, degenerate_vertices, random_vertices])
    
    complex_mesh = mesh.Mesh(np.zeros(all_vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(all_vertices):
        complex_mesh.vectors[i] = verts
    complex_mesh.save(filename)
    print(f"Created complex mesh: {filename}")


def create_pyramid(filename, base_size=10.0, height=8.0):
    """Create a pyramid STL file"""
    half_base = base_size / 2.0
    
    vertices = np.array([
        # Base triangles
        [[-half_base, -half_base, 0], [half_base, -half_base, 0], [half_base, half_base, 0]],
        [[-half_base, -half_base, 0], [half_base, half_base, 0], [-half_base, half_base, 0]],
        # Side faces
        [[-half_base, -half_base, 0], [half_base, -half_base, 0], [0, 0, height]],
        [[half_base, -half_base, 0], [half_base, half_base, 0], [0, 0, height]],
        [[half_base, half_base, 0], [-half_base, half_base, 0], [0, 0, height]],
        [[-half_base, half_base, 0], [-half_base, -half_base, 0], [0, 0, height]],
    ], dtype=np.float32)
    
    pyramid = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(vertices):
        pyramid.vectors[i] = verts
    pyramid.save(filename)
    print(f"Created pyramid: {filename}")


def create_cylinder(filename, radius=5.0, height=10.0, segments=16):
    """Create a cylinder STL file"""
    vertices = []
    
    # Create side faces
    for i in range(segments):
        angle1 = (i / segments) * 2 * np.pi
        angle2 = ((i + 1) / segments) * 2 * np.pi
        
        # Bottom vertices
        v1 = [radius * np.cos(angle1), radius * np.sin(angle1), 0]
        v2 = [radius * np.cos(angle2), radius * np.sin(angle2), 0]
        
        # Top vertices
        v3 = [radius * np.cos(angle2), radius * np.sin(angle2), height]
        v4 = [radius * np.cos(angle1), radius * np.sin(angle1), height]
        
        # Side face triangles
        vertices.append([v1, v2, v3])
        vertices.append([v1, v3, v4])
    
    # Create bottom face
    center_bottom = [0, 0, 0]
    for i in range(segments):
        angle1 = (i / segments) * 2 * np.pi
        angle2 = ((i + 1) / segments) * 2 * np.pi
        
        v1 = [radius * np.cos(angle1), radius * np.sin(angle1), 0]
        v2 = [radius * np.cos(angle2), radius * np.sin(angle2), 0]
        
        vertices.append([center_bottom, v2, v1])
    
    # Create top face
    center_top = [0, 0, height]
    for i in range(segments):
        angle1 = (i / segments) * 2 * np.pi
        angle2 = ((i + 1) / segments) * 2 * np.pi
        
        v1 = [radius * np.cos(angle1), radius * np.sin(angle1), height]
        v2 = [radius * np.cos(angle2), radius * np.sin(angle2), height]
        
        vertices.append([center_top, v1, v2])
    
    cylinder = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(vertices):
        cylinder.vectors[i] = verts
    cylinder.save(filename)
    print(f"Created cylinder: {filename}")


def create_torus(filename, major_radius=8.0, minor_radius=3.0, major_segments=16, minor_segments=8):
    """Create a torus STL file"""
    vertices = []
    
    for i in range(major_segments):
        theta1 = (i / major_segments) * 2 * np.pi
        theta2 = ((i + 1) / major_segments) * 2 * np.pi
        
        for j in range(minor_segments):
            phi1 = (j / minor_segments) * 2 * np.pi
            phi2 = ((j + 1) / minor_segments) * 2 * np.pi
            
            # Calculate vertices
            v1 = [
                (major_radius + minor_radius * np.cos(phi1)) * np.cos(theta1),
                (major_radius + minor_radius * np.cos(phi1)) * np.sin(theta1),
                minor_radius * np.sin(phi1)
            ]
            v2 = [
                (major_radius + minor_radius * np.cos(phi1)) * np.cos(theta2),
                (major_radius + minor_radius * np.cos(phi1)) * np.sin(theta2),
                minor_radius * np.sin(phi1)
            ]
            v3 = [
                (major_radius + minor_radius * np.cos(phi2)) * np.cos(theta2),
                (major_radius + minor_radius * np.cos(phi2)) * np.sin(theta2),
                minor_radius * np.sin(phi2)
            ]
            v4 = [
                (major_radius + minor_radius * np.cos(phi2)) * np.cos(theta1),
                (major_radius + minor_radius * np.cos(phi2)) * np.sin(theta1),
                minor_radius * np.sin(phi2)
            ]
            
            # Create two triangles for each quad
            vertices.append([v1, v2, v3])
            vertices.append([v1, v3, v4])
    
    torus = mesh.Mesh(np.zeros(len(vertices), dtype=mesh.Mesh.dtype))
    for i, verts in enumerate(vertices):
        torus.vectors[i] = verts
    torus.save(filename)
    print(f"Created torus: {filename}")


def main():
    """Create all test files"""
    # Create test_files directory if it doesn't exist
    test_dir = 'test_files'
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
    
    print("Creating test STL files...")
    
    # Create various test files
    create_cube(os.path.join(test_dir, 'test_cube.stl'))
    create_cube(os.path.join(test_dir, 'small_cube.stl'), size=2.0)
    create_cube(os.path.join(test_dir, 'large_cube.stl'), size=20.0)
    
    create_sphere(os.path.join(test_dir, 'test_sphere.stl'))
    create_sphere(os.path.join(test_dir, 'detailed_sphere.stl'), segments=32)
    
    create_pyramid(os.path.join(test_dir, 'test_pyramid.stl'))
    create_cylinder(os.path.join(test_dir, 'test_cylinder.stl'))
    create_torus(os.path.join(test_dir, 'test_torus.stl'))
    
    create_complex_mesh(os.path.join(test_dir, 'complex_mesh.stl'))
    create_complex_mesh(os.path.join(test_dir, 'very_complex_mesh.stl'))
    
    print(f"\nAll test files created in '{test_dir}' directory.")
    print("Files created:")
    for filename in os.listdir(test_dir):
        if filename.endswith('.stl'):
            filepath = os.path.join(test_dir, filename)
            size = os.path.getsize(filepath)
            print(f"  {filename} ({size} bytes)")


if __name__ == '__main__':
    main()
