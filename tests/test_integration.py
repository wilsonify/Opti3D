#!/usr/bin/env python
# coding: utf-8

"""
Integration tests for complete STL optimization workflow
"""

import unittest
import json
import tempfile
import os
import time
import requests
from stl import mesh
import numpy as np
from threading import Thread
import subprocess
import signal


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow"""

    @classmethod
    def setUpClass(cls):
        """Set up test server"""
        # Start Flask app in background
        cls.server_process = subprocess.Popen(
            ['python', 'app.py'],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for server to start
        time.sleep(3)
        
        cls.base_url = 'http://localhost:5000'
        cls.session = requests.Session()

    @classmethod
    def tearDownClass(cls):
        """Clean up test server"""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

    def setUp(self):
        """Set up test fixtures"""
        # Create test STL files
        self.create_test_files()

    def tearDown(self):
        """Clean up test fixtures"""
        # Clean up test files
        for file_path in self.test_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    def create_test_files(self):
        """Create various test STL files"""
        self.test_files = []
        
        # Create a simple cube
        cube_vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
            [[0, 0, 0], [0, 1, 0], [0, 1, 1]],
            [[0, 0, 0], [0, 1, 1], [0, 0, 1]],
            [[1, 0, 0], [1, 0, 1], [1, 1, 1]],
            [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
            [[0, 0, 0], [0, 0, 1], [1, 0, 1]],
            [[0, 0, 0], [1, 0, 1], [1, 0, 0]],
            [[0, 1, 0], [1, 1, 0], [1, 1, 1]],
            [[0, 1, 0], [1, 1, 1], [0, 1, 1]],
        ], dtype=np.float32)
        
        # Create mesh properly using numpy-stl format
        cube_mesh = mesh.Mesh(np.zeros(cube_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(cube_vertices):
            cube_mesh.vectors[i] = verts
        cube_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        cube_mesh.save(cube_file.name)
        cube_file.close()
        self.test_files.append(cube_file.name)

        # Create a more complex mesh with duplicates
        complex_vertices = np.array([
            # Cube faces
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
            # Duplicate vertices
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            # Some degenerate triangles
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
        ], dtype=np.float32)
        
        # Create mesh properly using numpy-stl format
        complex_mesh = mesh.Mesh(np.zeros(complex_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(complex_vertices):
            complex_mesh.vectors[i] = verts
        complex_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        complex_mesh.save(complex_file.name)
        complex_file.close()
        self.test_files.append(complex_file.name)

    def test_complete_workflow_light_optimization(self):
        """Test complete workflow with light optimization"""
        # Step 1: Access main page and get CSRF token
        response = self.session.get(self.base_url)
        self.assertEqual(response.status_code, 200)
        self.assertIn('Opti3D', response.text)
        
        # Extract CSRF token from the page
        csrf_token = None
        import re
        # Look for meta tag first
        csrf_match = re.search(r'<meta[^>]*name=["\']csrf-token["\'][^>]*content=["\']([^"\']+)["\']', response.text)
        if csrf_match:
            csrf_token = csrf_match.group(1)
        else:
            # Fallback to other patterns
            csrf_match = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', response.text)
            if csrf_match:
                csrf_token = csrf_match.group(1)
            else:
                # Try to get from JavaScript
                csrf_match = re.search(r'csrf_token["\']?\s*:\s*["\']([^"\']+)["\']', response.text)
                if csrf_match:
                    csrf_token = csrf_match.group(1)
        
        self.assertIsNotNone(csrf_token, "CSRF token not found in page")

        # Step 2: Upload STL file with CSRF token
        with open(self.test_files[0], 'rb') as test_file:
            files = {'file': ('test_cube.stl', test_file, 'application/octet-stream')}
            headers = {'X-CSRF-Token': csrf_token}
            upload_response = self.session.post(f"{self.base_url}/api/upload", files=files, headers=headers)
        
        self.assertEqual(upload_response.status_code, 200)
        upload_data = upload_response.json()
        
        # Verify upload response
        self.assertIn('file_id', upload_data)
        self.assertIn('analysis', upload_data)
        self.assertEqual(upload_data['analysis']['triangles'], 12)
        
        file_id = upload_data['file_id']

        # Step 3: Optimize file
        optimize_data = {
            'file_id': file_id,
            'level': 'light'
        }
        optimize_response = self.session.post(
            f"{self.base_url}/api/optimize",
            json=optimize_data,
            headers={'X-CSRF-Token': csrf_token}
        )
        
        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = optimize_response.json()
        
        # Verify optimization results
        self.assertIn('optimization_level', optimize_result)
        self.assertIn('original_size', optimize_result)
        self.assertIn('optimized_size', optimize_result)
        self.assertIn('compression_ratio', optimize_result)
        self.assertIn('download_id', optimize_result)
        
        # Step 4: Download optimized file
        download_id = optimize_result['download_id']
        download_response = self.session.get(f"{self.base_url}/api/download/{download_id}")
        
        self.assertEqual(download_response.status_code, 200)
        self.assertIn('Content-Disposition', download_response.headers)
        
        # Verify downloaded file is valid STL
        downloaded_content = download_response.content
        self.assertGreater(len(downloaded_content), 0)
        
        # Save downloaded file temporarily and verify it can be loaded
        temp_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        temp_file.write(downloaded_content)
        temp_file.close()
        
        try:
            loaded_mesh = mesh.Mesh.from_file(temp_file.name)
            self.assertGreater(len(loaded_mesh.vectors), 0)
        finally:
            os.unlink(temp_file.name)

        # Step 5: Cleanup
        cleanup_response = self.session.post(
            f"{self.base_url}/api/cleanup",
            json={'file_id': file_id},
            headers={'X-CSRF-Token': csrf_token}
        )
        self.assertEqual(cleanup_response.status_code, 200)

    def test_complete_workflow_medium_optimization(self):
        """Test complete workflow with medium optimization"""
        # Upload file with duplicates
        with open(self.test_files[1], 'rb') as test_file:
            files = {'file': ('complex_mesh.stl', test_file, 'application/octet-stream')}
            upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
        
        self.assertEqual(upload_response.status_code, 200)
        upload_data = upload_response.json()
        file_id = upload_data['file_id']

        # Optimize with medium level
        optimize_data = {
            'file_id': file_id,
            'level': 'medium'
        }
        optimize_response = self.session.post(
            f"{self.base_url}/api/optimize",
            json=optimize_data
        )
        
        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = optimize_response.json()
        
        # Medium optimization should show some compression
        self.assertGreaterEqual(optimize_result['compression_ratio'], 0)
        
        # Download and verify
        download_id = optimize_result['download_id']
        download_response = self.session.get(f"{self.base_url}/api/download/{download_id}")
        self.assertEqual(download_response.status_code, 200)

    def test_complete_workflow_aggressive_optimization(self):
        """Test complete workflow with aggressive optimization"""
        # Upload file
        with open(self.test_files[0], 'rb') as test_file:
            files = {'file': ('test_cube.stl', test_file, 'application/octet-stream')}
            upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
        
        upload_data = upload_response.json()
        file_id = upload_data['file_id']

        # Optimize with aggressive level
        optimize_data = {
            'file_id': file_id,
            'level': 'aggressive'
        }
        optimize_response = self.session.post(
            f"{self.base_url}/api/optimize",
            json=optimize_data
        )
        
        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = optimize_response.json()
        
        # Verify aggressive optimization results
        self.assertIn('optimized_analysis', optimize_result)
        optimized_analysis = optimize_result['optimized_analysis']
        self.assertIn('triangles', optimized_analysis)
        self.assertIn('vertices', optimized_analysis)

    def test_error_handling_workflow(self):
        """Test error handling in workflow"""
        # Try to optimize without uploading
        optimize_response = self.session.post(
            f"{self.base_url}/api/optimize",
            json={'file_id': 'nonexistent_id', 'level': 'medium'}
        )
        self.assertEqual(optimize_response.status_code, 404)
        
        # Try to download nonexistent file
        download_response = self.session.get(f"{self.base_url}/api/download/nonexistent.stl")
        self.assertEqual(download_response.status_code, 404)
        
        # Try to upload invalid file type
        files = {'file': ('test.txt', b'test content', 'text/plain')}
        upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
        self.assertEqual(upload_response.status_code, 400)

    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        def upload_file(filename):
            with open(self.test_files[0], 'rb') as test_file:
                files = {'file': (filename, test_file, 'application/octet-stream')}
                response = self.session.post(f"{self.base_url}/api/upload", files=files)
            return response
        
        # Upload multiple files concurrently
        threads = []
        responses = []
        
        for i in range(3):
            thread = Thread(target=lambda i=i: responses.append(upload_file(f'concurrent_test_{i}.stl')))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        # All uploads should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn('file_id', data)

    def test_large_file_handling(self):
        """Test handling of larger STL files"""
        # Create a larger mesh
        large_vertices = []
        for i in range(100):  # Create 100 triangles
            triangle = np.random.rand(3, 3).astype(np.float32) * 10
            large_vertices.append(triangle)
        
        large_vertices = np.array(large_vertices)
        # Create mesh properly using numpy-stl format
        large_mesh = mesh.Mesh(np.zeros(large_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(large_vertices):
            large_mesh.vectors[i] = verts
        
        large_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        large_mesh.save(large_file.name)
        large_file.close()
        
        try:
            # Upload large file
            with open(large_file.name, 'rb') as test_file:
                files = {'file': ('large_mesh.stl', test_file, 'application/octet-stream')}
                upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
            
            self.assertEqual(upload_response.status_code, 200)
            upload_data = upload_response.json()
            self.assertEqual(upload_data['analysis']['triangles'], 100)
            
            # Optimize large file
            optimize_response = self.session.post(
                f"{self.base_url}/api/optimize",
                json={'file_id': upload_data['file_id'], 'level': 'medium'}
            )
            self.assertEqual(optimize_response.status_code, 200)
            
        finally:
            os.unlink(large_file.name)

    def test_file_format_compatibility(self):
        """Test compatibility with different STL formats"""
        # Test binary STL (default)
        with open(self.test_files[0], 'rb') as test_file:
            files = {'file': ('binary_test.stl', test_file, 'application/octet-stream')}
            response = self.session.post(f"{self.base_url}/api/upload", files=files)
        self.assertEqual(response.status_code, 200)
        
        # Create ASCII STL for testing
        ascii_stl_content = """solid test_cube
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 0.0 0.0
      vertex 1.0 1.0 0.0
    endloop
  endfacet
  facet normal 0.0 0.0 1.0
    outer loop
      vertex 0.0 0.0 0.0
      vertex 1.0 1.0 0.0
      vertex 0.0 1.0 0.0
    endloop
  endfacet
endsolid test_cube
"""
        
        ascii_file = tempfile.NamedTemporaryFile(mode='w', suffix='.stl', delete=False)
        ascii_file.write(ascii_stl_content)
        ascii_file.close()
        
        try:
            with open(ascii_file.name, 'rb') as test_file:
                files = {'file': ('ascii_test.stl', test_file, 'application/octet-stream')}
                response = self.session.post(f"{self.base_url}/api/upload", files=files)
            self.assertEqual(response.status_code, 200)
            
        finally:
            os.unlink(ascii_file.name)

    def test_performance_metrics(self):
        """Test performance and timing"""
        # Upload file
        start_time = time.time()
        with open(self.test_files[0], 'rb') as test_file:
            files = {'file': ('performance_test.stl', test_file, 'application/octet-stream')}
            upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
        upload_time = time.time() - start_time
        
        self.assertEqual(upload_response.status_code, 200)
        self.assertLess(upload_time, 5.0)  # Should upload within 5 seconds
        
        upload_data = upload_response.json()
        
        # Optimize file
        start_time = time.time()
        optimize_response = self.session.post(
            f"{self.base_url}/api/optimize",
            json={'file_id': upload_data['file_id'], 'level': 'medium'}
        )
        optimize_time = time.time() - start_time
        
        self.assertEqual(optimize_response.status_code, 200)
        self.assertLess(optimize_time, 10.0)  # Should optimize within 10 seconds
        
        # Download file
        optimize_result = optimize_response.json()
        start_time = time.time()
        download_response = self.session.get(f"{self.base_url}/api/download/{optimize_result['download_id']}")
        download_time = time.time() - start_time
        
        self.assertEqual(download_response.status_code, 200)
        self.assertLess(download_time, 5.0)  # Should download within 5 seconds


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios"""

    @classmethod
    def setUpClass(cls):
        """Set up test server"""
        cls.server_process = subprocess.Popen(
            ['python', 'app.py'],
            cwd=os.path.dirname(__file__),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)
        cls.base_url = 'http://localhost:5000'
        cls.session = requests.Session()

    @classmethod
    def tearDownClass(cls):
        """Clean up test server"""
        if cls.server_process:
            cls.server_process.terminate()
            cls.server_process.wait()

    def test_3d_printing_preparation_workflow(self):
        """Test typical 3D printing preparation workflow"""
        # Simulate a user preparing a model for 3D printing
        
        # Step 1: User uploads a model
        vertices = np.array([
            [[0, 0, 0], [10, 0, 0], [10, 10, 0]],
            [[0, 0, 0], [10, 10, 0], [0, 10, 0]],
            [[0, 0, 5], [0, 10, 5], [10, 10, 5]],
            [[0, 0, 5], [10, 10, 5], [10, 0, 5]],
        ], dtype=np.float32)
        
        # Create mesh properly using numpy-stl format
        test_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(vertices):
            test_mesh.vectors[i] = verts
        test_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        test_mesh.save(test_file.name)
        test_file.close()
        
        try:
            with open(test_file.name, 'rb') as f:
                files = {'file': ('print_model.stl', f, 'application/octet-stream')}
                upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
            
            self.assertEqual(upload_response.status_code, 200)
            upload_data = upload_response.json()
            
            # Step 2: User checks model analysis
            analysis = upload_data['analysis']
            self.assertIn('dimensions', analysis)
            self.assertIn('volume', analysis)
            
            # Step 3: User tries different optimization levels
            for level in ['light', 'medium', 'aggressive']:
                optimize_response = self.session.post(
                    f"{self.base_url}/api/optimize",
                    json={'file_id': upload_data['file_id'], 'level': level}
                )
                self.assertEqual(optimize_response.status_code, 200)
                
                result = optimize_response.json()
                self.assertIn('compression_ratio', result)
                
                # Step 4: User downloads optimized model
                download_response = self.session.get(
                    f"{self.base_url}/api/download/{result['download_id']}"
                )
                self.assertEqual(download_response.status_code, 200)
                
                # Verify downloaded model is valid
                temp_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
                temp_file.write(download_response.content)
                temp_file.close()
                
                try:
                    loaded_mesh = mesh.Mesh.from_file(temp_file.name)
                    self.assertGreater(len(loaded_mesh.vectors), 0)
                finally:
                    os.unlink(temp_file.name)
        
        finally:
            os.unlink(test_file.name)

    def test_batch_processing_simulation(self):
        """Test simulated batch processing of multiple files"""
        file_ids = []
        
        # Upload multiple files
        for i in range(5):
            vertices = np.random.rand(20, 3, 3).astype(np.float32) * 10
            # Create mesh properly using numpy-stl format
            test_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
            for j, verts in enumerate(vertices):
                test_mesh.vectors[j] = verts
            test_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
            test_mesh.save(test_file.name)
            test_file.close()
            
            try:
                with open(test_file.name, 'rb') as f:
                    files = {'file': (f'batch_model_{i}.stl', f, 'application/octet-stream')}
                    upload_response = self.session.post(f"{self.base_url}/api/upload", files=files)
                
                self.assertEqual(upload_response.status_code, 200)
                file_ids.append(upload_response.json()['file_id'])
            
            finally:
                os.unlink(test_file.name)
        
        # Optimize all files
        optimization_results = []
        for file_id in file_ids:
            optimize_response = self.session.post(
                f"{self.base_url}/api/optimize",
                json={'file_id': file_id, 'level': 'medium'}
            )
            self.assertEqual(optimize_response.status_code, 200)
            optimization_results.append(optimize_response.json())
        
        # Verify all optimizations succeeded
        self.assertEqual(len(optimization_results), 5)
        for result in optimization_results:
            self.assertIn('compression_ratio', result)
            self.assertIn('download_id', result)
        
        # Cleanup all files
        cleanup_response = self.session.post(f"{self.base_url}/api/cleanup", json={})
        self.assertEqual(cleanup_response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
