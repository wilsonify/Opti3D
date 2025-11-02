#!/usr/bin/env python
# coding: utf-8

"""
Integration tests for complete STL optimization workflow
"""

import io
import json
import os
import tempfile
import time
import unittest

import numpy as np
from stl import mesh

# Import the Flask app
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import app


class TestIntegrationWorkflow(unittest.TestCase):
    """Test complete end-to-end workflow using Flask test client"""

    @classmethod
    def setUpClass(cls):
        """Set up test app"""
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.app_context = app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        """Clean up test app"""
        cls.app_context.pop()

    def get_csrf_token(self):
        """Get CSRF token from the main page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        # Extract CSRF token from the page
        import re
        csrf_match = re.search(r'<meta[^>]*name=["\']csrf-token["\'][^>]*content=["\']([^"\']+)["\']', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        # Fallback to other patterns
        csrf_match = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        # Try to get from JavaScript
        csrf_match = re.search(r'csrf_token["\']?\s*:\s*["\']([^"\']+)["\']', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        self.fail("CSRF token not found in page")

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
        complex_vertices = []
        for i in range(20):
            triangle = np.random.rand(3, 3).astype(np.float32) * 10
            complex_vertices.append(triangle)
            # Add some duplicates
            if i % 5 == 0:
                complex_vertices.append(triangle + np.array([0.01, 0.01, 0.01], dtype=np.float32))

        complex_vertices = np.array(complex_vertices)
        complex_mesh = mesh.Mesh(np.zeros(complex_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(complex_vertices):
            complex_mesh.vectors[i] = verts
        
        complex_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        complex_mesh.save(complex_file.name)
        complex_file.close()
        self.test_files.append(complex_file.name)

    def test_complete_workflow_light_optimization(self):
        """Test complete workflow with light optimization"""
        # Get CSRF token
        csrf_token = self.get_csrf_token()
        
        # Step 1: Upload STL file
        with open(self.test_files[0], 'rb') as test_file:
            data = {'file': (test_file, 'test_cube.stl', 'application/octet-stream')}
            upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

        self.assertEqual(upload_response.status_code, 200)
        upload_data = json.loads(upload_response.data)

        # Verify upload response
        self.assertIn('file_id', upload_data)
        self.assertIn('analysis', upload_data)
        self.assertEqual(upload_data['analysis']['triangles'], 12)

        file_id = upload_data['file_id']

        # Step 2: Optimize file
        optimize_data = {
            'file_id': file_id,
            'level': 'light'
        }
        optimize_response = self.client.post(
            '/api/optimize',
            data=json.dumps(optimize_data),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )

        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = json.loads(optimize_response.data)

        # Verify optimization results
        self.assertIn('optimization_level', optimize_result)
        self.assertIn('original_size', optimize_result)
        self.assertIn('optimized_size', optimize_result)
        self.assertIn('compression_ratio', optimize_result)

        # Step 3: Download optimized file
        download_id = optimize_result['download_id']
        download_response = self.client.get(f'/api/download/{download_id}')
        self.assertEqual(download_response.status_code, 200)
        self.assertIn('Content-Disposition', download_response.headers)

        # Verify downloaded file is valid STL
        downloaded_content = download_response.data
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

        # Step 4: Cleanup
        cleanup_response = self.client.post(
            '/api/cleanup',
            data=json.dumps({'file_id': file_id}),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )
        self.assertEqual(cleanup_response.status_code, 200)

    def test_complete_workflow_medium_optimization(self):
        """Test complete workflow with medium optimization"""
        csrf_token = self.get_csrf_token()
        
        # Upload file with duplicates
        with open(self.test_files[1], 'rb') as test_file:
            data = {'file': (test_file, 'complex_mesh.stl', 'application/octet-stream')}
            upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

        self.assertEqual(upload_response.status_code, 200)
        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']

        # Optimize with medium level
        optimize_data = {
            'file_id': file_id,
            'level': 'medium'
        }
        optimize_response = self.client.post(
            '/api/optimize',
            data=json.dumps(optimize_data),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )

        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = json.loads(optimize_response.data)

        # Medium optimization should show some compression
        self.assertGreaterEqual(optimize_result['compression_ratio'], 0)

        # Download and verify
        download_id = optimize_result['download_id']
        download_response = self.client.get(f'/api/download/{download_id}')
        self.assertEqual(download_response.status_code, 200)

    def test_complete_workflow_aggressive_optimization(self):
        """Test complete workflow with aggressive optimization"""
        csrf_token = self.get_csrf_token()
        
        # Upload file
        with open(self.test_files[0], 'rb') as test_file:
            data = {'file': (test_file, 'test_cube.stl', 'application/octet-stream')}
            upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']

        # Optimize with aggressive level
        optimize_data = {
            'file_id': file_id,
            'level': 'aggressive'
        }
        optimize_response = self.client.post(
            '/api/optimize',
            data=json.dumps(optimize_data),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )

        self.assertEqual(optimize_response.status_code, 200)
        optimize_result = json.loads(optimize_response.data)

        # Verify aggressive optimization results
        self.assertIn('optimized_analysis', optimize_result)
        optimized_analysis = optimize_result['optimized_analysis']
        self.assertIn('triangles', optimized_analysis)
        self.assertIn('vertices', optimized_analysis)

    def test_error_handling_workflow(self):
        """Test error handling in workflow"""
        csrf_token = self.get_csrf_token()
        
        # Try to optimize without uploading
        optimize_response = self.client.post(
            '/api/optimize',
            data=json.dumps({'file_id': 'nonexistent_id', 'level': 'medium'}),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )
        self.assertEqual(optimize_response.status_code, 400)

        # Try to download nonexistent file
        download_response = self.client.get('/api/download/nonexistent.stl')
        self.assertEqual(download_response.status_code, 400)

        # Try to upload invalid file type
        data = {'file': (io.BytesIO(b'test content'), 'test.txt', 'text/plain')}
        upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})
        self.assertEqual(upload_response.status_code, 400)

    def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        # Since we're using the test client, true concurrency isn't tested
        # but we can test multiple sequential requests work correctly
        responses = []
        
        for i in range(3):
            csrf_token = self.get_csrf_token()
            with open(self.test_files[0], 'rb') as test_file:
                data = {'file': (test_file, f'concurrent_test_{i}.stl', 'application/octet-stream')}
                response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})
                responses.append(response)

        # All uploads should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('file_id', data)
            self.assertIn('analysis', data)

    def test_file_format_compatibility(self):
        """Test compatibility with different STL formats"""
        csrf_token = self.get_csrf_token()
        
        # Test binary STL (default)
        with open(self.test_files[0], 'rb') as test_file:
            data = {'file': (test_file, 'binary_test.stl', 'application/octet-stream')}
            response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})
        self.assertEqual(response.status_code, 200)

        # Create ASCII STL for testing
        ascii_stl_content = b"""solid test_cube
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

        ascii_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        ascii_file.write(ascii_stl_content)
        ascii_file.close()

        try:
            # Get fresh CSRF token for second upload
            csrf_token = self.get_csrf_token()
            with open(ascii_file.name, 'rb') as test_file:
                data = {'file': (test_file, 'ascii_test.stl', 'application/octet-stream')}
                response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})
            self.assertEqual(response.status_code, 200)

        finally:
            os.unlink(ascii_file.name)

    def test_large_file_handling(self):
        """Test handling of larger STL files"""
        csrf_token = self.get_csrf_token()
        
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
                data = {'file': (test_file, 'large_mesh.stl', 'application/octet-stream')}
                upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

            self.assertEqual(upload_response.status_code, 200)
            upload_data = json.loads(upload_response.data)
            self.assertEqual(upload_data['analysis']['triangles'], 100)

            # Optimize large file
            optimize_data = {
                'file_id': upload_data['file_id'],
                'level': 'medium'
            }
            optimize_response = self.client.post(
                '/api/optimize',
                data=json.dumps(optimize_data),
                content_type='application/json',
                headers={'X-CSRF-Token': csrf_token}
            )
            self.assertEqual(optimize_response.status_code, 200)

        finally:
            os.unlink(large_file.name)

    def test_performance_metrics(self):
        """Test performance and timing"""
        csrf_token = self.get_csrf_token()
        
        # Upload file
        start_time = time.time()
        with open(self.test_files[0], 'rb') as test_file:
            data = {'file': (test_file, 'performance_test.stl', 'application/octet-stream')}
            upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})
        upload_time = time.time() - start_time

        self.assertEqual(upload_response.status_code, 200)
        self.assertLess(upload_time, 5.0)  # Should upload within 5 seconds

        upload_data = json.loads(upload_response.data)

        # Optimize file
        start_time = time.time()
        optimize_data = {
            'file_id': upload_data['file_id'],
            'level': 'medium'
        }
        optimize_response = self.client.post(
            '/api/optimize',
            data=json.dumps(optimize_data),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )
        optimize_time = time.time() - start_time

        self.assertEqual(optimize_response.status_code, 200)
        self.assertLess(optimize_time, 10.0)  # Should optimize within 10 seconds

        # Download file
        optimize_result = json.loads(optimize_response.data)
        start_time = time.time()
        download_response = self.client.get(f'/api/download/{optimize_result['download_id']}')
        download_time = time.time() - start_time

        self.assertEqual(download_response.status_code, 200)
        self.assertLess(download_time, 5.0)  # Should download within 5 seconds


class TestRealWorldScenarios(unittest.TestCase):
    """Test real-world usage scenarios"""

    @classmethod
    def setUpClass(cls):
        """Set up test app"""
        app.config['TESTING'] = True
        cls.client = app.test_client()
        cls.app_context = app.app_context()
        cls.app_context.push()

    @classmethod
    def tearDownClass(cls):
        """Clean up test app"""
        cls.app_context.pop()

    def get_csrf_token(self):
        """Get CSRF token from the main page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        
        # Extract CSRF token from the page
        import re
        csrf_match = re.search(r'<meta[^>]*name=["\']csrf-token["\'][^>]*content=["\']([^"\']+)["\']', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        # Fallback to other patterns
        csrf_match = re.search(r'name="csrf_token"[^>]*value="([^"]+)"', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        # Try to get from JavaScript
        csrf_match = re.search(r'csrf_token["\']?\s*:\s*["\']([^"\']+)["\']', response.data.decode())
        if csrf_match:
            return csrf_match.group(1)
        
        self.fail("CSRF token not found in page")

    def test_3d_printing_preparation_workflow(self):
        """Test typical 3D printing preparation workflow"""
        csrf_token = self.get_csrf_token()
        
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
            # Upload file
            with open(test_file.name, 'rb') as f:
                data = {'file': (f, 'print_model.stl', 'application/octet-stream')}
                upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

            self.assertEqual(upload_response.status_code, 200)
            upload_data = json.loads(upload_response.data)

            # Step 2: User checks model analysis
            analysis = upload_data['analysis']
            self.assertIn('dimensions', analysis)
            self.assertIn('volume', analysis)

            # Step 3: User tries different optimization levels
            for level in ['light', 'medium', 'aggressive']:
                optimize_data = {
                    'file_id': upload_data['file_id'],
                    'level': level
                }
                optimize_response = self.client.post(
                    '/api/optimize',
                    data=json.dumps(optimize_data),
                    content_type='application/json',
                    headers={'X-CSRF-Token': csrf_token}
                )
                self.assertEqual(optimize_response.status_code, 200)

                result = json.loads(optimize_response.data)
                self.assertIn('compression_ratio', result)

                # Step 4: User downloads optimized model
                download_response = self.client.get(f'/api/download/{result['download_id']}')
                self.assertEqual(download_response.status_code, 200)

                # Verify downloaded model is valid
                temp_file = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
                temp_file.write(download_response.data)
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
                csrf_token = self.get_csrf_token()
                with open(test_file.name, 'rb') as f:
                    data = {'file': (f, f'batch_model_{i}.stl', 'application/octet-stream')}
                    upload_response = self.client.post('/api/upload', data=data, headers={'X-CSRF-Token': csrf_token})

                self.assertEqual(upload_response.status_code, 200)
                file_ids.append(json.loads(upload_response.data)['file_id'])

            finally:
                os.unlink(test_file.name)

        # Optimize all files
        optimization_results = []
        for file_id in file_ids:
            csrf_token = self.get_csrf_token()
            optimize_data = {'file_id': file_id, 'level': 'medium'}
            optimize_response = self.client.post(
                '/api/optimize',
                data=json.dumps(optimize_data),
                content_type='application/json',
                headers={'X-CSRF-Token': csrf_token}
            )
            self.assertEqual(optimize_response.status_code, 200)
            optimization_results.append(json.loads(optimize_response.data))

        # Verify all optimizations succeeded
        self.assertEqual(len(optimization_results), 5)
        for result in optimization_results:
            self.assertIn('compression_ratio', result)
            self.assertIn('download_id', result)

        # Cleanup all files
        csrf_token = self.get_csrf_token()
        cleanup_response = self.client.post(
            '/api/cleanup',
            data=json.dumps({}),
            content_type='application/json',
            headers={'X-CSRF-Token': csrf_token}
        )
        self.assertEqual(cleanup_response.status_code, 200)


if __name__ == '__main__':
    unittest.main()
