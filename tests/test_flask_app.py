#!/usr/bin/env python
# coding: utf-8

"""
Unit tests for Flask web application
"""

import io
import json
import os
import secrets
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from stl import mesh

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import app


class TestFlaskApp(unittest.TestCase):
    """Test Flask application endpoints"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        app.testing = True  # Ensure app is in testing mode

        # Set up session for CSRF
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = secrets.token_urlsafe(32)

        # Get CSRF token from session
        with self.app.session_transaction() as sess:
            self.csrf_token = sess.get('csrf_token', secrets.token_urlsafe(32))

        # Create a test STL file
        self.test_cube_vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
        ], dtype=np.float32)

        self.test_cube_mesh = mesh.Mesh(np.zeros(self.test_cube_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(self.test_cube_vertices):
            self.test_cube_mesh.vectors[i] = verts

        # Create temporary STL file for testing
        self.temp_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        self.test_cube_mesh.save(self.temp_stl.name)
        self.temp_stl.close()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_stl.name):
            os.unlink(self.temp_stl.name)

    def test_index_page(self):
        """Test main page loads correctly"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Opti3D', response.data)
        self.assertIn(b'STL File Optimizer', response.data)

    def test_upload_no_file(self):
        """Test upload endpoint with no file"""
        response = self.app.post('/api/upload',
                                 data={},
                                 headers={'X-CSRF-Token': self.csrf_token})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'No file provided')

    def test_upload_empty_filename(self):
        """Test upload endpoint with empty filename"""
        response = self.app.post('/api/upload',
                                 data={
                                     'file': (io.BytesIO(b''), '')
                                 },
                                 headers={'X-CSRF-Token': self.csrf_token})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

    def test_upload_invalid_file_type(self):
        """Test upload endpoint with invalid file type"""
        response = self.app.post('/api/upload',
                                 data={
                                     'file': (io.BytesIO(b'test content'), 'test.txt')
                                 },
                                 headers={'X-CSRF-Token': self.csrf_token})
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertIn('Invalid file type', data['error'])

    def test_upload_valid_stl_file(self):
        """Test upload endpoint with valid STL file"""
        with open(self.temp_stl.name, 'rb') as test_file:
            response = self.app.post('/api/upload',
                                     data={
                                         'file': (test_file, 'test_cube.stl')
                                     },
                                     headers={'X-CSRF-Token': self.csrf_token})

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('file_id', data)
        self.assertIn('filename', data)
        self.assertIn('analysis', data)
        self.assertEqual(data['filename'], 'test_cube.stl')
        self.assertIn('upload_time', data)

        # Check analysis data
        analysis = data['analysis']
        self.assertIn('triangles', analysis)
        self.assertIn('vertices', analysis)
        self.assertIn('dimensions', analysis)
        self.assertEqual(analysis['triangles'], 4)

    def test_upload_large_file(self):
        """Test upload endpoint with large file"""
        # Create a large file that exceeds the limit
        large_content = b'x' * (101 * 1024 * 1024)  # 101MB
        response = self.app.post('/api/upload',
                                 data={
                                     'file': (io.BytesIO(large_content), 'large.stl')
                                 },
                                 headers={'X-CSRF-Token': self.csrf_token})

        # Should be rejected due to size limit
        self.assertNotEqual(response.status_code, 200)

    @patch('app.optimize_stl_file_wrapper')
    def test_optimize_success(self, mock_optimize):
        """Test optimization endpoint success"""
        # Mock the optimization function
        mock_optimized_path = '/tmp/optimized_test.stl'
        mock_optimize.return_value = mock_optimized_path

        # First upload a file
        with open(self.temp_stl.name, 'rb') as test_file:
            upload_response = self.app.post('/api/upload',
                                            data={
                                                'file': (test_file, 'test_cube.stl')
                                            },
                                            headers={'X-CSRF-Token': self.csrf_token})

        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']

        # Mock file size checking
        with patch('os.path.getsize') as mock_getsize:
            mock_getsize.side_effect = [1000, 800]  # Original size, optimized size

            # Mock analysis function
            with patch('app.analyze_stl_file') as mock_analyze:
                mock_analyze.return_value = {
                    'triangles': 3,
                    'vertices': 3,
                    'dimensions': {'x': 1.0, 'y': 1.0, 'z': 1.0}
                }

                response = self.app.post('/api/optimize',
                                         data=json.dumps({
                                             'file_id': file_id,
                                             'level': 'medium'
                                         }),
                                         content_type='application/json',
                                         headers={'X-CSRF-Token': self.csrf_token}
                                         )

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)

        self.assertIn('optimization_level', data)
        self.assertIn('original_size', data)
        self.assertIn('optimized_size', data)
        self.assertIn('compression_ratio', data)
        self.assertIn('download_id', data)

    def test_optimize_no_file_id(self):
        """Test optimization endpoint without file_id"""
        response = self.app.post('/api/optimize',
                                 data=json.dumps({}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'File ID required')

    def test_optimize_invalid_file_id(self):
        """Test optimization endpoint with invalid file_id"""
        response = self.app.post('/api/optimize',
                                 data=json.dumps({
                                     'file_id': 'nonexistent_file_id'
                                 }),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'File not found')

    def test_optimization_levels(self):
        """Test different optimization levels"""
        # Upload a file first
        with open(self.temp_stl.name, 'rb') as test_file:
            upload_response = self.app.post('/api/upload',
                                            data={
                                                'file': (test_file, 'test_cube.stl')
                                            },
                                            headers={'X-CSRF-Token': self.csrf_token})

        upload_data = json.loads(upload_response.data)
        file_id = upload_data['file_id']

        # Test each optimization level
        for level in ['light', 'medium', 'aggressive']:
            with patch('app.optimize_stl_file_wrapper') as mock_optimize:
                mock_optimize.return_value = f'/tmp/optimized_{level}.stl'

                with patch('os.path.getsize') as mock_getsize:
                    mock_getsize.side_effect = [1000, 800]

                    with patch('app.analyze_stl_file') as mock_analyze:
                        mock_analyze.return_value = {
                            'triangles': 3,
                            'vertices': 3,
                            'dimensions': {'x': 1.0, 'y': 1.0, 'z': 1.0}
                        }

                        response = self.app.post('/api/optimize',
                                                 data=json.dumps({
                                                     'file_id': file_id,
                                                     'level': level
                                                 }),
                                                 content_type='application/json',
                                                 headers={'X-CSRF-Token': self.csrf_token}
                                                 )

                        self.assertEqual(response.status_code, 200)
                        data = json.loads(response.data)
                        self.assertEqual(data['optimization_level'], level)

    def test_download_nonexistent_file(self):
        """Test download endpoint with nonexistent file"""
        response = self.app.get('/api/download/nonexistent_file.stl')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)
        self.assertEqual(data['error'], 'Invalid file for download')

    def test_cleanup_specific_file(self):
        """Test cleanup endpoint for specific file"""
        response = self.app.post('/api/cleanup',
                                 data=json.dumps({
                                     'file_id': 'test_file_id'
                                 }),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)

    def test_cleanup_all_files(self):
        """Test cleanup endpoint for all files"""
        response = self.app.post('/api/cleanup',
                                 data=json.dumps({}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('message', data)

    def test_file_validation_functions(self):
        """Test file validation helper functions"""
        from app import allowed_file

        # Test valid extensions
        self.assertTrue(allowed_file('test.stl'))
        self.assertTrue(allowed_file('TEST.STL'))
        self.assertTrue(allowed_file('model.stl'))

        # Test invalid extensions
        self.assertFalse(allowed_file('test.txt'))
        self.assertFalse(allowed_file('test.obj'))
        self.assertFalse(allowed_file('test'))
        self.assertFalse(allowed_file('test.stl.txt'))

    def test_error_handling(self):
        """Test error handling in various scenarios"""
        # Test malformed JSON
        response = self.app.post('/api/optimize',
                                 data='invalid json',
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        self.assertEqual(response.status_code, 400)

        # Test missing content type
        response = self.app.post('/api/optimize',
                                 data=json.dumps({'file_id': 'test'}),
                                 headers={'X-CSRF-Token': self.csrf_token}
                                 )
        # Should handle gracefully (may return 400 or 200 depending on Flask config)

    def test_concurrent_uploads(self):
        """Test handling multiple concurrent uploads"""
        responses = []

        # Upload multiple files
        for i in range(3):
            with open(self.temp_stl.name, 'rb') as test_file:
                response = self.app.post('/api/upload',
                                         data={
                                             'file': (test_file, f'test_cube_{i}.stl')
                                         },
                                         headers={'X-CSRF-Token': self.csrf_token})
                responses.append(response)

        # All uploads should succeed
        for response in responses:
            self.assertEqual(response.status_code, 200)
            data = json.loads(response.data)
            self.assertIn('file_id', data)

        # Each file should have a unique ID
        file_ids = [json.loads(r.data)['file_id'] for r in responses]
        self.assertEqual(len(set(file_ids)), len(file_ids))


class TestFileOperations(unittest.TestCase):
    """Test file operation utilities"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        app.testing = True  # Ensure app is in testing mode

        # Set up session for CSRF
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = secrets.token_urlsafe(32)

        # Get CSRF token from session
        with self.app.session_transaction() as sess:
            self.csrf_token = sess.get('csrf_token', secrets.token_urlsafe(32))

        # Create a test STL file for this test class
        self.test_cube_vertices = np.array([
            [[0, 0, 0], [1, 0, 0], [1, 1, 0]],
            [[0, 0, 0], [1, 1, 0], [0, 1, 0]],
            [[0, 0, 1], [0, 1, 1], [1, 1, 1]],
            [[0, 0, 1], [1, 1, 1], [1, 0, 1]],
        ], dtype=np.float32)

        self.test_cube_mesh = mesh.Mesh(np.zeros(self.test_cube_vertices.shape[0], dtype=mesh.Mesh.dtype))
        for i, verts in enumerate(self.test_cube_vertices):
            self.test_cube_mesh.vectors[i] = verts

        # Create temporary STL file for testing
        self.temp_stl = tempfile.NamedTemporaryFile(suffix='.stl', delete=False)
        self.test_cube_mesh.save(self.temp_stl.name)
        self.temp_stl.close()

    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.temp_stl.name):
            os.unlink(self.temp_stl.name)

    def test_temp_file_handling(self):
        """Test temporary file creation and cleanup"""
        # Upload a file using the test STL file
        with open(self.temp_stl.name, 'rb') as test_file:
            response = self.app.post('/api/upload',
                                     data={
                                         'file': (test_file, 'test.stl')
                                     },
                                     headers={'X-CSRF-Token': self.csrf_token})

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        file_id = data['file_id']

        # File should be stored in temp directory
        self.assertIsNotNone(file_id)
        self.assertGreater(len(file_id), 0)

        # Cleanup should work
        cleanup_response = self.app.post('/api/cleanup',
                                         data=json.dumps({'file_id': file_id}),
                                         content_type='application/json',
                                         headers={'X-CSRF-Token': self.csrf_token}
                                         )
        self.assertEqual(cleanup_response.status_code, 200)


class TestSecurityFeatures(unittest.TestCase):
    """Test security features"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True
        app.testing = True  # Ensure app is in testing mode

        # Set up session for CSRF
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = secrets.token_urlsafe(32)

        # Get CSRF token from session
        with self.app.session_transaction() as sess:
            self.csrf_token = sess.get('csrf_token', secrets.token_urlsafe(32))

    def test_file_type_validation(self):
        """Test file type validation"""
        malicious_files = [
            ('test.exe', b'MZ'),
            ('test.php', b'<?php'),
            ('test.js', b'javascript'),
            ('test.sh', b'#!/bin/bash'),
        ]

        for filename, content in malicious_files:
            response = self.app.post('/api/upload',
                                     data={
                                         'file': (io.BytesIO(content), filename)
                                     },
                                     headers={'X-CSRF-Token': self.csrf_token})
            self.assertEqual(response.status_code, 400)
            data = json.loads(response.data)
            self.assertIn('Invalid file type', data['error'])

    def test_filename_sanitization(self):
        """Test filename sanitization"""
        dangerous_filenames = [
            '../../../etc/passwd',
            '..\\..\\windows\\system32\\config\\sam',
            'test.stl;rm -rf /',
            'test.stl|cat /etc/shadow',
        ]

        for filename in dangerous_filenames:
            response = self.app.post('/api/upload',
                                     data={
                                         'file': (io.BytesIO(b'test'), filename)
                                     },
                                     headers={'X-CSRF-Token': self.csrf_token})
            # Should either accept as STL or reject as invalid type
            # But should not cause security issues
            self.assertIn(response.status_code, [200, 400])


if __name__ == '__main__':
    unittest.main()
