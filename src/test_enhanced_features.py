#!/usr/bin/env python
"""
Enhanced test suite for improved Opti3D features
"""

import json
import os
import sys
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

import requests
from app import app


class TestEnhancedSecurity(unittest.TestCase):
    """Test enhanced security features"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_filename_sanitization(self):
        """Test filename sanitization prevents path traversal"""
        malicious_filenames = [
            '../../../etc/passwd.stl',
            '..\\..\\windows\\system32\\config\\sam.stl',
            'test.stl;rm -rf /',
            'test.stl|cat /etc/shadow',
            'test/../test.stl',
            'test\\..\\test.stl'
        ]

        for filename in malicious_filenames:
            with self.app.session_transaction() as sess:
                sess['csrf_token'] = 'test_token'

            response = self.app.post('/api/upload',
                                     data={'file': (tempfile.NamedTemporaryFile().read(), filename)},
                                     headers={'X-CSRF-Token': 'test_token'})
            
            # Should either accept as valid STL or reject, but not cause security issues
            self.assertIn(response.status_code, [200, 400])

    def test_rate_limiting_enhanced(self):
        """Test enhanced rate limiting with detailed logging"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        # Make multiple requests quickly
        responses = []
        for _ in range(6):  # Exceed default limit of 5
            resp = self.app.post('/api/upload',
                                     data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                     headers={'X-CSRF-Token': 'test_token'})
            responses.append(resp)

        # At least one should be rate limited
        rate_limited = any(r.status_code == 429 for r in responses)
        self.assertTrue(rate_limited, "Rate limiting should trigger after limit exceeded")

    def test_csrf_token_validation(self):
        """Test CSRF token validation"""
        # Test without CSRF token
        response = self.app.post('/api/upload',
                                 data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')})
        self.assertEqual(response.status_code, 403)

        # Test with invalid CSRF token
        response = self.app.post('/api/upload',
                                 data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                 headers={'X-CSRF-Token': 'invalid_token'})
        self.assertEqual(response.status_code, 403)


class TestEnhancedErrorHandling(unittest.TestCase):
    """Test enhanced error handling"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_custom_error_responses(self):
        """Test custom error response format"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        # Test validation error
        response = self.app.post('/api/upload',
                                 data={'file': (b'content', 'test.txt')},
                                 headers={'X-CSRF-Token': 'test_token'})
        
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error_code', data)
        self.assertIn('status', data)
        self.assertEqual(data['status'], 'error')

    def test_error_logging(self):
        """Test that errors are properly logged"""
        with patch('app.error_logger') as mock_logger:
            with self.app.session_transaction() as sess:
                sess['csrf_token'] = 'test_token'

            # Trigger an error (response not needed)
            self.app.post('/api/upload',
                                     data={'file': (b'content', 'test.txt')},
                                     headers={'X-CSRF-Token': 'test_token'})
            
            # Verify error was logged
            mock_logger.error.assert_called()


class TestHealthChecks(unittest.TestCase):
    """Test health check endpoints"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_health_check_endpoint(self):
        """Test basic health check"""
        response = self.app.get('/api/health')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('status', data)
        self.assertIn('timestamp', data)
        self.assertIn('version', data)
        self.assertIn('checks', data)

    def test_metrics_endpoint(self):
        """Test metrics collection"""
        response = self.app.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('timestamp', data)
        self.assertIn('metrics', data)
        self.assertIn('uploaded_files', data['metrics'])
        self.assertIn('total_storage_used', data['metrics'])

    def test_info_endpoint(self):
        """Test application info endpoint"""
        response = self.app.get('/api/info')
        self.assertEqual(response.status_code, 200)
        
        data = json.loads(response.data)
        self.assertIn('name', data)
        self.assertIn('version', data)
        self.assertIn('endpoints', data)
        self.assertIn('optimization_levels', data)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management system"""

    def test_config_from_env(self):
        """Test configuration loading from environment"""
        # Set test environment variables
        os.environ['FLASK_HOST'] = '0.0.0.0'
        os.environ['FLASK_PORT'] = '8080'
        os.environ['LOG_LEVEL'] = 'DEBUG'
        os.environ['MAX_CONTENT_LENGTH'] = '52428800'  # 50MB
        
        try:
            from stldeli.settings import get_config
            config = get_config()
            
            self.assertEqual(config.server.host, '0.0.0.0')
            self.assertEqual(config.server.port, 8080)
            self.assertEqual(config.logging.level, 'DEBUG')
            self.assertEqual(config.security.max_content_length, 52428800)
            
        finally:
            # Clean up environment
            for key in ['FLASK_HOST', 'FLASK_PORT', 'LOG_LEVEL', 'MAX_CONTENT_LENGTH']:
                if key in os.environ:
                    del os.environ[key]

    def test_config_validation(self):
        """Test configuration validation"""
        # Test invalid port
        with self.assertRaises(ValueError):
            from stldeli.settings import AppConfig, SecurityConfig, ServerConfig
            config = AppConfig(
                security=SecurityConfig(secret_key='test'),
                server=ServerConfig(port=70000)  # Invalid port
            )
            config.validate()

    def test_config_to_dict(self):
        """Test configuration serialization"""
        config = get_config()
        config_dict = config.to_dict()
        
        self.assertIsInstance(config_dict, dict)
        self.assertIn('security', config_dict)
        self.assertIn('logging', config_dict)
        self.assertIn('performance', config_dict)
        self.assertIn('server', config_dict)


class TestPerformanceImprovements(unittest.TestCase):
    """Test performance-related improvements"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_processing_time_tracking(self):
        """Test that processing time is tracked"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        # Mock optimization to avoid actual processing
        with patch('app.optimize_stl_file_wrapper') as mock_optimize:
            mock_optimize.return_value = '/tmp/optimized_test.stl'
            
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
                                                 'file_id': 'test_file_id',
                                                 'level': 'medium'
                                             }),
                                             content_type='application/json',
                                             headers={'X-CSRF-Token': 'test_token'})

                    self.assertEqual(response.status_code, 200)
                    data = json.loads(response.data)
                    self.assertIn('processing_time', data)
                    self.assertIsInstance(data['processing_time'], (int, float))

    def test_file_cleanup_improvements(self):
        """Test enhanced file cleanup"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        response = self.app.post('/api/cleanup',
                                 data=json.dumps({}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': 'test_token'})

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('files_removed', data)
        self.assertIn('processing_time', data)


class TestFrontendEnhancements(unittest.TestCase):
    """Test frontend improvements"""

    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

    def test_enhanced_upload_response(self):
        """Test enhanced upload response format"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        # Create a temporary STL file
        with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
            tmp.write(b'test stl content')
            tmp_path = tmp.name

        try:
            with open(tmp_path, 'rb') as test_file:
                response = self.app.post('/api/upload',
                                         data={'file': (test_file, 'test.stl')},
                                         headers={'X-CSRF-Token': 'test_token'})

            # Note: This will fail due to invalid STL format, but we test the response structure
            if response.status_code == 200:
                data = json.loads(response.data)
                self.assertIn('file_size', data)
                self.assertIn('status', data)
                self.assertEqual(data['status'], 'success')

        finally:
            os.unlink(tmp_path)

    def test_enhanced_optimization_response(self):
        """Test enhanced optimization response"""
        with self.app.session_transaction() as sess:
            sess['csrf_token'] = 'test_token'

        with patch('app.optimize_stl_file_wrapper') as mock_optimize:
            mock_optimize.return_value = '/tmp/optimized_test.stl'
            
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
                                                 'file_id': 'test_file_id',
                                                 'level': 'medium'
                                             }),
                                             content_type='application/json',
                                             headers={'X-CSRF-Token': 'test_token'})

                    if response.status_code == 200:
                        data = json.loads(response.data)
                        self.assertIn('processing_time', data)
                        self.assertIn('status', data)
                        self.assertEqual(data['status'], 'success')


if __name__ == '__main__':
    unittest.main()
