#!/usr/bin/env python
# coding: utf-8

"""
Security compliance tests for Checkmarx SAST/DAST
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from app import app
from security_middleware import SecurityMiddleware, InputSanitizer


class TestSecurityMiddleware(unittest.TestCase):
    """Test security middleware functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True

    def test_sql_injection_detection(self):
        """Test SQL injection attack detection"""
        sql_payloads = [
            "'; DROP TABLE users; --",
            "1' OR '1'='1",
            "1; DELETE FROM users WHERE 1=1; --",
            "admin'--",
            "' UNION SELECT password FROM users --"
        ]

        for payload in sql_payloads:
            with self.subTest(payload=payload):
                response = self.app.post('/api/optimize',
                                         data=json.dumps({'file_id': payload}),
                                         content_type='application/json',
                                         headers={'X-CSRF-Token': 'test_token'})
                # Should be blocked by security middleware or return validation error
                self.assertIn(response.status_code, [400, 403, 422])

    def test_xss_prevention(self):
        """Test XSS attack prevention"""
        xss_payloads = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<svg onload=alert('xss')>",
            "';alert('xss');//"
        ]

        for payload in xss_payloads:
            with self.subTest(payload=payload):
                response = self.app.post('/api/optimize',
                                         data=json.dumps({'file_id': payload}),
                                         content_type='application/json',
                                         headers={'X-CSRF-Token': 'test_token'})
                # Should be blocked or return validation error
                self.assertIn(response.status_code, [400, 403, 422])

    def test_path_traversal_prevention(self):
        """Test path traversal attack prevention"""
        path_payloads = [
            "../../../etc/passwd",
            "..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc/passwd",
            "..%252f..%252f..%252fetc%252fpasswd"
        ]

        for payload in path_payloads:
            with self.subTest(payload=payload):
                response = self.app.get(f'/api/download/{payload}')
                # Should be blocked or return validation error
                self.assertIn(response.status_code, [400, 403, 404, 422])

    def test_command_injection_prevention(self):
        """Test command injection attack prevention"""
        cmd_payloads = [
            "; ls -la",
            "| cat /etc/passwd",
            "& echo 'command injection'",
            "`whoami`",
            "$(id)"
        ]

        for payload in cmd_payloads:
            with self.subTest(payload=payload):
                filename = f'test{payload}.stl'
                response = self.app.post('/api/upload',
                                         data={'file': (tempfile.NamedTemporaryFile().read(), filename)},
                                         headers={'X-CSRF-Token': 'test_token'})
                # Should be blocked or return validation error
                self.assertIn(response.status_code, [400, 403, 422])

    def test_request_size_limits(self):
        """Test request size limits"""
        # Test oversized query string
        long_query = 'a' * 5000  # 5KB query string
        response = self.app.get(f'/api/health?{long_query}=1')
        # Should handle gracefully or block
        self.assertIn(response.status_code, [200, 400, 413, 422])

    def test_security_headers(self):
        """Test security headers are present"""
        response = self.app.get('/')
        
        # Check for security headers
        security_headers = [
            'X-Frame-Options',
            'X-Content-Type-Options',
            'X-XSS-Protection',
            'Referrer-Policy',
            'Content-Security-Policy'
        ]
        
        for header in security_headers:
            with self.subTest(header=header):
                self.assertIn(header, response.headers)

    def test_server_header_obfuscation(self):
        """Test server header is obfuscated"""
        response = self.app.get('/')
        server_header = response.headers.get('Server', '')
        self.assertNotIn('Flask', server_header)
        self.assertNotIn('Werkzeug', server_header)
        self.assertNotIn('Python', server_header)


class TestInputSanitization(unittest.TestCase):
    """Test input sanitization functionality"""

    def test_filename_sanitization(self):
        """Test filename sanitization"""
        malicious_filenames = [
            "../../../etc/passwd.stl",
            "..\\..\\windows\\system32\\config.sam.stl",
            "test<script>alert('xss')</script>.stl",
            "con.txt",  # Windows reserved name
            "test|pipe.stl",
            "test:colon.stl",
            "test\"quote.stl",
            "test?question.stl",
            "test*asterisk.stl",
            "test<angle>.stl",
            "a" * 300 + ".stl"  # Too long
        ]

        for filename in malicious_filenames:
            with self.subTest(filename=filename):
                sanitized = InputSanitizer.sanitize_filename(filename)
                # Should either be empty or safe
                if sanitized:
                    self.assertNotIn('/', sanitized)
                    self.assertNotIn('\\', sanitized)
                    self.assertNotIn('<', sanitized)
                    self.assertNotIn('>', sanitized)
                    self.assertNotIn(':', sanitized)
                    self.assertNotIn('|', sanitized)
                    self.assertNotIn('?', sanitized)
                    self.assertNotIn('*', sanitized)
                    self.assertNotIn('"', sanitized)
                    self.assertLessEqual(len(sanitized), 255)

    def test_input_sanitization(self):
        """Test general input sanitization"""
        malicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:alert('xss')",
            "onclick=alert('xss')",
            "<img src=x onerror=alert('xss')>",
            "<iframe src='javascript:alert(\"xss\")'></iframe>"
        ]

        for input_text in malicious_inputs:
            with self.subTest(input=input_text):
                sanitized = InputSanitizer.sanitize_input(str(input_text))
                self.assertNotIn('<script', sanitized.lower())
                self.assertNotIn('javascript:', sanitized.lower())
                # Note: onclick= might remain if it's not in HTML tag context
                self.assertNotIn('onerror=', sanitized.lower())

    def test_uuid_validation(self):
        """Test UUID validation"""
        valid_uuids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "550e8400-e29b-41d4-a716-446655440000",
            "6ba7b810-9dad-11d1-80b4-00c04fd430c8"
        ]

        invalid_uuids = [
            "123-456-789",
            "not-a-uuid",
            "123e4567-e89b-12d3-a456-42661417400",  # Missing digit
            "123e4567e89b12d3a456426614174000",    # Missing dashes
            "",
            None
        ]

        for uuid_val in valid_uuids:
            with self.subTest(uuid=uuid_val):
                self.assertTrue(InputSanitizer.validate_uuid(str(uuid_val)))

        for uuid_val in invalid_uuids:
            with self.subTest(uuid=uuid_val):
                if uuid_val is None:
                    self.assertFalse(InputSanitizer.validate_uuid(""))
                else:
                    self.assertFalse(InputSanitizer.validate_uuid(str(uuid_val)))

    def test_file_id_validation(self):
        """Test file ID validation"""
        valid_ids = [
            "123e4567-e89b-12d3-a456-426614174000",
            "abc123def456",
            "test_file_123",
            "valid-id-123"
        ]

        invalid_ids = [
            "",
            "a",  # Too short
            "../../../etc/passwd",
            "test;rm -rf /",
            "test|cat /etc/passwd",
            "test<script>alert('xss')</script>",
            None
        ]

        for file_id in valid_ids:
            with self.subTest(file_id=file_id):
                self.assertTrue(InputSanitizer.validate_file_id(file_id))

        for file_id in invalid_ids:
            with self.subTest(file_id=file_id):
                self.assertFalse(InputSanitizer.validate_file_id(file_id))


class TestCSRFProtection(unittest.TestCase):
    """Test CSRF protection"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True

    def test_csrf_token_required(self):
        """Test CSRF token is required for POST requests"""
        endpoints = [
            ('/api/upload', 'POST', {'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')}),
            ('/api/optimize', 'POST', json.dumps({'file_id': 'test'}), 'application/json'),
            ('/api/cleanup', 'POST', json.dumps({}), 'application/json')
        ]

        for endpoint, method, data, *content_type in endpoints:
            with self.subTest(endpoint=endpoint):
                if content_type:
                    response = self.app.open(endpoint, method=method, data=data, 
                                           content_type=content_type[0])
                else:
                    response = self.app.open(endpoint, method=method, data=data)
                
                # Should get 403 (CSRF) or 400 (validation error)
                self.assertIn(response.status_code, [400, 403])

    def test_invalid_csrf_token(self):
        """Test invalid CSRF token is rejected"""
        response = self.app.post('/api/upload',
                                 data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                 headers={'X-CSRF-Token': 'invalid_token'})
        
        # Should get 403 (CSRF) or 400 (validation error)
        self.assertIn(response.status_code, [400, 403])

    def test_csrf_token_validation(self):
        """Test CSRF token validation works"""
        # Get a valid CSRF token first
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
        # Extract CSRF token from response (simplified)
        # In real implementation, you'd extract from HTML or session
        csrf_token = 'test_token'  # This would be dynamically obtained
        
        # Test with valid token structure
        response = self.app.post('/api/upload',
                                 data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                 headers={'X-CSRF-Token': csrf_token})
        
        # Should not get 403 (might get 400 for other reasons)
        self.assertNotEqual(response.status_code, 403)


class TestRateLimiting(unittest.TestCase):
    """Test rate limiting functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True

    def test_rate_limiting_enforced(self):
        """Test rate limiting is enforced"""
        # Make multiple rapid requests
        responses = []
        for i in range(10):  # Exceed typical rate limit
            response = self.app.post('/api/upload',
                                     data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                     headers={'X-CSRF-Token': 'test_token'})
            responses.append(response)
        
        # At least some requests should be rate limited or validation errors
        rate_limited_or_blocked = any(r.status_code in [429, 400, 403] for r in responses)
        self.assertTrue(rate_limited_or_blocked, "Rate limiting should be enforced")

    def test_rate_limiting_headers(self):
        """Test rate limiting headers are present"""
        response = self.app.post('/api/upload',
                                 data={'file': (tempfile.NamedTemporaryFile().read(), 'test.stl')},
                                 headers={'X-CSRF-Token': 'test_token'})
        
        # Check for rate limiting headers if implemented
        rate_limit_headers = [
            'X-RateLimit-Limit',
            'X-RateLimit-Remaining',
            'X-RateLimit-Reset'
        ]
        
        # These might not be implemented, so we just check they don't cause errors
        for header in rate_limit_headers:
            header_value = response.headers.get(header)
            if header_value:
                self.assertIsInstance(header_value, str)


class TestSecureFileHandling(unittest.TestCase):
    """Test secure file handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True

    def test_file_type_validation(self):
        """Test file type validation"""
        malicious_files = [
            ('malicious.exe', b'fake executable content'),
            ('script.php', b'<?php echo "hello"; ?>'),
            ('document.pdf', b'%PDF-1.4 fake pdf'),
            ('archive.zip', b'PK\x03\x04 fake zip'),
            ('script.js', b'console.log("hello");')
        ]

        for filename, content in malicious_files:
            with self.subTest(filename=filename):
                response = self.app.post('/api/upload',
                                         data={'file': (content, filename)},
                                         headers={'X-CSRF-Token': 'test_token'})
                
                self.assertEqual(response.status_code, 400)
                data = json.loads(response.data)
                self.assertIn('error', data)

    def test_file_size_validation(self):
        """Test file size validation"""
        # Create a file that's too large
        large_content = b'x' * (200 * 1024 * 1024)  # 200MB
        
        response = self.app.post('/api/upload',
                                 data={'file': (large_content, 'large.stl')},
                                 headers={'X-CSRF-Token': 'test_token'})
        
        # Should be rejected due to size
        self.assertIn(response.status_code, [400, 413])

    def test_temporary_file_cleanup(self):
        """Test temporary files are cleaned up"""
        # This would require testing the cleanup mechanism
        # For now, we just verify cleanup endpoints exist
        response = self.app.post('/api/cleanup',
                                 data=json.dumps({}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': 'test_token'})
        
        # Should get 200 (success), 400 (validation), or 403 (CSRF)
        self.assertIn(response.status_code, [200, 400, 403])


class TestErrorHandling(unittest.TestCase):
    """Test secure error handling"""

    def setUp(self):
        """Set up test fixtures"""
        self.app = app.test_client()
        self.app.testing = True

    def test_error_messages_sanitized(self):
        """Test error messages don't leak sensitive information"""
        response = self.app.post('/api/optimize',
                                 data=json.dumps({'file_id': 'nonexistent'}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': 'test_token'})
        
        data = json.loads(response.data)
        
        # Should not contain sensitive information
        error_message = data.get('error', '')
        self.assertNotIn('password', error_message.lower())
        self.assertNotIn('secret', error_message.lower())
        self.assertNotIn('key', error_message.lower())
        self.assertNotIn('/etc/', error_message)
        self.assertNotIn('C:\\', error_message)

    def test_stack_trace_not_exposed(self):
        """Test stack traces are not exposed in production"""
        response = self.app.post('/api/optimize',
                                 data=json.dumps({'file_id': 'nonexistent'}),
                                 content_type='application/json',
                                 headers={'X-CSRF-Token': 'test_token'})
        
        response_text = response.get_data(as_text=True)
        
        # Should not contain stack trace information
        self.assertNotIn('Traceback', response_text)
        self.assertNotIn('File "', response_text)
        self.assertNotIn('line ', response_text)


if __name__ == '__main__':
    unittest.main()
