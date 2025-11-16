"""
Security middleware for Checkmarx compliance
"""

import re
import hashlib
import hmac
import time
from typing import Dict, Any, Optional
from flask import Flask, request, g, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import logging

security_logger = logging.getLogger('opti3d.security')


class SecurityMiddleware:
    """Security middleware for enhanced protection"""
    
    def __init__(self, app: Flask = None):
        self.app = app
        if app is not None:
            self.init_app(app)
    
    def init_app(self, app: Flask):
        """Initialize security middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Configure security headers
        app.after_request(self.add_security_headers)
        
        # Configure rate limiting
        self.rate_limit_store: Dict[str, Dict[str, Any]] = {}
        
        # Configure request validation
        self.setup_request_validation()
    
    def setup_request_validation(self):
        """Setup request validation patterns"""
        # SQL injection patterns
        self.sql_injection_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|UNION|SCRIPT)\b)",
            r"(\b(OR|AND)\s+\d+\s*=\s*\d+)",
            r"(['\"];?\s*(OR|AND)\s+['\"]?\w+['\"]?\s*=\s*['\"]?\w+['\"]?)",
            r"(\bUNION\s+SELECT\b)",
            r"(\b(SCRIPT|JAVASCRIPT|VBSCRIPT|ONLOAD|ONERROR)\b)",
        ]
        
        # XSS patterns
        self.xss_patterns = [
            r"(<script[^>]*>.*?</script>)",
            r"(javascript\s*:)",
            r"(on\w+\s*=)",
            r"(<iframe[^>]*>)",
            r"(eval\s*\()",
            r"(document\.(cookie|location|write))",
        ]
        
        # Path traversal patterns
        self.path_traversal_patterns = [
            r"(\.\./)",
            r"(\.\.\\)",
            r"(%2e%2e%2f)",
            r"(%2e%2e\\)",
            r"(\.\.%2f)",
            r"(\.\.%5c)",
        ]
        
        # Command injection patterns
        self.command_injection_patterns = [
            r"(\||&|;|`|\$\(|\$\{)",
            r"(wget\s|curl\s|nc\s|netcat\s)",
            r"(rm\s|mv\s|cp\s|cat\s)",
            r"(chmod\s|chown\s)",
        ]
    
    def before_request(self):
        """Security checks before processing request"""
        client_ip = self.get_client_ip()
        
        # Log request details for security monitoring
        security_logger.info(
            "Request received - IP: %s, Path: %s, Method: %s, User-Agent: %s",
            client_ip, request.path, request.method, request.headers.get('User-Agent', 'Unknown')
        )
        
        # Check for common attack patterns
        self.check_sql_injection()
        self.check_xss()
        self.check_path_traversal()
        self.check_command_injection()
        
        # Check request size limits
        self.check_request_size()
        
        # Check for suspicious user agents
        self.check_user_agent()
        
        # Store request context for monitoring
        g.start_time = time.time()
        g.client_ip = client_ip
    
    def after_request(self, response):
        """Security logging after request"""
        if hasattr(g, 'start_time'):
            processing_time = time.time() - g.start_time
            security_logger.info(
                "Request completed - IP: %s, Path: %s, Status: %d, Time: %.3fs",
                g.client_ip, request.path, response.status_code, processing_time
            )
        
        return response
    
    def add_security_headers(self, response):
        """Add comprehensive security headers"""
        # Prevent clickjacking
        response.headers['X-Frame-Options'] = 'DENY'
        
        # Prevent MIME type sniffing
        response.headers['X-Content-Type-Options'] = 'nosniff'
        
        # Enable XSS protection
        response.headers['X-XSS-Protection'] = '1; mode=block'
        
        # Referrer policy
        response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        # Content Security Policy
        if request.endpoint != 'static':
            csp = (
                "default-src 'self'; "
                "script-src 'self' 'unsafe-inline'; "
                "style-src 'self' 'unsafe-inline'; "
                "img-src 'self' data:; "
                "font-src 'self'; "
                "connect-src 'self'; "
                "frame-ancestors 'none'; "
                "base-uri 'self'; "
                "form-action 'self';"
            )
            response.headers['Content-Security-Policy'] = csp
        
        # HSTS (only in HTTPS)
        if request.is_secure:
            response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
        
        # Remove server information
        response.headers['Server'] = 'SecureServer'
        
        return response
    
    def get_client_ip(self) -> str:
        """Get real client IP considering proxies"""
        # Check for forwarded headers
        if request.headers.get('X-Forwarded-For'):
            return request.headers.get('X-Forwarded-For').split(',')[0].strip()
        elif request.headers.get('X-Real-IP'):
            return request.headers.get('X-Real-IP')
        else:
            return request.remote_addr
    
    def check_sql_injection(self):
        """Check for SQL injection patterns"""
        for pattern in self.sql_injection_patterns:
            if self.check_pattern_in_request(pattern):
                security_logger.warning(
                    "SQL injection attempt detected - IP: %s, Path: %s, Pattern: %s",
                    self.get_client_ip(), request.path, pattern
                )
                self.block_request("Potential SQL injection detected")
    
    def check_xss(self):
        """Check for XSS patterns"""
        for pattern in self.xss_patterns:
            if self.check_pattern_in_request(pattern):
                security_logger.warning(
                    "XSS attempt detected - IP: %s, Path: %s, Pattern: %s",
                    self.get_client_ip(), request.path, pattern
                )
                self.block_request("Potential XSS attack detected")
    
    def check_path_traversal(self):
        """Check for path traversal patterns"""
        for pattern in self.path_traversal_patterns:
            if self.check_pattern_in_request(pattern):
                security_logger.warning(
                    "Path traversal attempt detected - IP: %s, Path: %s, Pattern: %s",
                    self.get_client_ip(), request.path, pattern
                )
                self.block_request("Potential path traversal attack detected")
    
    def check_command_injection(self):
        """Check for command injection patterns"""
        for pattern in self.command_injection_patterns:
            if self.check_pattern_in_request(pattern):
                security_logger.warning(
                    "Command injection attempt detected - IP: %s, Path: %s, Pattern: %s",
                    self.get_client_ip(), request.path, pattern
                )
                self.block_request("Potential command injection detected")
    
    def check_pattern_in_request(self, pattern: str) -> bool:
        """Check if pattern exists in any request parameter"""
        # Check query parameters
        for key, value in request.args.items():
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check form data
        for key, value in request.form.items():
            if re.search(pattern, value, re.IGNORECASE):
                return True
        
        # Check JSON data
        if request.is_json:
            try:
                json_data = request.get_json(silent=True)
                if json_data:
                    json_str = str(json_data)
                    if re.search(pattern, json_str, re.IGNORECASE):
                        return True
            except Exception:
                pass
        
        # Check path
        if re.search(pattern, request.path, re.IGNORECASE):
            return True
        
        return False
    
    def check_request_size(self):
        """Check request size limits"""
        content_length = request.content_length or 0
        
        # Define size limits (in bytes)
        max_content_length = 100 * 1024 * 1024  # 100MB
        max_query_length = 4096  # 4KB
        max_header_length = 8192  # 8KB
        
        if content_length > max_content_length:
            security_logger.warning(
                "Oversized request detected - IP: %s, Size: %d bytes",
                self.get_client_ip(), content_length
            )
            self.block_request("Request too large")
        
        # Check query string length
        if len(request.query_string) > max_query_length:
            security_logger.warning(
                "Oversized query string detected - IP: %s, Length: %d",
                self.get_client_ip(), len(request.query_string)
            )
            self.block_request("Query string too long")
        
        # Check total header size to defend against header attacks
        try:
            headers_total = sum(len(k) + len(v) for k, v in request.headers.items())
        except Exception:
            headers_total = 0

        if headers_total > max_header_length:
            security_logger.warning(
                "Oversized headers detected - IP: %s, TotalHeaderBytes: %d",
                self.get_client_ip(), headers_total
            )
            self.block_request("Headers too large")
    
    def check_user_agent(self):
        """Check for suspicious user agents"""
        suspicious_agents = [
            'sqlmap', 'nikto', 'nmap', 'masscan', 'zap', 'burp',
            'scanner', 'crawler', 'bot', 'spider'
        ]
        
        user_agent = request.headers.get('User-Agent', '').lower()
        
        for agent in suspicious_agents:
            if agent in user_agent:
                security_logger.warning(
                    "Suspicious user agent detected - IP: %s, User-Agent: %s",
                    self.get_client_ip(), user_agent
                )
                # Log but don't block - might be legitimate
                break
    
    def block_request(self, reason: str):
        """Block request and return security response"""
        security_logger.error(
            "Request blocked - IP: %s, Path: %s, Reason: %s",
            self.get_client_ip(), request.path, reason
        )
        
        response = jsonify({
            'error': 'Security violation detected',
            'error_code': 'SECURITY_VIOLATION',
            'status': 'error'
        })
        response.status_code = 403
        return response


class InputSanitizer:
    """Input sanitization utilities"""
    
    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        if not filename:
            return ""
        
        # Remove path separators
        filename = re.sub(r'[\\/]', '_', filename)
        
        # Remove dangerous characters
        filename = re.sub(r'[<>:"|?*]', '', filename)
        
        # Limit length
        filename = filename[:255]
        
        # Ensure it has valid extension
        if not filename.lower().endswith('.stl'):
            return ""
        
        return filename
    
    @staticmethod
    def sanitize_input(input_string: str) -> str:
        """Sanitize user input to prevent XSS"""
        if not input_string:
            return ""
        
        # Remove HTML tags
        input_string = re.sub(r'<[^>]*>', '', input_string)
        
        # Remove JavaScript events
        input_string = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', input_string)
        
        # Remove javascript: protocol
        input_string = re.sub(r'javascript\s*:', '', input_string, flags=re.IGNORECASE)
        
        # Limit length
        input_string = input_string[:1000]
        
        return input_string.strip()
    
    @staticmethod
    def validate_uuid(uuid_string: str) -> bool:
        """Validate UUID format"""
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
        return bool(re.match(uuid_pattern, uuid_string, re.IGNORECASE))
    
    @staticmethod
    def validate_file_id(file_id: str) -> bool:
        """Validate file ID format (UUID or similar)"""
        if not file_id or len(file_id) < 10:
            return False
        
        # Check for valid characters
        if not re.match(r'^[a-zA-Z0-9_-]+$', file_id):
            return False
        
        return True


def create_hmac_signature(data: str, secret: str) -> str:
    """Create HMAC signature for data integrity"""
    return hmac.new(
        secret.encode('utf-8'),
        data.encode('utf-8'),
        hashlib.sha256
    ).hexdigest()


def verify_hmac_signature(data: str, signature: str, secret: str) -> bool:
    """Verify HMAC signature"""
    expected_signature = create_hmac_signature(data, secret)
    return hmac.compare_digest(expected_signature, signature)
