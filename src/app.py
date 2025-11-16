#!/usr/bin/env python
# coding: utf-8

"""
Flask web application for STL file optimization
Provides frontend for uploading STL files and downloading optimized versions.
"""

import logging
import os
import secrets
import tempfile
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Union

from flask import Flask, request, jsonify, render_template, send_file, session, Response
from stl import mesh
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename
from werkzeug.wrappers import Response as WerkzeugResponse
from werkzeug.exceptions import HTTPException

from stldeli.stl_optimizer import analyze_stl_mesh, optimize_stl_file
from security_middleware import SecurityMiddleware, InputSanitizer

# ---------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100MB default
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())

# Initialize security middleware
security_middleware = SecurityMiddleware(app)

# Logging configuration
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
log_format = '%(asctime)s | %(levelname)s | %(filename)s | %(name)s | %(lineno)d | %(message)s'

# Configure logging with both file and console handlers
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format=log_format,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('logs/app.log', mode='a') if os.path.exists('logs') or os.makedirs('logs', exist_ok=True) else logging.NullHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create separate loggers for different components
security_logger = logging.getLogger('opti3d.security')
performance_logger = logging.getLogger('opti3d.performance')
error_logger = logging.getLogger('opti3d.errors')

# ---------------------------------------------------------------------
# Security and key management
# ---------------------------------------------------------------------

secret_key = os.environ.get('SECRET_KEY')
if secret_key:
    app.config['SECRET_KEY'] = secret_key
else:
    if os.environ.get('FLASK_ENV', 'development') == 'production':
        raise ValueError("SECRET_KEY environment variable must be set in production")
    app.config['SECRET_KEY'] = secrets.token_hex(32)
    logger.warning("Using auto-generated secret key. Set SECRET_KEY environment variable in production.")

# ---------------------------------------------------------------------
# Validate upload directory
# ---------------------------------------------------------------------

upload_folder = app.config['UPLOAD_FOLDER']
if not os.path.exists(upload_folder):
    try:
        os.makedirs(upload_folder, exist_ok=True)
        logger.info("Created upload directory: %s", upload_folder)
    except OSError as e:
        logger.error("Cannot create upload directory %s: %s", upload_folder, str(e))
        raise

if not os.access(upload_folder, os.W_OK):
    logger.error("Upload directory %s is not writable", upload_folder)
    raise PermissionError(f"Upload directory {upload_folder} is not writable")

# ---------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------

ALLOWED_EXTENSIONS = {'stl'}
RATE_LIMIT_MSG = 'Rate limit exceeded. Please try again later.'
CSRF_TOKEN_MSG = 'CSRF token missing or invalid'

# ---------------------------------------------------------------------
# In-memory rate limiter
# ---------------------------------------------------------------------

rate_limit_store = {}

def check_rate_limit(client_ip: str, limit: int = 5, window: int = 60) -> bool:
    """Simple rate limiting implementation with enhanced logging."""
    if app.testing:
        return True

    now = time.time()
    requests = rate_limit_store.setdefault(client_ip, [])
    rate_limit_store[client_ip] = [t for t in requests if now - t < window]

    current_count = len(rate_limit_store[client_ip])
    if current_count >= limit:
        security_logger.warning(
            "Rate limit exceeded for IP %s: %d requests (limit: %d)",
            client_ip, current_count, limit
        )
        return False

    rate_limit_store[client_ip].append(now)
    security_logger.debug(
        "Request allowed for IP %s: %d/%d requests used",
        client_ip, current_count + 1, limit
    )
    return True

# ---------------------------------------------------------------------
# CSRF Token Management
# ---------------------------------------------------------------------

def generate_csrf_token() -> str:
    """Generate CSRF token for session."""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return session['csrf_token']

def validate_csrf_token(token: str) -> bool:
    """Validate CSRF token."""
    if app.testing:
        return True
    return 'csrf_token' in session and session['csrf_token'] == token

# ---------------------------------------------------------------------
# Response Hardening
# ---------------------------------------------------------------------

@app.after_request
def add_security_headers(response: Response) -> Response:
    """Add security headers to prevent common web vulnerabilities."""
    response.headers.update({
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Referrer-Policy': 'strict-origin-when-cross-origin',
    })

    if os.environ.get('FLASK_ENV') == 'production' and request.is_secure:
        response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains; preload'

    if os.environ.get('FLASK_ENV') == 'production':
        csp = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
            "font-src 'self'; connect-src 'self'; frame-ancestors 'none'; form-action 'self';"
        )
    else:
        csp = (
            "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; font-src 'self'; connect-src 'self';"
        )

    response.headers['Content-Security-Policy'] = csp
    return response

# ---------------------------------------------------------------------
# Centralized Error Handling
# ---------------------------------------------------------------------

class Opti3DError(Exception):
    """Base exception class for Opti3D application."""
    def __init__(self, message: str, error_code: str = None, status_code: int = 500):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'GENERIC_ERROR'
        self.status_code = status_code

class ValidationError(Opti3DError):
    """Raised for validation errors."""
    def __init__(self, message: str):
        super().__init__(message, 'VALIDATION_ERROR', 400)

class FileProcessingError(Opti3DError):
    """Raised for file processing errors."""
    def __init__(self, message: str):
        super().__init__(message, 'FILE_PROCESSING_ERROR', 500)

class OptimizationError(Opti3DError):
    """Raised for optimization errors."""
    def __init__(self, message: str):
        super().__init__(message, 'OPTIMIZATION_ERROR', 500)

@app.errorhandler(Exception)
def handle_exception(e: Exception) -> tuple:
    """Global exception handler with proper logging."""
    # Log the error with context
    error_logger.error(
        "Unhandled exception: %s - %s\nPath: %s\nMethod: %s\nIP: %s",
        type(e).__name__, str(e), request.path, request.method, request.remote_addr,
        exc_info=True
    )
    
    # Don't expose internal errors in production
    if app.testing or app.debug:
        message = str(e)
    else:
        message = "An internal error occurred"
    
    return jsonify({
        'error': message,
        'error_code': 'INTERNAL_ERROR',
        'status': 'error'
    }), 500

@app.errorhandler(HTTPException)
def handle_http_exception(e: HTTPException) -> tuple:
    """HTTP exception handler."""
    security_logger.warning(
        "HTTP exception: %s - %s\nPath: %s\nMethod: %s\nIP: %s",
        e.code, e.name, request.path, request.method, request.remote_addr
    )
    
    return jsonify({
        'error': e.description,
        'error_code': f'HTTP_{e.code}',
        'status': 'error'
    }), e.code

@app.errorhandler(Opti3DError)
def handle_opti3d_error(e: Opti3DError) -> tuple:
    """Custom Opti3D exception handler."""
    error_logger.error(
        "Opti3D error: %s - %s\nPath: %s\nMethod: %s\nIP: %s",
        e.error_code, e.message, request.path, request.method, request.remote_addr
    )
    
    return jsonify({
        'error': e.message,
        'error_code': e.error_code,
        'status': 'error'
    }), e.status_code

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension with enhanced validation."""
    if not filename or not isinstance(filename, str):
        return False
    
    # Use input sanitizer for enhanced security
    sanitized_filename = InputSanitizer.sanitize_filename(filename)
    if not sanitized_filename:
        security_logger.warning("Filename failed sanitization: %s", filename)
        return False
    
    # Extract extension safely
    parts = sanitized_filename.rsplit('.', 1)
    if len(parts) != 2:
        return False
    
    extension = parts[1].lower()
    is_valid = extension in ALLOWED_EXTENSIONS
    
    if not is_valid:
        security_logger.warning("Invalid file extension attempted: %s", extension)
    
    return is_valid

def analyze_stl_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Analyze STL file and return metadata with enhanced error handling."""
    start_time = time.time()
    
    try:
        # Validate file exists and is readable
        if not os.path.exists(file_path):
            raise FileProcessingError(f"File not found: {file_path}")
        
        if not os.access(file_path, os.R_OK):
            raise FileProcessingError(f"File not readable: {file_path}")
        
        # Check file size
        file_size = os.path.getsize(file_path)
        if file_size == 0:
            raise ValidationError("STL file is empty")
        
        if file_size > app.config['MAX_CONTENT_LENGTH']:
            raise ValidationError(f"STL file too large: {file_size} bytes")
        
        mesh_data = mesh.Mesh.from_file(file_path)
        analysis = analyze_stl_mesh(mesh_data)
        
        if not analysis:
            raise FileProcessingError("Failed to analyze STL mesh structure")
        
        # Add file metadata
        analysis['file_size'] = file_size
        analysis['file_path'] = file_path
        
        processing_time = time.time() - start_time
        performance_logger.info(
            "STL analysis completed in %.3fs - File: %s, Triangles: %d, Vertices: %d",
            processing_time, os.path.basename(file_path), analysis.get('triangles', 0), analysis.get('vertices', 0)
        )
        
        return analysis
        
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e:
        error_logger.error("Error analyzing STL file %s: %s", file_path, str(e))
        raise FileProcessingError(f"STL file analysis failed: {str(e)}")

def optimize_stl_file_wrapper(file_path: str, optimization_level: str = 'medium') -> str:
    """Optimize STL file and return path to optimized version with enhanced error handling."""
    start_time = time.time()
    
    try:
        # Validate inputs
        if not os.path.exists(file_path):
            raise FileProcessingError(f"Source file not found: {file_path}")
        
        if optimization_level not in ['light', 'medium', 'aggressive']:
            raise ValidationError(f"Invalid optimization level: {optimization_level}")
        
        optimized_mesh = optimize_stl_file(file_path, optimization_level)
        
        if not optimized_mesh:
            raise OptimizationError("Optimization failed to produce valid mesh")
        
        # Generate secure filename
        optimized_filename = f"optimized_{uuid.uuid4().hex[:8]}_{int(time.time())}.stl"
        optimized_path = os.path.join(app.config['UPLOAD_FOLDER'], optimized_filename)
        
        # Save optimized mesh
        optimized_mesh.save(optimized_path)
        
        # Verify the optimized file was created successfully
        if not os.path.exists(optimized_path) or os.path.getsize(optimized_path) == 0:
            raise OptimizationError("Optimized file was not created properly")
        
        processing_time = time.time() - start_time
        performance_logger.info(
            "STL optimization completed in %.3fs - Level: %s, Output: %s",
            processing_time, optimization_level, optimized_filename
        )
        
        return optimized_path
        
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, KeyError) as e:
        error_logger.error("Error optimizing STL file %s: %s", file_path, str(e))
        raise OptimizationError(f"STL file optimization failed: {str(e)}")

# ---------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------

@app.route('/')
def index() -> str:
    """Serve the main page."""
    csrf_token = generate_csrf_token()
    return render_template('index.html', csrf_token=csrf_token)

@app.route('/api/upload', methods=['POST'])
def upload_file() -> tuple:
    """Handle STL file upload with enhanced validation and error handling."""
    start_time = time.time()
    client_ip = request.remote_addr
    
    try:
        # Rate limiting check
        if not check_rate_limit(client_ip):
            security_logger.warning("Rate limit exceeded for upload from IP: %s", client_ip)
            raise ValidationError(RATE_LIMIT_MSG)

        # CSRF token validation
        csrf_token = request.headers.get('X-CSRF-Token')
        if not csrf_token or not validate_csrf_token(csrf_token):
            security_logger.warning("Invalid CSRF token for upload from IP: %s", client_ip)
            raise ValidationError(CSRF_TOKEN_MSG)

        # Validate request structure and save file
        if 'file' not in request.files:
            raise ValidationError('No file provided')

        file = request.files['file']
        if file.filename == '':
            raise ValidationError('No file selected')

        # Delegate detailed validation and saving to helper
        upload_path, filename, file_size, file_id = _validate_and_save_upload(file)

        # Analyze and build response
        return _analyze_and_build_response(upload_path, filename, file_size, file_id, start_time, client_ip)

    except Opti3DError:
        raise
    except Exception as e:
        error_logger.error("Unexpected error during file upload: %s", str(e), exc_info=True)
        raise FileProcessingError('File upload failed')

@app.route('/api/optimize', methods=['POST'])
def optimize_file() -> tuple:
    """Optimize uploaded STL file with enhanced validation and error handling."""
    start_time = time.time()
    client_ip = request.remote_addr
    
    try:
        # Rate limiting check
        if not check_rate_limit(client_ip, limit=10):
            security_logger.warning("Rate limit exceeded for optimization from IP: %s", client_ip)
            raise ValidationError(RATE_LIMIT_MSG)
        # CSRF token validation and input parsing
        upload_path, file_id, optimization_level = _parse_optimize_request()
        
        # Perform optimization
        try:
            optimized_path = optimize_stl_file_wrapper(upload_path, optimization_level)
        except Exception as e:
            error_logger.error("Optimization failed for file %s: %s", file_id, str(e))
            raise e

        # Calculate compression metrics
        try:
            original_size = os.path.getsize(upload_path)
            optimized_size = os.path.getsize(optimized_path)
            
            if original_size == 0:
                raise OptimizationError('Original file size is zero')
            
            compression_ratio = (1 - optimized_size / original_size) * 100
            
            # Analyze optimized file
            optimized_analysis = analyze_stl_file(optimized_path)
            
        except Exception as e:
            # Clean up optimized file if metrics calculation fails
            _safe_remove(optimized_path)
            raise OptimizationError(f'Failed to calculate optimization metrics: {str(e)}')

        processing_time = time.time() - start_time
        performance_logger.info(
            "File optimization completed in %.3fs - Level: %s, Compression: %.1f%%, IP: %s",
            processing_time, optimization_level, compression_ratio, client_ip
        )

        return jsonify({
            'optimization_level': optimization_level,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'compression_ratio': round(compression_ratio, 2),
            'optimized_analysis': optimized_analysis,
            'download_id': os.path.basename(optimized_path),
            'processing_time': round(processing_time, 3),
            'status': 'success'
        }), 200

    except Opti3DError:
        raise
    except Exception as e:
        error_logger.error("Unexpected error during file optimization: %s", str(e), exc_info=True)
        raise OptimizationError('Optimization failed')

@app.route('/api/download/<filename>')
def download_file(filename: str) -> Union[WerkzeugResponse, tuple]:
    """Download optimized STL file with enhanced security."""
    try:
        # Validate filename parameter
        if not filename or not isinstance(filename, str):
            raise ValidationError('Invalid filename parameter')
        
        # Security checks for filename
        if '..' in filename or '/' in filename or '\\' in filename:
            security_logger.warning("Suspicious download filename attempted: %s", filename)
            raise ValidationError('Invalid filename format')
        
        # Ensure filename has expected pattern
        if not (filename.startswith('optimized_') and filename.endswith('.stl')):
            security_logger.warning("Invalid download file pattern: %s", filename)
            raise ValidationError('Invalid file for download')
        
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Validate file exists and is accessible
        if not os.path.exists(file_path):
            raise ValidationError('File not found')
        
        if not os.access(file_path, os.R_OK):
            raise FileProcessingError('File not accessible for download')
        
        # Log download attempt
        security_logger.info(
            "File download requested - File: %s, IP: %s",
            filename, request.remote_addr
        )
        
        return send_file(
            file_path,
            as_attachment=True,
            download_name=f'optimized_{filename.replace("optimized_", "")}',
            mimetype='application/octet-stream'
        )

    except Opti3DError:
        raise
    except Exception as e:
        error_logger.error("Unexpected error during file download: %s", str(e), exc_info=True)
        raise FileProcessingError('Download failed')

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files() -> tuple:
    """Clean up temporary files with enhanced validation."""
    start_time = time.time()
    client_ip = request.remote_addr
    
    try:
        # Rate limiting check
        if not check_rate_limit(client_ip, limit=20):
            security_logger.warning("Rate limit exceeded for cleanup from IP: %s", client_ip)
            raise ValidationError(RATE_LIMIT_MSG)

        # CSRF token validation
        csrf_token = request.headers.get('X-CSRF-Token')
        if not csrf_token or not validate_csrf_token(csrf_token):
            security_logger.warning("Invalid CSRF token for cleanup from IP: %s", client_ip)
            raise ValidationError(CSRF_TOKEN_MSG)

        # Validate and parse JSON data
        try:
            data = request.get_json() or {}
        except Exception:
            data = {}
        
        files_removed = 0
        
        if data and 'file_id' in data:
            # Clean up specific file
            file_id = str(data['file_id']).strip()
            if file_id and len(file_id) >= 10:
                files_removed = _cleanup_by_file_id(file_id)
                security_logger.info(
                    "Specific file cleanup completed - File ID: %s, Files removed: %d, IP: %s",
                    file_id, files_removed, client_ip
                )
        else:
            # Clean up expired files
            files_removed = _cleanup_expired_files()
            security_logger.info(
                "Expired file cleanup completed - Files removed: %d, IP: %s",
                files_removed, client_ip
            )

        processing_time = time.time() - start_time
        performance_logger.info(
            "File cleanup completed in %.3fs - Files removed: %d",
            processing_time, files_removed
        )

        return jsonify({
            'message': 'Cleanup completed',
            'files_removed': files_removed,
            'processing_time': round(processing_time, 3),
            'status': 'success'
        }), 200

    except Opti3DError:
        raise
    except Exception as e:
        error_logger.error("Unexpected error during cleanup: %s", str(e), exc_info=True)
        raise FileProcessingError('Cleanup failed')


def _cleanup_by_file_id(file_id: str) -> int:
    """Remove all temporary files associated with a given file_id and return count."""
    folder = app.config['UPLOAD_FOLDER']
    files_removed = 0
    
    try:
        upload_files = [f for f in os.listdir(folder) if f.startswith(file_id)]
        
        for filename in upload_files:
            if _safe_remove(os.path.join(folder, filename)):
                files_removed += 1
                
    except OSError as e:
        error_logger.error("Error during file cleanup for ID %s: %s", file_id, str(e))
    
    return files_removed


def _cleanup_expired_files(expiry_seconds: int = 3600) -> int:
    """Remove temporary STL files older than expiry_seconds and return count."""
    folder = app.config['UPLOAD_FOLDER']
    current_time = datetime.now()
    files_removed = 0

    try:
        for filename in os.listdir(folder):
            if not filename.endswith('.stl'):
                continue

            file_path = os.path.join(folder, filename)
            
            try:
                file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                
                if (current_time - file_time).seconds > expiry_seconds:
                    if _safe_remove(file_path):
                        files_removed += 1
                        
            except OSError:
                # Skip files that can't be accessed
                continue
                
    except OSError as e:
        error_logger.error("Error during expired files cleanup: %s", str(e))
    
    return files_removed


def _safe_remove(path: str) -> bool:
    """Attempt to remove a file, returning True if successful."""
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _parse_optimize_request():
    """Parse and validate the optimization request payload and return (upload_path, file_id, optimization_level)."""
    data = _validate_csrf_and_get_json()

    # Validate required fields
    if 'file_id' not in data:
        raise ValidationError('File ID required')

    file_id = str(data['file_id']).strip()
    if not InputSanitizer.validate_file_id(file_id):
        raise ValidationError('Invalid file ID format')

    optimization_level = data.get('level', 'medium')
    if optimization_level not in ['light', 'medium', 'aggressive']:
        raise ValidationError('Invalid optimization level')

    upload_path = _find_upload_path_for_file_id(file_id)

    return upload_path, file_id, optimization_level


def _validate_csrf_and_get_json():
    """Validate CSRF token and return parsed JSON body."""
    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        security_logger.warning("Invalid CSRF token for optimization from IP: %s", request.remote_addr)
        raise ValidationError(CSRF_TOKEN_MSG)

    try:
        data = request.get_json()
        if data is None:
            raise ValidationError('Invalid JSON data provided')
        return data
    except Exception:
        raise ValidationError('Invalid JSON format')


def _find_upload_path_for_file_id(file_id: str) -> str:
    """Return the upload path for a given file_id or raise ValidationError if not found."""
    upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
    if not upload_files:
        raise ValidationError('File not found')

    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
    if not os.path.exists(upload_path):
        raise ValidationError('Source file no longer exists')

    return upload_path


def _validate_and_save_upload(file) -> tuple:
    """Validate an uploaded FileStorage, save it to upload folder and return (path, filename, size).

    Raises ValidationError or FileProcessingError on failure.
    """
    # Enhanced file validation
    if not allowed_file(file.filename):
        raise ValidationError('Invalid file type. Only STL files are allowed')

    # Validate file content
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    if file_size == 0:
        raise ValidationError('Uploaded file is empty')

    if file_size > app.config['MAX_CONTENT_LENGTH']:
        raise ValidationError(f'File size {file_size} exceeds maximum allowed size')

    # Sanitize filename and generate unique ID
    filename = secure_filename(file.filename)
    if not filename:
        raise ValidationError('Invalid filename after sanitization')

    file_id = str(uuid.uuid4())
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")

    # Save file with error handling
    try:
        file.save(upload_path)
    except Exception as e:
        error_logger.error("Failed to save uploaded file: %s", str(e))
        raise FileProcessingError('Failed to save uploaded file')

    return upload_path, filename, file_size, file_id


def _analyze_and_build_response(upload_path: str, filename: str, file_size: int, file_id: str, start_time: float, client_ip: str):
    """Analyze uploaded file and return the Flask response tuple.

    This helper centralizes analysis, cleanup on failure and response formatting
    to keep the route handler small and focused.
    """
    try:
        analysis = analyze_stl_file(upload_path)
    except Exception as e:
        _safe_remove(upload_path)
        raise e

    if not analysis:
        _safe_remove(upload_path)
        raise FileProcessingError('Failed to analyze STL file')

    processing_time = time.time() - start_time
    performance_logger.info(
        "File upload completed in %.3fs - File: %s, Size: %d bytes, IP: %s",
        processing_time, filename, file_size, client_ip
    )

    return jsonify({
        'file_id': file_id,
        'filename': filename,
        'analysis': analysis,
        'upload_time': datetime.now().isoformat(),
        'file_size': file_size,
        'status': 'success'
    }), 200

# ---------------------------------------------------------------------
# Health Check and Monitoring Endpoints
# ---------------------------------------------------------------------

@app.route('/api/health')
def health_check() -> tuple:
    """Basic health check endpoint."""
    try:
        # Check upload directory
        upload_dir_accessible = os.access(app.config['UPLOAD_FOLDER'], os.W_OK)
        
        # Check disk space (basic check)
        try:
            import shutil
            disk_usage = shutil.disk_usage(app.config['UPLOAD_FOLDER'])
            disk_space_ok = disk_usage.free > 100 * 1024 * 1024  # At least 100MB free
        except Exception:
            disk_space_ok = False
        
        # Check memory usage (basic check)
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_ok = memory.percent < 90  # Less than 90% used
        except ImportError:
            memory_ok = True  # Can't check, assume OK
        
        status = 'healthy' if all([upload_dir_accessible, disk_space_ok, memory_ok]) else 'degraded'
        
        return jsonify({
            'status': status,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0.0',
            'checks': {
                'upload_directory': 'ok' if upload_dir_accessible else 'error',
                'disk_space': 'ok' if disk_space_ok else 'warning',
                'memory': 'ok' if memory_ok else 'warning'
            }
        }), 200 if status == 'healthy' else 503
        
    except Exception as e:
        error_logger.error("Health check failed: %s", str(e))
        return jsonify({
            'status': 'unhealthy',
            'timestamp': datetime.now().isoformat(),
            'error': 'Health check failed'
        }), 503

@app.route('/api/metrics')
def get_metrics() -> tuple:
    """Get application metrics."""
    try:
        # Count files in upload directory
        try:
            upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.endswith('.stl')]
            file_count = len(upload_files)
            
            # Calculate total size
            total_size = sum(
                os.path.getsize(os.path.join(app.config['UPLOAD_FOLDER'], f)) 
                for f in upload_files
            )
        except Exception:
            file_count = 0
            total_size = 0
        
        # Rate limit stats
        active_ips = len(rate_limit_store)
        
        return jsonify({
            'timestamp': datetime.now().isoformat(),
            'metrics': {
                'uploaded_files': file_count,
                'total_storage_used': total_size,
                'active_rate_limited_ips': active_ips,
                'uptime_seconds': time.time() - app.start_time if hasattr(app, 'start_time') else 0
            }
        }), 200
        
    except Exception as e:
        error_logger.error("Metrics collection failed: %s", str(e))
        return jsonify({'error': 'Failed to collect metrics'}), 500

@app.route('/api/info')
def get_app_info() -> tuple:
    """Get application information."""
    return jsonify({
        'name': 'Opti3D',
        'description': 'STL File Optimization Service',
        'version': '1.0.0',
        'endpoints': {
            'upload': 'POST /api/upload',
            'optimize': 'POST /api/optimize',
            'download': 'GET /api/download/<filename>',
            'cleanup': 'POST /api/cleanup',
            'health': 'GET /api/health',
            'metrics': 'GET /api/metrics'
        },
        'optimization_levels': ['light', 'medium', 'aggressive'],
        'max_file_size': app.config['MAX_CONTENT_LENGTH']
    }), 200

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == '__main__':
    # Record application start time
    app.start_time = time.time()
    
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', 5000))
    
    if host == '0.0.0.0':
        security_logger.warning("Binding to all interfaces (0.0.0.0). Ensure firewall is properly configured.")
    
    logger.info(
        "Starting Opti3D application - Host: %s, Port: %d, Debug: %s",
        host, port, debug_mode
    )
    
    try:
        app.run(debug=debug_mode, host=host, port=port)
    except Exception as e:
        error_logger.critical("Failed to start application: %s", str(e), exc_info=True)
        raise
