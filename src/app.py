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

from stldeli.stl_optimizer import analyze_stl_mesh, optimize_stl_file

# ---------------------------------------------------------------------
# Flask app configuration
# ---------------------------------------------------------------------

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024))  # 100MB default
app.config['UPLOAD_FOLDER'] = os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir())

# Logging configuration
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logging.basicConfig(level=getattr(logging, log_level, logging.INFO))
logger = logging.getLogger(__name__)

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
    """Simple rate limiting implementation."""
    if app.testing:
        return True

    now = time.time()
    requests = rate_limit_store.setdefault(client_ip, [])
    rate_limit_store[client_ip] = [t for t in requests if now - t < window]

    if len(rate_limit_store[client_ip]) >= limit:
        return False

    rate_limit_store[client_ip].append(now)
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
# Utility functions
# ---------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    """Check if file has allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_stl_file(file_path: str) -> Optional[Dict[str, Any]]:
    """Analyze STL file and return metadata."""
    try:
        mesh_data = mesh.Mesh.from_file(file_path)
        return analyze_stl_mesh(mesh_data)
    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e6:
        logger.error("Error analyzing STL file: %s", str(e6))
        return None

def optimize_stl_file_wrapper(file_path: str, optimization_level: str = 'medium') -> str:
    """Optimize STL file and return path to optimized version."""
    try:
        optimized_mesh = optimize_stl_file(file_path, optimization_level)
        optimized_filename = f"optimized_{uuid.uuid4().hex[:8]}.stl"
        optimized_path = os.path.join(app.config['UPLOAD_FOLDER'], optimized_filename)
        optimized_mesh.save(optimized_path)
        return optimized_path
    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, KeyError) as e5:
        logger.error("Error optimizing STL file: %s", str(e5))
        raise

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
    """Handle STL file upload."""
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip):
        return jsonify({'error': RATE_LIMIT_MSG}), 429

    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': CSRF_TOKEN_MSG}), 403

    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only STL files are allowed'}), 400

        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(upload_path)

        analysis = analyze_stl_file(upload_path)
        if not analysis:
            os.remove(upload_path)
            return jsonify({'error': 'Failed to analyze STL file'}), 400

        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'analysis': analysis,
            'upload_time': datetime.now().isoformat()
        }), 200

    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, KeyError) as e4:
        logger.error("Upload error: %s", str(e4))
        return jsonify({'error': 'File upload failed'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_file() -> tuple:
    """Optimize uploaded STL file."""
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip, limit=10):
        return jsonify({'error': RATE_LIMIT_MSG}), 429

    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': CSRF_TOKEN_MSG}), 403

    try:
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({'error': 'File ID required'}), 400

        file_id = data['file_id']
        optimization_level = data.get('level', 'medium')
        if optimization_level not in ['light', 'medium', 'aggressive']:
            return jsonify({'error': 'Invalid optimization level'}), 400

        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
        if not upload_files:
            return jsonify({'error': 'File not found'}), 404

        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
        optimized_path = optimize_stl_file_wrapper(upload_path, optimization_level)

        original_size = os.path.getsize(upload_path)
        optimized_size = os.path.getsize(optimized_path)
        compression_ratio = (1 - optimized_size / original_size) * 100
        optimized_analysis = analyze_stl_file(optimized_path)

        return jsonify({
            'optimization_level': optimization_level,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'compression_ratio': round(compression_ratio, 2),
            'optimized_analysis': optimized_analysis,
            'download_id': os.path.basename(optimized_path)
        }), 200

    except (OSError, ValueError, RuntimeError, TypeError, AttributeError, KeyError) as e3:
        logger.error("Optimization error: %s", str(e3))
        return jsonify({'error': 'Optimization failed'}), 500

@app.route('/api/download/<filename>')
def download_file(filename: str) -> Union[WerkzeugResponse, tuple]:
    """Download optimized STL file."""
    try:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'File not found'}), 404

        return send_file(
            file_path,
            as_attachment=True,
            download_name=f'optimized_{filename}',
            mimetype='application/octet-stream'
        )

    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e2:
        logger.error("Download error: %s", str(e2))
        return jsonify({'error': 'Download failed'}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files() -> tuple:
    """Clean up temporary files."""
    client_ip = request.remote_addr
    if not check_rate_limit(client_ip, limit=20):
        return jsonify({'error': RATE_LIMIT_MSG}), 429

    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': CSRF_TOKEN_MSG}), 403

    try:
        data = request.get_json()
        if data and 'file_id' in data:
            _cleanup_by_file_id(data['file_id'])
        else:
            _cleanup_expired_files()

        return jsonify({'message': 'Cleanup completed'}), 200

    except (OSError, ValueError, TypeError, AttributeError, KeyError) as e1:
        logger.error("Cleanup error: %s", str(e1))
        return jsonify({'error': 'Cleanup failed'}), 500


def _cleanup_by_file_id(file_id: str) -> None:
    """Remove all temporary files associated with a given file_id."""
    folder = app.config['UPLOAD_FOLDER']
    upload_files = [f for f in os.listdir(folder) if f.startswith(file_id)]

    for filename in upload_files:
        _safe_remove(os.path.join(folder, filename))


def _cleanup_expired_files(expiry_seconds: int = 3600) -> None:
    """Remove temporary STL files older than expiry_seconds."""
    folder = app.config['UPLOAD_FOLDER']
    current_time = datetime.now()

    for filename in os.listdir(folder):
        if not filename.endswith('.stl'):
            continue

        file_path = os.path.join(folder, filename)
        file_time = datetime.fromtimestamp(os.path.getctime(file_path))

        if (current_time - file_time).seconds > expiry_seconds:
            _safe_remove(file_path)


def _safe_remove(path: str) -> None:
    """Attempt to remove a file, ignoring common OS errors."""
    try:
        os.remove(path)
    except OSError:
        pass

# ---------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    if host == '0.0.0.0':
        logger.warning("Binding to all interfaces (0.0.0.0). Ensure firewall is properly configured.")
    app.run(debug=debug_mode, host=host, port=5000)
