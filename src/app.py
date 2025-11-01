#!/usr/bin/env python
# coding: utf-8

"""
Flask web application for STL file optimization
Provides frontend for uploading STL files and downloading optimized versions
"""

import os
import tempfile
import uuid
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, session
from werkzeug.utils import secure_filename
from werkzeug.middleware.proxy_fix import ProxyFix
import numpy as np
from stl import mesh
import logging
import secrets
import time

from stldeli.stl_optimizer import analyze_stl_mesh, optimize_stl_file

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['UPLOAD_FOLDER'] = tempfile.gettempdir()
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple rate limiting using in-memory store
rate_limit_store = {}

def check_rate_limit(client_ip, limit=5, window=60):
    """Simple rate limiting implementation"""
    # Disable rate limiting during testing
    if app.testing:
        return True
        
    now = time.time()
    if client_ip not in rate_limit_store:
        rate_limit_store[client_ip] = []
    
    # Remove old requests outside the window
    rate_limit_store[client_ip] = [req_time for req_time in rate_limit_store[client_ip] 
                                   if now - req_time < window]
    
    # Check if under limit
    if len(rate_limit_store[client_ip]) >= limit:
        return False
    
    # Add current request
    rate_limit_store[client_ip].append(now)
    return True

def generate_csrf_token():
    """Generate CSRF token for session"""
    if 'csrf_token' not in session:
        session['csrf_token'] = secrets.token_urlsafe(32)
    return session['csrf_token']

def validate_csrf_token(token):
    """Validate CSRF token"""
    return 'csrf_token' in session and session['csrf_token'] == token

@app.after_request
def add_security_headers(response):
    """Add security headers to prevent common web vulnerabilities"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Strict-Transport-Security'] = 'max-age=31536000; includeSubDomains'
    response.headers['Content-Security-Policy'] = "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline';"
    return response

ALLOWED_EXTENSIONS = {'stl'}

def allowed_file(filename):
    """Check if file has allowed extension"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def analyze_stl_file(file_path):
    """Analyze STL file and return metadata"""
    try:
        mesh_data = mesh.Mesh.from_file(file_path)
        analysis = analyze_stl_mesh(mesh_data)
        return analysis
    except (IOError, OSError) as e:
        logger.error(f"File system error analyzing STL file: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error analyzing STL file: {str(e)}")
        return None

def optimize_stl_file_wrapper(file_path, optimization_level='medium'):
    """
    Optimize STL file based on optimization level
    Returns path to optimized file
    """
    try:
        # Use the optimizer module to get optimized mesh
        optimized_mesh = optimize_stl_file(file_path, optimization_level)
        
        # Save optimized file
        optimized_filename = f"optimized_{uuid.uuid4().hex[:8]}.stl"
        optimized_path = os.path.join(app.config['UPLOAD_FOLDER'], optimized_filename)
        optimized_mesh.save(optimized_path)
        
        return optimized_path
        
    except (IOError, OSError) as e:
        logger.error(f"File system error optimizing STL file: {str(e)}")
        raise
    except ValueError as e:
        logger.error(f"Invalid parameters for STL optimization: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error optimizing STL file: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the main page"""
    # Generate CSRF token for the session
    csrf_token = generate_csrf_token()
    return render_template('index.html', csrf_token=csrf_token)

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle STL file upload"""
    client_ip = request.remote_addr
    
    # Check rate limiting
    if not check_rate_limit(client_ip, limit=5, window=60):
        return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
    
    # Check CSRF token for API requests
    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': 'CSRF token missing or invalid'}), 403
    
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Only STL files are allowed'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}_{filename}")
        file.save(upload_path)
        
        # Analyze the file
        analysis = analyze_stl_file(upload_path)
        if not analysis:
            os.remove(upload_path)
            return jsonify({'error': 'Failed to analyze STL file'}), 400
        
        return jsonify({
            'file_id': file_id,
            'filename': filename,
            'analysis': analysis,
            'upload_time': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': 'File upload failed'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_file():
    """Optimize uploaded STL file"""
    client_ip = request.remote_addr
    
    # Check rate limiting
    if not check_rate_limit(client_ip, limit=10, window=60):
        return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
    
    # Check CSRF token for API requests
    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': 'CSRF token missing or invalid'}), 403
    
    try:
        data = request.get_json()
        if not data or 'file_id' not in data:
            return jsonify({'error': 'File ID required'}), 400
        
        file_id = data['file_id']
        optimization_level = data.get('level', 'medium')
        
        # Validate optimization level
        if optimization_level not in ['light', 'medium', 'aggressive']:
            return jsonify({'error': 'Invalid optimization level'}), 400
        
        # Find the uploaded file
        upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
        if not upload_files:
            return jsonify({'error': 'File not found'}), 404
        
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], upload_files[0])
        
        # Optimize the file
        optimized_path = optimize_stl_file_wrapper(upload_path, optimization_level)
        
        # Get file info
        original_size = os.path.getsize(upload_path)
        optimized_size = os.path.getsize(optimized_path)
        compression_ratio = (1 - optimized_size / original_size) * 100
        
        # Analyze optimized file
        optimized_analysis = analyze_stl_file(optimized_path)
        
        return jsonify({
            'optimization_level': optimization_level,
            'original_size': original_size,
            'optimized_size': optimized_size,
            'compression_ratio': round(compression_ratio, 2),
            'optimized_analysis': optimized_analysis,
            'download_id': os.path.basename(optimized_path)
        })
        
    except Exception as e:
        logger.error(f"Optimization error: {str(e)}")
        return jsonify({'error': 'Optimization failed'}), 500

@app.route('/api/download/<filename>')
def download_file(filename):
    """Download optimized STL file"""
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
        
    except Exception as e:
        logger.error(f"Download error: {str(e)}")
        return jsonify({'error': 'Download failed'}), 500

@app.route('/api/cleanup', methods=['POST'])
def cleanup_files():
    """Clean up temporary files"""
    client_ip = request.remote_addr
    
    # Check rate limiting (more lenient for cleanup)
    if not check_rate_limit(client_ip, limit=20, window=60):
        return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429
    
    # Check CSRF token for API requests
    csrf_token = request.headers.get('X-CSRF-Token')
    if not csrf_token or not validate_csrf_token(csrf_token):
        return jsonify({'error': 'CSRF token missing or invalid'}), 403
    
    try:
        data = request.get_json()
        if data and 'file_id' in data:
            # Clean up specific files
            file_id = data['file_id']
            upload_files = [f for f in os.listdir(app.config['UPLOAD_FOLDER']) if f.startswith(file_id)]
            for file in upload_files:
                try:
                    os.remove(os.path.join(app.config['UPLOAD_FOLDER'], file))
                except:
                    pass
        else:
            # Clean up old files (older than 1 hour)
            current_time = datetime.now()
            for filename in os.listdir(app.config['UPLOAD_FOLDER']):
                if filename.endswith('.stl'):
                    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                    file_time = datetime.fromtimestamp(os.path.getctime(file_path))
                    if (current_time - file_time).seconds > 3600:
                        try:
                            os.remove(file_path)
                        except:
                            pass
        
        return jsonify({'message': 'Cleanup completed'})
        
    except Exception as e:
        logger.error(f"Cleanup error: {str(e)}")
        return jsonify({'error': 'Cleanup failed'}), 500

if __name__ == '__main__':
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(debug=debug_mode, host='0.0.0.0', port=5000)
