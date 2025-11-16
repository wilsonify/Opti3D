import io
import os
import uuid
import time
from werkzeug.datastructures import FileStorage
from flask import json

import src.app as app_module


def test_validate_and_save_upload(tmp_path, monkeypatch):
    # Create a small fake STL content
    content = b"solid test\nendsolid test\n"
    stream = io.BytesIO(content)
    fs = FileStorage(stream=stream, filename="test.stl", content_type="application/octet-stream")

    # Ensure upload folder is a temp dir for the test
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)

    upload_path, filename, file_size, file_id = app_module._validate_and_save_upload(fs)

    assert os.path.exists(upload_path)
    assert filename == "test.stl"
    assert file_size == len(content)
    # cleanup
    os.remove(upload_path)


def test_analyze_and_build_response_monkeypatched(tmp_path, monkeypatch):
    # Create dummy upload file
    file_path = tmp_path / "dummy.stl"
    file_path.write_bytes(b"solid test\nendsolid test\n")

    # Monkeypatch analyze_stl_file to avoid heavy parsing
    monkey_result = {'triangles': 1, 'vertices': 3}
    monkeypatch.setattr(app_module, 'analyze_stl_file', lambda p: dict(monkey_result))

    start_time = time.time()
    client_ip = '127.0.0.1'

    # Need app context for jsonify
    with app_module.app.test_request_context():
        response, status = app_module._analyze_and_build_response(str(file_path), 'dummy.stl', file_path.stat().st_size, 'fid123', start_time, client_ip)

        assert status == 200
        data = json.loads(response.get_data(as_text=True))
        assert data['file_id'] == 'fid123'
        assert data['filename'] == 'dummy.stl'
        assert 'analysis' in data


def test_parse_optimize_request_and_find_upload_path(tmp_path):
    # Ensure app is in testing mode so CSRF passes
    app_module.app.testing = True

    # Create a fake uploaded file with file_id prefix
    file_id = str(uuid.uuid4())
    filename = f"{file_id}_uploaded.stl"
    upload_folder = tmp_path
    app_module.app.config['UPLOAD_FOLDER'] = str(upload_folder)
    file_path = upload_folder / filename
    file_path.write_bytes(b"solid test\nendsolid test\n")

    # Prepare JSON body
    json_body = {'file_id': file_id, 'level': 'medium'}

    # Call _parse_optimize_request within a request context (include CSRF header)
    headers = {'X-CSRF-Token': 'testtoken'}
    with app_module.app.test_request_context(json=json_body, headers=headers):
        upload_path, returned_file_id, level = app_module._parse_optimize_request()

        assert returned_file_id == file_id
        assert level == 'medium'
        assert upload_path == str(file_path)


    def test_upload_errors_and_success(tmp_path, monkeypatch):
        app_module.app.testing = True
        app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
        client = app_module.app.test_client()

        # No file provided
        r = client.post('/api/upload', data={})
        assert r.status_code == 400

        # Invalid extension
        data = {'file': (io.BytesIO(b"not an stl"), 'bad.txt')}
        r = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert r.status_code == 400

        # Successful upload (monkeypatch analysis to avoid heavy work)
        def fake_analyze(upload_path, filename, file_size, file_id, start_time, client_ip):
            return app_module.jsonify({'file_id': file_id, 'filename': filename, 'analysis': {}}), 200

        monkeypatch.setattr(app_module, '_analyze_and_build_response', fake_analyze)

        data = {'file': (io.BytesIO(b"solid test\nendsolid test\n"), 'good.stl')}
        r = client.post('/api/upload', data=data, content_type='multipart/form-data')
        assert r.status_code == 200


    def test_optimize_endpoints(tmp_path, monkeypatch):
        app_module.app.testing = True
        app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
        client = app_module.app.test_client()

        # Missing file_id
        r = client.post('/api/optimize', json={})
        assert r.status_code == 400

        # Successful optimization path (monkeypatch wrapper and analysis)
        file_id = 'fid-opt-12345'
        filename = f"{file_id}_in.stl"
        orig = tmp_path / filename
        orig.write_bytes(b"solid\nendsolid\n")

        optpath = tmp_path / 'opt.stl'
        optpath.write_bytes(b"solid\nendsolid\n")

        monkeypatch.setattr(app_module, 'optimize_stl_file_wrapper', lambda p, level='medium': str(optpath))
        monkeypatch.setattr(app_module, 'analyze_stl_file', lambda p: {'triangles': 1})

        r = client.post('/api/optimize', json={'file_id': file_id, 'level': 'medium'})
        assert r.status_code == 200
        j = r.get_json()
        assert 'optimized_size' in j or 'optimized_analysis' in j


    def test_health_and_info_and_metrics_endpoints(tmp_path):
        # Set upload folder
        app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)

        client = app_module.app.test_client()

        # Health check
        resp = client.get('/api/health')
        assert resp.status_code in (200, 503)
        data = resp.get_json()
        assert 'status' in data

        # App info
        info = client.get('/api/info')
        assert info.status_code == 200
        js = info.get_json()
        assert js.get('name') == 'Opti3D'

        # Metrics with a file
        fpath = tmp_path / 'somefile.stl'
        fpath.write_bytes(b"solid t\nendsolid t\n")
        metrics = client.get('/api/metrics')
        assert metrics.status_code == 200
        mjs = metrics.get_json()
        assert 'metrics' in mjs


    def test_download_and_cleanup_helpers(tmp_path):
        # Prepare upload folder and files
        app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)

        # Invalid download filename patterns should be rejected
        client = app_module.app.test_client()
        r = client.get('/api/download/../../etc')
        assert r.status_code == 400

        # Create a file with file_id prefix and test cleanup helper
        fid = 'cleanupid12345'
        fname = f"{fid}_f.stl"
        p = tmp_path / fname
        p.write_bytes(b"solid\nendsolid\n")

        removed = app_module._cleanup_by_file_id(fid)
        # Should remove the file we just created
        assert removed >= 1
