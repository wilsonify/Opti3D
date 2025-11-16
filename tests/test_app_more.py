import io
import os
import time
import uuid
import pytest
from werkzeug.datastructures import FileStorage

import src.app as app_module
from src.app import ValidationError, OptimizationError


def test_allowed_file_edge_cases(monkeypatch):
    # sanitize returns None -> invalid
    monkeypatch.setattr(app_module.InputSanitizer, 'sanitize_filename', staticmethod(lambda s: None))
    assert app_module.allowed_file('test.stl') is False

    # sanitize returns filename without extension
    monkeypatch.setattr(app_module.InputSanitizer, 'sanitize_filename', staticmethod(lambda s: 'noext'))
    assert app_module.allowed_file('noext') is False

    # sanitize returns filename with bad extension
    monkeypatch.setattr(app_module.InputSanitizer, 'sanitize_filename', staticmethod(lambda s: 'bad.txt'))
    assert app_module.allowed_file('bad.txt') is False

    # good filename
    monkeypatch.setattr(app_module.InputSanitizer, 'sanitize_filename', staticmethod(lambda s: 'good.stl'))
    assert app_module.allowed_file('good.stl') is True


def test_validate_and_save_upload_empty(tmp_path):
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    stream = io.BytesIO(b"")
    fs = FileStorage(stream=stream, filename='empty.stl')

    with pytest.raises(ValidationError):
        app_module._validate_and_save_upload(fs)


def test_analyze_and_build_response_cleans_file_on_error(tmp_path, monkeypatch):
    # create a dummy file
    p = tmp_path / 'to_analyze.stl'
    p.write_bytes(b"solid\nendsolid\n")

    # monkeypatch analyze_stl_file to raise
    def raise_fp(path):
        raise app_module.FileProcessingError('boom')

    monkeypatch.setattr(app_module, 'analyze_stl_file', raise_fp)

    with pytest.raises(app_module.FileProcessingError):
        app_module._analyze_and_build_response(str(p), 'to_analyze.stl', p.stat().st_size, 'fidx', time.time(), '127.0.0.1')

    assert not p.exists()


def test_safe_remove_and_cleanup_by_file_id(tmp_path):
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    # create some files
    fid = 'cleanupme12345'
    f1 = tmp_path / f"{fid}_a.stl"
    f1.write_bytes(b"solid\nendsolid\n")
    f2 = tmp_path / f"{fid}_b.txt"
    f2.write_text('noop')

    # safe remove returns True
    assert app_module._safe_remove(str(f1)) is True
    # removing again returns False
    assert app_module._safe_remove(str(f1)) is False

    # create files to test _cleanup_by_file_id
    f3 = tmp_path / f"{fid}_c.stl"
    f3.write_bytes(b"solid\nendsolid\n")
    removed = app_module._cleanup_by_file_id(fid)
    assert removed >= 1


def test_find_upload_path_for_file_id_not_found(tmp_path):
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    with pytest.raises(ValidationError):
        app_module._find_upload_path_for_file_id('doesnotexist')


def test_cleanup_expired_files_removes(tmp_path):
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    f = tmp_path / 'oldfile.stl'
    f.write_bytes(b"solid\nendsolid\n")
    # force expiry by passing negative expiry_seconds
    removed = app_module._cleanup_expired_files(expiry_seconds=-1)
    assert removed >= 1


def test_download_file_success(tmp_path):
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)
    filename = 'optimized_test123.stl'
    p = tmp_path / filename
    p.write_bytes(b"solid\nendsolid\n")

    client = app_module.app.test_client()
    resp = client.get(f'/api/download/{filename}')
    assert resp.status_code == 200
    # content-disposition should indicate attachment
    assert 'attachment' in resp.headers.get('Content-Disposition', '')


def test_optimize_wrapper_raises_on_none(tmp_path, monkeypatch):
    # create source file
    src = tmp_path / 's.stl'
    src.write_bytes(b"solid\nendsolid\n")

    # monkeypatch optimize_stl_file to return None
    monkeypatch.setattr(app_module, 'optimize_stl_file', lambda p, lvl='medium': None)

    with pytest.raises(OptimizationError):
        app_module.optimize_stl_file_wrapper(str(src), 'medium')
