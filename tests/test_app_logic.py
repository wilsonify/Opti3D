import time
from flask import session
import src.app as app_module
from src.app import ValidationError, FileProcessingError, OptimizationError


def test_rate_limiter(tmp_path):
    # ensure deterministic
    orig_testing = app_module.app.testing
    app_module.app.testing = False
    client_ip = '1.2.3.4'

    # Reset store
    if client_ip in app_module.rate_limit_store:
        del app_module.rate_limit_store[client_ip]

    assert app_module.check_rate_limit(client_ip, limit=2, window=10) is True
    assert app_module.check_rate_limit(client_ip, limit=2, window=10) is True
    # third should be blocked
    assert app_module.check_rate_limit(client_ip, limit=2, window=10) is False

    # restore
    app_module.app.testing = orig_testing


def test_csrf_token_generation_and_validation():
    orig_testing = app_module.app.testing
    app_module.app.testing = False
    try:
        with app_module.app.test_request_context('/'):
            # token generation places token in session
            token = app_module.generate_csrf_token()
            assert 'csrf_token' in session
            assert app_module.validate_csrf_token(token) is True
            assert app_module.validate_csrf_token('wrong') is False
    finally:
        app_module.app.testing = orig_testing


def test_analyze_stl_file_not_found():
    # Depending on implementation details this may raise FileProcessingError or return None
    try:
        result = app_module.analyze_stl_file('/path/does/not/exist.stl')
        assert result is None
    except FileProcessingError:
        # acceptable
        pass


def test_optimize_route_handles_optimization_error(tmp_path, monkeypatch):
    app_module.app.testing = True
    app_module.app.config['UPLOAD_FOLDER'] = str(tmp_path)

    # create a fake original file so _find_upload_path_for_file_id passes
    fid = 'othertestfid'
    fname = f"{fid}_a.stl"
    p = tmp_path / fname
    p.write_bytes(b"solid\nendsolid\n")

    # Monkeypatch wrapper to raise
    def raise_opt(path, level='medium'):
        raise OptimizationError('boom')

    monkeypatch.setattr(app_module, 'optimize_stl_file_wrapper', raise_opt)

    client = app_module.app.test_client()
    resp = client.post('/api/optimize', json={'file_id': fid, 'level': 'medium'}, headers={'X-CSRF-Token': 'testtoken'})
    assert resp.status_code == 500
