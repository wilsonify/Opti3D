"""
Configuration management for Opti3D application
"""

import os
import tempfile
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SecurityConfig:
    """Security-related configuration."""
    secret_key: str
    csrf_enabled: bool = True
    rate_limit_enabled: bool = True
    rate_limits: Dict[str, int] = field(default_factory=lambda: {
        'upload': 5,
        'optimize': 10,
        'cleanup': 20,
        'download': 30
    })
    rate_limit_window: int = 60  # seconds
    max_content_length: int = 100 * 1024 * 1024  # 100MB
    allowed_extensions: set = field(default_factory=lambda: {'stl'})


@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = 'INFO'
    format: str = '%(asctime)s | %(levelname)s | %(filename)s | %(name)s | %(lineno)d | %(message)s'
    log_dir: str = 'logs'
    enable_file_logging: bool = True
    enable_console_logging: bool = True
    log_rotation: bool = True
    max_log_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


@dataclass
class PerformanceConfig:
    """Performance-related configuration."""
    upload_folder: str = '/tmp/uploads'
    file_expiry_time: int = 3600  # seconds
    cleanup_interval: int = 300  # seconds
    max_concurrent_uploads: int = 10
    optimization_timeout: int = 300  # seconds
    enable_compression: bool = True
    cache_enabled: bool = False
    cache_ttl: int = 3600  # seconds


@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = '127.0.0.1'
    port: int = 5000
    debug: bool = False
    environment: str = 'development'
    proxy_fix: bool = True
    workers: int = 1


@dataclass
class OptimizationConfig:
    """STL optimization configuration."""
    default_level: str = 'medium'
    tolerance_light: float = 0.001
    tolerance_medium: float = 0.01
    tolerance_aggressive: float = 0.1
    smoothing_iterations: int = 1
    min_triangle_area: float = 1e-10
    max_vertices: int = 1000000  # Safety limit


@dataclass
class AppConfig:
    """Main application configuration."""
    security: SecurityConfig
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        # Security configuration
        secret_key = os.environ.get('SECRET_KEY')
        if not secret_key and os.environ.get('FLASK_ENV', 'development') == 'production':
            raise ValueError("SECRET_KEY environment variable must be set in production")
        
        if not secret_key:
            import secrets
            secret_key = secrets.token_hex(32)
        
        security = SecurityConfig(
            secret_key=secret_key,
            csrf_enabled=os.environ.get('CSRF_ENABLED', 'true').lower() == 'true',
            rate_limit_enabled=os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true',
            max_content_length=int(os.environ.get('MAX_CONTENT_LENGTH', 100 * 1024 * 1024)),
        )
        
        # Logging configuration
        logging_config = LoggingConfig(
            level=os.environ.get('LOG_LEVEL', 'INFO').upper(),
            log_dir=os.environ.get('LOG_DIR', 'logs'),
            enable_file_logging=os.environ.get('ENABLE_FILE_LOGGING', 'true').lower() == 'true',
            enable_console_logging=os.environ.get('ENABLE_CONSOLE_LOGGING', 'true').lower() == 'true',
        )
        
        # Performance configuration
        performance = PerformanceConfig(
            upload_folder=os.environ.get('UPLOAD_FOLDER', tempfile.gettempdir()),
            file_expiry_time=int(os.environ.get('FILE_EXPIRY_TIME', 3600)),
            cleanup_interval=int(os.environ.get('CLEANUP_INTERVAL', 300)),
            max_concurrent_uploads=int(os.environ.get('MAX_CONCURRENT_UPLOADS', 10)),
            optimization_timeout=int(os.environ.get('OPTIMIZATION_TIMEOUT', 300)),
            enable_compression=os.environ.get('ENABLE_COMPRESSION', 'true').lower() == 'true',
            cache_enabled=os.environ.get('CACHE_ENABLED', 'false').lower() == 'true',
            cache_ttl=int(os.environ.get('CACHE_TTL', 3600)),
        )
        
        # Server configuration
        server = ServerConfig(
            host=os.environ.get('FLASK_HOST', '127.0.0.1'),
            port=int(os.environ.get('FLASK_PORT', 5000)),
            debug=os.environ.get('FLASK_DEBUG', 'false').lower() == 'true',
            environment=os.environ.get('FLASK_ENV', 'development'),
            proxy_fix=os.environ.get('PROXY_FIX', 'true').lower() == 'true',
            workers=int(os.environ.get('WORKERS', 1)),
        )
        
        # Optimization configuration
        optimization = OptimizationConfig(
            default_level=os.environ.get('DEFAULT_OPTIMIZATION_LEVEL', 'medium'),
            tolerance_light=float(os.environ.get('TOLERANCE_LIGHT', 0.001)),
            tolerance_medium=float(os.environ.get('TOLERANCE_MEDIUM', 0.01)),
            tolerance_aggressive=float(os.environ.get('TOLERANCE_AGGRESSIVE', 0.1)),
            smoothing_iterations=int(os.environ.get('SMOOTHING_ITERATIONS', 1)),
            min_triangle_area=float(os.environ.get('MIN_TRIANGLE_AREA', 1e-10)),
            max_vertices=int(os.environ.get('MAX_VERTICES', 1000000)),
        )
        
        return cls(
            security=security,
            logging=logging_config,
            performance=performance,
            server=server,
            optimization=optimization,
        )
    
    def validate(self) -> None:
        """Validate configuration values."""
        # Validate security
        if not self.security.secret_key:
            raise ValueError("Secret key is required")
        
        if self.security.max_content_length <= 0:
            raise ValueError("Max content length must be positive")
        
        # Validate logging
        valid_log_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if self.logging.level not in valid_log_levels:
            raise ValueError(f"Invalid log level: {self.logging.level}")
        
        # Validate performance
        if not os.path.exists(self.performance.upload_folder):
            try:
                os.makedirs(self.performance.upload_folder, exist_ok=True)
            except OSError as e:
                raise ValueError(f"Cannot create upload directory: {e}")
        
        if not os.access(self.performance.upload_folder, os.W_OK):
            raise ValueError(f"Upload directory is not writable: {self.performance.upload_folder}")
        
        # Validate server
        if self.server.port < 1 or self.server.port > 65535:
            raise ValueError("Port must be between 1 and 65535")
        
        # Validate optimization
        valid_levels = {'light', 'medium', 'aggressive'}
        if self.optimization.default_level not in valid_levels:
            raise ValueError(f"Invalid optimization level: {self.optimization.default_level}")
        
        if any(t < 0 for t in [
            self.optimization.tolerance_light,
            self.optimization.tolerance_medium,
            self.optimization.tolerance_aggressive
        ]):
            raise ValueError("Tolerance values must be non-negative")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'security': {
                'csrf_enabled': self.security.csrf_enabled,
                'rate_limit_enabled': self.security.rate_limit_enabled,
                'rate_limits': self.security.rate_limits,
                'max_content_length': self.security.max_content_length,
                'allowed_extensions': list(self.security.allowed_extensions),
            },
            'logging': {
                'level': self.logging.level,
                'log_dir': self.logging.log_dir,
                'enable_file_logging': self.logging.enable_file_logging,
                'enable_console_logging': self.logging.enable_console_logging,
            },
            'performance': {
                'upload_folder': self.performance.upload_folder,
                'file_expiry_time': self.performance.file_expiry_time,
                'cleanup_interval': self.performance.cleanup_interval,
                'max_concurrent_uploads': self.performance.max_concurrent_uploads,
                'optimization_timeout': self.performance.optimization_timeout,
                'enable_compression': self.performance.enable_compression,
                'cache_enabled': self.performance.cache_enabled,
            },
            'server': {
                'host': self.server.host,
                'port': self.server.port,
                'debug': self.server.debug,
                'environment': self.server.environment,
            },
            'optimization': {
                'default_level': self.optimization.default_level,
                'tolerance_light': self.optimization.tolerance_light,
                'tolerance_medium': self.optimization.tolerance_medium,
                'tolerance_aggressive': self.optimization.tolerance_aggressive,
                'smoothing_iterations': self.optimization.smoothing_iterations,
            }
        }


# Global configuration instance
config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global config
    if config is None:
        config = AppConfig.from_env()
        config.validate()
    return config


def reload_config() -> AppConfig:
    """Reload configuration from environment."""
    global config
    config = AppConfig.from_env()
    config.validate()
    return config
