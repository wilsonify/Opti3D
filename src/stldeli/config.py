import os

log_dir = 'logs'

log_dict_config = {
    'formatters': {
        'default': {
            'format': '%(asctime)s | %(levelname)s | %(filename)s | %(name)s | %(lineno)d | %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            'class': 'logging.StreamHandler',
            'formatter': 'default'
        },
        'file': {
            'level': 'DEBUG',
            'class': ' logging.FileHandler',
            'formatter': 'default',
            'filename': os.path.join(log_dir, 'deli.log'),
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'DEBUG',
    },
}
