import logging


LOG_LEVEL = {
    'critical': logging.CRITICAL,
    'error': logging.ERROR,
    'warning': logging.WARNING,
    'info': logging.INFO,
    'debug': logging.DEBUG,
}


def set_log_level(level: int) -> None:
    """Set log level."""
    logging.basicConfig(
        format='[%(levelname)s:%(asctime)s:%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=level,
    )
