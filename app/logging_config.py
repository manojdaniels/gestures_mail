import logging
import os
from logging.handlers import RotatingFileHandler
from .config import settings


def configure_logging() -> None:
    os.makedirs(os.path.join(settings.storage_dir, "logs"), exist_ok=True)
    log_file_path = os.path.join(settings.storage_dir, "logs", "app.log")

    root_logger = logging.getLogger()
    if root_logger.handlers:
        # Already configured
        return

    root_logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(log_file_path, maxBytes=5_000_000, backupCount=3)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)