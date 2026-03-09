import logging.config
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import yaml


def setup_logging():
    config_file = (
        Path(__file__).resolve().parent.parent.parent / "config" / "logging_config.yaml"
    )
    if config_file.exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            try:
                logging.config.dictConfig(config)
            except Exception as e:
                logging.basicConfig(level=logging.INFO)
                print(f"DEBUG: Logging config failed, using basic config. Error: {e}")
    else:
        logging.basicConfig(level=logging.INFO)

    # Suppress noisy third-party loggers
    logging.getLogger("RapidOCR").setLevel(logging.WARNING)
    logging.getLogger("docling").setLevel(logging.WARNING)

    # Suppress harmless Pydantic serialization warning emitted by LangChain's
    # with_structured_output() when it stores a parsed model on AIMessage.parsed.
    warnings.filterwarnings(
        "ignore",
        message=r"Pydantic serializer warnings",
        category=UserWarning,
        module=r"pydantic\.main",
    )


@contextmanager
def suppress_stderr():
    """Temporarily suppress C++ stderr output (e.g. std::bad_alloc from docling)."""
    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, stderr_fd)
        yield
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull)


setup_logging()
logger = logging.getLogger("src")
