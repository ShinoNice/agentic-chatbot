import logging
import logging.config
import os
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import yaml


class _RequestIdFilter(logging.Filter):
    """Inject the current request's correlation ID into every log record.

    Reads from ``src.api.middleware.request_id_var`` lazily so non-API
    entry points (CLI, scripts, tests) that don't import middleware still
    log cleanly — the field just defaults to ``"-"``.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            from src.api.middleware import get_request_id

            record.request_id = get_request_id() or "-"
        except Exception:
            record.request_id = "-"
        return True


def setup_logging():
    config_file = Path(__file__).resolve().parent.parent.parent / "config" / "logging_config.yaml"
    if config_file.exists():
        with open(config_file) as f:
            config = yaml.safe_load(f)
        # Pre-create parent dirs for any file-based handlers so the config
        # doesn't crash on fresh machines / containers where data/logs/ is
        # gitignored or .dockerignored.
        for handler in (config.get("handlers") or {}).values():
            filename = handler.get("filename")
            if filename:
                Path(filename).parent.mkdir(parents=True, exist_ok=True)
        try:
            logging.config.dictConfig(config)
        except Exception as e:
            logging.basicConfig(level=logging.INFO)
            print(f"DEBUG: Logging config failed, using basic config. Error: {e}")
    else:
        logging.basicConfig(level=logging.INFO)

    # Attach the correlation-ID filter to the project logger so every record
    # emitted under the "src" namespace carries a request_id field.
    _src_logger = logging.getLogger("src")
    if not any(isinstance(f, _RequestIdFilter) for f in _src_logger.filters):
        _src_logger.addFilter(_RequestIdFilter())

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
