import logging
from typing import Optional


def setup_logger(
    name: str, log_file: Optional[str] = None, level=logging.INFO, propaget: bool = True
) -> logging.Logger:
    """
    Creates and configures a logger with the specified name.

    If a logger with the given name already exists, it reuses it. If no file handler has been added yet,
    and a log_file is provided, it adds a FileHandler to write logs to the specified file.

    Note:
        - Logger names are unique. If you reuse the same name with a different log_file, the logger will still
          write to the original file unless you first clean it up using `cleanup_logger`.
        - By default, log messages propagate to the root logger, which may result in console output.
          Set `propaget=False` to prevent this.

    Args:
        name (str): The name of the logger.
        log_file (Optional[str]): Path to the log file. If None, no file handler is added.
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG).
        propaget (bool): Whether to propagate log messages to the root logger.

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = propaget

    # Prevent adding multiple handlers if logger already exists
    if log_file is not None and not logger.handlers:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def cleanup_logger(logger: logging.Logger):
    """
    Removes all handlers from the given logger and closes them.

    This is useful when you want to reconfigure a logger with a new file handler or prevent duplicate logging.

    Args:
        logger (logging.Logger): The logger instance to clean up.
    """
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)
