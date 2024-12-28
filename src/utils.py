import logging
import os
import yaml
from datetime import datetime


def read_yaml(file_path):
    """
    Reads a YAML file and returns its contents as a Python object.
    :param file_path: Path to the YAML file.
    :return: Parsed contents of the YAML file as a Python dictionary or list.
    """
    try:
        with open(file_path, "r") as file:
            data = yaml.safe_load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return None


def init_logger(log_level=logging.INFO, log_format=None):
    """
    Initializes a logger that logs to both a file and the console.
    :param log_file: Path to the log file.
    :param log_level: Logging level (e.g., logging.DEBUG, logging.INFO).
    :param log_format: Custom format for log messages. If None, a default format is used.
    :return: Configured logger instance.
    """
    log_file = os.path.join(
        "..",
        "log",
        datetime.today().date().strftime(format="%d-%m-%Y"),
        f"logger_{datetime.now().time().strftime(format='%H hrs:%M min')}.log",
    )
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Create logs directory if it doesn't exist
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Create logger
    logger = logging.getLogger()
    logger.setLevel(log_level)

    # Avoid duplicate logs by clearing existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(logging.Formatter(log_format))

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


config = read_yaml(os.path.join("..", "config", "rag_config.yaml"))
logger = init_logger()
