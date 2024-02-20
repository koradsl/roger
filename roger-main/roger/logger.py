import logging
import os


def set_logger() -> None:
    """Logging level : dev, prod"""
    logging_level = os.getenv("ROGER_LOG_LEVEL", "dev")

    if logging_level == "dev":
        level = logging.DEBUG
    elif logging_level == "prod":
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logging.basicConfig(format="[%(asctime)s][%(levelname)s] %(message)s", level=level)
