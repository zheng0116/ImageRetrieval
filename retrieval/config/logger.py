import sys
from loguru import logger


def set_logger(
    logger_name: str = "main",
    level: str = "INFO",
):
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        f"<level>{{level: <8}}</level> | "
        f"<cyan>{logger_name}</cyan>:<cyan>{{name}}</cyan>:<cyan>{{function}}</cyan>:<cyan>{{line}}</cyan> - "
        "<level>{message}</level>",
        level=level,
    )

    return logger
