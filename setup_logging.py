# collect all loggers in one file
import logging

consoleLogger = logging.getLogger("consoleLogger")
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
consoleLogger.addHandler(console_handler)
consoleLogger.setLevel(logging.INFO)


def setup_fileLogger(logger_name: str, log_name: str):
    logger = logging.getLogger(logger_name)
    handler = logging.FileHandler(f"logs/{log_name}")
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    return logger
