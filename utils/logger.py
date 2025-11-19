"""
logger.py

Provides utilities to create a timestamped experiment logger.

This module simplifies logging for experiments or scripts by automatically
creating a timestamped experiment folder and configuring a logger that writes
to both a log file and optionally to the console. Useful for machine learning
experiments, data processing scripts, or any project where organized logging
is desired.

Functions:
----------
get_logger(exp_root="experiments", log_name="train.log")
    Creates a logger and a timestamped experiment folder, returning both
    for use in scripts.
"""

from datetime import datetime
import logging
import os


def get_logger(exp_root="experiments", log_name="train.log", exp_dir=None):
    """
    Set up a logger that writes logs to a timestamped experiment folder.

    Args:
        exp_root (str, optional): Root directory to store experiment folders. Defaults to "experiments".
        log_name (str, optional): Name of the log file. Defaults to "train.log".

    Returns:
        tuple:
            logger (logging.Logger): Configured logger object.
            exp_dir (str): Path to the created timestamped experiment folder.

    Notes:
        - A new folder is created inside `exp_root` with the current timestamp (YYYYMMDD_HHMMSS).
        - Logger writes to both the log file and the console.
        - Calling this function multiple times avoids adding duplicate handlers.
    """

    # Ensure root experiments folder exists
    os.makedirs(exp_root, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create a timestamped experiment folder
    if exp_dir is None:
        exp_dir = os.path.join(exp_root, timestamp)
        os.makedirs(exp_dir, exist_ok=True)
    else:
        # Make sure folder exists
        os.makedirs(exp_dir, exist_ok=True)

    # Setup logger
    logger = logging.getLogger(timestamp)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if this function is called multiple times
    if not logger.handlers:
        fh = logging.FileHandler(os.path.join(exp_dir, log_name))
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        # Optional: also log to console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger, exp_dir
