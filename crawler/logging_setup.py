"""
logging_setup.py — call setup() once at startup before anything else.
"""

import logging
import sys


def setup(level: int = logging.INFO):
    root = logging.getLogger()
    if root.handlers:
        return  # already configured

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s  %(levelname)-8s  %(message)s",
            datefmt="%H:%M:%S",
        )
    )

    root.setLevel(level)
    root.addHandler(handler)
