# src/config.py
from pathlib import Path
from dotenv import load_dotenv
import os
import random
import numpy as np
from loguru import logger


def set_seed(seed: int = 42):
    """
    Set Python and NumPy seeds for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)


def init_logger(level: str = "INFO"):
    """
    Configure loguru logger with only level and message.
    """
    # Remove default handler to avoid duplicate logs
    logger.remove()

    # Console sink: only level + message
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        enqueue=False,
        backtrace=True,
        diagnose=True,
        format="<level>{level: <5}</level> | {message}"
    )

def load_config():
    """
    Load configuration from a .env file located at the repo root.
    Returns a dictionary of configuration values.
    Also applies seeding and sets up logging.
    """
    # Load closest .env up from repo root
    env_path = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(env_path, override=False)

    cfg = {
        "OPENAI_BASE_URL": os.getenv("OPENAI_BASE_URL", "").strip(),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "").strip(),
        "MB_CHAT_MODEL": os.getenv("MB_CHAT_MODEL", "gemini-flash").strip(),
        "MB_EMBED_MODEL": os.getenv("MB_EMBED_MODEL", "gemini-embedding").strip(),
        "MB_TOP_K": int(os.getenv("MB_TOP_K", "5")),
        "MB_RESULTS_DIR": os.getenv("MB_RESULTS_DIR", "./benchmark/results"),
        "SEED": int(os.getenv("SEED", "42")),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }

    # Apply seed globally
    set_seed(cfg["SEED"])

    # Initialize logging
    init_logger(level=cfg["LOG_LEVEL"])

    return cfg
