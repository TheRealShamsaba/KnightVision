import logging
import os


def configure_logging(level: str | None = None) -> None:
    """Configure basic logging for the project."""
    level_str = level or os.getenv("LOG_LEVEL", "INFO")
    numeric_level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(level=numeric_level,
                        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    logging.getLogger().setLevel(numeric_level)
