import json
from pathlib import Path

from utils import logger

logger = logger.AppLogger(name="ConfigManager").get_logger()


def get_default_config() -> dict:
    """
    Returns the default configuration for the application.
    """
    return {
        "upload_dir": "data/uploads",
        "library_path": "data/library.json",
        "allowed_extensions": [
            "txt",
            "pdf",
            "docx",
            "json",
            "csv",
            "md",
            "html",
            "rtf",
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".tiff",
        ],
        "watch_folders": [],
        "page_title": "InfoSynth",
        "page_layout": "wide",
        "top_k": 5,
        "max_search_steps": 3,
    }


def update_config_key(config_path: Path, key: str, value):
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        config[key] = value
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        return config
    except Exception as e:
        logger.error(f"Failed to update config key {key}: {e}")
        return None
