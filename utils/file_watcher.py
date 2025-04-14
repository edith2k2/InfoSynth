import time
from pathlib import Path
from typing import List
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from utils.file_utils import (
    get_mtime,
    load_file_library,
    process_uploaded_files,
)
from utils.logger import AppLogger
from utils.watcher_state import watcher_state

logger = AppLogger("Watcher").get_logger()


class UploadFolderHandler(FileSystemEventHandler):
    def __init__(self, upload_dir: Path, library_path: Path):
        self.upload_dir = upload_dir
        self.library_path = library_path
        mtime = get_mtime(self.library_path)
        self.library = load_file_library(library_path, mtime=mtime)

    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            ext = file_path.suffix.lower().lstrip(".")

            allowed_exts = [
                e.lower().lstrip(".") for e in self.config.get("allowed_extensions", [])
            ]

            if ext in allowed_exts:
                logger.info(f"New downloaded file detected: {file_path.name}")
                try:
                    mtime = get_mtime(self.library_path)
                    self.library = load_file_library(self.library_path, mtime=mtime)
                    self.library = process_uploaded_files(
                        [file_path], self.upload_dir, self.library, self.library_path
                    )
                    watcher_state.files_changed = True  # Signal change
                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")
