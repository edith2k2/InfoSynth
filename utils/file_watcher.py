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
        if not event.is_directory and event.src_path.endswith(
            (
                ".pdf",
                ".txt",
                ".docx",
                ".json",
                ".csv",
                ".md",
                ".html",
                ".rtf",
                ".jpg",
                ".jpeg",
                ".png",
                "bmp",
                ".tiff",
            )
        ):
            file_path = Path(event.src_path)
            logger.info(f"New downloaded file detected: {file_path.name}")
            try:
                mtime = get_mtime(self.library_path)
                self.library = load_file_library(self.library_path, mtime=mtime)
                self.library = process_uploaded_files(
                    [file_path], self.upload_dir, self.library, self.library_path
                )
                watcher_state.files_changed = True  # Signal change
            except Exception as e:
                logger.error(f"Error: {e}")


def start_watcher(watch_dirs: List[Path], library_path: Path):
    if not watch_dirs:
        logger.warning("No watch folders provided. File watcher is disabled.")
        return

    logger.info(f"Monitoring {len(watch_dirs)} folder(s)...")
    mtime = get_mtime(library_path)
    library = load_file_library(library_path, mtime=mtime)

    observer = Observer()
    for directory in watch_dirs:
        directory.mkdir(parents=True, exist_ok=True)

        existing_files = {meta["file_name"] for meta in library.values()}
        all_files = (
            list(directory.glob("*.pdf"))
            + list(directory.glob("*.txt"))
            + list(directory.glob("*.docx"))
            + list(directory.glob("*.json"))
            + list(directory.glob("*.csv"))
            + list(directory.glob("*.md"))
            + list(directory.glob("*.html"))
            + list(directory.glob("*.rtf"))
            + list(directory.glob("*.jpg"))
            + list(directory.glob("*.jpeg"))
            + list(directory.glob("*.png"))
            + list(directory.glob("*.bmp"))
            + list(directory.glob("*.tiff"))
        )

        new_files = [f for f in all_files if f.name not in existing_files]

        if new_files:
            library = process_uploaded_files(
                new_files, directory, library, library_path
            )
            watcher_state.files_changed = True  # Signal change

        event_handler = UploadFolderHandler(directory, library_path)
        observer.schedule(event_handler, path=str(directory), recursive=False)
        logger.info(f"\t* {directory.resolve()}")

    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
