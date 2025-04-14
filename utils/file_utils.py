from core.retriever import Retriever
from pathlib import Path
from datetime import datetime
from typing import Dict
import json
import streamlit as st

from utils.logger import AppLogger

logger = AppLogger(name="FileUtils").get_logger()


def process_uploaded_files(
    uploaded_files, destination_dir: Path, library: dict, library_path: Path
):
    """Save uploaded files, update metadata, chunk documents, and save library"""
    saved_files = []

    # Save uploaded files to destination_dir
    for uploaded_file in uploaded_files:
        if hasattr(uploaded_file, "getbuffer"):
            # Streamlit uploaded file
            file_path = destination_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        else:
            # Directly passed from filesystem (e.g. watcher)
            file_path = Path(uploaded_file)
            if not file_path.exists():
                continue  # Skip if file doesn't exist

        saved_files.append(file_path)

        # Register file into library so retriever sees it
        file_info = file_path.stat()
        library[file_path.name] = {
            "file_name": file_path.name,
            "file_path": str(file_path.resolve()),
            "size_kb": round(file_info.st_size / 1024, 2),
            "created_at": datetime.fromtimestamp(file_info.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(file_info.st_mtime).isoformat(),
        }

        file_info = file_path.stat()
        st.success(f"✅ Uploaded: {uploaded_file.name}")
        st.markdown(f"- **Size:** {file_info.st_size / 1024:.2f} KB")
        st.markdown(
            f"- **Uploaded At:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        st.markdown("---")

    # Chunk and update library with chunks/num_chunks
    library, chunks, sources = Retriever.load_and_chunk_files(library, library_path)

    return library, chunks, sources


def load_file_library(path: Path) -> Dict[str, dict]:
    """Load metadata from local JSON cache"""
    if not path.exists():
        path.write_text(json.dumps({}), encoding="utf-8")
        return {}

    with open(path, "r") as f:
        library = json.load(f)
        return library


def save_file_library(library: Dict[str, dict], path: Path):
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(library, f, indent=2)
    print(f"Saved library to {path}")


def update_library_with_file(file_path: Path) -> dict:
    """Create a metadata entry for a file"""
    file_info = file_path.stat()
    return {
        "file_name": file_path.name,
        "file_path": str(file_path.resolve()),
        "size_kb": round(file_info.st_size / 1024, 2),
        "created_at": datetime.fromtimestamp(file_info.st_ctime).isoformat(),
        "modified_at": datetime.fromtimestamp(file_info.st_mtime).isoformat(),
    }

def load_config( path: str) -> dict:
    """Load configuration file from JSON file."""
    try:
        import json

        with open(path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}