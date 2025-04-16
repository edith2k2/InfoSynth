from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import json
import streamlit as st
import orjson
from utils.watcher_state import watcher_state
import re
import fitz
import docx
import csv
import markdown
from PIL import Image
from striprtf.striprtf import rtf_to_text
from bs4 import BeautifulSoup
import pytesseract
from math import isclose
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy

from utils.logger import AppLogger

logger = AppLogger(name="FileUtils").get_logger()


def show_status_message(message, type="info"):
    css_class = f"status-message status-{type}"
    st.markdown(
        f'<div class="{css_class}">{message}</div>',
        unsafe_allow_html=True,
    )


def get_mtime(file_path: Path) -> float:
    """Get the last modified time of a file"""
    try:
        return file_path.stat().st_mtime
    except FileNotFoundError:
        return 0.0


@st.cache_data(show_spinner=False)
def load_file_library(path_str: str, mtime: float) -> dict:
    path = Path(path_str)

    if not path.exists():
        path.write_text("{}", encoding="utf-8")
        return {}

    try:
        logger.info(f"Loading library from {path}")
        return orjson.loads(path.read_bytes())
    except orjson.JSONDecodeError:
        return {}


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


def load_and_chunk_files(
    library: dict, file_library_path: Path, chunk_size: int = 500
) -> Tuple[List[str], List[str]]:
    logger.info(f"Loading and chunking files from {file_library_path}")

    all_chunks = []
    all_sources = []
    updated = False
    new_library = copy.deepcopy(library)
    missing_files = set()

    def process_file(file_name: str):
        file_meta = library[file_name]
        file_path = Path(file_meta["file_path"])

        # Check if the file exists, if not, remove from library
        if not file_path.exists():
            logger.warning(f"File {file_name} does not exist. Marking for deletion.")
            missing_files.add(file_name)
            return None

        file_info = file_path.stat()
        current_mtime = file_info.st_mtime
        created_time = file_info.st_ctime
        cached_mtime = file_meta.get("last_modified", 0)
        # Check if the file has been modified since last chunking or newly added
        needs_chunking = "chunks" not in file_meta or not isclose(
            cached_mtime, current_mtime, abs_tol=1e-6
        )

        was_updated = False
        if needs_chunking:
            chunks, _ = read_and_chunk_file(file_path)
            file_meta["chunks"] = chunks
            file_meta["num_chunks"] = len(chunks)
            file_meta["last_modified"] = current_mtime
            was_updated = True
        else:
            chunks = file_meta.get("chunks", [])

        file_meta["file_name"] = file_path.name
        file_meta["file_path"] = str(file_path.resolve())
        file_meta["size_kb"] = round(file_info.st_size / 1024, 2)
        file_meta["created_at"] = datetime.fromtimestamp(created_time).isoformat()

        return file_name, file_meta, chunks, was_updated

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_file, fname) for fname in list(library.keys())
        ]

        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
            file_name, file_meta, chunks, was_updated = result
            new_library[file_name] = file_meta
            all_chunks.extend(chunks)
            all_sources.extend([file_meta["file_path"]] * len(chunks))

            if was_updated:
                updated = True

    for missing in missing_files:
        new_library.pop(missing, None)
        updated = True

    if updated:
        with open(file_library_path, "w") as f:
            json.dump(new_library, f, indent=2)

    return new_library, all_chunks, all_sources


def read_text(file_path: Path) -> str:
    try:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                return "\n".join([page.get_text() for page in doc])
        elif file_path.suffix.lower() == ".docx":
            doc = docx.Document(file_path)
            return "\n".join([para.text for para in doc.paragraphs])
        elif file_path.suffix.lower() == ".json":
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return json.dumps(data, indent=2)
        elif file_path.suffix.lower() == ".csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                return "\n".join([", ".join(row) for row in reader])
        elif file_path.suffix.lower() == ".md":
            with open(file_path, "r", encoding="utf-8") as f:
                md_content = f.read()
                html_content = markdown.markdown(md_content)
                # need to convert HTML to text using BeautifulSoup
                soup = BeautifulSoup(html_content, "html.parser")
                return soup.get_text()
        elif file_path.suffix.lower() == ".html":
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
                soup = BeautifulSoup(html_content, "html.parser")
                return soup.get_text()
        elif file_path.suffix.lower() == ".rtf":
            with open(file_path, "r", encoding="utf-8") as f:
                rtf_content = f.read()
                return rtf_to_text(rtf_content)
        elif file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]:
            # Perform OCR on image files
            image = Image.open(file_path)
            return pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Failed to read file {file_path.name}: {e}")
    return ""


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    paragraphs = re.split(r"\n{2,}|(?<=\n)\s*(?=\S)", text.strip())
    paragraphs = [p.strip().replace("\n", " ") for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = []
    current_length = 0

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        if current_length + len(para_words) <= chunk_size:
            current_chunk.extend(para_words)
            current_length += len(para_words)
        else:
            chunks.append(" ".join(current_chunk))
            overlap_words = current_chunk[-overlap:] if overlap > 0 else []
            current_chunk = overlap_words + para_words
            current_length = len(current_chunk)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def read_and_chunk_file(file_path: Path) -> Tuple[List[str], str]:
    text = read_text(file_path)
    chunks = chunk_text(text)
    return chunks, str(file_path)


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

        show_status_message(f"{len(saved_files)} new file(s) uploaded.", type="success")

    # Chunk and update library with chunks/num_chunks
    library, _, _ = load_and_chunk_files(library, library_path)
    return library

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
