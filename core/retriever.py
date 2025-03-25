from datetime import datetime
from typing import List, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import re
import fitz

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


class Retriever:
    def __init__(
        self, documents: List[str], doc_paths: List[str], max_results: int = 5
    ):
        self.documents = documents
        self.doc_paths = doc_paths
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        self.max_results = max_results

    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Search the documents using TF-IDF + cosine similarity.
        Returns: List of (document snippet, source path, score)
        """
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        ranked_indices = similarities.argsort()[::-1][: self.max_results]

        results = []
        for idx in ranked_indices:
            results.append(
                (self.documents[idx], self.doc_paths[idx], similarities[idx])
            )
        return results

    @staticmethod
    def load_and_chunk_files(
        library: dict, file_library_path: Path, chunk_size: int = 500
    ) -> Tuple[List[str], List[str]]:
        all_chunks = []
        all_sources = []
        updated = False

        for file_name in list(library.keys()):
            file_meta = library[file_name]
            file_path = Path(file_meta["file_path"])

            if not file_path.exists():
                continue

            # Get fresh file info
            file_info = file_path.stat()
            current_mtime = file_info.st_mtime
            created_time = file_info.st_ctime

            cached_mtime = file_meta.get("last_modified")
            needs_chunking = (
                "chunks" not in file_meta
                or not file_meta["chunks"]
                or cached_mtime != current_mtime
            )

            if needs_chunking:
                chunks, _ = read_and_chunk_file(file_path)
                file_meta["chunks"] = chunks
                file_meta["num_chunks"] = len(chunks)
                file_meta["last_modified"] = current_mtime
                updated = True
            else:
                chunks = file_meta["chunks"]

            # Always update core metadata
            file_meta["file_name"] = file_path.name
            file_meta["file_path"] = str(file_path.resolve())
            file_meta["size_kb"] = round(file_info.st_size / 1024, 2)
            file_meta["created_at"] = datetime.fromtimestamp(created_time).isoformat()

            library[file_name] = file_meta
            all_chunks.extend(chunks)
            all_sources.extend([str(file_path)] * len(chunks))

        if updated:
            with open(file_library_path, "w") as f:
                json.dump(library, f, indent=2)

        return library, all_chunks, all_sources


def read_text(file_path: Path) -> str:
    try:
        if file_path.suffix.lower() == ".txt":
            return file_path.read_text(encoding="utf-8")
        elif file_path.suffix.lower() == ".pdf":
            with fitz.open(file_path) as doc:
                return "\n".join([page.get_text() for page in doc])
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
