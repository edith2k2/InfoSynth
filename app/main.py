import threading
from typing import List
import streamlit as st
from pathlib import Path
import os
import sys
import multiprocessing as mp
import platform
import shutil
import pytesseract
from dotenv import load_dotenv
import json

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from core.query_classifier import QueryClassifier
from core.retriever import Retriever
from core.llm import generate_answer
from utils.logger import AppLogger
from utils.file_utils import (
    get_mtime,
    process_uploaded_files,
    load_file_library,
    load_and_chunk_files,
)
from utils.config_manager import get_default_config, update_config_key
from utils.watcher_state import watcher_state
from utils.file_watcher import UploadFolderHandler
from utils.file_utils import show_status_message
from watchdog.observers import Observer

load_dotenv()

logger = AppLogger("Watcher").get_logger()


from utils.file_utils import load_config
from utils.logger import AppLogger

embedding_available = False
logger = AppLogger("Watcher").get_logger()

def configure_tesseract():
    system_name = platform.system()

    # Attempt to locate tesseract using shutil
    tesseract_path = shutil.which("tesseract")

    if tesseract_path:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        print(f"Tesseract found at: {tesseract_path}")
    else:
        if system_name == "Darwin":  # macOS
            possible_paths = [
                "/opt/homebrew/bin/tesseract",  # Apple Silicon
                "/usr/local/bin/tesseract",  # Intel Mac
            ]
        elif system_name == "Linux":
            possible_paths = ["/usr/bin/tesseract", "/usr/local/bin/tesseract"]
        else:
            print(f"Unsupported OS: {system_name}")
            return

        for path in possible_paths:
            if Path(path).exists():
                pytesseract.pytesseract.tesseract_cmd = path
                print(f"Tesseract configured using: {path}")
                return
        raise FileNotFoundError(
            "Tesseract not found. Please install Tesseract or add it to your PATH."
        )

def configure_embeddings():
    config = load_config("config.json")
    
    if not config.get("use_embedding", False):
        return
    
    try:
        import spacy
        try:
            embedding_available = True
            logger.info("Using existing SpaCy model for embeddings...")
        except OSError:
            logger.info("SpaCy model not found. Attempting to install...")
            import subprocess
            subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_lg"], 
                        check=True)
            logger.info("SpaCy model installed successfully.")
            embedding_available = True
    except ImportError:
        logger.error("SpaCy is not installed. Please install SpaCy to use embeddings.")
        config["use_embedding"] = False
    except Exception as e:
        raise RuntimeError(f"Error loading SpaCy model: {e}")
        
from dotenv import load_dotenv



def apply_external_styles():
    st.markdown(
        """
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
    """,
        unsafe_allow_html=True,
    )
    style_path = Path(__file__).parent / "styles" / "styles.css"
    with open(style_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def start_watcher(watch_dirs: List[Path], library_path: Path, config: dict):
    if not watch_dirs:
        logger.warning("No watch folders provided. File watcher is disabled.")
        return

    # 🔁 Stop old observer
    if watcher_state.observer:
        logger.info("Stopping previous file watcher...")
        watcher_state.observer.stop()
        watcher_state.observer.join()
        watcher_state.observer = None
        watcher_state.watcher_started = False

    allowed_exts = [
        ext.lower().lstrip(".") for ext in config.get("allowed_extensions", [])
    ]
    logger.info(
        f"Monitoring {len(watch_dirs)} folder(s) with extensions: {allowed_exts}"
    )

    mtime = get_mtime(library_path)
    library = load_file_library(library_path, mtime=mtime)

    observer = Observer()

    for directory in watch_dirs:
        # Find files matching allowed extensions
        all_files = []
        for ext in allowed_exts:
            all_files.extend(directory.glob(f"*.{ext}"))

        existing_files = {meta["file_name"] for meta in library.values()}
        new_files = [f for f in all_files if f.name not in existing_files]

        if new_files:
            library = process_uploaded_files(
                new_files, directory, library, library_path
            )
            watcher_state.files_changed = True

        handler = UploadFolderHandler(directory, library_path, config)
        observer.schedule(handler, str(directory), recursive=False)
        logger.info(f"\t* Watching: {directory.resolve()}")

    observer.start()
    watcher_state.observer = observer
    watcher_state.watcher_started = True
    watcher_state.watched_paths = set(map(str, watch_dirs))


class InfoSynthApp:
    def __init__(self, config: dict = None):
        self.config = config or self.load_config("config.json")
        self.config_path = Path("config.json")
        self.upload_dir = Path(self.config.get("upload_dir", "data/uploads"))
        self.library_path = Path(self.config.get("library_path", "data/library.json"))
        self.allowed_extensions = self.config.get("allowed_extensions", ["pdf"])
        self.watch_folders = [Path(p) for p in self.config.get("watch_folders")]
        self.page_title = self.config.get("page_title", "InfoSynth")
        self.page_layout = self.config.get("page_layout", "wide")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        if not self.gemini_api_key:
            raise ValueError(
                "GEMINI_API_KEY is not set. Please set it in your environment variables."
            )

        self.classifier = QueryClassifier()
        self.retriever = None

        self._setup()

    @staticmethod
    @st.cache_resource(show_spinner=False)
    def build_retriever(library: dict, file_library_path: Path, use_embedding: bool = False):
        logger.info("Building retriever...")
        _, chunks, sources = load_and_chunk_files(library, file_library_path)
        return Retriever(chunks, sources, max_results=watcher_state.max_results, use_embedding=use_embedding)

    def load_config(self, path: str) -> dict:
        """Load configuration file from JSON file."""
        DEFAULT_CONFIG = get_default_config()
        try:
            with open(path, "r") as f:
                config = json.load(f)
                return {**DEFAULT_CONFIG, **config}
        except (FileNotFoundError, json.JSONDecodeError):
            # Create default config if file not found or invalid JSON
            with open(path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            logger.info(f"Created default config file at {path}")
            return DEFAULT_CONFIG
    def _setup(self):
        logger.info("Setting up InfoSynth...")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.library_path.parent.mkdir(parents=True, exist_ok=True)
        st.set_page_config(page_title=self.page_title, layout=self.page_layout)
        self.icons = {
            ".pdf": '<i class="fas fa-file-pdf" style="color: #e74c3c;"></i>',
            ".docx": '<i class="fas fa-file-word" style="color: #2980b9;"></i>',
            ".csv": '<i class="fas fa-file-csv" style="color: #27ae60;"></i>',
            ".jpg": '<i class="fas fa-file-image" style="color: #8e44ad;"></i>',
            ".png": '<i class="fas fa-file-image" style="color: #8e44ad;"></i>',
            ".json": '<i class="fas fa-file-code" style="color: #f39c12;"></i>',
            ".txt": '<i class="fas fa-file-alt" style="color: #34495e;"></i>',
        }

        mtime = get_mtime(self.library_path)
        self.file_library = load_file_library(self.library_path, mtime=mtime)
        self.retriever = self.build_retriever(self.file_library, self.library_path, use_embedding=embedding_available)

    def handle_query(self, query: str, retriever=None):
        """Handle search queries with basic classification."""
        # Check number of documents
        if not self.file_library:
            show_status_message(
                "No documents available. Please upload documents to the library.",
                "error",
            )
            return
        analysis = self.classifier.analyze_query(query)
        retriever = retriever or self.retriever

        st.session_state.answer = None
        st.session_state.results = []

        if retriever:
            results = retriever.search(analysis.corrected_query)

            if results:
                with st.spinner("Generating answer with Gemini..."):
                    top_chunks = [r[0] for r in results]
                    sources = [r[1] for r in results]
                    scores = [r[2] for r in results]
                    answer = generate_answer(query, top_chunks, scores)

                    st.session_state.answer = answer
                    st.session_state.results = results
            else:
                st.info("No results found for your query.")

        st.session_state.query_input = ""

    def render_ui(self):
        """Render the main user interface"""
        apply_external_styles()

        st.title("InfoSynth")

        left_col, spacer, right_col = st.columns([13, 0.5, 7])

        # LEFT
        @st.fragment(run_every=5)
        def show_recent_documents():
            mtime = get_mtime(self.library_path)
            self.file_library = load_file_library(self.library_path, mtime=mtime)
            st.markdown("### Recent Documents")

            recent_files = []
            if self.file_library:
                recent_files = sorted(
                    self.file_library.items(),
                    key=lambda x: x[1].get("created_at", ""),
                    reverse=True,
                )[:4]

            cols = st.columns(5)

            for i, (file_name, meta) in enumerate(recent_files[:4]):
                file_ext = Path(file_name).suffix.lower()
                with cols[i]:
                    icon_html = self.icons.get(
                        file_ext, '<i class="fas fa-file" style="color: #7f8c8d;"></i>'
                    )
                    st.markdown(
                        f"""
                        <div class="doc-card">
                            <div class="doc-icon">
                                {icon_html}
                            </div>
                            <div class="doc-name" title="{file_name}">
                                {file_name}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # Upload tile (last column)
            with cols[4]:
                uploaded_files = st.file_uploader(
                    label="Upload new documents",
                    type=self.allowed_extensions,
                    accept_multiple_files=True,
                    key="styled_uploader",
                    label_visibility="hidden",
                )

                if uploaded_files:
                    new_files = [
                        f for f in uploaded_files if f.name not in self.file_library
                    ]

                    if new_files:
                        self.file_library = process_uploaded_files(
                            new_files,
                            self.upload_dir,
                            self.file_library,
                            self.library_path,
                        )
                        self.retriever = self.build_retriever(
                            self.file_library, self.library_path, use_embedding=embedding_available
                        )

        with left_col:

            @st.fragment(run_every=5)
            def refresh_library():
                mtime = get_mtime(self.library_path)
                current_library = load_file_library(self.library_path, mtime=mtime)
                st.subheader("📚 Document Library")

                if not current_library:
                    st.info("No documents available.")
                    return

                # Scroll container
                st.markdown(
                    """
                    <div style='
                        max-height: 250px;
                        overflow-y: auto;
                        padding-right: 1rem;
                        margin-top: 0.5rem;
                    '>
                """,
                    unsafe_allow_html=True,
                )

                # Render each file
                for file_name, meta in current_library.items():
                    st.markdown(
                        f"""
                        <div style='margin-bottom: 0.75rem;'>
                            <strong>{file_name}</strong><br>
                            📦 {meta['size_kb']} KB |
                            📅 {meta['created_at'].split('T')[0]} |
                            📑 {meta.get('num_chunks', 0)} chunks
                            <hr style='margin-top: 0.4rem; margin-bottom: 0.4rem; border-top: 1px solid #333;'>
                        </div>
                    """,
                        unsafe_allow_html=True,
                    )

                st.markdown("</div>", unsafe_allow_html=True)

            # Call the fragment
            show_recent_documents()
            st.markdown("---")
            refresh_library()

        # RIGHT
        with right_col:
            with st.form("search_form"):
                query = st.text_input(
                    "Enter your search query",
                    value=st.session_state.get("query_input", ""),
                    key="search_input",
                )
                st.session_state.query_input = query

                submitted = st.form_submit_button("Search")
                if submitted:
                    if not query.strip():
                        st.warning("Please enter a query.")
                    else:
                        self.handle_query(query)

            answer = st.session_state.get("answer")
            results = st.session_state.get("results", [])

            if answer:
                st.markdown("### 💬 Answer")
                st.success(answer)  # this is the LLM's answer

                st.markdown("### 📂 Sources for LLM's answer")
                # NOTE: results is a list of tuples (e.g., its length is 5 when k for topk = 5). The first element of the tuple is the chunk, the second is the filepath,
                # the third is the combined score, and the fourth is a dict containing combined_score, tfidf_score, and bm25_score.
                for i, (_, path, _, _) in enumerate(
                    results
                ):  # TODO: Is this what we want to display here?
                    st.markdown(f"**{i+1}.** `{path}`")

                st.markdown("---")
                st.markdown("### BM25 Results")
                for i, (chunk, source, score, additional) in enumerate(
                    results
                ):  # these are BM25 results
                    st.markdown(f"**{i+1}.** 📄 **Source:** `{source}`")
                    st.markdown(f"🧩 **Score:** {score:.4f}")
                    st.markdown(
                        f"> {chunk[:300]}..."
                    )  # TODO: do we want to allow the user to see the whole chunk? Can implement an on-demand dropdown if needed on the UI
                    st.markdown("---")
        # Sidebar
        with st.sidebar:
            st.header("Configuration")

            with st.expander("Time Range", expanded=True):
                st.markdown("Temporal controls placeholder")

            with st.expander("Search Configuration", expanded=True):
                with st.form(key="config_form"):
                    st.markdown('<div class="form-block">', unsafe_allow_html=True)

                    new_top_k = st.number_input(
                        "Number of results to retrieve (top-k)",
                        min_value=1,
                        max_value=20,
                        value=self.config.get("top_k", 5),
                        step=1,
                        key="form_top_k_input",
                    )

                    submitted = st.form_submit_button("Save")
                    st.markdown("</div>", unsafe_allow_html=True)

                if submitted:
                    watcher_state.max_results = new_top_k
                    updated = update_config_key(self.config_path, "top_k", new_top_k)
                    if updated:
                        self.config = updated
                        show_status_message("Configuration updated.", "success")

            with st.expander("Monitored Folders", expanded=True):
                st.markdown("### Currently Watched Folders")
                current_folders = self.config.get("watch_folders", [])

                for idx, folder in enumerate(current_folders):
                    col1, col2 = st.columns([4, 1])
                    folder_path = Path(folder).resolve()
                    try:
                        relative_path = str(folder_path.relative_to(ROOT_DIR))
                    except ValueError:
                        relative_path = str(folder_path)

                    with col1:
                        st.markdown(
                            f"<div class='folder-box'>{relative_path}</div>",
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("X", key=f"remove_folder_{idx}"):
                            updated_folders = (
                                current_folders[:idx] + current_folders[idx + 1 :]
                            )
                            self.config = update_config_key(
                                self.config_path, "watch_folders", updated_folders
                            )
                            self.watch_folders = [Path(p) for p in updated_folders]
                            watcher_state.watcher_started = False
                            show_status_message(
                                "Folder removed from watch list.", "info"
                            )
                            st.rerun()

                st.markdown("---")
                st.markdown("### Add New Folder")

                with st.form("watcher_folder_form"):
                    new_folder = st.text_input("Folder path", key="new_watch_folder")
                    submitted = st.form_submit_button("Add Folder")

                if submitted:
                    if new_folder:
                        new_folder_path = Path(new_folder.strip()).resolve()
                        existing_paths = set(
                            map(lambda p: str(Path(p).resolve()), current_folders)
                        )

                        if not new_folder_path.exists() or not new_folder_path.is_dir():
                            show_status_message("Folder does not exist.", "error")
                            return
                        elif str(new_folder_path) in existing_paths:
                            show_status_message("Folder is already watched.", "warning")
                            return
                        else:
                            current_folders.append(str(new_folder_path))

                    self.config = update_config_key(
                        self.config_path, "watch_folders", current_folders
                    )
                    self.watch_folders = [Path(p) for p in current_folders]
                    watcher_state.watcher_started = False
                    show_status_message("Configuration updated.", "success")
                    st.rerun()

    def run(self):
        """Main application entry point"""
        current_paths = set(map(str, self.watch_folders))
        previous_paths = watcher_state.watched_paths

        if current_paths != previous_paths or not watcher_state.watcher_started:
            watcher_state.watched_paths = current_paths
            logger.info("🔁 Starting/restarting file watcher...")

            threading.Thread(
                target=start_watcher,
                args=(self.watch_folders, self.library_path, self.config),
                daemon=True,
            ).start()
        self.render_ui()


if __name__ == "__main__":
    mp.freeze_support()
    app = InfoSynthApp()
    configure_tesseract()
    configure_embeddings()

    app.run()
