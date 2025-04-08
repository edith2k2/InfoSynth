import threading
import time
import streamlit as st
from pathlib import Path
import os
import sys
import multiprocessing as mp
import platform
import shutil
import pytesseract


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


from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.query_classifier import QueryClassifier
from core.retriever import Retriever
from core.llm import generate_answer
from utils.file_utils import process_uploaded_files, load_file_library
from utils.watcher_state import watcher_state
from utils.file_watcher import start_watcher

load_dotenv()


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


class InfoSynthApp:
    def __init__(self, config: dict = None):
        self.config = config or self.load_config("config.json")
        self.upload_dir = Path(self.config.get("upload_dir", "data/uploads"))
        self.library_path = Path(self.config.get("library_path", "data/library.json"))
        self.allowed_extensions = self.config.get("allowed_extensions", ["pdf"])
        self.watch_folders = [Path(p) for p in self.config.get("watch_folders")]
        self.page_title = self.config.get("page_title", "InfoSynth")
        self.page_layout = self.config.get("page_layout", "wide")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")

        self.classifier = QueryClassifier()
        self.retriever = None

        self._setup()

    def load_config(self, path: str) -> dict:
        """Load configuration file from JSON file."""
        try:
            import json

            with open(path, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _setup(self):
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

        self.file_library = load_file_library(self.library_path)

        if not self.file_library:
            return

        _, chunks, sources = Retriever.load_and_chunk_files(
            self.file_library, self.library_path
        )
        self.retriever = Retriever(chunks, sources, max_results=5)

    def handle_query(self, query: str, retriever=None):
        """Handle search queries with basic classification."""
        analysis = self.classifier.analyze_query(query)
        retriever = retriever or self.retriever

        st.session_state.analysis = analysis
        st.session_state.answer = None
        st.session_state.results = []

        if retriever:
            results = retriever.search(analysis.corrected_query)

            if results:
                with st.spinner("Generating answer with Gemini..."):
                    top_chunks = [r[0] for r in results]
                    sources = [r[1] for r in results]
                    answer = generate_answer(query, top_chunks)

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
            st.markdown("### Recent Documents")

            recent_files = []
            if self.file_library:
                sorted_files = sorted(
                    self.file_library.items(),
                    key=lambda x: x[1].get("created_at", ""),
                    reverse=True,
                )
                recent_files = sorted_files[:4]

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
                    with st.spinner("Processing documents..."):
                        self.file_library, chunks, sources = process_uploaded_files(
                            uploaded_files,
                            self.upload_dir,
                            self.file_library,
                            self.library_path,
                        )
                        self.retriever = (
                            Retriever(chunks, sources, max_results=5)
                            if chunks
                            else None
                        )

                        uploaded_files = []

        with left_col:

            @st.fragment(run_every=5)
            def refresh_library():
                current_library = load_file_library(self.library_path)
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
            query = st.text_input(
                "Enter your search query",
                value=st.session_state.get("query_input", ""),
                key="search_input",
            )
            st.session_state.query_input = query

            if st.button("Search", key="search_button_top"):
                if not query.strip():
                    st.warning("Please enter a query.")
                else:
                    self.handle_query(query)

            analysis = st.session_state.get("analysis")
            answer = st.session_state.get("answer")
            results = st.session_state.get("results", [])

            if analysis:
                st.markdown(f"🔍 **Query:** {analysis.corrected_query}")
                st.markdown(f"🧠 **Intent:** `{analysis.query_type.value}`")
                st.markdown(f"📊 **Confidence:** {analysis.confidence:.2f}")
                if analysis.corrections:
                    st.markdown("✏️ **Corrections:**")
                    for orig, corr in analysis.corrections.items():
                        st.markdown(f"- `{orig}` → `{corr}`")

            if answer:
                st.markdown("### 💬 Answer")
                st.success(answer)

                st.markdown("### 📂 Sources")
                for i, (_, src, _) in enumerate(results):
                    st.markdown(f"**{i+1}.** `{src}`")

                st.markdown("---")
                for chunk, source, score in results:
                    st.markdown(f"📄 **Source:** `{source}`")
                    st.markdown(f"🧩 **Score:** {score:.4f}")
                    st.markdown(f"> {chunk[:300]}...")
                    st.markdown("---")
        # Sidebar
        with st.sidebar:
            st.header("Configuration")

            with st.expander("🕒 Time Range", expanded=False):
                st.markdown("Temporal controls placeholder")

            with st.expander("⚙️ Search Configuration", expanded=False):
                st.number_input(
                    "Maximum chain of thought search steps",
                    min_value=0,
                    max_value=5,
                    value=1,
                    key="steps_input",
                )
                st.number_input(
                    "Number of results to retrieve (top-k)",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    key="top_k_input",
                )

    def run(self):
        """Main application entry point"""
        self.render_ui()


if __name__ == "__main__":
    mp.freeze_support()
    app = InfoSynthApp()
    configure_tesseract()

    if not watcher_state.watcher_started:
        threading.Thread(
            target=start_watcher,
            args=(app.watch_folders, app.library_path),
            daemon=True,
        ).start()
        watcher_state.watcher_started = True
    app.run()
