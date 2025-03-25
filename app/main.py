import streamlit as st
from pathlib import Path
import os
import sys
import multiprocessing as mp

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from core.query_classifier import QueryClassifier
from core.retriever import Retriever
from core.llm import generate_answer
from utils.file_utils import process_uploaded_files, load_file_library

load_dotenv()


class InfoSynthApp:
    def __init__(self, config: dict = None):
        self.config = config or self.load_config("config.json")
        self.upload_dir = Path(self.config.get("upload_dir", "data/uploads"))
        self.library_path = Path(self.config.get("library_path", "data/library.json"))
        self.allowed_extensions = self.config.get("allowed_extensions", ["pdf"])
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

        self.file_library = load_file_library(self.library_path)

        if not self.file_library:
            return

        library, chunks, sources = Retriever.load_and_chunk_files(
            self.file_library, self.library_path
        )
        self.retriever = Retriever(chunks, sources, max_results=5)

    def handle_query(self, query: str):
        """Handle search queries with basic classification."""
        analysis = self.classifier.analyze_query(query)

        st.markdown(f"🔍 **Query:** {analysis.corrected_query}")
        st.markdown(f"🧠 **Detected Intent:** `{analysis.query_type.value}`")
        st.markdown(f"📊 **Confidence:** {analysis.confidence:.2f}")
        if analysis.corrections:
            st.markdown("✏️ **Corrections:**")
            for orig, corr in analysis.corrections.items():
                st.markdown(f"- `{orig}` → `{corr}`")

        if self.retriever:
            st.markdown("---")
            st.subheader("🔎 Top Retrieved Chunks")

            results = self.retriever.search(analysis.corrected_query)

            if results:
                with st.spinner("Generating answer with Gemini..."):
                    top_chunks = [r[0] for r in results]
                    sources = [r[1] for r in results]

                    answer = generate_answer(query, top_chunks)

                    st.markdown("### 💬 Answer")
                    st.success(answer)

                    st.markdown("### 📂 Sources")
                    for i, src in enumerate(sources):
                        st.markdown(f"**{i+1}.** `{src}`")

                st.markdown("---")
                for chunk, source, score in results:
                    st.markdown(f"📄 **Source:** `{source}`")
                    st.markdown(f"🧩 **Score:** {score:.4f}")
                    st.markdown(f"> {chunk[:300]}...")
                    st.markdown("---")

    def render_ui(self):
        """Render the main user interface"""
        st.title("Document Search Engine")

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
                    help="Set the maximum number iterations for chain of thought search",
                )
                top_k = st.number_input(
                    "Number of results to retrieve (top-k)",
                    min_value=1,
                    max_value=20,
                    value=5,
                    step=1,
                    help="Set the number of top results to retrieve from search",
                )

            with st.expander("📁 Document Upload", expanded=True):
                with st.form("document_upload_form"):
                    uploaded_files = st.file_uploader(
                        "Upload Documents",
                        accept_multiple_files=True,
                        type=self.allowed_extensions,
                        help="Upload text or PDF documents for processing",
                    )

                    submit_button = st.form_submit_button("Process Documents")

                    if submit_button and uploaded_files:
                        with st.spinner("Processing documents..."):
                            self.file_library, chunks, sources = process_uploaded_files(
                                uploaded_files,
                                self.upload_dir,
                                self.file_library,
                                self.library_path,
                            )

                            if chunks:
                                self.retriever = Retriever(
                                    chunks, sources, max_results=5
                                )
                            else:
                                self.retriever = None

        if self.file_library:
            with st.expander("📚 Document Library", expanded=False):
                for file_name, meta in self.file_library.items():
                    st.markdown(
                        f"**{file_name}**  ",
                    )
                    st.markdown(
                        f"🗂 {meta['size_kb']} KB | 📅 {meta['created_at'].split('T')[0]} | 📑 {meta.get('num_chunks', 0)} chunks",
                    )
                    st.markdown("---")

        st.markdown("---")
        query = st.text_input("Enter your search query")
        if st.button("Search"):
            if not query.strip():
                st.warning("Please enter a query.")
            else:
                self.handle_query(query)

    def run(self):
        """Main application entry point"""
        self.render_ui()


if __name__ == "__main__":
    mp.freeze_support()
    app = InfoSynthApp()
    app.run()
