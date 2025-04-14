from datetime import datetime
from typing import List, Tuple, Optional
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import streamlit as st
import json
import re
import fitz
import docx
import csv
import markdown
from bs4 import BeautifulSoup
from striprtf.striprtf import rtf_to_text
import pytesseract
from PIL import Image
import numpy as np
from rank_bm25 import BM25Okapi
import spacy

import importlib.util

ROOT_DIR = Path(__file__).resolve().parent.parent

from core.query_classifier import QueryClassifier, QueryType
from core.llm import llm_query_expansion
class Retriever:
    def __init__(
        self, documents: List[str], doc_paths: List[str], max_results: int = 5, use_embedding: bool = False
    ):
        self.documents = documents
        self.doc_paths = doc_paths
        # TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_vectors = self.vectorizer.fit_transform(documents)
        # BM25
        self.tokenized_docs = [self._tokenize_text(doc) for doc in documents]
        self.bm25_index = BM25Okapi(self.tokenized_docs)
        self.max_results = max_results
        #dense embeddign
        self.document_embeddings = None
        self.use_embedding = use_embedding
        if self.use_embedding:
            try:
                self.nlp = spacy.load("en_core_web_lg")
                self.document_embeddings = self._compute_document_embeddings()
            except Exception as e:
                print(f"Error loading spaCy model: {e}")
                self.use_embedding = False
        # Query Classifier
        self.query_classifier = QueryClassifier()

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using sklearn's vectorizer"""
        vectorizer = CountVectorizer(
            lowercase=True, 
            stop_words='english',
            token_pattern=r'(?u)\b\w\w+\b'  
        )
        analyzer = vectorizer.build_analyzer()
        return analyzer(text)
    
    def _compute_document_embeddings(self):
        document_embeddings = []
        for doc in self.documents:
            # Get document vector (using spaCy's built-in word vectors)
            doc_vector = self.nlp(doc).vector
            document_embeddings.append(doc_vector)
        return np.array(document_embeddings)
    
    
    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Perform hybrid search using TF-IDF, BM25, and optionally dense embeddings
        Returns: List of (document snippet, source path, score)
        """
        # Classify query to determine optimal weights
        query_analysis = self.query_classifier.analyze_query(query)
        
        # Get retrieval weights based on query type
        weights = query_analysis.weights
        sparse_weight = weights.get('sparse', 0.5)
        dense_weight = weights.get('dense', 0.5)
        
        # Get TF-IDF results
        query_vector = self.vectorizer.transform([query])
        tfidf_similarities = cosine_similarity(query_vector, self.doc_vectors).flatten()
        
        # BM25 results
        tokenized_query = self._tokenize_text(query)
        bm25_scores = np.array(self.bm25_index.get_scores(tokenized_query))
        
        # Normalize scores
        max_tfidf = max(tfidf_similarities) if max(tfidf_similarities) > 0 else 1
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

        tfidf_norm = tfidf_similarities / max_tfidf
        bm25_norm = bm25_scores / max_bm25
        
        # Only use dense embeddings if enabled
        if self.use_embedding and hasattr(self, 'nlp') and self.document_embeddings is not None:
            semantic_weight = weights.get('semantic', 0.33)
            # Dense embedding similarity
            query_embedding = self.nlp(query).vector
            dense_similarities = np.array([
                np.dot(query_embedding, doc_embedding) / 
                (np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8)
                for doc_embedding in self.document_embeddings
            ])
            
            # Normalize dense scores
            max_dense = max(dense_similarities) if max(dense_similarities) > 0 else 1
            dense_norm = dense_similarities / max_dense
            
            # Combine all scores using weights from query classifier
            combined_scores = (sparse_weight * tfidf_norm) + \
                            (dense_weight * bm25_norm) + \
                            (semantic_weight * dense_norm)
        else:
            # If embeddings not available, just use sparse search with equal weights
            combined_scores = (sparse_weight * tfidf_norm) + (dense_weight * bm25_norm)
        
        # Get top results
        ranked_indices = combined_scores.argsort()[::-1][:self.max_results]
        
        results = []
        for idx in ranked_indices:
            # Include all scores for debug/comparison
            result_meta = {
                'combined_score': combined_scores[idx],
                'tfidf_score': tfidf_similarities[idx],
                'bm25_score': bm25_scores[idx]
            }
            
            # Only include dense score if embeddings were used
            if self.use_embedding and hasattr(self, 'nlp') and self.document_embeddings is not None:
                result_meta['dense_score'] = dense_similarities[idx]
            
            # Return the document, source and combined score
            results.append(
                (self.documents[idx], self.doc_paths[idx], combined_scores[idx], result_meta)
            )
            
        return results
    
    def query_expansion(self, query: str, num_k: int = 5) -> List[str]:
        """
        Perform query expansion with the help of LLM
        Returns: List of expanded queries
        """
        print("query", query, "num", num_k)
        expanded_queries = []
        for _ in range(num_k):
            expanded_query = llm_query_expansion(query, expanded_queries)
            expanded_queries.append(expanded_query)
        # print(expanded_queries)
        return expanded_queries

    def expanded_search(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Perform search with query expansion
        Returns: List of (document snippet, source path, score)
        """
        expanded_queries = [query] + self.query_expansion(query)
        all_results = defaultdict(list)
        for eq in expanded_queries:
            results = self.search(eq)
            all_results[eq].extend(results)
        return dict(all_results)

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
