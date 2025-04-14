from typing import List, Tuple
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

import numpy as np
from rank_bm25 import BM25Okapi
from pathlib import Path
from core.query_classifier import QueryClassifier
from utils.logger import AppLogger

ROOT_DIR = Path(__file__).resolve().parent.parent
logger = AppLogger(name="Retriever").get_logger()


class Retriever:
    def __init__(
        self, documents: List[str], doc_paths: List[str], max_results: int = 5
    ):
        self.documents = documents
        self.doc_paths = doc_paths
        self.vectorizer = None
        self.doc_vectors = None
        self.max_results = max_results
        if self.documents:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self.doc_vectors = self.vectorizer.fit_transform(self.documents)

            # BM25
            self.tokenized_docs = [self._tokenize_text(doc) for doc in documents]
            self.bm25_index = BM25Okapi(self.tokenized_docs)

        # Query Classifier
        self.query_classifier = QueryClassifier()

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text using sklearn's vectorizer"""
        vectorizer = CountVectorizer(
            lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w\w+\b"
        )
        analyzer = vectorizer.build_analyzer()
        return analyzer(text)

    def search(self, query: str) -> List[Tuple[str, str, float]]:
        """
        Perform hybrid search using both TF-IDF and BM25
        Returns: List of (document snippet, source path, score, additional scores information)
        """
        # Classify query to determine optimal weights
        query_analysis = self.query_classifier.analyze_query(query)

        # Get retrieval weights based on query type
        weights = query_analysis.weights
        sparse_weight = weights.get("sparse", 0.5)
        dense_weight = weights.get("dense", 0.5)

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

        # Combine scores with weights
        combined_scores = (dense_weight * tfidf_norm) + (sparse_weight * bm25_norm)

        # Get top results
        ranked_indices = combined_scores.argsort()[::-1][: self.max_results]

        # # Log some information about the search
        # st.session_state.setdefault('last_search_info', {})
        # st.session_state.last_search_info = {
        #     'query_type': query_analysis.query_type.value,
        #     'weights': {
        #         'sparse': sparse_weight,
        #         'dense': dense_weight,
        #     },
        #     'confidence': query_analysis.confidence
        # }

        results = []
        for idx in ranked_indices:
            # Include both scores for debug/comparison
            result_meta = {
                "combined_score": combined_scores[idx],
                "tfidf_score": tfidf_similarities[idx],
                "bm25_score": bm25_scores[idx],
            }

            # Return the document, source and combined score
            results.append(
                (
                    self.documents[idx],
                    self.doc_paths[idx],
                    combined_scores[idx],
                    result_meta,
                )
            )

        return results
