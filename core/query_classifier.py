from dataclasses import dataclass
from typing import Dict, Tuple
import spacy
import logging
from enum import Enum
from symspellpy import SymSpell, Verbosity
import pkg_resources
import re

logger = logging.getLogger(__name__)


class QueryType(Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    COMPARISON = "comparison"
    EXPLORATORY = "exploratory"
    PROCEDURAL = "procedural"


@dataclass
class QueryAnalysis:
    query_type: QueryType
    weights: Dict[str, float]
    confidence: float = 1.0
    features: Dict[str, float] = None
    corrections: Dict[str, str] = None
    original_query: str = None
    corrected_query: str = None


class QueryClassifier:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            import subprocess

            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")

        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)

        dictionary_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        bigram_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_bigramdictionary_en_243_342.txt"
        )

        self.sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
        self.sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

        self.intent_patterns = {
            QueryType.FACTUAL: {
                "question_words": ["what", "when", "where", "who", "which"],
                "verbs": ["is", "are", "was", "were", "does"],
                "patterns": ["define", "meaning of", "definition of"],
            },
            QueryType.REASONING: {
                "question_words": ["why", "how"],
                "verbs": ["explain", "causes", "affects", "influences", "works"],
                "patterns": ["reason for", "because", "explain", "understand"],
            },
            QueryType.COMPARISON: {
                "markers": ["compare", "versus", "vs", "difference", "better", "worse"],
                "patterns": ["compared to", "differences between", "pros and cons"],
            },
            QueryType.EXPLORATORY: {
                "verbs": ["tell", "describe", "elaborate", "discuss"],
                "patterns": [
                    "tell me about",
                    "what are",
                    "information about",
                    "learn about",
                ],
            },
            QueryType.PROCEDURAL: {
                "markers": ["how to", "steps", "guide", "tutorial", "instructions"],
                "verbs": ["make", "create", "build", "implement", "setup", "configure"],
            },
        }

        self.retrieval_weights = {
            QueryType.FACTUAL: {"dense": 0.2, "sparse": 0.4, "semantic": 0.4},
            QueryType.REASONING: {"dense": 0.3, "sparse": 0.2, "semantic": 0.5},
            QueryType.COMPARISON: {"dense": 0.25, "sparse": 0.25, "semantic": 0.5},
            QueryType.EXPLORATORY: {"dense": 0.2, "sparse": 0.1, "semantic": 0.7},
            QueryType.PROCEDURAL: {"dense": 0.3, "sparse": 0.3, "semantic": 0.4},
        }

    def correct_query(self, query: str) -> Tuple[str, Dict[str, str]]:
        suggestions = self.sym_spell.lookup_compound(
            query, max_edit_distance=2, transfer_casing=True
        )
        if not suggestions:
            return query, {}

        corrected_query = suggestions[0].term
        corrections = {}
        for o, c in zip(query.split(), corrected_query.split()):
            if o.lower() != c.lower():
                corrections[o] = c
        return corrected_query, corrections

    def analyze_query(self, query: str) -> QueryAnalysis:
        try:
            corrected_query, corrections = self.correct_query(query)
            doc = self.nlp(corrected_query)
            type_scores = self._calculate_type_scores(doc, corrected_query)
            predicted_type = max(type_scores.items(), key=lambda x: x[1])
            query_type, confidence = predicted_type

            weights = self.retrieval_weights[query_type].copy()
            if confidence < 0.5:
                for key in weights:
                    weights[key] = 0.5 + (weights[key] - 0.5) * confidence

            return QueryAnalysis(
                query_type=query_type,
                weights=weights,
                confidence=confidence,
                features=type_scores,
                corrections=corrections or None,
                original_query=query,
                corrected_query=corrected_query,
            )
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return QueryAnalysis(
                query_type=QueryType.EXPLORATORY,
                weights={"dense": 0.5, "sparse": 0.5},
                confidence=0.0,
                original_query=query,
            )

    def _calculate_type_scores(self, doc, query: str) -> Dict[QueryType, float]:
        scores = {qt: 0.0 for qt in QueryType}
        query_lower = query.lower()

        for query_type, patterns in self.intent_patterns.items():
            score = 0.0
            for key in ["question_words", "verbs", "markers"]:
                if key in patterns:
                    score += (
                        sum(word in query_lower.split() for word in patterns[key]) * 0.3
                    )
            if "patterns" in patterns:
                score += sum(p in query_lower for p in patterns["patterns"]) * 0.5

            if query_type == QueryType.FACTUAL and any(
                t.tag_ in ["WDT", "WP", "WRB"] for t in doc
            ):
                score += 0.4
            elif query_type == QueryType.REASONING and any(
                t.text.lower() == "why" for t in doc
            ):
                score += 0.6
            elif query_type == QueryType.COMPARISON and any(
                t.dep_ == "amod" for t in doc
            ):
                score += 0.4
            elif query_type == QueryType.PROCEDURAL and doc[0].pos_ == "VERB":
                score += 0.4

            scores[query_type] = min(score, 1.0)

        return scores
