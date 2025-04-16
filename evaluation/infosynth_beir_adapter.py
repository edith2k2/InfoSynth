import os
import sys
from pathlib import Path
import json
import numpy as np
from typing import Dict, List

# Add InfoSynth to path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Import InfoSynth components
from core.retriever import Retriever, read_and_chunk_file
from core.query_classifier import QueryClassifier
from utils.file_utils import load_file_library

# BEIR imports
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.datasets.data_loader import GenericDataLoader

class InfoSynthBeirRetriever:
    """
    Adapter class to use InfoSynth's existing retriever with BEIR evaluation framework.
    """
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.corpus = {}
        self.corpus_ids = []
        
        # Use InfoSynth's native components
        self.query_classifier = QueryClassifier()

    def index(self, corpus: Dict[str, Dict[str, str]]):
        """
        Index the corpus for retrieval.
        
        Args:
            corpus: Dictionary with document IDs as keys and documents (containing 'title' and 'text') as values
        """
        print(f"Indexing {len(corpus)} documents...")
        
        self.corpus = corpus
        self.corpus_ids = list(corpus.keys())
        
        # Prepare documents for InfoSynth retriever format
        documents = []
        doc_paths = []
        
        for doc_id in self.corpus_ids:
            title = corpus[doc_id].get("title", "").strip()
            text = corpus[doc_id].get("text", "").strip()
            document = f"{title} {text}".strip()
            documents.append(document)
            doc_paths.append(doc_id)  # Use document ID as path
        
        # Initialize the InfoSynth retriever with our documents
        self.retriever = Retriever(documents, doc_paths, max_results=self.top_k)
        
        print("Indexing complete.")
        return self

    def retrieve(self, queries: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        Retrieve documents for each query.
        
        Args:
            queries: Dictionary with query IDs as keys and queries as values
            
        Returns:
            Dictionary with query IDs as keys and dictionaries (document ID -> score) as values
        """
        print(f"Retrieving documents for {len(queries)} queries...")
        
        results = {}
        for query_id, query in queries.items():
            # Use InfoSynth's native search method
            search_results = self.retriever.search(query)
            
            # Extract document IDs and scores
            doc_scores = {}
            for _, doc_path, score, _ in search_results:
                doc_scores[doc_path] = float(score)
            
            results[query_id] = doc_scores
        
        return results


def evaluate_infosynth_on_beir(dataset_path, output_file="beir_evaluation_results.json"):
    """
    Evaluate InfoSynth on a BEIR dataset.
    
    Args:
        dataset_path: Path to the BEIR dataset
        output_file: Path to save evaluation results
    """
    # Load BEIR dataset
    print(f"Loading dataset from {dataset_path}...")
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")

    # Print 5 of corpus, queries, qrels
    print(" 5 documents in corpus:")
    for doc_id, doc in list(corpus.items())[:5]:
        print(f"{doc_id}: {doc}")
    
    print("\n 5 queries:")
    for query_id, query in list(queries.items())[:5]:
        print(f"{query_id}: {query}")
    
    print("\n 5 query relevance judgments (qrels):")
    for query_id, doc_rels in list(qrels.items())[:5]:
        print(f"{query_id}: {doc_rels}")
    
    # Initialize InfoSynth retriever
    print("Initializing InfoSynth retriever...")
    retriever = InfoSynthBeirRetriever(top_k=100)
    retriever.index(corpus)
    
    # Initialize BEIR evaluator
    print("Setting up evaluation...")
    evaluator = EvaluateRetrieval()
    
    # Retrieve and evaluate
    print("Running retrieval and evaluation...")
    results = retriever.retrieve(queries)
    
    # Calculate metrics
    print("Computing metrics...")
    k_values = [1, 3, 5, 10, 20, 50, 100]
    ndcg, _map, recall, precision = evaluator.evaluate(qrels, results, k_values)
    
    # Combine all metrics
    metrics = {
        "ndcg": ndcg,
        "map": _map,
        "recall": recall,
        "precision": precision
    }
    
    # Save results
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Print key metrics
    print("\nEvaluation Results:")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"MAP@100: {_map['MAP@100']:.4f}")
    print(f"Recall@100: {recall['Recall@100']:.4f}")
    print(f"Precision@10: {precision['P@10']:.4f}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate InfoSynth on BEIR datasets")
    parser.add_argument("--dataset", type=str, required=True, 
                        help="Path to BEIR dataset directory")
    parser.add_argument("--output", type=str, default="beir_evaluation_results.json",
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    evaluate_infosynth_on_beir(args.dataset, args.output)