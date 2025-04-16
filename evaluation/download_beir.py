import os
import argparse
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from pathlib import Path

def download_beir_dataset(dataset_name, output_dir="./beir_datasets"):
    """
    Download a BEIR dataset and save to the specified directory.
    
    Args:
        dataset_name: Name of the BEIR dataset (e.g., 'scifact', 'nfcorpus', 'fiqa', etc.)
        output_dir: Directory to save the downloaded dataset
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define the dataset path
    dataset_path = os.path.join(output_dir, dataset_name)
    
    # Download the dataset if it doesn't exist
    if not os.path.exists(dataset_path):
        print(f"Downloading {dataset_name} dataset...")
        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
        try:
            util.download_and_unzip(url, output_dir)
            print(f"Dataset downloaded successfully to {dataset_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None
    else:
        print(f"Dataset already exists at {dataset_path}")
    
    # Load and verify the dataset
    corpus, queries, qrels = GenericDataLoader(dataset_path).load(split="test")
    
    print(f"Dataset statistics:")
    print(f"Number of documents: {len(corpus)}")
    print(f"Number of queries: {len(queries)}")
    print(f"Number of query-document pairs: {sum(len(qr) for qr in qrels.values())}")
    
    return dataset_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BEIR datasets")
    parser.add_argument("--dataset", type=str, default="scifact", 
                        help="Name of the BEIR dataset to download")
    parser.add_argument("--output_dir", type=str, default="./beir_datasets", 
                        help="Directory to save the downloaded dataset")
    
    args = parser.parse_args()
    
    print(f"Downloading BEIR dataset: {args.dataset}")
    download_beir_dataset(args.dataset, args.output_dir)
    
    print("\nAvailable BEIR datasets:")
    print("- scifact (Scientific fact-checking)")
    print("- nfcorpus (News and medical information)")
    print("- fiqa (Financial domain QA)")
    print("- arguana (Argument retrieval)")
    print("- scidocs (Scientific documents)")
    print("- climate-fever (Climate fact checking)")
    print("- quora (Question duplicates)")
    print("- dbpedia-entity (Entity retrieval)")
    print("- hotpotqa (Multi-hop QA)")
    print("- nq (Natural Questions)")