# 📚 InfoSynth

---

## 🚀 Getting Started

Follow these steps to get up and running:

### 1. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
.venv\Scripts\activate      # Windows
```

### 2. Install dependencies

```python
pip install -r requirements.txt
```

#### Note: You will also need to install the Tesseract OCR engine on your local machine, which is needed for processing and extracting text from image files. The path to the installed binary is automatically configured in the code based on the architecture of your machine.

If you are on MacOS:
```sh
brew install tesseract
```

OR if you are on Linux (Debian based distro):
```sh
sudo apt install tesseract-ocr
```

### 3. Configure your environment

Copy the .env.example file to .env and add your Gemini API Key:

```python
cp .env.example .env
```

Inside .env:

```
GEMINI_API_KEY=your-gemini-api-key-here
```

[ 🔑 ] Get your Gemini API key from: https://makersuite.google.com/app/apikey

### 4. Run the application

```
streamlit run app/main.py
```

This will launch a browser tab with the full UI.

### 5. Project Structure
```
infosynth/
├── app/
│   └── main.py                # Main application entry-point
│
├── core/
│   ├── retriever.py           # Chunking + TF-IDF search
│   ├── query_classifier.py    # Query intent classification
│   └── llm.py                 # Gemini API integration
│
├── utils/
│   ├── file_utils.py          # File upload, metadata, JSON saving
│   └── logger.py              # Console log formatting
│
├── data/
│   ├── uploads/               # Uploaded files
│   └── library.json           # Document metadata + chunk cache
│
├── .env                       # Your Gemini API key lives here
├── .env.example               # Template for .env
├── requirements.txt           # Project Dependencies
└── README.md
```
## 📊 Evaluation

### Step 1: Download a BEIR dataset

```bash
python evaluation/download_beir.py --dataset scifact --output_dir ./evaluation/beir_datasets
```

Available datasets include:

- **scifact**: Scientific fact-checking (smaller dataset, good for testing)
- **nfcorpus**: News and medical articles
- **fiqa**: Financial domain Q&A
- **arguana**: Argumentative question answering
- **scidocs**: Scientific documents
- **hotpotqa**: Multi-hop question answering

### Step 2: Run the evaluation

```bash
python evaluation/infosynth_beir_adapter.py --dataset ./evaluation/beir_datasets/scifact --output evaluation/results.json
```

This will:

- Load the specified BEIR dataset.
- Use InfoSynth's existing retriever for document search.
- Evaluate the retrieval performance using standard IR metrics.
- Save the results to the specified output file.

### 📈 Evaluation Metrics

The evaluation computes several standard information retrieval metrics:

- **NDCG (Normalized Discounted Cumulative Gain)**: Measures ranking quality.
- **MAP (Mean Average Precision)**: Measures precision at different recall levels.
- **Recall**: Proportion of relevant documents retrieved.
- **Precision**: Proportion of retrieved documents that are relevant.

Each metric is calculated at various cutoff points (k=1, 3, 5, 10, 20, 50, 100).