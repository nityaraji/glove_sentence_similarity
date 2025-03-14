# Semantic Similarity using GloVe

## Overview
This project calculates the **semantic similarity** between sentences using **GloVe word embeddings**. It utilizes **pre-trained GloVe vectors** to generate sentence embeddings and compute similarity scores.

## Features
- **Uses GloVe (Global Vectors for Word Representation)** for word embeddings
- **Computes sentence similarity** using cosine similarity
- **Lightweight & efficient** for NLP tasks

## Installation
### 1. Install Dependencies
Ensure you have Python installed, then install the required libraries:
```bash
pip install numpy scipy
```

### 2. Download GloVe Model
Manually download the GloVe model (e.g., `glove.6B.50d.txt`) from:
[https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)

Save it in the `glove_similarity/data/` directory.

## Usage
### Running the Script
Run the Python script to compute sentence similarities:
```bash
python similarity.py
```

## Directory Structure
```
GloVe_Similarity/
│── glove_similarity/
│   ├── src/
│   │   ├── glove_loader.py
│   │   ├── sentence_embedding.py
│   │   ├── similarity_calculator.py
│   ├── data/
│   │   ├── glove.6B.50d.txt
│── similarity.py
│── README.md
```

## Troubleshooting
- **Error: GloVe file not found** → Ensure that `glove.6B.50d.txt` is in `glove_similarity/data/`
- **Missing dependencies** → Install all required libraries using `pip install numpy scipy`

## License
This project is open-source under the MIT License.

