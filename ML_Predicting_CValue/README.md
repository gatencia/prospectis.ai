# Prospectis ML Commercial Value Prediction

This repository contains the ML components for the Prospectis platform, which assigns commercial value to research papers by matching them to real-world business problems.

## Project Overview

Prospectis bridges academic research and business problems by:

1. Generating embeddings for research papers and business problem descriptions
2. Predicting the commercial value/potential of research papers using weak supervision
3. Matching papers to relevant business problems based on embedding similarity
4. Providing APIs to score papers and find relevant solutions to business challenges

## Directory Structure

```
/ML_Predicting_CValue/
├── config/                       # Configuration files
├── data/                         # Data storage and processing
│   ├── preprocessing/            # Data preprocessing scripts
│   ├── ingestion/                # Data ingestion scripts
│   └── connectors/               # Database connectors
├── embeddings/                   # Embedding generation and management
├── models/                       # Model definitions and training
│   ├── commercial_value/         # Commercial value prediction models
│   ├── matching/                 # Paper-problem matching models
│   └── evaluation/               # Model evaluation scripts
├── inference/                    # Inference utilities
├── feedback/                     # Feedback collection and processing
├── utils/                        # Utility scripts
│   └── visualization/            # Visualization utilities
└── scripts/                      # Automation scripts
```

## Setup and Installation

1. Create a Python virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Configure MongoDB connection in `config/db_config.py`

## Key Components

### Data Processing

- **Embedding Generation**: Create vector representations of papers (`embed_papers.py`) and business problems (`embed_problems.py`)
- **Vector Database**: Manage embeddings for fast similarity search using FAISS

### ML Model Pipeline

- **Feature Extraction**: Extract features from papers for predicting commercial value
- **Proxy Label Generation**: Generate weak supervision signals for training
- **Commercial Value Model**: Train and evaluate models to predict commercial potential
- **Paper-Problem Matching**: Match papers to relevant business problems

### Usage

#### Generate Embeddings
```bash
python embeddings/embed_papers.py
python embeddings/embed_problems.py
```

#### Train Commercial Value Model
```bash
python models/commercial_value/train.py --cross-validate
```

#### Score a Paper
```bash
python inference/score_paper.py --id <paper_id>
```

#### Find Papers for a Business Problem
```bash
python inference/find_related_papers.py --text "How to optimize database performance for large-scale IoT applications"
```

#### Run Daily Updates
```bash
python scripts/daily_update.py
```

## Model Details

The commercial value prediction model uses a combination of signals:

1. **Patent Citations**: Papers cited in patents likely have industrial relevance
2. **Industry Mentions**: Papers mentioned in industry forums, blogs, or news
3. **Problem Similarity**: Papers that match multiple business problems
4. **Author Affiliations**: Papers with industry-affiliated authors

We use weak supervision to train the model without requiring manual labels, then continuously improve with user feedback.

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation as needed