# Prospectis Research Pipeline

This repository contains the data pipeline for Prospectis, a platform designed to bridge academic research and real-world business problems.

## Project Overview

Prospectis is a data-driven platform that assigns commercial value to research papers by matching their content to specific operational challenges faced by companies. The research pipeline is responsible for collecting recent academic publications from multiple sources and storing them in a structured format for further analysis.

## Features

- Automated data ingestion from multiple research paper sources:
  - arXiv
  - CrossRef
  - IEEE Xplore
  - Semantic Scholar
- Scheduled jobs for regular updates
- MongoDB storage for structured data
- Configurable parameters and filtering

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/prospectis.git
   cd prospectis
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```
   cp .env.example .env
   # Edit the .env file with your API keys and configuration
   ```

5. Set up MongoDB:
   - Install MongoDB or use a cloud instance
   - Update the MongoDB connection string in your `.env` file

## Usage

### Running the Pipeline Once

To run the pipeline once:

```
python scripts/run_pipeline.py
```

Options:
- `--days`: Number of days to look back (default: 3)
- `--limit`: Maximum number of papers to fetch per source (default: 100)
- `--sources`: Specific sources to use (e.g., `--sources arxiv crossref`)

Example:
```
python scripts/run_pipeline.py --days 7 --limit 200 --sources arxiv ieee
```

### Running the Pipeline as a Scheduled Job

To run the pipeline on a schedule:

```
python scripts/scheduler.py
```

Options:
- Same options as `run_pipeline.py`
- `--interval`: Interval between runs in hours (default: 24)

Example:
```
python scripts/scheduler.py --interval 12 --days 1 --limit 50
```

## Project Structure

```
prospectis/
├── README.md                       # Project documentation
├── requirements.txt                # Python dependencies
├── .env.example                    # Example environment variables 
├── .gitignore                      # Git ignore file
│
├── research_pipeline/              # Main package
│   ├── __init__.py                 # Package initialization
│   ├── main.py                     # Entry point for the pipeline
│   ├── config.py                   # Configuration and environment variables
│   ├── db/                         # Database related modules
│   │   ├── __init__.py
│   │   ├── connection.py           # MongoDB connection handling
│   │   └── schema.py               # Database schema definition
│   │
│   ├── models/                     # Data models
│   │   ├── __init__.py
│   │   └── research_paper.py       # Research paper model definition
│   │
│   ├── apis/                       # API client implementations
│   │   ├── __init__.py
│   │   ├── base.py                 # Base API client class
│   │   ├── arxiv.py                # arXiv API client
│   │   ├── crossref.py             # CrossRef API client
│   │   ├── ieee.py                 # IEEE Xplore API client
│   │   └── semantic_scholar.py     # Semantic Scholar API client
│   │
│   └── utils/                      # Utility functions
│       ├── __init__.py
│       └── logging_config.py       # Logging configuration
│
├── scripts/                        # Standalone scripts
│   ├── run_pipeline.py             # Script to run pipeline once
│   └── scheduler.py                # Scheduler for regular execution
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── test_apis/                  # API tests
│   │   ├── __init__.py
│   │   ├── test_arxiv.py
│   │   ├── test_crossref.py
│   │   ├── test_ieee.py
│   │   └── test_semantic_scholar.py
│   │
│   ├── test_models/                # Model tests
│   │   ├── __init__.py
│   │   └── test_research_paper.py
│   │
│   └── test_db/                    # Database tests
│       ├── __init__.py
│       └── test_connection.py
│
└── docs/                           # Additional documentation
    ├── mongodb_setup.md            # MongoDB setup instructions
    ├── api_schemas.md              # Documentation of API schemas
    └── deployment.md               # Deployment instructions
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.