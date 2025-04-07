# API Schemas Documentation

This document provides detailed information about the APIs used in the Prospectis research pipeline and their data schemas.

## Research Paper Model

All APIs convert their response data to a standardized `ResearchPaper` model with the following structure:

```python
class ResearchPaper:
    paper_id: str                # Unique identifier for the paper
    source: str                  # Source of the paper (arxiv, crossref, ieee, semantic_scholar)
    source_id: str               # ID of the paper in the original source
    title: str                   # Title of the paper
    abstract: Optional[str]      # Abstract of the paper
    authors: List[Author]        # List of authors
    published_date: Optional[datetime]  # Publication date
    url: Optional[str]           # URL to the paper
    doi: Optional[str]           # DOI of the paper
    categories: List[str]        # Categories/topics of the paper
    pdf_url: Optional[str]       # URL to the PDF of the paper
    journal: Optional[str]       # Journal where the paper was published
    metadata: Dict[str, Any]     # Additional metadata specific to the source
    ingestion_date: datetime     # Date when the paper was ingested
    last_updated: datetime       # Date when the paper was last updated
```

## arXiv API

### API Documentation

Official arXiv API documentation: https://arxiv.org/help/api/

### Details

- **Format**: Atom XML feed
- **Rate Limits**: Maximum of 1 request per 3 seconds
- **Authentication**: Not required

### Example Query

```
http://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+submittedDate:[20230101 TO 20230105]&start=0&max_results=10&sortBy=submittedDate&sortOrder=descending
```

### Response Format

The arXiv API returns data in Atom XML format, which is parsed using the `arxiv` Python package. Key fields include:

- `id`: arXiv ID
- `title`: Paper title
- `summary`: Abstract
- `published`: Publication date
- `authors`: List of authors
- `categories`: List of arXiv categories

## CrossRef API

### API Documentation

Official CrossRef API documentation: https://github.com/CrossRef/rest-api-doc

### Details

- **Format**: JSON
- **Rate Limits**: 
  - Anonymous: 50 requests per second
  - With API token: 100 requests per second
- **Authentication**: Optional (Polite Pool)

### Example Query

```
https://api.crossref.org/works?filter=from-pub-date:2023-01-01,has-abstract:true&sort=published&order=desc&rows=10
```

### Response Format

The CrossRef API returns data in JSON format. Key fields include:

- `DOI`: Digital Object Identifier
- `title`: Array of title strings
- `abstract`: HTML-formatted abstract
- `author`: Array of author objects with given, family, and affiliation
- `published-print` / `published-online`: Publication date information
- `subject`: Array of subject categories
- `URL`: Link to the paper

## IEEE Xplore API

### API Documentation

Official IEEE Xplore API documentation: https://developer.ieee.org/docs

### Details

- **Format**: JSON
- **Rate Limits**: Varies by subscription level
- **Authentication**: Required (API key)

### Example Query

```
https://ieeexploreapi.ieee.org/api/v1/search/articles?apikey=YOUR_API_KEY&format=json&max_records=10&start_record=1&sort_order=desc&sort_field=publication_date&start_date=20230101&end_date=20230105
```

### Response Format

The IEEE API returns data in JSON format. Key fields include:

- `article_number`: Unique identifier
- `title`: Paper title
- `abstract`: Abstract text
- `authors`: Array of author objects
- `publication_date`: Publication date
- `doi`: DOI
- `html_url`: Link to paper
- `pdf_url`: Link to PDF
- `index_terms`: Keywords and IEEE terms
- `publication_title`: Journal/conference name

## Semantic Scholar API

### API Documentation

Official Semantic Scholar API documentation: https://api.semanticscholar.org/api-docs/

### Details

- **Format**: JSON
- **Rate Limits**: 
  - Without API key: 100 requests per 5 minutes
  - With API key: Higher limits (varies)
- **Authentication**: Optional (API key)

### Example Query

```
https://api.semanticscholar.org/graph/v1/paper/search?query=computer%20science&fields=paperId,externalIds,url,title,abstract,venue,year,publicationDate,authors,fieldsOfStudy&limit=10
```

### Response Format

The Semantic Scholar API returns data in JSON format. Key fields include:

- `paperId`: Unique identifier
- `externalIds`: Object with DOI and other identifiers
- `title`: Paper title
- `abstract`: Abstract text
- `authors`: Array of author objects
- `publicationDate`: ISO format publication date
- `year`: Publication year
- `venue`: Journal/conference
- `fieldsOfStudy`: Array of field names
- `openAccessPdf`: Object with PDF URL

## MongoDB Schema

The research papers are stored in MongoDB with the following schema:

```javascript
{
  "paper_id": String,         // Unique identifier
  "source": String,           // API source
  "source_id": String,        // Original ID in source
  "title": String,            // Paper title
  "abstract": String,         // Abstract text
  "authors": [                // List of authors
    {
      "name": String,
      "affiliation": String,
      "email": String
    }
  ],
  "published_date": Date,     // Publication date
  "url": String,              // URL to paper
  "doi": String,              // DOI
  "categories": [String],     // List of categories
  "pdf_url": String,          // URL to PDF
  "journal": String,          // Journal/conference name
  "metadata": Object,         // Source-specific metadata
  "ingestion_date": Date,     // When paper was first ingested
  "last_updated": Date        // When paper was last updated
}
```

## API Error Handling

All API clients implement robust error handling with:

1. **Retry logic**: Automatic retry for network errors and timeouts
2. **Rate limiting**: Automatic backoff when approaching rate limits
3. **Logging**: Detailed logging of API interactions
4. **Exception handling**: Graceful handling of API errors to prevent pipeline failures

## Adding a