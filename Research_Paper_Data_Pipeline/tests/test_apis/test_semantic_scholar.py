"""
Tests for the Semantic Scholar API client.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from research_pipeline.apis.semantic_scholar import SemanticScholarClient
from research_pipeline.models.research_paper import ResearchPaper


@pytest.fixture
def semantic_scholar_client():
    """Create a Semantic Scholar client instance with a mock API key."""
    os.environ["SEMANTIC_SCHOLAR_API_KEY"] = "test_api_key"
    return SemanticScholarClient()


def test_semantic_scholar_client_initialization(semantic_scholar_client):
    """Test that the Semantic Scholar client initializes correctly."""
    assert semantic_scholar_client.name == "Semantic Scholar"
    assert semantic_scholar_client.api_key == "test_api_key"
    assert semantic_scholar_client.session.headers.get("x-api-key") == "test_api_key"


@patch('research_pipeline.apis.semantic_scholar.SemanticScholarClient._make_request')
def test_fetch_recent_papers(mock_make_request, semantic_scholar_client):
    """Test fetching recent papers from Semantic Scholar."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "data": [
            {
                "paperId": "1234567890",
                "externalIds": {
                    "DOI": "10.1234/test.1"
                },
                "url": "https://semanticscholar.org/paper/1234567890",
                "title": "Test Paper 1",
                "abstract": "This is a test abstract 1",
                "venue": "Journal of Testing",
                "year": 2021,
                "publicationDate": "2021-01-01",
                "authors": [
                    {
                        "authorId": "123456",
                        "name": "John Doe"
                    }
                ],
                "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
                "s2FieldsOfStudy": [
                    {
                        "category": "Computer Science",
                        "source": "s2-fos-model"
                    }
                ],
                "openAccessPdf": {
                    "url": "https://arxiv.org/pdf/test.1.pdf"
                }
            },
            {
                "paperId": "0987654321",
                "externalIds": {
                    "DOI": "10.1234/test.2"
                },
                "url": "https://semanticscholar.org/paper/0987654321",
                "title": "Test Paper 2",
                "abstract": "This is a test abstract 2",
                "venue": "Conference on Testing",
                "year": 2021,
                "publicationDate": "2021-01-02",
                "authors": [
                    {
                        "authorId": "654321",
                        "name": "Jane Smith"
                    }
                ],
                "fieldsOfStudy": ["Computer Science", "Machine Learning"],
                "s2FieldsOfStudy": [
                    {
                        "category": "Computer Science > Machine Learning",
                        "source": "s2-fos-model"
                    }
                ]
            }
        ]
    }
    mock_make_request.return_value = mock_response
    
    # Patch the filtering methods to ensure both papers are processed
    with patch('research_pipeline.apis.semantic_scholar.SemanticScholarClient._is_cs_paper', return_value=True):
        with patch('research_pipeline.apis.semantic_scholar.SemanticScholarClient._is_within_date_range', return_value=True):
            # Call fetch_recent_papers
            papers = semantic_scholar_client.fetch_recent_papers(days_back=7, limit=2)
    
    # Assertions
    assert len(papers) == 2
    
    # Check first paper
    assert papers[0].paper_id == "semanticscholar_1234567890"
    assert papers[0].source == "semantic_scholar"
    assert papers[0].source_id == "1234567890"
    assert papers[0].title == "Test Paper 1"
    assert papers[0].abstract == "This is a test abstract 1"
    assert len(papers[0].authors) == 1
    assert papers[0].authors[0].name == "John Doe"
    assert papers[0].published_date.strftime("%Y-%m-%d") == "2021-01-01"
    assert papers[0].url == "https://semanticscholar.org/paper/1234567890"
    assert papers[0].doi == "10.1234/test.1"
    assert set(papers[0].categories) == set(["Computer Science", "Artificial Intelligence"])
    assert papers[0].pdf_url == "https://arxiv.org/pdf/test.1.pdf"
    assert papers[0].journal == "Journal of Testing"
    assert papers[0].metadata["year"] == 2021
    
    # Check second paper
    assert papers[1].paper_id == "semanticscholar_0987654321"
    assert papers[1].title == "Test Paper 2"
    assert papers[1].published_date.strftime("%Y-%m-%d") == "2021-01-02"
    assert "Computer Science" in papers[1].categories
    assert "Machine Learning" in papers[1].categories


def test_is_cs_paper(semantic_scholar_client):
    """Test the _is_cs_paper method."""
    # Test with CS in fieldsOfStudy
    cs_paper = {
        "fieldsOfStudy": ["Computer Science", "Mathematics"]
    }
    assert semantic_scholar_client._is_cs_paper(cs_paper) is True
    
    # Test with CS in s2FieldsOfStudy
    s2_paper = {
        "fieldsOfStudy": ["Mathematics"],
        "s2FieldsOfStudy": [
            {
                "category": "Computer Science",
                "source": "s2-fos-model"
            }
        ]
    }
    assert semantic_scholar_client._is_cs_paper(s2_paper) is True
    
    # Test non-CS paper
    non_cs_paper = {
        "fieldsOfStudy": ["Biology"],
        "s2FieldsOfStudy": [
            {
                "category": "Biology",
                "source": "s2-fos-model"
            }
        ]
    }
    assert semantic_scholar_client._is_cs_paper(non_cs_paper) is False


def test_is_within_date_range(semantic_scholar_client):
    """Test the _is_within_date_range method."""
    date_since = datetime(2021, 1, 1)
    
    # Test with valid date string
    valid_paper = {
        "publicationDate": "2021-01-02"
    }
    assert semantic_scholar_client._is_within_date_range(valid_paper, date_since) is True
    
    # Test with older date
    old_paper = {
        "publicationDate": "2020-12-31"
    }
    assert semantic_scholar_client._is_within_date_range(old_paper, date_since) is False
    
    # Test with invalid date format but valid year
    year_paper = {
        "publicationDate": None,
        "year": 2021
    }
    assert semantic_scholar_client._is_within_date_range(year_paper, date_since) is True
    
    # Test with invalid date format and old year
    old_year_paper = {
        "publicationDate": None,
        "year": 2020
    }
    assert semantic_scholar_client._is_within_date_range(old_year_paper, date_since) is False


def test_convert_to_model(semantic_scholar_client):
    """Test converting Semantic Scholar item to ResearchPaper model."""
    # Create a mock Semantic Scholar item
    ss_item = {
        "paperId": "1234567890",
        "externalIds": {
            "DOI": "10.1234/test.1"
        },
        "url": "https://semanticscholar.org/paper/1234567890",
        "title": "Test Paper",
        "abstract": "This is a test abstract",
        "venue": "Journal of Testing",
        "year": 2021,
        "publicationDate": "2021-01-01",
        "authors": [
            {
                "authorId": "123456",
                "name": "John Doe"
            }
        ],
        "fieldsOfStudy": ["Computer Science", "Artificial Intelligence"],
        "s2FieldsOfStudy": [
            {
                "category": "Computer Science",
                "source": "s2-fos-model"
            }
        ],
        "openAccessPdf": {
            "url": "https://arxiv.org/pdf/test.1.pdf"
        },
        "citationCount": 10,
        "referenceCount": 20,
        "influentialCitationCount": 5,
        "publicationTypes": ["Journal Article"]
    }
    
    # Convert to model
    paper = semantic_scholar_client._convert_to_model(ss_item)
    
    # Assertions
    assert isinstance(paper, ResearchPaper)
    assert paper.paper_id == "semanticscholar_1234567890"
    assert paper.source == "semantic_scholar"
    assert paper.source_id == "1234567890"
    assert paper.title == "Test Paper"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 1
    assert paper.authors[0].name == "John Doe"
    assert paper.published_date.strftime("%Y-%m-%d") == "2021-01-01"
    assert paper.url == "https://semanticscholar.org/paper/1234567890"
    assert paper.doi == "10.1234/test.1"
    assert set(paper.categories) == set(["Computer Science", "Artificial Intelligence"])
    assert paper.pdf_url == "https://arxiv.org/pdf/test.1.pdf"
    assert paper.journal == "Journal of Testing"
    assert paper.metadata["year"] == 2021
    assert paper.metadata["citation_count"] == 10
    assert paper.metadata["reference_count"] == 20
    assert paper.metadata["influential_citation_count"] == 5
    assert paper.metadata["publication_types"] == ["Journal Article"]