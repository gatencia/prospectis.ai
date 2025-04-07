"""
Tests for the CrossRef API client.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from research_pipeline.apis.crossref import CrossRefClient
from research_pipeline.models.research_paper import ResearchPaper


@pytest.fixture
def crossref_client():
    """Create a CrossRef client instance."""
    return CrossRefClient()


def test_crossref_client_initialization(crossref_client):
    """Test that the CrossRef client initializes correctly."""
    assert crossref_client.name == "CrossRef"
    # Remove the email check since we're now using the environment variable


@patch('research_pipeline.apis.crossref.CrossRefClient._make_request')
def test_fetch_recent_papers(mock_make_request, crossref_client):
    """Test fetching recent papers from CrossRef."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "message": {
            "items": [
                {
                    "DOI": "10.1234/test.1",
                    "title": ["Test Paper 1"],
                    "abstract": "This is a test abstract 1",
                    "author": [
                        {
                            "given": "John",
                            "family": "Doe",
                            "affiliation": [
                                {"name": "Test University"}
                            ]
                        }
                    ],
                    "published-online": {
                        "date-parts": [[2021, 1, 1]]
                    },
                    "URL": "https://doi.org/10.1234/test.1",
                    "subject": ["Computer Science", "Artificial Intelligence"],
                    "container-title": ["Journal of Testing"]
                },
                {
                    "DOI": "10.1234/test.2",
                    "title": ["Test Paper 2"],
                    "abstract": "This is a test abstract 2",
                    "author": [
                        {
                            "given": "Jane",
                            "family": "Smith"
                        }
                    ],
                    "published-print": {
                        "date-parts": [[2021, 1, 2]]
                    },
                    "URL": "https://doi.org/10.1234/test.2",
                    "subject": ["Machine Learning"],
                    "container-title": ["Conference on Testing"]
                }
            ]
        }
    }
    mock_make_request.return_value = mock_response
    
    # Add _is_cs_paper patch to ensure both papers are processed
    with patch('research_pipeline.apis.crossref.CrossRefClient._is_cs_paper', return_value=True):
        # Call fetch_recent_papers
        papers = crossref_client.fetch_recent_papers(days_back=7, limit=2)
    
    # Assertions
    assert len(papers) == 2
    
    # Check first paper
    assert papers[0].paper_id == "crossref_10.1234/test.1"
    assert papers[0].source == "crossref"
    assert papers[0].source_id == "10.1234/test.1"
    assert papers[0].title == "Test Paper 1"
    assert papers[0].abstract == "This is a test abstract 1"
    assert len(papers[0].authors) == 1
    assert papers[0].authors[0].name == "John Doe"
    assert papers[0].authors[0].affiliation == "Test University"
    assert papers[0].published_date == datetime(2021, 1, 1)
    assert papers[0].url == "https://doi.org/10.1234/test.1"
    assert papers[0].doi == "10.1234/test.1"
    assert set(papers[0].categories) == set(["Computer Science", "Artificial Intelligence"])
    assert papers[0].journal == "Journal of Testing"
    
    # Check second paper
    assert papers[1].paper_id == "crossref_10.1234/test.2"
    assert papers[1].title == "Test Paper 2"
    assert len(papers[1].authors) == 1
    assert papers[1].authors[0].name == "Jane Smith"
    assert papers[1].published_date == datetime(2021, 1, 2)


def test_is_cs_paper(crossref_client):
    """Test the _is_cs_paper method."""
    # Test with CS subject
    cs_paper = {
        "subject": ["Computer Science", "Mathematics"]
    }
    assert crossref_client._is_cs_paper(cs_paper) is True
    
    # Test with CS term in title/abstract
    term_paper = {
        "subject": ["Mathematics"],
        "title": ["A Study on Machine Learning Algorithms"]
    }
    assert crossref_client._is_cs_paper(term_paper) is True
    
    # Test with CS journal
    journal_paper = {
        "subject": ["Biology"],
        "title": ["A Study on Evolution"],
        "container-title": ["IEEE Transactions on Something"]
    }
    assert crossref_client._is_cs_paper(journal_paper) is True
    
    # Test non-CS paper
    non_cs_paper = {
        "subject": ["Biology"],
        "title": ["A Study on Evolution"],
        "container-title": ["Journal of Biology"]
    }
    assert crossref_client._is_cs_paper(non_cs_paper) is False


def test_convert_to_model(crossref_client):
    """Test converting CrossRef item to ResearchPaper model."""
    # Create a mock CrossRef item
    crossref_item = {
        "DOI": "10.1234/test.1",
        "title": ["Test Paper Title"],
        "abstract": "This is a test abstract",
        "author": [
            {
                "given": "John",
                "family": "Doe",
                "affiliation": [
                    {"name": "Test University"}
                ]
            }
        ],
        "published-online": {
            "date-parts": [[2021, 1, 1]]
        },
        "URL": "https://doi.org/10.1234/test.1",
        "subject": ["Computer Science", "Artificial Intelligence"],
        "container-title": ["Journal of Testing"],
        "publisher": "Test Publisher",
        "issue": "1",
        "volume": "10",
        "page": "100-110"
    }
    
    # Convert to model
    paper = crossref_client._convert_to_model(crossref_item)
    
    # Assertions
    assert isinstance(paper, ResearchPaper)
    assert paper.paper_id == "crossref_10.1234/test.1"
    assert paper.source == "crossref"
    assert paper.source_id == "10.1234/test.1"
    assert paper.title == "Test Paper Title"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 1
    assert paper.authors[0].name == "John Doe"
    assert paper.authors[0].affiliation == "Test University"
    assert paper.published_date == datetime(2021, 1, 1)
    assert paper.url == "https://doi.org/10.1234/test.1"
    assert paper.doi == "10.1234/test.1"
    assert paper.categories == ["Computer Science", "Artificial Intelligence"]
    assert paper.journal == "Journal of Testing"
    assert paper.metadata["publisher"] == "Test Publisher"
    assert paper.metadata["issue"] == "1"
    assert paper.metadata["volume"] == "10"
    assert paper.metadata["page"] == "100-110"