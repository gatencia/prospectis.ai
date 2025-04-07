"""
Tests for the IEEE Xplore API client.
"""

import pytest
import os
from unittest.mock import patch, MagicMock
from datetime import datetime

from research_pipeline.apis.ieee import IEEEClient
from research_pipeline.models.research_paper import ResearchPaper


@pytest.fixture
def ieee_client():
    """Create an IEEE client instance with a mock API key."""
    os.environ["IEEE_API_KEY"] = "test_api_key"
    return IEEEClient()


def test_ieee_client_initialization(ieee_client):
    """Test that the IEEE client initializes correctly."""
    assert ieee_client.name == "IEEE"
    assert ieee_client.api_key == "test_api_key"


@patch('research_pipeline.apis.ieee.IEEEClient._make_request')
def test_fetch_recent_papers(mock_make_request, ieee_client):
    """Test fetching recent papers from IEEE."""
    # Create mock response
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "total_records": 2,
        "articles": [
            {
                "article_number": "1234567",
                "title": "Test Paper 1",
                "abstract": "This is a test abstract 1",
                "authors": {
                    "authors": [
                        {
                            "full_name": "John Doe",
                            "affiliation": "Test University"
                        }
                    ]
                },
                "publication_date": "2021-01-01",
                "doi": "10.1109/test.1",
                "html_url": "https://ieeexplore.ieee.org/document/1234567",
                "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234567",
                "index_terms": {
                    "ieee_terms": {
                        "terms": ["Artificial Intelligence", "Machine Learning"]
                    },
                    "author_terms": {
                        "terms": ["Deep Learning"]
                    }
                },
                "publication_title": "IEEE Transactions on Testing",
                "publisher": "IEEE",
                "publication_year": "2021",
                "content_type": "Journals",
                "access_type": "Open Access",
                "is_open_access": True
            },
            {
                "article_number": "7654321",
                "title": "Test Paper 2",
                "abstract": "This is a test abstract 2",
                "authors": {
                    "authors": [
                        {
                            "full_name": "Jane Smith"
                        }
                    ]
                },
                "publication_date": "2021-01",
                "doi": "10.1109/test.2",
                "html_url": "https://ieeexplore.ieee.org/document/7654321",
                "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7654321",
                "keywords": ["Software Engineering", "Testing"],
                "publication_title": "IEEE Conference on Testing",
                "publisher": "IEEE",
                "publication_year": "2021",
                "content_type": "Conferences",
                "conference_location": "New York, NY, USA",
                "conference_dates": "Jan. 15-17, 2021"
            }
        ]
    }
    mock_make_request.return_value = mock_response
    
    # Call fetch_recent_papers
    papers = ieee_client.fetch_recent_papers(days_back=7, limit=2)
    
    # Assertions
    assert len(papers) == 2
    
    # Check first paper
    assert papers[0].paper_id == "ieee_1234567"
    assert papers[0].source == "ieee"
    assert papers[0].source_id == "1234567"
    assert papers[0].title == "Test Paper 1"
    assert papers[0].abstract == "This is a test abstract 1"
    assert len(papers[0].authors) == 1
    assert papers[0].authors[0].name == "John Doe"
    assert papers[0].authors[0].affiliation == "Test University"
    assert papers[0].published_date.strftime("%Y-%m-%d") == "2021-01-01"
    assert papers[0].url == "https://ieeexplore.ieee.org/document/1234567"
    assert papers[0].doi == "10.1109/test.1"
    assert set(papers[0].categories) == set(["Artificial Intelligence", "Machine Learning", "Deep Learning"])
    assert papers[0].pdf_url == "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234567"
    assert papers[0].journal == "IEEE Transactions on Testing"
    assert papers[0].metadata["publisher"] == "IEEE"
    assert papers[0].metadata["is_open_access"] is True
    
    # Check second paper
    assert papers[1].paper_id == "ieee_7654321"
    assert papers[1].title == "Test Paper 2"
    assert len(papers[1].authors) == 1
    assert papers[1].authors[0].name == "Jane Smith"
    assert papers[1].published_date.year == 2021
    assert papers[1].published_date.month == 1
    assert set(papers[1].categories) == set(["Software Engineering", "Testing"])
    assert papers[1].metadata["conference_location"] == "New York, NY, USA"


def test_convert_to_model(ieee_client):
    """Test converting IEEE item to ResearchPaper model."""
    # Create a mock IEEE item
    ieee_item = {
        "article_number": "1234567",
        "title": "Test Paper",
        "abstract": "This is a test abstract",
        "authors": {
            "authors": [
                {
                    "full_name": "John Doe",
                    "affiliation": "Test University"
                }
            ]
        },
        "publication_date": "2021-01-01",
        "doi": "10.1109/test.1",
        "html_url": "https://ieeexplore.ieee.org/document/1234567",
        "pdf_url": "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234567",
        "index_terms": {
            "ieee_terms": {
                "terms": ["Artificial Intelligence", "Machine Learning"]
            },
            "author_terms": {
                "terms": ["Deep Learning"]
            }
        },
        "keywords": ["Algorithm"],
        "publication_title": "IEEE Transactions on Testing",
        "publisher": "IEEE",
        "publication_year": "2021",
        "content_type": "Journals",
        "access_type": "Open Access",
        "is_open_access": True
    }
    
    # Convert to model
    paper = ieee_client._convert_to_model(ieee_item)
    
    # Assertions
    assert isinstance(paper, ResearchPaper)
    assert paper.paper_id == "ieee_1234567"
    assert paper.source == "ieee"
    assert paper.source_id == "1234567"
    assert paper.title == "Test Paper"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 1
    assert paper.authors[0].name == "John Doe"
    assert paper.authors[0].affiliation == "Test University"
    assert paper.published_date.strftime("%Y-%m-%d") == "2021-01-01"
    assert paper.url == "https://ieeexplore.ieee.org/document/1234567"
    assert paper.doi == "10.1109/test.1"
    # Check that all terms are included in categories
    for term in ["Artificial Intelligence", "Machine Learning", "Deep Learning", "Algorithm"]:
        assert term in paper.categories
    assert paper.pdf_url == "https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=1234567"
    assert paper.journal == "IEEE Transactions on Testing"
    assert paper.metadata["publisher"] == "IEEE"
    assert paper.metadata["publication_year"] == "2021"
    assert paper.metadata["content_type"] == "Journals"
    assert paper.metadata["is_open_access"] is True