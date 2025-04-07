"""
Tests for the arXiv API client.
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

from research_pipeline.apis.arxiv import ArxivClient
from research_pipeline.models.research_paper import ResearchPaper


@pytest.fixture
def arxiv_client():
    """Create an arXiv client instance."""
    return ArxivClient()


def test_arxiv_client_initialization(arxiv_client):
    """Test that the arXiv client initializes correctly."""
    assert arxiv_client.name == "arXiv"


@patch('arxiv.Client')
def test_fetch_recent_papers(mock_arxiv_client, arxiv_client):
    """Test fetching recent papers from arXiv."""
    # Create mock arXiv results
    mock_result1 = MagicMock()
    mock_result1.get_short_id.return_value = "2101.00001"
    mock_result1.title = "Test Paper 1"
    mock_result1.summary = "This is a test abstract 1"
    mock_result1.authors = [MagicMock(name="Author One"), MagicMock(name="Author Two")]
    mock_result1.published = datetime(2021, 1, 1)
    mock_result1.entry_id = "http://arxiv.org/abs/2101.00001"
    mock_result1.pdf_url = "http://arxiv.org/pdf/2101.00001"
    mock_result1.comment = "Test comment"
    mock_result1.journal_ref = None
    mock_result1.primary_category = "cs.AI"
    mock_result1.categories = ["cs.AI", "cs.LG"]
    mock_result1.updated = datetime(2021, 1, 2)
    
    mock_result2 = MagicMock()
    mock_result2.get_short_id.return_value = "2101.00002"
    mock_result2.title = "Test Paper 2"
    mock_result2.summary = "This is a test abstract 2"
    mock_result2.authors = [MagicMock(name="Author Three")]
    mock_result2.published = datetime(2021, 1, 2)
    mock_result2.entry_id = "http://arxiv.org/abs/2101.00002"
    mock_result2.pdf_url = "http://arxiv.org/pdf/2101.00002"
    mock_result2.comment = None
    mock_result2.journal_ref = None
    mock_result2.primary_category = "cs.CL"
    mock_result2.categories = ["cs.CL"]
    mock_result2.updated = None
    
    # Setup mock client to return our mock results
    mock_client_instance = MagicMock()
    mock_client_instance.results.return_value = [mock_result1, mock_result2]
    mock_arxiv_client.return_value = mock_client_instance
    
    # Call fetch_recent_papers
    papers = arxiv_client.fetch_recent_papers(days_back=7, limit=2)
    
    # Assertions
    assert len(papers) == 2
    
    # Check first paper
    assert papers[0].paper_id == "arxiv_2101.00001"
    assert papers[0].source == "arxiv"
    assert papers[0].source_id == "2101.00001"
    assert papers[0].title == "Test Paper 1"
    assert papers[0].abstract == "This is a test abstract 1"
    assert len(papers[0].authors) == 2
    assert papers[0].authors[0].name == "Author One"
    assert papers[0].published_date == datetime(2021, 1, 1)
    assert papers[0].url == "http://arxiv.org/abs/2101.00001"
    assert papers[0].pdf_url == "http://arxiv.org/pdf/2101.00001"
    assert papers[0].categories == ["cs.AI", "cs.LG"]
    assert papers[0].metadata["comment"] == "Test comment"
    assert papers[0].metadata["primary_category"] == "cs.AI"
    assert papers[0].metadata["updated"] == "2021-01-02T00:00:00"
    
    # Check second paper
    assert papers[1].paper_id == "arxiv_2101.00002"
    assert papers[1].title == "Test Paper 2"
    assert len(papers[1].authors) == 1
    assert papers[1].metadata["updated"] is None


def test_convert_to_model(arxiv_client):
    """Test converting arXiv result to ResearchPaper model."""
    # Create a mock arXiv result
    mock_result = MagicMock()
    mock_result.get_short_id.return_value = "2101.00001"
    mock_result.title = "Test Paper"
    mock_result.summary = "This is a test abstract"
    mock_result.authors = [MagicMock(name="Author One")]
    mock_result.published = datetime(2021, 1, 1)
    mock_result.entry_id = "http://arxiv.org/abs/2101.00001"
    mock_result.pdf_url = "http://arxiv.org/pdf/2101.00001"
    mock_result.comment = "Test comment"
    mock_result.journal_ref = None
    mock_result.primary_category = "cs.AI"
    mock_result.categories = ["cs.AI", "cs.LG"]
    mock_result.updated = datetime(2021, 1, 2)
    
    # Convert to model
    paper = arxiv_client._convert_to_model(mock_result)
    
    # Assertions
    assert isinstance(paper, ResearchPaper)
    assert paper.paper_id == "arxiv_2101.00001"
    assert paper.source == "arxiv"
    assert paper.source_id == "2101.00001"
    assert paper.title == "Test Paper"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 1
    assert paper.authors[0].name == "Author One"
    assert paper.published_date == datetime(2021, 1, 1)
    assert paper.url == "http://arxiv.org/abs/2101.00001"
    assert paper.pdf_url == "http://arxiv.org/pdf/2101.00001"
    assert paper.categories == ["cs.AI", "cs.LG"]
    assert paper.metadata["comment"] == "Test comment"
    assert paper.metadata["primary_category"] == "cs.AI"
    assert paper.metadata["updated"] == "2021-01-02T00:00:00"