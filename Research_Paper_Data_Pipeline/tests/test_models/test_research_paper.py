"""
Tests for the ResearchPaper model.
"""

import pytest
from datetime import datetime

from research_pipeline.models.research_paper import ResearchPaper, Author


def test_author_model():
    """Test the Author model."""
    # Create an Author instance
    author = Author(
        name="John Doe",
        affiliation="Example University",
        email="john.doe@example.edu"
    )
    
    # Assertions
    assert author.name == "John Doe"
    assert author.affiliation == "Example University"
    assert author.email == "john.doe@example.edu"
    
    # Test with minimal fields
    minimal_author = Author(name="Jane Smith")
    assert minimal_author.name == "Jane Smith"
    assert minimal_author.affiliation is None
    assert minimal_author.email is None


def test_research_paper_model():
    """Test the ResearchPaper model."""
    # Create a ResearchPaper instance
    now = datetime.utcnow()
    
    paper = ResearchPaper(
        paper_id="test_123",
        source="test_source",
        source_id="123",
        title="Test Paper Title",
        abstract="This is a test abstract",
        authors=[
            Author(name="John Doe", affiliation="Example University"),
            Author(name="Jane Smith")
        ],
        published_date=datetime(2021, 1, 1),
        url="https://example.com/paper",
        doi="10.1234/test.123",
        categories=["cs.AI", "cs.LG"],
        pdf_url="https://example.com/paper.pdf",
        journal="Journal of Testing",
        metadata={"key1": "value1", "key2": 42},
        ingestion_date=now,
        last_updated=now
    )
    
    # Assertions
    assert paper.paper_id == "test_123"
    assert paper.source == "test_source"
    assert paper.source_id == "123"
    assert paper.title == "Test Paper Title"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 2
    assert paper.authors[0].name == "John Doe"
    assert paper.authors[0].affiliation == "Example University"
    assert paper.authors[1].name == "Jane Smith"
    assert paper.published_date == datetime(2021, 1, 1)
    assert paper.url == "https://example.com/paper"
    assert paper.doi == "10.1234/test.123"
    assert paper.categories == ["cs.AI", "cs.LG"]
    assert paper.pdf_url == "https://example.com/paper.pdf"
    assert paper.journal == "Journal of Testing"
    assert paper.metadata == {"key1": "value1", "key2": 42}
    assert paper.ingestion_date == now
    assert paper.last_updated == now


def test_research_paper_to_dict():
    """Test the to_dict method of ResearchPaper."""
    # Create a ResearchPaper instance
    published_date = datetime(2021, 1, 1)
    ingestion_date = datetime(2021, 1, 2, 12, 0, 0)
    
    paper = ResearchPaper(
        paper_id="test_123",
        source="test_source",
        source_id="123",
        title="Test Paper Title",
        abstract="This is a test abstract",
        authors=[
            Author(name="John Doe", affiliation="Example University")
        ],
        published_date=published_date,
        categories=["cs.AI"],
        ingestion_date=ingestion_date,
        last_updated=ingestion_date
    )
    
    # Convert to dict
    paper_dict = paper.to_dict()
    
    # Assertions
    assert paper_dict["paper_id"] == "test_123"
    assert paper_dict["source"] == "test_source"
    assert paper_dict["source_id"] == "123"
    assert paper_dict["title"] == "Test Paper Title"
    assert paper_dict["abstract"] == "This is a test abstract"
    assert len(paper_dict["authors"]) == 1
    assert paper_dict["authors"][0]["name"] == "John Doe"
    assert paper_dict["authors"][0]["affiliation"] == "Example University"
    assert paper_dict["published_date"] == published_date
    assert paper_dict["categories"] == ["cs.AI"]
    assert paper_dict["ingestion_date"] == ingestion_date
    assert paper_dict["last_updated"] == ingestion_date


def test_research_paper_from_dict():
    """Test the from_dict method of ResearchPaper."""
    # Create a dictionary representation
    published_date = datetime(2021, 1, 1)
    ingestion_date = datetime(2021, 1, 2, 12, 0, 0)
    
    paper_dict = {
        "paper_id": "test_123",
        "source": "test_source",
        "source_id": "123",
        "title": "Test Paper Title",
        "abstract": "This is a test abstract",
        "authors": [
            {"name": "John Doe", "affiliation": "Example University", "email": None}
        ],
        "published_date": published_date,
        "url": "https://example.com/paper",
        "doi": "10.1234/test.123",
        "categories": ["cs.AI"],
        "pdf_url": "https://example.com/paper.pdf",
        "journal": "Journal of Testing",
        "metadata": {"key1": "value1"},
        "ingestion_date": ingestion_date,
        "last_updated": ingestion_date
    }
    
    # Create from dict
    paper = ResearchPaper.from_dict(paper_dict)
    
    # Assertions
    assert paper.paper_id == "test_123"
    assert paper.source == "test_source"
    assert paper.source_id == "123"
    assert paper.title == "Test Paper Title"
    assert paper.abstract == "This is a test abstract"
    assert len(paper.authors) == 1
    assert paper.authors[0].name == "John Doe"
    assert paper.authors[0].affiliation == "Example University"
    assert paper.published_date == published_date
    assert paper.url == "https://example.com/paper"
    assert paper.doi == "10.1234/test.123"
    assert paper.categories == ["cs.AI"]
    assert paper.pdf_url == "https://example.com/paper.pdf"
    assert paper.journal == "Journal of Testing"
    assert paper.metadata == {"key1": "value1"}
    assert paper.ingestion_date == ingestion_date
    assert paper.last_updated == ingestion_date


def test_research_paper_minimal():
    """Test the ResearchPaper model with minimal fields."""
    # Create a ResearchPaper instance with only required fields
    paper = ResearchPaper(
        paper_id="test_123",
        source="test_source",
        source_id="123",
        title="Test Paper Title"
    )
    
    # Assertions
    assert paper.paper_id == "test_123"
    assert paper.source == "test_source"
    assert paper.source_id == "123"
    assert paper.title == "Test Paper Title"
    assert paper.abstract is None
    assert paper.authors == []
    assert paper.published_date is None
    assert paper.url is None
    assert paper.doi is None
    assert paper.categories == []
    assert paper.pdf_url is None
    assert paper.journal is None
    assert paper.metadata == {}
    assert paper.ingestion_date is not None  # Default is now
    assert paper.last_updated is not None  # Default is now