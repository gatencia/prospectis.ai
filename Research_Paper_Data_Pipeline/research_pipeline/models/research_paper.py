"""
Research paper model definition for Prospectis pipeline.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Author(BaseModel):
    """Author model for research papers."""
    name: str
    affiliation: Optional[str] = None
    email: Optional[str] = None


class ResearchPaper(BaseModel):
    """Research paper model for standardizing data from different sources."""
    
    # Core fields
    paper_id: str = Field(..., description="Unique identifier for the paper")
    source: str = Field(..., description="Source of the paper (arxiv, crossref, ieee, semantic_scholar)")
    source_id: str = Field(..., description="ID of the paper in the original source")
    title: str = Field(..., description="Title of the paper")
    abstract: Optional[str] = Field(None, description="Abstract of the paper")
    
    # Metadata fields
    authors: List[Author] = Field(default_factory=list, description="Authors of the paper")
    published_date: Optional[datetime] = Field(None, description="Publication date of the paper")
    url: Optional[str] = Field(None, description="URL to the paper")
    doi: Optional[str] = Field(None, description="DOI of the paper")
    categories: List[str] = Field(default_factory=list, description="Categories/topics of the paper")
    pdf_url: Optional[str] = Field(None, description="URL to the PDF of the paper")
    journal: Optional[str] = Field(None, description="Journal where the paper was published")
    
    # Source-specific metadata
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata specific to the source")
    
    # System fields
    ingestion_date: datetime = Field(default_factory=datetime.utcnow, description="Date when the paper was ingested")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Date when the paper was last updated")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the model to a dictionary suitable for MongoDB storage."""
        return self.model_dump(by_alias=True)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResearchPaper':
        """Create a ResearchPaper instance from a dictionary."""
        return cls(**data)


class StoredResearchPaper(ResearchPaper):
    """Model for research papers retrieved from storage, including MongoDB ID."""
    _id: str = Field(..., description="MongoDB document ID")