"""
Database schema definition for Prospectis research pipeline.
"""

from pymongo import ASCENDING, DESCENDING, IndexModel


# Define indexes for the research papers collection
RESEARCH_PAPER_INDEXES = [
    IndexModel([("paper_id", ASCENDING)], unique=True),
    IndexModel([("source", ASCENDING), ("source_id", ASCENDING)], unique=True),
    IndexModel([("published_date", DESCENDING)]),
    IndexModel([("title", "text"), ("abstract", "text")]),
    IndexModel([("categories", ASCENDING)]),
    IndexModel([("authors.name", ASCENDING)]),
    IndexModel([("ingestion_date", DESCENDING)])
]


def setup_db_indexes(db):
    """
    Set up database indexes for collections.
    
    Args:
        db: MongoDB database instance
    """
    papers_collection = db.get_collection(
        name=db.get_collection_name("MONGODB_COLLECTION_PAPERS", "research_papers")
    )
    papers_collection.create_indexes(RESEARCH_PAPER_INDEXES)


def get_research_paper_schema():
    """
    Get the JSON schema for research papers.
    This is useful for validation and documentation.
    
    Returns:
        dict: JSON schema for research papers
    """
    return {
        "bsonType": "object",
        "required": ["paper_id", "title", "source", "source_id", "ingestion_date"],
        "properties": {
            "paper_id": {
                "bsonType": "string",
                "description": "Unique identifier for the paper"
            },
            "source": {
                "bsonType": "string",
                "description": "Source of the paper (arxiv, crossref, ieee, semantic_scholar)"
            },
            "source_id": {
                "bsonType": "string",
                "description": "ID of the paper in the original source"
            },
            "title": {
                "bsonType": "string",
                "description": "Title of the paper"
            },
            "abstract": {
                "bsonType": ["string", "null"],
                "description": "Abstract of the paper"
            },
            "authors": {
                "bsonType": "array",
                "description": "Authors of the paper",
                "items": {
                    "bsonType": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {
                            "bsonType": "string",
                            "description": "Name of the author"
                        },
                        "affiliation": {
                            "bsonType": ["string", "null"],
                            "description": "Affiliation of the author"
                        },
                        "email": {
                            "bsonType": ["string", "null"],
                            "description": "Email of the author"
                        }
                    }
                }
            },
            "published_date": {
                "bsonType": ["date", "null"],
                "description": "Publication date of the paper"
            },
            "url": {
                "bsonType": ["string", "null"],
                "description": "URL to the paper"
            },
            "doi": {
                "bsonType": ["string", "null"],
                "description": "DOI of the paper"
            },
            "categories": {
                "bsonType": "array",
                "description": "Categories/topics of the paper",
                "items": {
                    "bsonType": "string"
                }
            },
            "pdf_url": {
                "bsonType": ["string", "null"],
                "description": "URL to the PDF of the paper"
            },
            "journal": {
                "bsonType": ["string", "null"],
                "description": "Journal where the paper was published"
            },
            "metadata": {
                "bsonType": "object",
                "description": "Additional metadata specific to the source"
            },
            "ingestion_date": {
                "bsonType": "date",
                "description": "Date when the paper was ingested into the system"
            },
            "last_updated": {
                "bsonType": "date",
                "description": "Date when the paper was last updated in the system"
            }
        }
    }