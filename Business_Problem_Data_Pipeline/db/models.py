"""MongoDB models for business problems."""

import json
import uuid
from datetime import datetime
from bson import ObjectId

class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder for MongoDB objects."""
    def default(self, obj):
        if isinstance(obj, ObjectId):
            return str(obj)
        if isinstance(obj, datetime):
            return obj.isoformat()
        return super(JSONEncoder, self).default(obj)

class BusinessProblem:
    """Business problem model."""
    
    def __init__(self, 
                 text, 
                 source,
                 source_id=None,
                 title=None,
                 url=None,
                 author=None,
                 created_at=None,
                 tags=None,
                 metadata=None):
        """
        Initialize a new business problem.
        
        Args:
            text (str): The problem description text
            source (str): Source name (e.g., 'reddit', 'stackoverflow')
            source_id (str, optional): Original ID from the source
            title (str, optional): Problem title if available
            url (str, optional): URL to the original content
            author (str, optional): Author of the content
            created_at (datetime, optional): When the content was created
            tags (list, optional): List of tags/categories
            metadata (dict, optional): Additional source-specific metadata
        """
        self.id = str(uuid.uuid4())
        self.text = text
        self.source = source
        self.source_id = source_id
        self.title = title
        self.url = url
        self.author = author
        self.created_at = created_at or datetime.utcnow()
        self.tags = tags or []
        self.metadata = metadata or {}
        self.collected_at = datetime.utcnow()
        self.processed = False
        self.keywords = []
    
    def to_dict(self):
        """Convert the business problem to a dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "source_id": self.source_id,
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "created_at": self.created_at,
            "tags": self.tags,
            "metadata": self.metadata,
            "collected_at": self.collected_at,
            "processed": self.processed,
            "keywords": self.keywords
        }
    
    def to_json(self):
        """Convert the business problem to a JSON string."""
        return json.dumps(self.to_dict(), cls=JSONEncoder)
    
    @classmethod
    def from_dict(cls, data):
        """Create a business problem from a dictionary."""
        problem = cls(
            text=data["text"],
            source=data["source"],
            source_id=data.get("source_id"),
            title=data.get("title"),
            url=data.get("url"),
            author=data.get("author"),
            created_at=data.get("created_at"),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )
        problem.id = data.get("id", problem.id)
        problem.collected_at = data.get("collected_at", problem.collected_at)
        problem.processed = data.get("processed", problem.processed)
        problem.keywords = data.get("keywords", problem.keywords)
        return problem