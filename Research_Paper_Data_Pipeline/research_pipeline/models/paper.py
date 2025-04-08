"""
Compatibility module that re-exports ResearchPaper as Paper for backward compatibility.
"""

from research_pipeline.models.research_paper import ResearchPaper as Paper

# Export the ResearchPaper class as Paper for compatibility
__all__ = ['Paper']
