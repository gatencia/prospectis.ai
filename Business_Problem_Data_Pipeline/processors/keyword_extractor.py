"""Extract keywords from problem descriptions."""

import re
import logging
from collections import Counter

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Extract keywords from business problem text."""
    
    def __init__(self):
        """Initialize the keyword extractor."""
        # Common tech terms that are good keywords
        self.tech_terms = set([
            "api", "algorithm", "automation", "backend", "bug", "cloud", 
            "database", "deployment", "frontend", "infrastructure", "interface",
            "latency", "memory", "network", "optimization", "performance", 
            "pipeline", "scaling", "security", "server", "storage", 
            "throughput", "ui", "ux", "validation"
        ])
        
        # Stop words to filter out
        self.stop_words = set([
            "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
            "be", "been", "being", "have", "has", "had", "do", "does", "did",
            "will", "would", "shall", "should", "can", "could", "may", "might",
            "must", "to", "of", "in", "for", "on", "by", "at", "with", "about",
            "against", "between", "into", "through", "during", "before", "after",
            "above", "below", "from", "up", "down", "out", "off", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "any", "both", "each", "few", "more", "most",
            "other", "some", "such", "no", "nor", "not", "only", "own", "same",
            "so", "than", "too", "very", "s", "t", "just", "don", "now"
        ])
        
        # Word tokenization pattern
        self.word_pattern = re.compile(r'\b[a-zA-Z]\w+\b')
    
    def extract_keywords(self, problem):
        """
        Extract keywords from a business problem.
        
        Args:
            problem (BusinessProblem): Business problem to process
            
        Returns:
            BusinessProblem: Business problem with extracted keywords
        """
        # Extract keywords from text
        text = problem.text.lower()
        if problem.title:
            # Title words are given more weight
            text = problem.title.lower() + " " + text
        
        # Extract keywords from the text
        problem.keywords = self._extract_from_text(text)
        
        # Add tags as keywords if available
        if problem.tags:
            for tag in problem.tags:
                if tag.lower() not in problem.keywords:
                    problem.keywords.append(tag.lower())
        
        logger.debug(f"Extracted {len(problem.keywords)} keywords from problem {problem.id}")
        return problem
    
    def _extract_from_text(self, text, max_keywords=10):
        """
        Extract keywords from text.
        
        Args:
            text (str): Text to extract keywords from
            max_keywords (int): Maximum number of keywords to extract
            
        Returns:
            list: List of extracted keywords
        """
        # Find all words
        words = self.word_pattern.findall(text)
        
        # Filter out stop words and short words
        filtered_words = [word.lower() for word in words 
                          if word.lower() not in self.stop_words 
                          and len(word) > 2]
        
        # Count word frequencies
        word_counts = Counter(filtered_words)
        
        # Prioritize tech terms
        tech_keywords = [word for word in word_counts if word in self.tech_terms]
        
        # Get the most common words
        common_keywords = [word for word, _ in word_counts.most_common(max_keywords)]
        
        # Combine and deduplicate
        all_keywords = []
        for keyword in tech_keywords + common_keywords:
            if keyword not in all_keywords:
                all_keywords.append(keyword)
        
        # Limit to max_keywords
        return all_keywords[:max_keywords]