"""Clean and normalize problem text."""

import re
import logging
import html
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class TextProcessor:
    """Processor for cleaning and normalizing text."""
    
    def __init__(self):
        """Initialize the text processor."""
        self.boilerplate_patterns = [
            r"thanks\s+(?:in\s+advance|for\s+your\s+help|for\s+any\s+help)",
            r"any\s+help\s+(?:is|would\s+be)\s+(?:greatly\s+)?appreciated",
            r"i'm\s+new\s+to\s+this",
            r"i've\s+been\s+struggling\s+with\s+this\s+for\s+(?:hours|days)",
            r"i've\s+searched\s+(?:everywhere|google|stackoverflow)",
            r"sorry\s+for\s+(?:the\s+noob\s+question|being\s+a\s+newbie)"
        ]
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) for pattern in self.boilerplate_patterns]
    
    def process(self, problem):
        """
        Process a business problem.
        
        Args:
            problem (BusinessProblem): Business problem to process
            
        Returns:
            BusinessProblem: Processed business problem
        """
        logger.debug(f"Processing problem: {problem.id}")
        
        # Clean the text
        problem.text = self._clean_text(problem.text, problem.source)
        
        # Clean the title if available
        if problem.title:
            problem.title = self._clean_text(problem.title, problem.source, is_title=True)
        
        # Mark as processed
        problem.processed = True
        
        return problem
    
    def _clean_text(self, text, source, is_title=False):
        """
        Clean and normalize text.
        
        Args:
            text (str): Text to clean
            source (str): Source of the text
            is_title (bool): Whether the text is a title
            
        Returns:
            str: Cleaned text
        """
        if not text:
            return ""
        
        # Decode HTML entities
        text = html.unescape(text)
        
        # Remove HTML tags
        if source in ["stack_exchange", "reddit"]:
            text = self._remove_html(text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+', '', text)
        
        # Remove code blocks for non-titles
        if not is_title and source in ["stack_exchange", "reddit"]:
            text = self._remove_code_blocks(text, source)
        
        # Remove markdown formatting for non-titles
        if not is_title:
            text = self._remove_markdown(text)
        
        # Remove boilerplate text for non-titles
        if not is_title:
            text = self._remove_boilerplate(text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _remove_html(self, text):
        """Remove HTML tags from text."""
        try:
            soup = BeautifulSoup(text, 'lxml')
            return soup.get_text()
        except Exception as e:
            logger.warning(f"Error removing HTML: {e}")
            # Fallback to regex if BeautifulSoup fails
            return re.sub(r'<.*?>', '', text)
    
    def _remove_code_blocks(self, text, source):
        """Remove code blocks from text based on source format."""
        if source == "stack_exchange":
            # Remove <code> and <pre> blocks
            text = re.sub(r'<code>.*?</code>', '', text, flags=re.DOTALL)
            text = re.sub(r'<pre>.*?</pre>', '', text, flags=re.DOTALL)
        
        # Remove markdown code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`.*?`', '', text)
        
        return text
    
    def _remove_markdown(self, text):
        """Remove markdown formatting from text."""
        # Remove headers
        text = re.sub(r'#+\s+', '', text)
        
        # Remove bold and italic
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
        text = re.sub(r'\*(.*?)\*', r'\1', text)
        text = re.sub(r'__(.*?)__', r'\1', text)
        text = re.sub(r'_(.*?)_', r'\1', text)
        
                # Remove lists
        text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        return text
    
    def _remove_boilerplate(self, text):
        """Remove common boilerplate phrases from text."""
        for pattern in self.compiled_patterns:
            text = pattern.sub('', text)
        return text