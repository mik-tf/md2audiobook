"""
md2audiobook - Professional Markdown to Audiobook Pipeline
Part of the ucli-tools ecosystem

A comprehensive toolkit for converting markdown documents into professional-quality
audiobooks, with special focus on academic and technical content containing
mathematical expressions, citations, and complex formatting.
"""

__version__ = "0.1.0"
__author__ = "ucli-tools"
__email__ = "contact@ucli-tools.org"
__license__ = "Apache 2.0"

from markdown_processor import MarkdownProcessor
from text_enhancer import TextEnhancer
from audiobook_generator import AudiobookGenerator

__all__ = [
    "MarkdownProcessor",
    "TextEnhancer", 
    "AudiobookGenerator",
]
