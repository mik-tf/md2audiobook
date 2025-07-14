"""
Markdown Processor - Core markdown parsing and structure extraction
Part of md2audiobook pipeline
"""

import re
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import frontmatter
import markdown
from markdown.extensions import codehilite, tables, toc


@dataclass
class DocumentStructure:
    """Structured representation of a markdown document for audiobook generation"""
    metadata: Dict[str, Any]
    title: str
    chapters: List['Chapter']
    math_expressions: List['MathExpression']
    citations: List['Citation']
    footnotes: List['Footnote']
    code_blocks: List['CodeBlock']


@dataclass
class Chapter:
    """Chapter or section within the document"""
    level: int  # Header level (1, 2, 3, etc.)
    title: str
    content: str
    start_line: int
    end_line: int
    subsections: List['Chapter']


@dataclass
class MathExpression:
    """Mathematical expression (LaTeX)"""
    latex: str  # The LaTeX content
    is_block: bool  # True for $$...$$, False for $...$
    line_number: int
    context: str  # Surrounding text for context


@dataclass
class Citation:
    """Academic citation reference"""
    original: str
    author: str
    year: str
    line_number: int
    context: str


@dataclass
class Footnote:
    """Footnote or reference"""
    marker: str
    content: str
    line_number: int


@dataclass
class CodeBlock:
    """Code block or inline code"""
    content: str
    language: Optional[str]
    is_block: bool  # True for ```...```, False for `...`
    line_number: int


class MarkdownProcessor:
    """
    Core markdown processor for audiobook generation
    
    Parses markdown documents and extracts structured information
    optimized for text-to-speech conversion, including:
    - YAML frontmatter metadata
    - Chapter/section hierarchy
    - Mathematical expressions (LaTeX)
    - Academic citations
    - Code blocks and technical content
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.markdown_config = config.get('markdown', {})
        
        # Initialize markdown parser with extensions
        self.md = markdown.Markdown(
            extensions=[
                'codehilite',
                'tables', 
                'toc',
                'footnotes',
                'attr_list',
                'def_list'
            ],
            extension_configs={
                'codehilite': {
                    'css_class': 'highlight',
                    'use_pygments': False
                },
                'toc': {
                    'permalink': False
                }
            }
        )
        
        # Regex patterns for content extraction
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for content extraction"""
        # Mathematical expressions
        self.math_block_pattern = re.compile(r'\$\$(.*?)\$\$', re.DOTALL)
        self.math_inline_pattern = re.compile(r'\$(.*?)\$')
        
        # Citations (common academic formats)
        self.citation_patterns = [
            re.compile(r'\(([A-Za-z\s]+),\s*(\d{4})\)'),  # (Author, Year)
            re.compile(r'\[([A-Za-z\s]+)\s+(\d{4})\]'),   # [Author Year]
            re.compile(r'\(([A-Za-z\s]+)\s+(\d{4})\)'),   # (Author Year)
        ]
        
        # Headers for chapter detection
        self.header_pattern = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
        
        # Code blocks
        self.code_block_pattern = re.compile(r'```(\w+)?\n(.*?)```', re.DOTALL)
        self.code_inline_pattern = re.compile(r'`([^`]+)`')
        
        # Footnotes
        self.footnote_pattern = re.compile(r'\[\^([^\]]+)\]:\s*(.+)')
    
    def process_document(self, file_path: Path) -> DocumentStructure:
        """
        Process a markdown document into structured format
        
        Args:
            file_path: Path to the markdown file
            
        Returns:
            DocumentStructure: Parsed and structured document
        """
        # Read and parse frontmatter
        with open(file_path, 'r', encoding='utf-8') as f:
            post = frontmatter.load(f)
        
        content = post.content
        metadata = post.metadata
        
        # Extract document title
        title = self._extract_title(content, metadata)
        
        # Extract structured content
        chapters = self._extract_chapters(content)
        math_expressions = self._extract_math_expressions(content)
        citations = self._extract_citations(content)
        footnotes = self._extract_footnotes(content)
        code_blocks = self._extract_code_blocks(content)
        
        return DocumentStructure(
            metadata=metadata,
            title=title,
            chapters=chapters,
            math_expressions=math_expressions,
            citations=citations,
            footnotes=footnotes,
            code_blocks=code_blocks
        )
    
    def _extract_title(self, content: str, metadata: Dict[str, Any]) -> str:
        """Extract document title from metadata or first header"""
        # Try metadata first
        if 'title' in metadata:
            return metadata['title']
        
        # Try first H1 header
        h1_match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if h1_match:
            return h1_match.group(1).strip()
        
        return "Untitled Document"
    
    def _extract_chapters(self, content: str) -> List[Chapter]:
        """Extract chapter structure from headers"""
        chapters = []
        lines = content.split('\n')
        chapter_levels = self.markdown_config.get('chapter_levels', [1, 2, 3, 4, 5, 6])
        
        current_chapters = {}  # Track chapters by level
        
        for i, line in enumerate(lines):
            header_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            if header_match:
                level = len(header_match.group(1))
                title = header_match.group(2).strip()
                
                if level in chapter_levels:
                    # Find chapter content (until next header of ANY level)
                    content_lines = []
                    j = i + 1
                    while j < len(lines):
                        next_header = re.match(r'^(#{1,6})\s+', lines[j])
                        if next_header:  # Stop at ANY header
                            break
                        content_lines.append(lines[j])
                        j += 1
                    
                    chapter = Chapter(
                        level=level,
                        title=title,
                        content='\n'.join(content_lines).strip(),
                        start_line=i,
                        end_line=j-1,
                        subsections=[]
                    )
                    
                    # Handle chapter hierarchy
                    if level == min(chapter_levels):
                        # Top-level chapter
                        chapters.append(chapter)
                        current_chapters[level] = chapter
                    else:
                        # Subsection - add to parent chapter
                        parent_level = level - 1
                        while parent_level >= min(chapter_levels):
                            if parent_level in current_chapters:
                                current_chapters[parent_level].subsections.append(chapter)
                                break
                            parent_level -= 1
                        else:
                            # No parent found, treat as top-level
                            chapters.append(chapter)
                    
                    current_chapters[level] = chapter
        
        return chapters
    
    def _extract_math_expressions(self, content: str) -> List[MathExpression]:
        """Extract LaTeX mathematical expressions"""
        expressions = []
        lines = content.split('\n')
        
        # Block math expressions ($$...$$)
        for match in self.math_block_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            context = self._get_context(content, match.start(), match.end())
            
            expressions.append(MathExpression(
                latex=match.group(1).strip(),
                is_block=True,
                line_number=line_num,
                context=context
            ))
        
        # Inline math expressions ($...$)
        # Remove block expressions first to avoid conflicts
        content_no_blocks = self.math_block_pattern.sub('', content)
        for match in self.math_inline_pattern.finditer(content_no_blocks):
            line_num = content_no_blocks[:match.start()].count('\n') + 1
            context = self._get_context(content_no_blocks, match.start(), match.end())
            
            expressions.append(MathExpression(
                latex=match.group(1).strip(),
                is_block=False,
                line_number=line_num,
                context=context
            ))
        
        return expressions
    
    def _extract_citations(self, content: str) -> List[Citation]:
        """Extract academic citations"""
        citations = []
        
        for pattern in self.citation_patterns:
            for match in pattern.finditer(content):
                line_num = content[:match.start()].count('\n') + 1
                context = self._get_context(content, match.start(), match.end())
                
                citations.append(Citation(
                    original=match.group(0),
                    author=match.group(1).strip(),
                    year=match.group(2).strip(),
                    line_number=line_num,
                    context=context
                ))
        
        return citations
    
    def _extract_footnotes(self, content: str) -> List[Footnote]:
        """Extract footnotes and references"""
        footnotes = []
        
        for match in self.footnote_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            
            footnotes.append(Footnote(
                marker=match.group(1),
                content=match.group(2).strip(),
                line_number=line_num
            ))
        
        return footnotes
    
    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract code blocks and inline code"""
        code_blocks = []
        
        # Block code (```...```)
        for match in self.code_block_pattern.finditer(content):
            line_num = content[:match.start()].count('\n') + 1
            language = match.group(1) if match.group(1) else None
            
            code_blocks.append(CodeBlock(
                content=match.group(2).strip(),
                language=language,
                is_block=True,
                line_number=line_num
            ))
        
        # Inline code (`...`)
        # Remove block code first to avoid conflicts
        content_no_blocks = self.code_block_pattern.sub('', content)
        for match in self.code_inline_pattern.finditer(content_no_blocks):
            line_num = content_no_blocks[:match.start()].count('\n') + 1
            
            code_blocks.append(CodeBlock(
                content=match.group(1).strip(),
                language=None,
                is_block=False,
                line_number=line_num
            ))
        
        return code_blocks
    
    def _get_context(self, content: str, start: int, end: int, 
                    context_chars: int = 100) -> str:
        """Get surrounding context for an expression"""
        context_start = max(0, start - context_chars)
        context_end = min(len(content), end + context_chars)
        return content[context_start:context_end].strip()
    
    def get_audiobook_metadata(self, doc_structure: DocumentStructure) -> Dict[str, Any]:
        """
        Extract metadata suitable for audiobook generation
        
        Args:
            doc_structure: Parsed document structure
            
        Returns:
            Dict containing audiobook metadata
        """
        metadata = doc_structure.metadata.copy()
        
        # Ensure required fields
        audiobook_metadata = {
            'title': doc_structure.title,
            'author': metadata.get('author', 'Unknown Author'),
            'description': metadata.get('description', ''),
            'date': metadata.get('date', ''),
            'language': metadata.get('language', 'en'),
            'genre': metadata.get('genre', 'Educational'),
            'narrator': metadata.get('narrator', metadata.get('author', 'AI Narrator')),
            'chapters': len(doc_structure.chapters),
            'has_math': len(doc_structure.math_expressions) > 0,
            'has_citations': len(doc_structure.citations) > 0,
            'has_code': len(doc_structure.code_blocks) > 0,
        }
        
        # Add custom audiobook settings from frontmatter
        for key in ['narrator_voice', 'chapter_voice', 'math_voice', 'reading_speed']:
            if key in metadata:
                audiobook_metadata[key] = metadata[key]
        
        return audiobook_metadata
    
    def validate_document(self, doc_structure: DocumentStructure) -> Tuple[bool, List[str]]:
        """
        Validate document structure for audiobook generation
        
        Args:
            doc_structure: Parsed document structure
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for title
        if not doc_structure.title or doc_structure.title == "Untitled Document":
            issues.append("Document lacks a clear title")
        
        # Check for content
        if not doc_structure.chapters:
            issues.append("Document has no identifiable chapters or sections")
        
        # Check for very short content
        total_content = sum(len(chapter.content) for chapter in doc_structure.chapters)
        if total_content < 100:
            issues.append("Document content is very short (< 100 characters)")
        
        # Validate math expressions
        for expr in doc_structure.math_expressions:
            if not expr.latex.strip():
                issues.append(f"Empty math expression found at line {expr.line_number}")
        
        # Check for balanced math delimiters
        content = '\n'.join(chapter.content for chapter in doc_structure.chapters)
        if content.count('$') % 2 != 0:
            issues.append("Unbalanced math delimiters ($) detected")
        
        return len(issues) == 0, issues
