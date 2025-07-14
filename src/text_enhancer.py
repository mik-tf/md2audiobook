"""
Text Enhancer - Academic text optimization for speech synthesis
Part of md2audiobook pipeline
"""

import re
import yaml
import json
import subprocess
import tempfile
import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import requests
from .markdown_processor import DocumentStructure, MathExpression, Citation


@dataclass
class EnhancedText:
    """Enhanced text optimized for speech synthesis"""
    content: str
    voice_assignments: Dict[str, str]  # text_segment -> voice_id
    pause_markers: List[Tuple[int, float]]  # (position, duration)
    pronunciation_guides: Dict[str, str]  # term -> pronunciation
    chapter_breaks: List[int]  # positions of chapter breaks
    chapter_titles: List[str]  # original chapter titles


class TextEnhancer:
    """
    Text enhancement for academic and technical content
    
    Optimizes markdown content for natural speech synthesis by:
    - Converting LaTeX math to spoken form
    - Naturalizing academic citations
    - Adding pronunciation guides for technical terms
    - Optimizing sentence structure for speech
    - Adding appropriate pauses and emphasis
    """
    
    def __init__(self, config: Dict[str, Any], processing_mode: str = 'basic'):
        self.config = config
        self.processing_mode = processing_mode
        self.enhancement_config = config.get('text_enhancement', {})
        self.academic_config = config.get('academic', {})
        self.logger = logging.getLogger(__name__)
        
        # Check if pandoc is available
        try:
            subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
            self.pandoc_available = True
            self.logger.info("Pandoc available for math processing")
        except (subprocess.CalledProcessError, FileNotFoundError):
            self.pandoc_available = False
            self.logger.warning("Pandoc not available, falling back to basic math processing")
        
        # Comprehensive LaTeX to speech mappings for math teacher-style narration
        self.latex_to_speech = {
            # Complex fractions and nested structures
            r'\\frac\{([^{}]+)\}\{([^{}]+)\}': r'\g<1> over \g<2>',
            
            # Roots with proper handling
            r'\\sqrt\{([^{}]+)\}': r'the square root of \g<1>',
            r'\\sqrt\[([^]]+)\]\{([^{}]+)\}': r'the \g<1>-th root of \g<2>',
            
            # Summation and integration with bounds
            r'\\sum_\{([^}]+)\}\^\{([^}]+)\}': r'the sum from \g<1> to \g<2> of',
            r'\\sum': ' the sum of ',
            r'\\int_\{([^}]+)\}\^\{([^}]+)\}': r'the integral from \g<1> to \g<2> of',
            r'\\int': ' the integral of ',
            r'\\oint': ' the contour integral of ',
            
            # Limits with proper phrasing
            r'\\lim_\{([^}]+)\\to\s*([^}]+)\}': r'the limit as \g<1> approaches \g<2> of',
            r'\\lim_\{([^}]+)\}': r'the limit as \g<1> of',
            
            # Products
            r'\\prod_\{([^}]+)\}\^\{([^}]+)\}': r'the product from \g<1> to \g<2> of',
            r'\\prod': ' the product of ',
            
            # Derivatives and differentials
            r'\\frac\{d\}\{d([^}]+)\}': r'the derivative with respect to \g<1> of',
            r'\\frac\{\\partial\}\{\\partial\s*([^}]+)\}': r'the partial derivative with respect to \g<1> of',
            r'\\partial\^\{([^}]+)\}': r'partial to the power of \g<1>',
            r'\\partial': ' partial ',
            r'\\nabla\^\{([^}]+)\}': r'del operator to the power of \g<1>',
            r'\\nabla': ' del operator ',
            
            # Superscripts and subscripts with context
            r'([a-zA-Z])\^\{([^}]+)\}': r'\g<1> to the power of \g<2>',
            r'([a-zA-Z])_\{([^}]+)\}': r'\g<1> subscript \g<2>',
            r'\^\{([^}]+)\}': r' to the power of \g<1>',
            r'_\{([^}]+)\}': r' subscript \g<1>',
            
            # Quantum mechanics specific (MUST be processed before Greek letters and absolute values)
            # Process inner products first (most specific patterns)
            r'\\langle\s*([^|]+)\s*\|\s*([^\rangle]+)\s*\\rangle': r'the inner product of \g<1> and \g<2>',
            r'\\braket\{([^}]+)\}\{([^}]+)\}': r'the inner product of \g<1> and \g<2>',
            # Then individual bra and ket notation
            r'\|([^\rangle|]+)\\rangle': r'ket \g<1>',
            r'\\langle([^|]+)\|': r'bra \g<1>',
            r'\\bra\{([^}]+)\}': r'bra \g<1>',
            r'\\ket\{([^}]+)\}': r'ket \g<1>',
            
            # Greek letters (lowercase)
            r'\\alpha': ' alpha ',
            r'\\beta': ' beta ',
            r'\\gamma': ' gamma ',
            r'\\delta': ' delta ',
            r'\\epsilon': ' epsilon ',
            r'\\varepsilon': ' epsilon ',
            r'\\zeta': ' zeta ',
            r'\\eta': ' eta ',
            r'\\theta': ' theta ',
            r'\\vartheta': ' theta ',
            r'\\iota': ' iota ',
            r'\\kappa': ' kappa ',
            r'\\lambda': ' lambda ',
            r'\\mu': ' mu ',
            r'\\nu': ' nu ',
            r'\\xi': ' xi ',
            r'\\pi': ' pi ',
            r'\\varpi': ' pi ',
            r'\\rho': ' rho ',
            r'\\varrho': ' rho ',
            r'\\sigma': ' sigma ',
            r'\\varsigma': ' sigma ',
            r'\\tau': ' tau ',
            r'\\upsilon': ' upsilon ',
            r'\\phi': ' phi ',
            r'\\varphi': ' phi ',
            r'\\chi': ' chi ',
            r'\\psi': ' psi ',
            r'\\omega': ' omega ',
            
            # Greek letters (uppercase)
            r'\\Gamma': ' capital gamma ',
            r'\\Delta': ' capital delta ',
            r'\\Theta': ' capital theta ',
            r'\\Lambda': ' capital lambda ',
            r'\\Xi': ' capital xi ',
            r'\\Pi': ' capital pi ',
            r'\\Sigma': ' capital sigma ',
            r'\\Upsilon': ' capital upsilon ',
            r'\\Phi': ' capital phi ',
            r'\\Psi': ' capital psi ',
            r'\\Omega': ' capital omega ',
            
            # Mathematical operators with context
            r'\\cdot': ' times ',
            r'\\times': ' cross product ',
            r'\\div': ' divided by ',
            r'\\pm': ' plus or minus ',
            r'\\mp': ' minus or plus ',
            r'\\leq': ' is less than or equal to ',
            r'\\le\b': ' is less than or equal to ',
            r'\\geq': ' is greater than or equal to ',
            r'\\ge': ' is greater than or equal to ',
            r'\\neq': ' is not equal to ',
            r'\\approx': ' is approximately equal to ',
            r'\\equiv': ' is equivalent to ',
            r'\\sim': ' is similar to ',
            r'\\propto': ' is proportional to ',
            
            # Set theory and logic (use word boundaries to prevent partial matches)
            r'\\in\b': ' is an element of ',
            r'\\notin\b': ' is not an element of ',
            r'\\subset': ' is a subset of ',
            r'\\subseteq': ' is a subset of or equal to ',
            r'\\supset': ' is a superset of ',
            r'\\supseteq': ' is a superset of or equal to ',
            r'\\cup': ' union ',
            r'\\cap': ' intersection ',
            r'\\emptyset': ' the empty set ',
            r'\\varnothing': ' the empty set ',
            r'\\forall': ' for all ',
            r'\\exists': ' there exists ',
            r'\\nexists': ' there does not exist ',
            
            # Functions and special expressions
            r'\\sin': ' sine of ',
            r'\\cos': ' cosine of ',
            r'\\tan': ' tangent of ',
            r'\\sec': ' secant of ',
            r'\\csc': ' cosecant of ',
            r'\\cot': ' cotangent of ',
            r'\\arcsin': ' arcsine of ',
            r'\\arccos': ' arccosine of ',
            r'\\arctan': ' arctangent of ',
            r'\\sinh': ' hyperbolic sine of ',
            r'\\cosh': ' hyperbolic cosine of ',
            r'\\tanh': ' hyperbolic tangent of ',
            r'\\ln': ' natural log of ',
            r'\\log': ' log of ',
            r'\\exp': ' exponential of ',
            
            # Vectors and matrices
            r'\\mathbf\{([^}]+)\}': r'bold \g<1>',
            r'\\vec\{([^}]+)\}': r'vector \g<1>',
            r'\\hat\{([^}]+)\}': r'\g<1> hat',
            r'\\bar\{([^}]+)\}': r'\g<1> bar',
            r'\\tilde\{([^}]+)\}': r'\g<1> tilde',
            r'\\dot\{([^}]+)\}': r'\g<1> dot',
            r'\\ddot\{([^}]+)\}': r'\g<1> double dot',
            
            # Matrix environments
            r'\\begin\{pmatrix\}([^\\]+)\\end\{pmatrix\}': r'the matrix \g<1>',
            r'\\begin\{bmatrix\}([^\\]+)\\end\{bmatrix\}': r'the matrix \g<1>',
            r'\\begin\{vmatrix\}([^\\]+)\\end\{vmatrix\}': r'the determinant of \g<1>',
            r'\\\\': ' and ',  # Matrix row separator
            r'&': ' ',  # Matrix column separator
            
            # Special symbols and constants
            r'\\infty': ' infinity ',
            r'\\ldots': ' dot dot dot ',
            r'\\cdots': ' dot dot dot ',
            r'\\vdots': ' vertical dots ',
            r'\\ddots': ' diagonal dots ',
            r'\\hbar': ' h-bar ',
            r'\\ell': ' script l ',
            
            # Brackets and delimiters
            r'\\langle': ' left angle bracket ',
            r'\\rangle': ' right angle bracket ',
            r'\\lfloor': ' floor of ',
            r'\\rfloor': '',
            r'\\lceil': ' ceiling of ',
            r'\\rceil': '',
            r'\\left\(': ' ',
            r'\\right\)': ' ',
            r'\\left\[': ' open bracket ',
            r'\\right\]': ' close bracket ',
            r'\\left\{': ' open brace ',
            r'\\right\}': ' close brace ',
            
            # Absolute value and norms (order matters - double bars first)
            r'\|\|([^|]+)\|\|': r'the norm of \g<1>',
            r'\|([^|]+)\|': r'the absolute value of \g<1>',
            
            # Arrows and relations
            r'\\rightarrow': ' implies ',
            r'\\leftarrow': ' is implied by ',
            r'\\leftrightarrow': ' if and only if ',
            r'\\Rightarrow': ' implies ',
            r'\\Leftarrow': ' is implied by ',
            r'\\Leftrightarrow': ' if and only if ',
            r'\\uparrow': ' up arrow ',
            r'\\downarrow': ' down arrow ',
            r'\\mapsto': ' maps to ',
        }
        
        # Load pronunciation dictionaries
        self._load_pronunciation_dictionaries()
        
        # Initialize AI providers based on mode
        self._init_ai_providers()
        
        # Math symbol mappings
        self.math_symbols = self.academic_config.get('math', {}).get('symbols', {})
        
        # Citation patterns
        self.citation_patterns = self.academic_config.get('citations', {}).get('patterns', [])
    
    def _load_pronunciation_dictionaries(self):
        """Load pronunciation dictionaries for technical terms"""
        self.pronunciation_dict = {}
        
        # Load from config
        terminology = self.academic_config.get('terminology', {})
        for domain, terms in terminology.items():
            self.pronunciation_dict.update(terms)
        
        # Load from external file if specified
        dict_file = self.enhancement_config.get('technical_terms', {}).get('dictionary_file')
        if dict_file and Path(dict_file).exists():
            with open(dict_file, 'r', encoding='utf-8') as f:
                external_dict = yaml.safe_load(f)
                self.pronunciation_dict.update(external_dict)
    
    def _init_ai_providers(self):
        """Initialize AI providers based on processing mode"""
        self.ai_providers = {}
        
        if self.processing_mode in ['local_ai', 'hybrid']:
            # Initialize Ollama for local AI
            ollama_config = self.config.get('ai_providers', {}).get('ollama', {})
            if ollama_config.get('enabled', False):
                self.ai_providers['ollama'] = {
                    'host': ollama_config.get('host', 'http://localhost:11434'),
                    'model': ollama_config.get('default_model', 'llama2:7b'),
                    'timeout': ollama_config.get('timeout', 30)
                }
        
        if self.processing_mode in ['api', 'hybrid']:
            # Initialize API providers
            openai_config = self.config.get('ai_providers', {}).get('openai', {})
            if openai_config.get('enabled', False):
                self.ai_providers['openai'] = {
                    'api_key': openai_config.get('api_key'),
                    'model': openai_config.get('default_model', 'gpt-3.5-turbo'),
                    'max_tokens': openai_config.get('max_tokens', 2000)
                }
    
    def enhance_document(self, doc_structure: DocumentStructure) -> EnhancedText:
        """
        Enhance document text for speech synthesis
        
        Args:
            doc_structure: Parsed document structure
            
        Returns:
            EnhancedText: Optimized text with speech annotations
        """
        enhanced_content = []
        voice_assignments = {}
        pause_markers = []
        chapter_breaks = []
        chapter_titles = []
        current_position = 0
        
        # Process each chapter
        for chapter in doc_structure.chapters:
            # Store original chapter title
            chapter_titles.append(chapter.title)
            # Add chapter break marker
            chapter_breaks.append(current_position)
            
            # Process chapter title
            chapter_title = self._enhance_chapter_title(chapter.title)
            enhanced_content.append(chapter_title)
            
            # Assign voice for chapter title
            title_start = current_position
            title_end = current_position + len(chapter_title)
            voice_assignments[f"{title_start}:{title_end}"] = "chapter_voice"
            
            # Add pause after chapter title
            current_position = title_end
            pause_markers.append((current_position, 2.0))  # 2 second pause
            
            # Process chapter content
            enhanced_chapter_content = self._enhance_chapter_content(
                chapter.content, doc_structure
            )
            enhanced_content.append(enhanced_chapter_content)
            
            current_position += len(enhanced_chapter_content)
        
        # Combine all content
        full_content = '\n\n'.join(enhanced_content)
        
        # Apply global enhancements
        if self.processing_mode in ['local_ai', 'api', 'hybrid']:
            full_content = self._apply_ai_enhancement(full_content)
        
        return EnhancedText(
            content=full_content,
            voice_assignments=voice_assignments,
            pause_markers=pause_markers,
            pronunciation_guides=self.pronunciation_dict,
            chapter_breaks=chapter_breaks,
            chapter_titles=chapter_titles
        )
    
    def _enhance_chapter_title(self, title: str) -> str:
        """Enhance chapter title for speech"""
        # Add emphasis markers for TTS
        enhanced_title = f"Chapter: {title}"
        
        # Apply pronunciation fixes
        for term, pronunciation in self.pronunciation_dict.items():
            enhanced_title = enhanced_title.replace(term, pronunciation)
        
        return enhanced_title
    
    def _clean_markdown_for_speech(self, content: str) -> str:
        """Clean markdown formatting for natural speech conversion"""
        cleaned = content
        
        # Remove markdown headers (hashtags) - Phase 1 fix
        cleaned = re.sub(r'^#{1,6}\s+', '', cleaned, flags=re.MULTILINE)
        
        # Clean other markdown elements that affect speech
        # Remove horizontal rules
        cleaned = re.sub(r'^---+$', '', cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r'^\*\*\*+$', '', cleaned, flags=re.MULTILINE)
        
        # Remove blockquote markers but keep content
        cleaned = re.sub(r'^>\s*', '', cleaned, flags=re.MULTILINE)
        
        # Remove list markers but keep content
        cleaned = re.sub(r'^\s*[*+-]\s+', '', cleaned, flags=re.MULTILINE)  # Unordered lists
        cleaned = re.sub(r'^\s*\d+\.\s+', '', cleaned, flags=re.MULTILINE)  # Ordered lists
        
        # Remove inline code backticks (keep content)
        cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
        
        # Remove links but keep text
        cleaned = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', cleaned)  # [text](url)
        cleaned = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', cleaned)      # [text][ref]
        
        # Clean up extra whitespace and empty lines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)  # Multiple newlines -> single
        cleaned = ' '.join(cleaned.split())  # Normalize whitespace
        
        return cleaned
    
    def _enhance_chapter_content(self, content: str, doc_structure: DocumentStructure) -> str:
        """Enhance chapter content for speech synthesis"""
        enhanced_content = content
        
        # Process mathematical expressions
        if self.enhancement_config.get('math_processing', {}).get('enabled', True):
            enhanced_content = self._process_math_expressions(
                enhanced_content, doc_structure.math_expressions
            )
        
        # Process citations
        if self.enhancement_config.get('citation_handling', {}).get('enabled', True):
            enhanced_content = self._process_citations(
                enhanced_content, doc_structure.citations
            )
        
        # Apply pronunciation guides
        enhanced_content = self._apply_pronunciation_guides(enhanced_content)
        
        # Optimize sentence structure for speech
        enhanced_content = self._optimize_for_speech(enhanced_content)
        
        return enhanced_content
    
    def _process_math_expressions(self, content: str, math_expressions: List[MathExpression]) -> str:
        """Convert LaTeX math expressions to spoken form using Pandoc"""
        if not self.pandoc_available:
            return self._fallback_math_processing(content, math_expressions)
        
        processed_content = content
        
        # Process each math expression using Pandoc
        for math_expr in math_expressions:
            try:
                spoken_math = self._pandoc_latex_to_speech(math_expr.latex, math_expr.is_block)
                
                if math_expr.is_block:
                    # Block math ($$...$$)
                    replacement = f"[MATH_BLOCK] {spoken_math} [/MATH_BLOCK]"
                    pattern = rf"\$\$\s*{re.escape(math_expr.latex)}\s*\$\$"
                else:
                    # Inline math ($...$)
                    replacement = f"[MATH] {spoken_math} [/MATH]"
                    pattern = rf"\$\s*{re.escape(math_expr.latex)}\s*\$"
                
                # Use raw string replacement to avoid escape issues
                processed_content = re.sub(pattern, lambda m: replacement, processed_content, count=1)
                
            except Exception as e:
                self.logger.warning(f"Failed to process math expression '{math_expr.latex}': {e}")
                # Fall back to basic processing for this expression
                spoken_math = self._fallback_latex_to_speech(math_expr.latex)
                if math_expr.is_block:
                    replacement = f"[MATH_BLOCK] {spoken_math} [/MATH_BLOCK]"
                    pattern = rf"\$\$\s*{re.escape(math_expr.latex)}\s*\$\$"
                else:
                    replacement = f"[MATH] {spoken_math} [/MATH]"
                    pattern = rf"\$\s*{re.escape(math_expr.latex)}\s*\$"
                
                # Use raw string replacement to avoid escape issues
                processed_content = re.sub(pattern, lambda m: replacement, processed_content, count=1)
        
        return processed_content
    
    def _pandoc_latex_to_speech(self, latex: str, is_block: bool = False) -> str:
        """Convert LaTeX to speech using Pandoc AST processing"""
        try:
            # Create temporary markdown with the math expression
            if is_block:
                temp_md = f"$$\n{latex}\n$$"
            else:
                temp_md = f"${latex}$"
            
            # Use Pandoc to parse to JSON AST
            result = subprocess.run(
                ['pandoc', '-f', 'markdown', '-t', 'json'],
                input=temp_md,
                text=True,
                capture_output=True,
                check=True
            )
            
            # Parse the JSON AST
            ast = json.loads(result.stdout)
            
            # Extract and convert math expressions from AST
            spoken_text = self._extract_math_from_ast(ast)
            
            return spoken_text if spoken_text else self._fallback_latex_to_speech(latex)
            
        except Exception as e:
            self.logger.warning(f"Pandoc processing failed for '{latex}': {e}")
            return self._fallback_latex_to_speech(latex)
    
    def _extract_math_from_ast(self, ast: dict) -> str:
        """Extract and convert math expressions from Pandoc AST"""
        def process_element(element):
            if isinstance(element, dict):
                if element.get('t') == 'Math':
                    # Found a math element
                    math_type = element['c'][0]['t']  # InlineMath or DisplayMath
                    latex_content = element['c'][1]   # The actual LaTeX
                    return self._convert_latex_ast_to_speech(latex_content)
                elif 'c' in element:
                    # Process children
                    if isinstance(element['c'], list):
                        results = []
                        for child in element['c']:
                            result = process_element(child)
                            if result:
                                results.append(result)
                        return ' '.join(results) if results else None
                    else:
                        return process_element(element['c'])
            elif isinstance(element, list):
                results = []
                for item in element:
                    result = process_element(item)
                    if result:
                        results.append(result)
                return ' '.join(results) if results else None
            elif isinstance(element, str):
                return element
            return None
        
        return process_element(ast) or ""
    
    def _convert_latex_ast_to_speech(self, latex: str) -> str:
        """Convert LaTeX content to natural speech"""
        # This is where we apply sophisticated LaTeX-to-speech conversion
        spoken = latex
        
        # Apply LaTeX command mappings
        for pattern, replacement in self.latex_to_speech.items():
            spoken = re.sub(pattern, replacement, spoken)
        
        # Handle complex structures
        spoken = self._handle_complex_latex_structures(spoken)
        
        # Clean up
        spoken = re.sub(r'\s+', ' ', spoken).strip()
        
        return spoken
    
    def _handle_complex_latex_structures(self, latex: str) -> str:
        """Handle complex LaTeX structures with math teacher-style narration"""
        # Handle nested fractions with proper phrasing
        latex = re.sub(r'\\frac\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}\{([^{}]+(?:\{[^{}]*\}[^{}]*)*)\}', 
                      lambda m: f"the fraction {m.group(1)} over {m.group(2)}", latex)
        
        # Handle matrix and vector notation
        latex = re.sub(r'\\begin\{pmatrix\}([^\\]+)\\end\{pmatrix\}', 
                      r'the matrix \g<1>', latex)
        latex = re.sub(r'\\begin\{bmatrix\}([^\\]+)\\end\{bmatrix\}', 
                      r'the matrix \g<1>', latex)
        latex = re.sub(r'\\begin\{vmatrix\}([^\\]+)\\end\{vmatrix\}', 
                      r'the determinant of \g<1>', latex)
        
        # Handle equation environments
        latex = re.sub(r'\\begin\{equation\}([^\\]+)\\end\{equation\}', 
                      r'the equation \g<1>', latex)
        latex = re.sub(r'\\begin\{align\}([^\\]+)\\end\{align\}', 
                      r'the aligned equations \g<1>', latex)
        
        # Handle cases and piecewise functions
        latex = re.sub(r'\\begin\{cases\}([^\\]+)\\end\{cases\}', 
                      r'the piecewise function \g<1>', latex)
        
        # Handle binomial coefficients
        latex = re.sub(r'\\binom\{([^}]+)\}\{([^}]+)\}', 
                      r'\g<1> choose \g<2>', latex)
        
        # Handle complex superscripts and subscripts with context
        latex = re.sub(r'([a-zA-Z])\^\{([^}]+)\}_\{([^}]+)\}', 
                      r'\g<1> to the power of \g<2> subscript \g<3>', latex)
        latex = re.sub(r'([a-zA-Z])_\{([^}]+)\}\^\{([^}]+)\}', 
                      r'\g<1> subscript \g<2> to the power of \g<3>', latex)
        
        # Handle simple superscripts and subscripts
        latex = re.sub(r'\^\{([^}]+)\}', r' to the power of \g<1>', latex)
        latex = re.sub(r'_\{([^}]+)\}', r' subscript \g<1>', latex)
        latex = re.sub(r'\^(\w+)', r' to the power of \g<1>', latex)
        latex = re.sub(r'_(\w+)', r' subscript \g<1>', latex)
        
        # Handle special function notation
        latex = re.sub(r'\\operatorname\{([^}]+)\}', r'\g<1>', latex)
        latex = re.sub(r'\\text\{([^}]+)\}', r'\g<1>', latex)
        latex = re.sub(r'\\mathrm\{([^}]+)\}', r'\g<1>', latex)
        
        # Handle absolute values and norms
        latex = re.sub(r'\\left\|([^\\]+)\\right\|', r'the norm of \g<1>', latex)
        latex = re.sub(r'\|([^|]+)\|', r'the absolute value of \g<1>', latex)
        
        # Handle floor and ceiling functions
        latex = re.sub(r'\\lfloor([^\\]+)\\rfloor', r'the floor of \g<1>', latex)
        latex = re.sub(r'\\lceil([^\\]+)\\rceil', r'the ceiling of \g<1>', latex)
        
        # Handle complex numbers
        latex = re.sub(r'\\mathbb\{([^}]+)\}', r'the \g<1> numbers', latex)
        latex = re.sub(r'\\mathcal\{([^}]+)\}', r'script \g<1>', latex)
        
        # Handle spacing and alignment
        latex = re.sub(r'\\\\', ' and ', latex)  # Line breaks in equations
        latex = re.sub(r'\\quad', ' ', latex)  # Spacing
        latex = re.sub(r'\\qquad', ' ', latex)  # More spacing
        latex = re.sub(r'\\,', ' ', latex)  # Small space
        latex = re.sub(r'\\;', ' ', latex)  # Medium space
        latex = re.sub(r'\\:', ' ', latex)  # Medium space
        latex = re.sub(r'\\!', '', latex)  # Negative space
        
        # Clean up multiple spaces and trim
        latex = re.sub(r'\s+', ' ', latex).strip()
        
        # Add natural pauses for complex expressions
        if len(latex.split()) > 10:
            # Add pauses after major mathematical operations
            latex = re.sub(r'(equals?|is|are)\s+', r'\g<1> [PAUSE] ', latex)
            latex = re.sub(r'(therefore|thus|hence)\s+', r'\g<1> [PAUSE] ', latex)
            latex = re.sub(r'(where|such that|given that)\s+', r'\g<1> [PAUSE] ', latex)
        
        return latex
    
    def _fallback_math_processing(self, content: str, math_expressions: List[MathExpression]) -> str:
        """Fallback math processing when Pandoc is not available"""
        processed_content = content
        
        for math_expr in math_expressions:
            spoken_math = self._fallback_latex_to_speech(math_expr.latex)
            
            if math_expr.is_block:
                replacement = f"[MATH_BLOCK] {spoken_math} [/MATH_BLOCK]"
                pattern = rf"\$\$\s*{re.escape(math_expr.latex)}\s*\$\$"
            else:
                replacement = f"[MATH] {spoken_math} [/MATH]"
                pattern = rf"\$\s*{re.escape(math_expr.latex)}\s*\$"
            
            processed_content = re.sub(pattern, replacement, processed_content, count=1)
        
        return processed_content
    
    def _fallback_latex_to_speech(self, latex: str) -> str:
        """Fallback LaTeX to speech conversion without Pandoc"""
        spoken = latex
        
        # Apply basic LaTeX command mappings
        for pattern, replacement in self.latex_to_speech.items():
            spoken = re.sub(pattern, replacement, spoken)
        
        # Handle basic structures
        spoken = self._handle_complex_latex_structures(spoken)
        
        # Clean up
        spoken = re.sub(r'\s+', ' ', spoken).strip()
        
        return spoken
    
    def _process_citations(self, content: str, citations: List[Citation]) -> str:
        """Convert academic citations to natural speech"""
        enhanced_content = content
        
        for citation in citations:
            original = citation.original
            author = citation.author
            year = citation.year
            
            # Convert to natural speech
            if ',' in original:
                # (Author, Year) format
                spoken_citation = f"{author}, {self._year_to_speech(year)}"
            else:
                # [Author Year] or (Author Year) format
                spoken_citation = f"{author} {self._year_to_speech(year)}"
            
            enhanced_content = enhanced_content.replace(original, spoken_citation)
        
        return enhanced_content
    
    def _year_to_speech(self, year: str) -> str:
        """Convert year to natural speech (e.g., 1964 -> nineteen sixty-four)"""
        try:
            year_int = int(year)
            if 1000 <= year_int <= 2099:
                if year_int < 2000:
                    # 19xx format
                    century = year_int // 100
                    remainder = year_int % 100
                    if remainder == 0:
                        return f"{self._number_to_words(century)} hundred"
                    elif remainder < 10:
                        return f"{self._number_to_words(century)} oh {self._number_to_words(remainder)}"
                    else:
                        return f"{self._number_to_words(century)} {self._number_to_words(remainder)}"
                else:
                    # 20xx format
                    return f"twenty {self._number_to_words(year_int % 100)}" if year_int % 100 != 0 else "twenty hundred"
            else:
                return year  # Return as-is for unusual years
        except ValueError:
            return year  # Return as-is if not a valid integer
    
    def _number_to_words(self, num: int) -> str:
        """Convert number to words (simplified for years)"""
        ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        teens = ["ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", 
                "sixteen", "seventeen", "eighteen", "nineteen"]
        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]
        
        if num == 0:
            return "zero"
        elif num < 10:
            return ones[num]
        elif num < 20:
            return teens[num - 10]
        elif num < 100:
            return tens[num // 10] + ("" if num % 10 == 0 else " " + ones[num % 10])
        else:
            return str(num)  # Fallback for larger numbers
    
    def _apply_pronunciation_guides(self, content: str) -> str:
        """Apply pronunciation guides for technical terms"""
        enhanced_content = content
        
        for term, pronunciation in self.pronunciation_dict.items():
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(term) + r'\b'
            enhanced_content = re.sub(pattern, pronunciation, enhanced_content, flags=re.IGNORECASE)
        
        return enhanced_content
    
    def _optimize_for_speech(self, content: str) -> str:
        """Optimize text structure for natural speech"""
        # Clean markdown formatting first
        cleaned_content = self._clean_markdown_for_speech(content)
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', cleaned_content)
        optimized_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Break up very long sentences
            if len(sentence) > 200:
                # Split on conjunctions and add pauses
                sentence = re.sub(r'\b(and|but|however|therefore|moreover|furthermore)\b', 
                                r'[PAUSE] \g<1>', sentence)
            
            # Add emphasis markers for important terms (already converted from markdown)
            sentence = re.sub(r'\*\*([^*]+)\*\*', r'[EMPHASIS] \g<1> [/EMPHASIS]', sentence)
            sentence = re.sub(r'\*([^*]+)\*', r'[SLIGHT_EMPHASIS] \g<1> [/SLIGHT_EMPHASIS]', sentence)
            
            optimized_sentences.append(sentence)
        
        return '. '.join(optimized_sentences)
    
    def _apply_ai_enhancement(self, content: str) -> str:
        """Apply AI-powered text enhancement"""
        if 'ollama' in self.ai_providers:
            return self._enhance_with_ollama(content)
        elif 'openai' in self.ai_providers:
            return self._enhance_with_openai(content)
        else:
            return content
    
    def _enhance_with_ollama(self, content: str) -> str:
        """Enhance text using local Ollama AI"""
        try:
            ollama_config = self.ai_providers['ollama']
            
            prompt = f"""
            Please optimize the following academic text for text-to-speech conversion.
            Make it more natural for spoken delivery while preserving all technical accuracy.
            Add natural pauses and improve flow for audio consumption.
            
            Text to optimize:
            {content}
            
            Optimized text:
            """
            
            response = requests.post(
                f"{ollama_config['host']}/api/generate",
                json={
                    "model": ollama_config['model'],
                    "prompt": prompt,
                    "stream": False
                },
                timeout=ollama_config['timeout']
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', content)
            else:
                return content
                
        except Exception as e:
            print(f"Ollama enhancement failed: {e}")
            return content
    
    def _enhance_with_openai(self, content: str) -> str:
        """Enhance text using OpenAI API"""
        try:
            openai_config = self.ai_providers['openai']
            
            headers = {
                'Authorization': f"Bearer {openai_config['api_key']}",
                'Content-Type': 'application/json'
            }
            
            data = {
                "model": openai_config['model'],
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at optimizing academic text for text-to-speech conversion. Make text more natural for spoken delivery while preserving technical accuracy."
                    },
                    {
                        "role": "user", 
                        "content": f"Optimize this text for audiobook narration:\n\n{content}"
                    }
                ],
                "max_tokens": openai_config['max_tokens']
            }
            
            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                return content
                
        except Exception as e:
            print(f"OpenAI enhancement failed: {e}")
            return content
    
    def validate_enhancement(self, enhanced_text: EnhancedText) -> Tuple[bool, List[str]]:
        """
        Validate enhanced text quality
        
        Args:
            enhanced_text: Enhanced text structure
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        content = enhanced_text.content
        
        # Check for unprocessed LaTeX
        if re.search(r'\$[^$]+\$', content):
            issues.append("Unprocessed inline math expressions found")
        
        if re.search(r'\$\$[^$]+\$\$', content):
            issues.append("Unprocessed block math expressions found")
        
        # Check for very long sentences (> 300 chars)
        sentences = re.split(r'[.!?]+', content)
        long_sentences = [s for s in sentences if len(s.strip()) > 300]
        if long_sentences:
            issues.append(f"Found {len(long_sentences)} very long sentences that may be hard to narrate")
        
        # Check for balanced emphasis markers
        emphasis_starts = content.count('[EMPHASIS]')
        emphasis_ends = content.count('[/EMPHASIS]')
        if emphasis_starts != emphasis_ends:
            issues.append("Unbalanced emphasis markers")
        
        # Check for reasonable content length
        if len(content.strip()) < 50:
            issues.append("Enhanced content is very short")
        
        return len(issues) == 0, issues
