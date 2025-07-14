#!/usr/bin/env python3
"""
Debug Bell's theorem math processing
"""

import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.markdown_processor import MarkdownProcessor
from src.text_enhancer import TextEnhancer

def debug_bells_math():
    """Debug Bell's theorem math processing"""
    
    config = {
        'processing_mode': 'basic',
        'text_enhancement': {
            'enabled': True,
            'academic_content': True
        },
        'academic': {
            'math': {'enabled': True},
            'citations': {'enabled': True}
        }
    }
    
    # Process the Bell's theorem document
    input_file = "../bells_theorem/bells_theorem.md"
    
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return
    
    print("=" * 60)
    print("DEBUGGING BELL'S THEOREM MATH PROCESSING")
    print("=" * 60)
    
    # Stage 1: Process markdown
    print("\nStage 1: Processing Markdown...")
    processor = MarkdownProcessor(config)
    document = processor.process_document(Path(input_file))
    
    print(f"Document title: {document.title}")
    print(f"Number of chapters: {len(document.chapters)}")
    print(f"Total math expressions found: {len(document.math_expressions)}")
    
    # Stage 2: Enhance text
    print("\nStage 2: Enhancing text...")
    enhancer = TextEnhancer(config)
    enhanced_document = enhancer.enhance_document(document)
    
    # Check first few math expressions
    print("\nFirst 5 math expressions and their conversions:")
    for i, expr in enumerate(enhanced_document.math_expressions[:5]):
        print(f"  Math {i + 1}: {expr.latex}")
        print(f"    Block: {expr.is_block}")
    
    # Check a sample of the enhanced content
    print(f"\nSample of enhanced content from Chapter 1:")
    if enhanced_document.chapters:
        sample_content = enhanced_document.chapters[0].content[:500]
        print(f"'{sample_content}...'")
        
        # Check if "math" appears in the content
        if "math" in sample_content.lower():
            print("\n⚠️  WARNING: Found literal 'math' in enhanced content!")
            # Find the context around "math"
            import re
            math_matches = list(re.finditer(r'\bmath\b', sample_content, re.IGNORECASE))
            for match in math_matches:
                start = max(0, match.start() - 50)
                end = min(len(sample_content), match.end() + 50)
                context = sample_content[start:end]
                print(f"Context: '...{context}...'")

if __name__ == "__main__":
    debug_bells_math()
