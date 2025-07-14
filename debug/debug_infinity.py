#!/usr/bin/env python3
"""
Debug infinity symbol conversion
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with absolute path to avoid relative import issues
from src.text_enhancer import TextEnhancer

def debug_infinity():
    """Debug infinity symbol conversion"""
    
    # Initialize text enhancer directly
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
    
    text_enhancer = TextEnhancer(config)
    
    # Test infinity expressions
    test_expressions = [
        "\\infty",
        "\\int_0^{\\infty}",
        "\\sum_{n=1}^{\\infty}",
        "\\lim_{x \\to \\infty}",
        "x \\to \\infty"
    ]
    
    print("=" * 60)
    print("DEBUGGING INFINITY SYMBOL CONVERSION")
    print("=" * 60)
    
    for i, latex_expr in enumerate(test_expressions, 1):
        print(f"\n{i}. Testing: {latex_expr}")
        
        # Test step by step
        print(f"   Original: {latex_expr}")
        
        # Check if the pattern exists in the dictionary
        infty_pattern = r'\\infty'
        if infty_pattern in text_enhancer.latex_to_speech:
            replacement = text_enhancer.latex_to_speech[infty_pattern]
            print(f"   Pattern found: '{infty_pattern}' -> '{replacement}'")
        else:
            print(f"   Pattern NOT found in dictionary!")
        
        # Test direct regex replacement
        import re
        direct_result = re.sub(r'\\infty', ' infinity ', latex_expr)
        print(f"   Direct regex: {direct_result}")
        
        # Test fallback conversion
        try:
            fallback_result = text_enhancer._fallback_latex_to_speech(latex_expr)
            print(f"   Fallback result: {fallback_result}")
        except Exception as e:
            print(f"   Fallback error: {e}")
        
        # Test Pandoc conversion if available
        try:
            if text_enhancer.pandoc_available:
                pandoc_result = text_enhancer._pandoc_latex_to_speech(latex_expr, is_block=False)
                print(f"   Pandoc result: {pandoc_result}")
        except Exception as e:
            print(f"   Pandoc error: {e}")

if __name__ == "__main__":
    debug_infinity()
