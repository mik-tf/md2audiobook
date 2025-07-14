#!/usr/bin/env python3
"""
Debug Euler's limit expression conversion
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.text_enhancer import TextEnhancer

def debug_euler():
    """Debug Euler's limit expression"""
    
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
    
    # Test expressions step by step
    test_expressions = [
        "\\left(1\\right)",
        "\\left(1 + \\frac{1}{n}\\right)",
        "\\left(1 + \\frac{1}{n}\\right)^n",
        "\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n = e"
    ]
    
    print("=" * 60)
    print("DEBUGGING EULER'S LIMIT EXPRESSION")
    print("=" * 60)
    
    for i, latex_expr in enumerate(test_expressions, 1):
        print(f"\n{i}. Testing: {latex_expr}")
        
        # Test fallback conversion
        try:
            fallback_result = text_enhancer._fallback_latex_to_speech(latex_expr)
            print(f"   Fallback result: {fallback_result}")
        except Exception as e:
            print(f"   Fallback error: {e}")
        
        # Check specific problematic patterns
        left_paren_pattern = r'\\left\\\('
        right_paren_pattern = r'\\right\\\)'
        
        left_match = re.search(left_paren_pattern, latex_expr)
        right_match = re.search(right_paren_pattern, latex_expr)
        
        if left_match:
            print(f"   Found \\left\\( at position {left_match.start()}")
        if right_match:
            print(f"   Found \\right\\) at position {right_match.start()}")

if __name__ == "__main__":
    debug_euler()
