#!/usr/bin/env python3
"""
Test pattern order in LaTeX-to-speech dictionary
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.text_enhancer import TextEnhancer

def test_pattern_order():
    """Test the order of pattern processing"""
    
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
    
    print("=" * 60)
    print("TESTING PATTERN ORDER IN LATEX-TO-SPEECH DICTIONARY")
    print("=" * 60)
    
    # Print all patterns in order
    print("\nPatterns in processing order:")
    for i, (pattern, replacement) in enumerate(text_enhancer.latex_to_speech.items(), 1):
        if 'rangle' in pattern or 'langle' in pattern or '|' in pattern:
            print(f"{i:3d}. {pattern:<50} -> {replacement}")
    
    # Test specific quantum expression
    test_expr = "|\\psi\\rangle"
    print(f"\nTesting expression: {test_expr}")
    
    # Apply patterns one by one to see which one matches first
    result = test_expr
    for i, (pattern, replacement) in enumerate(text_enhancer.latex_to_speech.items(), 1):
        old_result = result
        try:
            result = re.sub(pattern, replacement, result)
            if result != old_result:
                print(f"Pattern {i:3d} matched: {pattern}")
                print(f"  Before: {old_result}")
                print(f"  After:  {result}")
                break
        except Exception as e:
            print(f"Pattern {i:3d} error: {pattern} -> {e}")

if __name__ == "__main__":
    test_pattern_order()
