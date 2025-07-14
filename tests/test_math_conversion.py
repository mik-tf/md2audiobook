#!/usr/bin/env python3
"""
Test script to examine LaTeX-to-speech conversion
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import with absolute path to avoid relative import issues
from src.markdown_processor import MarkdownProcessor
from src.text_enhancer import TextEnhancer

def test_math_conversion():
    """Test LaTeX-to-speech conversion on sample expressions"""
    
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
    
    # Test expressions from Bell's theorem document
    test_expressions = [
        "P(A)",
        "P(A \\cup B) = P(A) + P(B)",
        "P(A \\cap B) = P(A)P(B)",
        "E[X] = \\sum_{i=1}^{n} x_i P(x_i)",
        "\\langle A, B \\rangle = \\frac{1}{4}[P(A^+, B^+) + P(A^-, B^-) - P(A^+, B^-) - P(A^-, B^+)]",
        "\\rho_{AB}",
        "|\\psi\\rangle = \\frac{1}{\\sqrt{2}}(|\\uparrow\\downarrow\\rangle - |\\downarrow\\uparrow\\rangle)",
        "S = E(a, b) - E(a, b') + E(a', b) + E(a', b') \\leq 2",
        "\\frac{d}{dx} f(x)",
        "\\int_0^{\\infty} e^{-x} dx",
        "\\sum_{n=1}^{\\infty} \\frac{1}{n^2}",
        "\\sqrt{x^2 + y^2}",
        "\\alpha + \\beta = \\gamma"
    ]
    
    print("=" * 80)
    print("TESTING ENHANCED LATEX-TO-SPEECH CONVERSION")
    print("=" * 80)
    
    for i, latex_expr in enumerate(test_expressions, 1):
        print(f"\n{i}. Original LaTeX:")
        print(f"   {latex_expr}")
        
        # Test both Pandoc and fallback methods
        try:
            # Test Pandoc conversion if available
            if text_enhancer.pandoc_available:
                pandoc_result = text_enhancer._pandoc_latex_to_speech(latex_expr, is_block=False)
                print(f"   Pandoc Speech: {pandoc_result}")
        except Exception as e:
            print(f"   Pandoc Speech: [Error: {e}]")
        
        try:
            # Test fallback conversion
            fallback_result = text_enhancer._fallback_latex_to_speech(latex_expr)
            print(f"   Fallback Speech: {fallback_result}")
        except Exception as e:
            print(f"   Fallback Speech: [Error: {e}]")
    
    # Test complex structures
    print("\n" + "=" * 80)
    print("TESTING COMPLEX LATEX STRUCTURES")
    print("=" * 80)
    
    complex_expressions = [
        "\\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u",
        "\\int_{-\\infty}^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}",
        "\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n = e",
        "\\mathbf{F} = m\\mathbf{a}",
        "\\hat{H}|\\psi\\rangle = E|\\psi\\rangle",
        "\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}"
    ]
    
    for i, latex_expr in enumerate(complex_expressions, 1):
        print(f"\n{i}. Complex LaTeX:")
        print(f"   {latex_expr}")
        
        try:
            result = text_enhancer._handle_complex_latex_structures(latex_expr)
            print(f"   Enhanced Speech: {result}")
        except Exception as e:
            print(f"   Enhanced Speech: [Error: {e}]")

if __name__ == "__main__":
    test_math_conversion()
