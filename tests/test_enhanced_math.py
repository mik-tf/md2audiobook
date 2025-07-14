#!/usr/bin/env python3
"""
Test enhanced LaTeX-to-speech conversion with new improvements
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.text_enhancer import TextEnhancer

def test_enhanced_math():
    """Test enhanced math conversion with new patterns"""
    
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
    
    # Test expressions with new improvements
    test_expressions = [
        # Partial derivatives
        ("\\frac{\\partial^2 u}{\\partial t^2}", "Partial derivative test"),
        ("\\partial u", "Simple partial"),
        ("\\nabla^2 u", "Laplacian operator"),
        
        # Bold math
        ("\\mathbf{F} = m\\mathbf{a}", "Bold vectors"),
        ("\\mathbf{A}\\mathbf{x} = \\mathbf{b}", "Matrix equation"),
        
        # Matrix environments
        ("\\begin{pmatrix} a & b \\\\ c & d \\end{pmatrix}", "2x2 matrix"),
        ("\\begin{vmatrix} x & y \\\\ z & w \\end{vmatrix}", "Determinant"),
        
        # Absolute values and norms
        ("|x|", "Absolute value"),
        ("||\\mathbf{v}||", "Vector norm"),
        
        # Complex expressions
        ("\\hat{H}|\\psi\\rangle = E|\\psi\\rangle", "Quantum operator"),
        ("\\int_0^{\\infty} e^{-x^2} dx = \\sqrt{\\pi}", "Gaussian integral"),
        ("\\lim_{n \\to \\infty} \\left(1 + \\frac{1}{n}\\right)^n = e", "Euler's limit"),
    ]
    
    print("=" * 80)
    print("TESTING ENHANCED LATEX-TO-SPEECH CONVERSION")
    print("=" * 80)
    
    for i, (latex_expr, description) in enumerate(test_expressions, 1):
        print(f"\n{i}. {description}")
        print(f"   LaTeX: {latex_expr}")
        
        # Test fallback conversion
        try:
            fallback_result = text_enhancer._fallback_latex_to_speech(latex_expr)
            print(f"   Fallback: {fallback_result}")
        except Exception as e:
            print(f"   Fallback error: {e}")
        
        # Test Pandoc conversion if available
        try:
            if text_enhancer.pandoc_available:
                pandoc_result = text_enhancer._pandoc_latex_to_speech(latex_expr, is_block=False)
                print(f"   Pandoc: {pandoc_result}")
        except Exception as e:
            print(f"   Pandoc error: {e}")

if __name__ == "__main__":
    test_enhanced_math()
