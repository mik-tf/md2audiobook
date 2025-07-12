#!/usr/bin/env python3
"""
Debug quantum notation conversion
"""

import sys
import os
import re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.text_enhancer import TextEnhancer

def debug_quantum():
    """Debug quantum notation conversion"""
    
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
    
    # Test quantum expressions
    test_expressions = [
        "|\\psi\\rangle",
        "\\langle\\phi|",
        "\\langle\\phi|\\psi\\rangle",
        "\\hat{H}|\\psi\\rangle = E|\\psi\\rangle",
        "|0\\rangle + |1\\rangle",
        "\\langle\\psi|\\hat{A}|\\phi\\rangle"
    ]
    
    print("=" * 60)
    print("DEBUGGING QUANTUM NOTATION CONVERSION")
    print("=" * 60)
    
    for i, latex_expr in enumerate(test_expressions, 1):
        print(f"\n{i}. Testing: {latex_expr}")
        
        # Check specific patterns
        ket_pattern = r'\|([^\rangle|]+)\\rangle'
        bra_pattern = r'\\langle([^|]+)\|'
        
        print(f"   Original: {latex_expr}")
        
        # Test ket pattern
        ket_match = re.search(ket_pattern, latex_expr)
        if ket_match:
            print(f"   Ket match: '{ket_match.group(0)}' -> group(1): '{ket_match.group(1)}'")
        else:
            print(f"   No ket match for pattern: {ket_pattern}")
        
        # Test bra pattern
        bra_match = re.search(bra_pattern, latex_expr)
        if bra_match:
            print(f"   Bra match: '{bra_match.group(0)}' -> group(1): '{bra_match.group(1)}'")
        else:
            print(f"   No bra match for pattern: {bra_pattern}")
        
        # Test direct regex replacement
        direct_ket = re.sub(ket_pattern, r'ket \g<1>', latex_expr)
        direct_bra = re.sub(bra_pattern, r'bra \g<1>', direct_ket)
        print(f"   Direct regex: {direct_bra}")
        
        # Test fallback conversion
        try:
            fallback_result = text_enhancer._fallback_latex_to_speech(latex_expr)
            print(f"   Fallback result: {fallback_result}")
        except Exception as e:
            print(f"   Fallback error: {e}")

if __name__ == "__main__":
    debug_quantum()
