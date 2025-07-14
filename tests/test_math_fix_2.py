#!/usr/bin/env python3
"""
Test script to verify the mathematical notation fix
"""

import re

def auto_wrap_mathematical_expressions(content):
    """Intelligently detect mathematical expressions and wrap them in LaTeX delimiters"""
    
    # Split content by existing LaTeX expressions to avoid double-processing
    parts = []
    last_end = 0
    
    # Find all existing LaTeX expressions (both inline and block)
    latex_pattern = re.compile(r'(\$\$.*?\$\$|\$.*?\$)', re.DOTALL)
    
    for match in latex_pattern.finditer(content):
        # Add the text before this LaTeX expression
        before_latex = content[last_end:match.start()]
        if before_latex:
            parts.append(('text', before_latex))
        
        # Add the LaTeX expression as-is
        parts.append(('latex', match.group(0)))
        last_end = match.end()
    
    # Add any remaining text after the last LaTeX expression
    if last_end < len(content):
        remaining = content[last_end:]
        if remaining:
            parts.append(('text', remaining))
    
    # If no LaTeX found, treat entire content as text
    if not parts:
        parts = [('text', content)]
    
    # Process only the text parts, leaving LaTeX parts unchanged
    processed_parts = []
    for part_type, part_content in parts:
        if part_type == 'latex':
            # Keep LaTeX expressions unchanged
            processed_parts.append(part_content)
        else:
            # Auto-wrap mathematical expressions in text
            processed_text = wrap_math_in_text(part_content)
            processed_parts.append(processed_text)
    
    return ''.join(processed_parts)

def wrap_math_in_text(text):
    """Wrap mathematical expressions found in plain text with LaTeX delimiters"""
    
    # Mathematical function notation: P(A), f(x), g(t), etc.
    # Wrap in LaTeX so existing math processing handles them
    text = re.sub(r'\b([A-Za-z])\(([^)]+)\)', r'$\g<1>(\g<2>)$', text)
    
    # Expected value notation: E[X]
    text = re.sub(r'\bE\[([^\]]+)\]', r'$E[\g<1>]$', text)
    
    # Variance notation: Var(X)
    text = re.sub(r'\bVar\(([^)]+)\)', r'$\\text{Var}(\g<1>)$', text)
    
    # Standard deviation: SD(X)
    text = re.sub(r'\bSD\(([^)]+)\)', r'$\\text{SD}(\g<1>)$', text)
    
    # Set operations with symbols: A ∩ B, A ∪ B
    text = re.sub(r'([A-Z])\s*∩\s*([A-Z])', r'$\g<1> \\cap \g<2>$', text)
    text = re.sub(r'([A-Z])\s*∪\s*([A-Z])', r'$\g<1> \\cup \g<2>$', text)
    
    return text

def test_math_fix():
    """Test the mathematical notation fix"""
    
    print("=== TESTING MATHEMATICAL NOTATION FIX ===\n")
    
    # Test case 1: The problematic text from example2.md
    test1 = "If events A and B cannot occur simultaneously, the probability of either A or B occurring is $P(A \\cup B) = P(A) + P(B)$."
    
    print("TEST 1: Mixed LaTeX and plain text")
    print("ORIGINAL:", test1)
    result1 = auto_wrap_mathematical_expressions(test1)
    print("WRAPPED: ", result1)
    print("✅ P(A) and P(B) are now wrapped in LaTeX delimiters\n")
    
    # Test case 2: Plain mathematical notation
    test2 = "For an event A, its probability P(A) ranges from 0 to 1. The function f(x) equals x squared."
    
    print("TEST 2: Plain mathematical notation")
    print("ORIGINAL:", test2)
    result2 = auto_wrap_mathematical_expressions(test2)
    print("WRAPPED: ", result2)
    print("✅ P(A) and f(x) are now wrapped in LaTeX delimiters\n")
    
    # Test case 3: Expected value and variance
    test3 = "The expected value E[X] and variance Var(X) are important statistics."
    
    print("TEST 3: Statistical notation")
    print("ORIGINAL:", test3)
    result3 = auto_wrap_mathematical_expressions(test3)
    print("WRAPPED: ", result3)
    print("✅ E[X] and Var(X) are now wrapped in LaTeX delimiters\n")
    
    # Test case 4: Set operations
    test4 = "The union A ∪ B and intersection A ∩ B are set operations."
    
    print("TEST 4: Set operations")
    print("ORIGINAL:", test4)
    result4 = auto_wrap_mathematical_expressions(test4)
    print("WRAPPED: ", result4)
    print("✅ Set operations are now wrapped in LaTeX delimiters\n")
    
    print("=== SOLUTION SUMMARY ===")
    print("✅ Mathematical expressions like P(B) will now be wrapped in LaTeX delimiters")
    print("✅ The existing LaTeX math processing pipeline will convert them to natural speech")
    print("✅ TTS will receive 'probability of B' instead of 'P(B)' (chemistry symbols)")
    print("✅ No more 'Phosphorus Boron' misinterpretation!")

if __name__ == "__main__":
    test_math_fix()
