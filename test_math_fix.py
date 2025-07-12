#!/usr/bin/env python3
"""
Test the math marker fix
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.audiobook_generator import AudiobookGenerator

def test_math_marker_fix():
    """Test that math markers are properly removed"""
    
    config = {
        'processing_mode': 'api',
        'output_format': 'M4B'
    }
    
    generator = AudiobookGenerator(config)
    
    # Test text with math markers
    test_text = """
    The probability of [MATH] P(A|B) = 0.5 [/MATH] is important.
    
    We also have the equation:
    [MATH_BLOCK] E = mc^2 [/MATH_BLOCK]
    
    And inline math like [MATH] x = 2 [/MATH] should work.
    """
    
    print("=" * 60)
    print("TESTING MATH MARKER REMOVAL")
    print("=" * 60)
    
    print("Original text:")
    print(repr(test_text))
    
    cleaned_text = generator._clean_text_for_tts(test_text)
    
    print("\nCleaned text:")
    print(repr(cleaned_text))
    
    print("\nReadable cleaned text:")
    print(cleaned_text)
    
    # Check if markers are removed
    if '[MATH]' in cleaned_text or '[/MATH]' in cleaned_text:
        print("\n❌ ERROR: Math markers still present!")
    elif '[MATH_BLOCK]' in cleaned_text or '[/MATH_BLOCK]' in cleaned_text:
        print("\n❌ ERROR: Math block markers still present!")
    else:
        print("\n✅ SUCCESS: All math markers removed!")
        
    # Check if the actual math content is preserved
    if 'P(A|B) = 0.5' in cleaned_text and 'E = mc^2' in cleaned_text and 'x = 2' in cleaned_text:
        print("✅ SUCCESS: Math content preserved!")
    else:
        print("❌ ERROR: Math content missing!")

if __name__ == "__main__":
    test_math_marker_fix()
