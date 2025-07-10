#!/usr/bin/env python3
"""
Main processing script for md2audiobook pipeline
Orchestrates the complete Markdown to Audiobook conversion process
"""

import sys
import argparse
from pathlib import Path
import yaml
import json
from typing import Dict, Any, Optional
import os
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from markdown_processor import MarkdownProcessor
from text_enhancer import TextEnhancer
from audiobook_generator import AudiobookGenerator


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if config_path is None:
        # Try default locations
        possible_configs = [
            Path('config/default.yaml'),
            Path('config.yaml'),
            Path(__file__).parent.parent / 'config' / 'default.yaml'
        ]
        
        for config_file in possible_configs:
            if config_file.exists():
                config_path = config_file
                break
        else:
            raise FileNotFoundError("No configuration file found. Please specify --config or create config/default.yaml")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def setup_environment():
    """Setup environment variables"""
    # Load .env file if it exists
    env_file = Path('.env')
    if env_file.exists():
        load_dotenv(env_file)
    
    # Load from parent directory if not found
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(parent_env)


def validate_input_file(input_path: Path) -> bool:
    """Validate input markdown file"""
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return False
    
    if input_path.suffix.lower() not in ['.md', '.markdown']:
        print(f"Warning: Input file doesn't have .md extension: {input_path}")
    
    return True


def determine_processing_mode(config: Dict[str, Any], args: argparse.Namespace) -> str:
    """Determine processing mode based on config and arguments"""
    if args.mode:
        return args.mode
    
    return config.get('processing', {}).get('default_mode', 'basic')


def create_output_path(input_path: Path, output_dir: Optional[Path], 
                      config: Dict[str, Any]) -> Path:
    """Create output path for audiobook"""
    if output_dir is None:
        output_dir = Path(config.get('output', {}).get('directory', 'output'))
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename based on input
    base_name = input_path.stem
    audio_format = config.get('audio', {}).get('output_format', 'm4b')
    
    return output_dir / f"{base_name}.{audio_format}"


def extract_metadata(doc_structure, config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract audiobook metadata from document and config"""
    metadata = doc_structure.metadata.copy()
    
    # Add default metadata if missing
    defaults = {
        'title': doc_structure.title or 'Untitled Document',
        'author': 'Unknown Author',
        'genre': 'Educational',
        'date': '',
        'description': f'Audiobook generated from markdown document'
    }
    
    for key, default_value in defaults.items():
        if key not in metadata:
            metadata[key] = default_value
    
    # Override with config values if specified
    config_metadata = config.get('output', {}).get('metadata', {})
    metadata.update(config_metadata)
    
    return metadata


def print_processing_info(input_path: Path, output_path: Path, mode: str, config: Dict[str, Any]):
    """Print processing information"""
    print("=" * 60)
    print("MD2AUDIOBOOK - Markdown to Audiobook Pipeline")
    print("=" * 60)
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {mode}")
    print(f"Format: {config.get('audio', {}).get('output_format', 'm4b').upper()}")
    print("-" * 60)


def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description='Convert Markdown documents to professional audiobooks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Processing Modes:
  basic     - Simple markdown parsing + local TTS (fastest, $0 cost)
  local_ai  - AI text enhancement + local TTS (better quality, $0 cost)
  api       - AI enhancement + premium TTS APIs (highest quality, API costs)
  hybrid    - Intelligent fallback chain (best reliability)

Examples:
  %(prog)s document.md
  %(prog)s document.md --mode api --output audiobooks/
  %(prog)s document.md --config custom_config.yaml
        """
    )
    
    parser.add_argument('input', type=Path, help='Input markdown file')
    parser.add_argument('--output', '-o', type=Path, help='Output directory (default: output/)')
    parser.add_argument('--config', '-c', type=Path, help='Configuration file (default: config/default.yaml)')
    parser.add_argument('--mode', '-m', choices=['basic', 'local_ai', 'api', 'hybrid'], 
                       help='Processing mode (overrides config)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be processed without generating audio')
    
    args = parser.parse_args()
    
    try:
        # Setup environment
        setup_environment()
        
        # Validate input
        if not validate_input_file(args.input):
            sys.exit(1)
        
        # Load configuration
        config = load_config(args.config)
        
        # Determine processing mode
        processing_mode = determine_processing_mode(config, args)
        
        # Create output path
        output_path = create_output_path(args.input, args.output, config)
        
        # Print processing info
        print_processing_info(args.input, output_path, processing_mode, config)
        
        if args.dry_run:
            print("DRY RUN - No audio will be generated")
            print("-" * 60)
        
        # Stage 1: Markdown Processing
        print("Stage 1: Processing Markdown...")
        markdown_processor = MarkdownProcessor(config)
        doc_structure = markdown_processor.process_file(args.input)
        
        if args.verbose:
            print(f"  - Found {len(doc_structure.chapters)} chapters")
            print(f"  - Found {len(doc_structure.math_expressions)} math expressions")
            print(f"  - Found {len(doc_structure.citations)} citations")
        
        # Stage 2: Text Enhancement
        print("Stage 2: Enhancing text for speech...")
        text_enhancer = TextEnhancer(config, processing_mode)
        enhanced_text = text_enhancer.enhance_document(doc_structure)
        
        # Validate enhancement
        is_valid, issues = text_enhancer.validate_enhancement(enhanced_text)
        if not is_valid:
            print("Warning: Text enhancement issues detected:")
            for issue in issues:
                print(f"  - {issue}")
        
        if args.verbose:
            print(f"  - Enhanced text length: {len(enhanced_text.content)} characters")
            print(f"  - Voice assignments: {len(enhanced_text.voice_assignments)}")
            print(f"  - Chapter breaks: {len(enhanced_text.chapter_breaks)}")
        
        if args.dry_run:
            print("\nDRY RUN COMPLETE")
            print(f"Would generate audiobook: {output_path}")
            sys.exit(0)
        
        # Stage 3: Audio Generation
        print("Stage 3: Generating audiobook...")
        audiobook_generator = AudiobookGenerator(config, processing_mode)
        
        try:
            # Extract metadata
            metadata = extract_metadata(doc_structure, config)
            
            # Generate audiobook
            audiobook = audiobook_generator.generate_audiobook(
                enhanced_text, metadata, output_path
            )
            
            if args.verbose:
                print(f"  - Generated {len(audiobook.chapters)} audio chapters")
                print(f"  - Total duration: {len(audiobook.audio) / 1000:.1f} seconds")
                print(f"  - Output format: {config.get('audio', {}).get('output_format', 'm4b').upper()}")
            
            print("-" * 60)
            print("SUCCESS: Audiobook generation complete!")
            print(f"Output: {audiobook.file_path}")
            
            # Print chapter information
            if audiobook.chapters:
                print(f"\nChapters ({len(audiobook.chapters)}):")
                for chapter in audiobook.chapters:
                    duration_str = f"{chapter.duration:.1f}s"
                    print(f"  {chapter.chapter_number:2d}. {chapter.title} ({duration_str})")
        
        finally:
            # Cleanup temporary files
            audiobook_generator.cleanup()
    
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
