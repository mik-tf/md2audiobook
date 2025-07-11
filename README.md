# md2audiobook

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://opensource.org/licenses/Apache-2.0)
[![ucli-tools](https://img.shields.io/badge/ucli--tools-ecosystem-green.svg)](https://github.com/ucli-tools)

Professional Markdown-to-Audiobook Pipeline for Academic and Technical Content

## 🎯 Overview

`md2audiobook` transforms markdown documents into professional-quality audiobooks, designed specifically for academic and technical content with mathematical expressions, citations, and complex formatting. To be part of the [ucli-tools](https://github.com/ucli-tools) ecosystem, complementing [mdtexpdf](https://github.com/ucli-tools/mdtexpdf).

### **Unified Workflow**
```
📝 Markdown Source Document
    ├── 🎵 md2audiobook → 📚 Professional Audiobook (.m4b)
    └── 📄 mdtexpdf → 📄 LaTeX PDF Document (.pdf)
```

## ✨ Features

### **Core Capabilities**
- **🎵 Professional Audio Generation** - High-quality TTS with multi-voice support
- **🧮 Mathematical Expression Handling** - LaTeX math → natural speech conversion
- **📚 Academic Content Optimization** - Citations, footnotes, technical terms
- **🔄 Multi-Mode Processing** - Local, API, and hybrid processing options
- **📖 Chapter Structure Preservation** - Headers → audiobook navigation

### **Processing Modes**
- **📝 Basic Mode** - Local processing, $0 cost
- **🤖 Local AI Mode** - AI enhancement + local TTS, $0 cost
- **🌐 API Mode** - Premium TTS + AI enhancement, highest quality
- **🔄 Hybrid Mode** - Intelligent fallback for reliability

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **FFmpeg** (for audio processing)
- **Git** (for installation)

### Installation
```bash
# Clone the repository
git clone https://github.com/mik-tf/md2audiobook.git
cd md2audiobook

# Setup environment
make setup

# Test with example
make demo
```

### Basic Usage
```bash
# Process a markdown file
make audiobook SOURCE=document.md

# Different processing modes
make audiobook-basic SOURCE=document.md      # Local processing
make audiobook-local-ai SOURCE=document.md   # AI enhancement
make audiobook-api SOURCE=document.md        # Premium quality
make audiobook-hybrid SOURCE=document.md     # Best reliability

# Batch processing
make process-all
```

## 📁 Project Structure

```
md2audiobook/
├── README.md                    # This file
├── LICENSE                     # Apache 2.0 license
├── Makefile                    # Build and processing commands
├── requirements.txt            # Python dependencies
├── setup.py                    # Package installation
├── .env.example               # Environment variables template
├── .gitignore                 # Git ignore patterns
│
├── config/
│   └── default.yaml.example   # Configuration template
│
├── src/
│   ├── __init__.py
│   ├── markdown_processor.py  # Core markdown parsing
│   ├── text_enhancer.py      # Academic text optimization
│   ├── audiobook_generator.py # TTS and audio processing
│   ├── utils.py              # Common utilities
│   └── providers/            # TTS and AI providers
│       ├── __init__.py
│       ├── local_tts.py      # Local TTS providers
│       ├── api_tts.py        # API TTS providers
│       └── ai_providers.py   # AI enhancement providers
│
├── templates/
│   └── academic_audiobook.yaml # Academic content settings
│
├── scripts/
│   ├── process_audiobook.py   # Main processing script
│   ├── setup_environment.sh  # Environment setup
│   └── validate_output.py    # Quality assurance
│
├── documents/                 # Input markdown files
├── output/
│   ├── audiobooks/           # Generated audiobook files
│   └── enhanced_text/        # Processed text files
│
├── tests/
│   ├── __init__.py
│   ├── test_markdown_processor.py
│   ├── test_text_enhancer.py
│   └── test_audiobook_generator.py
│
└── docs/
    ├── architecture.md        # Detailed architecture
    ├── configuration.md       # Configuration guide
    └── examples.md           # Usage examples
```

## 🔧 Configuration

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Edit with your API keys (optional)
nano .env
```

### Processing Configuration
```yaml
# config/default.yaml
processing_mode: "hybrid"  # basic, local_ai, api, hybrid
output_format: "m4b"      # m4b, mp3, wav

voices:
  main_narrator: "piper_en_us"
  math_voice: "technical_clear"
  chapter_voice: "emphasis"
```

## 📚 Documentation

### Core Documentation
- **[Architecture](docs/architecture.md)** - Detailed system design and pipeline overview
- **[Configuration](docs/configuration.md)** - Setup and customization options
- **[Examples](docs/examples.md)** - Usage examples and tutorials

### TTS Provider Guides
- **[TTS Providers Overview](docs/tts-providers.md)** - Complete comparison of all TTS options
- **[Setup Guides](docs/setup-guides.md)** - Step-by-step provider configuration
- **[Cost Calculator](docs/cost-calculator.md)** - Pricing analysis and ROI calculator
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions

### Quick Reference
- **Free Options**: Piper TTS (offline), Hugging Face (API)
- **Premium Options**: ElevenLabs, Google Cloud, Azure, Amazon Polly
- **Best Value**: Google Cloud Standard ($4/1M characters)
- **Highest Quality**: ElevenLabs ($5-99/month) or Google Neural2

## 🧪 Testing

```bash
# Run all tests
make test

# Test specific components
python -m pytest tests/test_markdown_processor.py

# Integration test with example document
make test-integration
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone and setup development environment
git clone https://github.com/ucli-tools/md2audiobook.git
cd md2audiobook
make dev-setup

# Run tests
make test

# Check code quality
make lint
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **[ucli-tools](https://github.com/ucli-tools)** - Unified CLI tools ecosystem
- **[mdtexpdf](https://github.com/ucli-tools/mdtexpdf)** - Companion PDF generation tool
- **OpenAI Whisper** - Speech recognition technology
- **Piper TTS** - High-quality local text-to-speech

## 🔗 Related Projects

- **[mdtexpdf](https://github.com/ucli-tools/mdtexpdf)** - Markdown to PDF conversion
- **[ucli-tools](https://github.com/ucli-tools)** - Unified CLI tools ecosystem

---

*Part of the ucli-tools ecosystem - Professional tools for academic and technical content creation.*
