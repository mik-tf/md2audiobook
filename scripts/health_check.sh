#!/bin/bash
# health_check.sh - Comprehensive health check for md2audiobook pipeline

echo "🔍 md2audiobook Health Check"
echo "============================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "📋 System Requirements"
echo "----------------------"
if python3 --version >/dev/null 2>&1; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo -e "✅ ${GREEN}Python: $PYTHON_VERSION${NC}"
else
    echo -e "❌ ${RED}Python not found${NC}"
fi

# Check FFmpeg
if ffmpeg -version >/dev/null 2>&1; then
    FFMPEG_VERSION=$(ffmpeg -version 2>&1 | head -n1 | cut -d' ' -f3)
    echo -e "✅ ${GREEN}FFmpeg: $FFMPEG_VERSION${NC}"
else
    echo -e "❌ ${RED}FFmpeg not found${NC}"
    echo -e "   ${YELLOW}Install with: sudo apt install ffmpeg${NC}"
fi

echo ""

# Check Python dependencies
echo "📦 Python Dependencies"
echo "----------------------"
if python3 -c "import yaml, pydub, mutagen, librosa, soundfile" 2>/dev/null; then
    echo -e "✅ ${GREEN}Core audio libraries OK${NC}"
else
    echo -e "❌ ${RED}Missing core dependencies${NC}"
    echo -e "   ${YELLOW}Run: pip install -r requirements.txt${NC}"
fi

if python3 -c "import markdown, frontmatter, jinja2" 2>/dev/null; then
    echo -e "✅ ${GREEN}Markdown processing libraries OK${NC}"
else
    echo -e "❌ ${RED}Missing markdown dependencies${NC}"
fi

if python3 -c "import requests, httpx, aiohttp" 2>/dev/null; then
    echo -e "✅ ${GREEN}HTTP client libraries OK${NC}"
else
    echo -e "⚠️  ${YELLOW}Some HTTP libraries missing (API features may not work)${NC}"
fi

echo ""

# Check TTS providers
echo "🎵 TTS Providers"
echo "----------------"
if which piper >/dev/null 2>&1; then
    echo -e "✅ ${GREEN}Piper TTS available${NC}"
else
    echo -e "⚠️  ${YELLOW}Piper TTS not found${NC}"
    echo -e "   ${YELLOW}Install from: https://github.com/rhasspy/piper${NC}"
fi

if which espeak >/dev/null 2>&1; then
    echo -e "✅ ${GREEN}espeak available${NC}"
else
    echo -e "⚠️  ${YELLOW}espeak not found${NC}"
    echo -e "   ${YELLOW}Install with: sudo apt install espeak-ng${NC}"
fi

if which festival >/dev/null 2>&1; then
    echo -e "✅ ${GREEN}Festival available${NC}"
else
    echo -e "⚠️  ${YELLOW}Festival not found (optional)${NC}"
fi

echo ""

# Check configuration files
echo "⚙️  Configuration"
echo "-----------------"
if [ -f config/default.yaml ]; then
    echo -e "✅ ${GREEN}Configuration file exists${NC}"
    # Validate YAML syntax
    if python3 -c "import yaml; yaml.safe_load(open('config/default.yaml'))" 2>/dev/null; then
        echo -e "✅ ${GREEN}Configuration syntax valid${NC}"
    else
        echo -e "❌ ${RED}Configuration syntax error${NC}"
    fi
else
    echo -e "❌ ${RED}Configuration missing${NC}"
    echo -e "   ${YELLOW}Run: cp config/default.yaml.example config/default.yaml${NC}"
fi

if [ -f .env ]; then
    echo -e "✅ ${GREEN}Environment file exists${NC}"
else
    echo -e "⚠️  ${YELLOW}Environment file missing (API features disabled)${NC}"
    echo -e "   ${YELLOW}Run: cp .env.example .env${NC}"
fi

echo ""

# Check API keys
echo "🔑 API Keys"
echo "-----------"
if [ -f .env ]; then
    source .env
    
    API_COUNT=0
    TOTAL_APIS=5
    
    if [ ! -z "$ELEVENLABS_API_KEY" ]; then
        echo -e "✅ ${GREEN}ElevenLabs API key configured${NC}"
        ((API_COUNT++))
    fi
    
    if [ ! -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && [ ! -z "$GOOGLE_CLOUD_PROJECT" ]; then
        echo -e "✅ ${GREEN}Google Cloud credentials configured${NC}"
        ((API_COUNT++))
    fi
    
    if [ ! -z "$AZURE_SPEECH_KEY" ] && [ ! -z "$AZURE_SPEECH_REGION" ]; then
        echo -e "✅ ${GREEN}Azure Speech credentials configured${NC}"
        ((API_COUNT++))
    fi
    
    if [ ! -z "$AWS_ACCESS_KEY_ID" ] && [ ! -z "$AWS_SECRET_ACCESS_KEY" ]; then
        echo -e "✅ ${GREEN}AWS Polly credentials configured${NC}"
        ((API_COUNT++))
    fi
    
    if [ ! -z "$HUGGINGFACE_API_KEY" ]; then
        echo -e "✅ ${GREEN}Hugging Face API key configured${NC}"
        ((API_COUNT++))
    fi
    
    if [ $API_COUNT -eq 0 ]; then
        echo -e "⚠️  ${YELLOW}No API keys configured (local TTS only)${NC}"
    else
        echo -e "📊 ${GREEN}API providers configured: $API_COUNT/$TOTAL_APIS${NC}"
    fi
else
    echo -e "⚠️  ${YELLOW}No environment file found${NC}"
fi

echo ""

# Check directories
echo "📁 Directories"
echo "--------------"
DIRS=("documents" "output" "output/audiobooks" "output/enhanced_text" "logs")
for dir in "${DIRS[@]}"; do
    if [ -d "$dir" ]; then
        echo -e "✅ ${GREEN}$dir/ exists${NC}"
    else
        echo -e "⚠️  ${YELLOW}$dir/ missing (will be created)${NC}"
        mkdir -p "$dir"
    fi
done

echo ""

# Test basic functionality
echo "🧪 Basic Functionality Test"
echo "---------------------------"
if [ -f documents/example.md ]; then
    echo -e "✅ ${GREEN}Example document available${NC}"
    
    # Test markdown processing
    if python3 -c "
from src.markdown_processor import MarkdownProcessor
processor = MarkdownProcessor()
result = processor.process_file('documents/example.md')
print(f'✅ Markdown processing: {len(result.chapters)} chapters found')
" 2>/dev/null; then
        echo -e "✅ ${GREEN}Markdown processing works${NC}"
    else
        echo -e "❌ ${RED}Markdown processing failed${NC}"
    fi
    
    # Test text enhancement
    if python3 -c "
from src.text_enhancer import TextEnhancer
enhancer = TextEnhancer()
print('✅ Text enhancement module loads')
" 2>/dev/null; then
        echo -e "✅ ${GREEN}Text enhancement module OK${NC}"
    else
        echo -e "❌ ${RED}Text enhancement module failed${NC}"
    fi
    
else
    echo -e "⚠️  ${YELLOW}Example document missing${NC}"
fi

echo ""

# Performance check
echo "⚡ Performance Check"
echo "-------------------"
MEMORY=$(free -m | awk 'NR==2{printf "%.1f", $3*100/$2}')
DISK=$(df -h . | awk 'NR==2{print $5}' | sed 's/%//')

echo -e "💾 Memory usage: ${MEMORY}%"
echo -e "💿 Disk usage: ${DISK}%"

if (( $(echo "$MEMORY > 80" | bc -l) )); then
    echo -e "⚠️  ${YELLOW}High memory usage detected${NC}"
fi

if (( DISK > 90 )); then
    echo -e "⚠️  ${YELLOW}Low disk space${NC}"
fi

echo ""

# Summary
echo "📋 Health Check Summary"
echo "======================="

# Count issues
ISSUES=0
if ! python3 --version >/dev/null 2>&1; then ((ISSUES++)); fi
if ! ffmpeg -version >/dev/null 2>&1; then ((ISSUES++)); fi
if ! python3 -c "import yaml, pydub, mutagen" 2>/dev/null; then ((ISSUES++)); fi
if [ ! -f config/default.yaml ]; then ((ISSUES++)); fi

if [ $ISSUES -eq 0 ]; then
    echo -e "🎉 ${GREEN}All systems operational!${NC}"
    echo -e "   Ready to process audiobooks"
elif [ $ISSUES -le 2 ]; then
    echo -e "⚠️  ${YELLOW}Minor issues detected ($ISSUES)${NC}"
    echo -e "   Basic functionality should work"
else
    echo -e "❌ ${RED}Major issues detected ($ISSUES)${NC}"
    echo -e "   Setup required before processing"
fi

echo ""
echo "💡 Next steps:"
echo "   • Run 'make setup' to initialize configuration"
echo "   • Run 'make demo' to test with example document"
echo "   • Check docs/troubleshooting.md for specific issues"
echo ""
echo "Health check complete! 🏁"
