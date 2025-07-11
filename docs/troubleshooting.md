# Troubleshooting Guide

Common issues and solutions for the md2audiobook pipeline.

## 🚨 Common Issues

### 1. Chapter Title Problems

#### Issue: Chapter titles are truncated or corrupted
```
Expected: "Chapter 1: Introduction to Machine Learning"
Actual: "Chapter 1: Intro..."
```

**Solution**: This was fixed in the latest version. Update your installation:
```bash
git pull origin main
pip install -r requirements.txt
```

**Root Cause**: Previous versions extracted chapter titles from enhanced text, causing truncation. Now uses original titles directly.

#### Issue: Chapter titles missing "Chapter:" prefix
**Solution**: Check your markdown headers:
```markdown
# Chapter 1: Introduction  ✅ Correct
## Introduction           ❌ Will not be treated as chapter
```

### 2. TTS Provider Issues

#### Issue: Piper TTS not found
```
Error: piper: command not found
```

**Solutions**:
```bash
# Option 1: Install via package manager
sudo apt update && sudo apt install piper-tts

# Option 2: Download binary
wget https://github.com/rhasspy/piper/releases/latest/download/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper /usr/local/bin/

# Option 3: Use fallback
# Edit config/default.yaml
tts:
  provider: "espeak"  # or "huggingface"
```

#### Issue: ElevenLabs API errors
```
Error: 401 Unauthorized
Error: 429 Too Many Requests
```

**Solutions**:
```bash
# Check API key
echo $ELEVENLABS_API_KEY

# Check usage at https://elevenlabs.io/usage
# Verify voice ID exists
curl -X GET "https://api.elevenlabs.io/v1/voices" \
  -H "xi-api-key: $ELEVENLABS_API_KEY"

# Rate limiting: add delays
# config/default.yaml
tts:
  elevenlabs:
    request_delay: 1.0  # seconds between requests
```

#### Issue: Google Cloud authentication
```
Error: Could not automatically determine credentials
```

**Solutions**:
```bash
# Check credentials file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS

# Check project ID
echo $GOOGLE_CLOUD_PROJECT

# Test authentication
gcloud auth application-default login

# Verify API is enabled
gcloud services list --enabled | grep texttospeech
```

#### Issue: Azure region mismatch
```
Error: The specified region is not valid
```

**Solution**: Ensure region matches your Speech resource:
```bash
# Common regions (must match exactly):
# eastus, westus2, northeurope, southeastasia
export AZURE_SPEECH_REGION=eastus  # Replace with your region
```

### 3. Audio Quality Issues

#### Issue: Robotic or poor quality audio
**Solutions**:
1. **Upgrade TTS provider**:
   ```yaml
   # From espeak to Piper
   tts:
     provider: "piper"  # Much better quality
   ```

2. **Use premium providers**:
   ```yaml
   tts:
     provider: "elevenlabs"  # Highest quality
   ```

3. **Adjust voice settings**:
   ```yaml
   tts:
     elevenlabs:
       stability: 0.3      # Lower = more expressive
       similarity_boost: 0.7  # Higher = more consistent
   ```

#### Issue: Mathematical expressions sound wrong
```
"x^2 + y^2 = z^2" → "x caret 2 plus y caret 2 equals z caret 2"
```

**Solution**: The text enhancer should handle this automatically. Check configuration:
```yaml
text_enhancement:
  math_processing: true
  latex_to_speech: true
```

**Manual fix** in your markdown:
```markdown
<!-- Before -->
The equation $x^2 + y^2 = z^2$ represents...

<!-- After -->
The equation x squared plus y squared equals z squared represents...
```

### 4. Processing Errors

#### Issue: Out of memory during processing
```
Error: MemoryError: Unable to allocate array
```

**Solutions**:
```bash
# Process in smaller chunks
# config/default.yaml
processing:
  chunk_size: 1000      # Reduce from default
  max_concurrent: 1     # Reduce parallel processing

# Or process chapters individually
python scripts/process_audiobook.py document.md --chapter-by-chapter
```

#### Issue: FFmpeg not found
```
Error: ffmpeg: command not found
```

**Solution**:
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Verify installation
ffmpeg -version
```

#### Issue: Long processing times
**Optimization strategies**:
```yaml
# Use faster TTS for drafts
tts:
  provider: "piper"  # Faster than API providers

# Reduce audio quality for testing
audio:
  sample_rate: 22050  # Lower than 44100
  bitrate: 64         # Lower than 128

# Process smaller sections first
processing:
  test_mode: true     # Process first chapter only
```

### 5. Configuration Issues

#### Issue: Configuration file not found
```
Error: Config file not found: config/default.yaml
```

**Solution**:
```bash
# Copy template
cp config/default.yaml.example config/default.yaml

# Or use make setup
make setup
```

#### Issue: Environment variables not loaded
```
Error: API key not found
```

**Solutions**:
```bash
# Check .env file exists
ls -la .env

# Copy template if missing
cp .env.example .env

# Edit with your keys
nano .env

# Verify loading
python -c "
from dotenv import load_dotenv
import os
load_dotenv()
print('Keys loaded:', len([k for k in os.environ if 'API' in k]))
"
```

### 6. Output Issues

#### Issue: No audio output generated
**Debugging steps**:
```bash
# Check output directory
ls -la output/audiobooks/

# Check logs
tail -f logs/audiobook_generation.log

# Test with minimal example
echo "# Test Chapter\nHello world." > test.md
python scripts/process_audiobook.py test.md --verbose
```

#### Issue: Audio files are corrupted
**Solutions**:
```bash
# Check FFmpeg installation
ffmpeg -version

# Verify audio files
ffprobe output/audiobooks/your_book.m4b

# Regenerate with different format
python scripts/process_audiobook.py document.md --format mp3
```

---

## 🔍 Debugging Tools

### 1. Verbose Logging
```bash
# Enable detailed logging
python scripts/process_audiobook.py document.md --verbose --debug

# Check specific component
python scripts/process_audiobook.py document.md --debug-component text_enhancer
```

### 2. Test Individual Components
```bash
# Test markdown processing only
python -c "
from src.markdown_processor import MarkdownProcessor
processor = MarkdownProcessor()
result = processor.process_file('document.md')
print(f'Processed {len(result.chapters)} chapters')
"

# Test TTS provider
python -c "
from src.providers.local_tts import PiperTTS
tts = PiperTTS()
tts.synthesize('Hello world', 'test.wav')
print('TTS test complete')
"
```

### 3. Validate Dependencies
```bash
# Run dependency check
python test_dependencies.py

# Check specific packages
python -c "
import pydub, mutagen, librosa
print('Audio libraries: OK')
"
```

### 4. Configuration Validation
```bash
# Validate YAML syntax
python -c "
import yaml
with open('config/default.yaml') as f:
    config = yaml.safe_load(f)
print('Config valid:', bool(config))
"

# Check provider configuration
python scripts/validate_config.py
```

---

## 📊 Performance Optimization

### 1. Speed Optimization
```yaml
# Fastest configuration
processing_mode: "basic"
tts:
  provider: "piper"
  piper:
    model: "en_US-lessac-low"  # Smaller model
audio:
  sample_rate: 22050
  bitrate: 64
text_enhancement:
  ai_enhancement: false
```

### 2. Quality Optimization
```yaml
# Highest quality configuration
processing_mode: "api"
tts:
  provider: "elevenlabs"
  elevenlabs:
    model: "eleven_multilingual_v2"
    stability: 0.3
    similarity_boost: 0.8
audio:
  sample_rate: 44100
  bitrate: 192
text_enhancement:
  ai_enhancement: true
```

### 3. Balanced Configuration
```yaml
# Good balance of speed and quality
processing_mode: "hybrid"
tts:
  provider: "piper"
  fallback_chain: ["google", "espeak"]
audio:
  sample_rate: 44100
  bitrate: 128
```

---

## 🆘 Getting Help

### 1. Check Logs
```bash
# Main log file
tail -f logs/md2audiobook.log

# Component-specific logs
ls logs/
```

### 2. Create Minimal Test Case
```bash
# Create simple test
echo "# Test Chapter
This is a test document with math: $x^2$.
" > minimal_test.md

# Process with verbose output
python scripts/process_audiobook.py minimal_test.md --verbose
```

### 3. System Information
```bash
# Gather system info for bug reports
python -c "
import sys, platform
print(f'Python: {sys.version}')
print(f'Platform: {platform.platform()}')
print(f'Architecture: {platform.architecture()}')
"

# Check available memory
free -h

# Check disk space
df -h
```

### 4. Report Issues
When reporting issues, include:
- **Error message** (full traceback)
- **System information** (OS, Python version)
- **Configuration** (sanitized config/default.yaml)
- **Input sample** (minimal markdown that reproduces issue)
- **Expected vs actual behavior**

### 5. Community Resources
- **GitHub Issues**: Report bugs and feature requests
- **Discussions**: Ask questions and share tips
- **Wiki**: Community-maintained guides and examples

---

## ✅ Health Check Script

Create a comprehensive health check:

```bash
#!/bin/bash
# health_check.sh

echo "🔍 md2audiobook Health Check"
echo "=========================="

# Check Python
python3 --version || echo "❌ Python not found"

# Check dependencies
python3 -c "import yaml, pydub, mutagen" 2>/dev/null && echo "✅ Core dependencies OK" || echo "❌ Missing dependencies"

# Check FFmpeg
ffmpeg -version >/dev/null 2>&1 && echo "✅ FFmpeg OK" || echo "❌ FFmpeg not found"

# Check TTS providers
which piper >/dev/null && echo "✅ Piper TTS available" || echo "⚠️  Piper TTS not found"
which espeak >/dev/null && echo "✅ espeak available" || echo "⚠️  espeak not found"

# Check configuration
[ -f config/default.yaml ] && echo "✅ Configuration file exists" || echo "❌ Configuration missing"
[ -f .env ] && echo "✅ Environment file exists" || echo "⚠️  Environment file missing"

# Check API keys
python3 -c "
import os
from dotenv import load_dotenv
load_dotenv()
keys = ['ELEVENLABS_API_KEY', 'GOOGLE_APPLICATION_CREDENTIALS', 'AZURE_SPEECH_KEY', 'HUGGINGFACE_API_KEY']
found = [k for k in keys if os.getenv(k)]
print(f'✅ API keys configured: {len(found)}/{len(keys)}')
"

echo "=========================="
echo "Health check complete!"
```

Run with: `bash health_check.sh`

This troubleshooting guide should help resolve most common issues with the md2audiobook pipeline!
