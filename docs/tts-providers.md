# TTS Providers Guide

Complete guide to Text-to-Speech providers supported by md2audiobook, including setup instructions, costs, and quality comparisons.

## 🆓 Free TTS Providers

### 1. Local TTS (Completely Free)

#### Piper TTS (Recommended)
- **Cost**: $0 (completely free)
- **Quality**: High quality neural voices
- **Setup**: System installation required
- **Voices**: 200+ voices in 50+ languages
- **Speed**: Fast local processing

```bash
# Ubuntu/Debian
sudo apt install piper-tts

# Or download from: https://github.com/rhasspy/piper
```

**Configuration**: No API keys needed - works offline!

#### espeak-ng (Fallback)
- **Cost**: $0 (completely free)
- **Quality**: Basic robotic voice
- **Setup**: System installation
- **Speed**: Very fast

```bash
sudo apt install espeak-ng
```

#### Festival TTS
- **Cost**: $0 (completely free)
- **Quality**: Basic synthetic voice
- **Setup**: System installation

```bash
sudo apt install festival
```

### 2. Hugging Face (Free Tier)

#### Hugging Face Inference API
- **Cost**: FREE up to rate limits
- **Quality**: High quality (Bark, SpeechT5, etc.)
- **Models**: Multiple TTS models available
- **Rate Limits**: Generous free tier

**Setup**:
```bash
# Get free API key from https://huggingface.co/settings/tokens
HUGGINGFACE_API_KEY=hf_your_free_token_here
```

**Available Models**:
- `microsoft/speecht5_tts` - High quality English
- `suno/bark` - Very natural, supports emotions
- `facebook/mms-tts` - 1000+ languages

**Example Cost**: FREE for personal use

---

## 💰 Premium TTS Providers

### 1. ElevenLabs (Highest Quality)

#### Pricing
- **Free Tier**: 10,000 characters/month
- **Starter**: $5/month - 30,000 characters
- **Creator**: $22/month - 100,000 characters
- **Pro**: $99/month - 500,000 characters

#### Quality Features
- Ultra-realistic AI voices
- Voice cloning capabilities
- Emotion and style control
- Multiple languages

#### Setup
```bash
# Get API key from https://elevenlabs.io/
ELEVENLABS_API_KEY=your_elevenlabs_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM  # Default voice
```

#### Cost Example
- **50,000 word audiobook** ≈ 300,000 characters
- **Cost**: $22/month (Creator plan)
- **Quality**: Professional audiobook grade

### 2. Google Cloud Text-to-Speech

#### Pricing (Pay-per-use)
- **Standard voices**: $4.00 per 1M characters
- **WaveNet voices**: $16.00 per 1M characters  
- **Neural2 voices**: $16.00 per 1M characters
- **Free tier**: 1M characters/month for WaveNet

#### Setup
1. Create Google Cloud Project
2. Enable Text-to-Speech API
3. Create service account & download JSON key
4. Configure environment:

```bash
GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

#### Cost Example
- **50,000 word audiobook** ≈ 300,000 characters
- **Standard**: $1.20
- **WaveNet/Neural2**: $4.80
- **Quality**: Professional grade

### 3. Azure Cognitive Services Speech

#### Pricing
- **Free tier**: 500,000 characters/month
- **Standard**: $4.00 per 1M characters (Standard)
- **Neural**: $16.00 per 1M characters

#### Setup
1. Create Azure account
2. Create Speech Services resource
3. Get API key and region

```bash
AZURE_SPEECH_KEY=your_azure_speech_key_here
AZURE_SPEECH_REGION=eastus
```

#### Cost Example
- **50,000 word audiobook** ≈ 300,000 characters
- **Free tier**: $0 (under monthly limit)
- **Standard**: $1.20
- **Neural**: $4.80

### 4. Amazon Polly

#### Pricing
- **Free tier**: 5M characters/month (12 months)
- **Standard**: $4.00 per 1M characters
- **Neural**: $16.00 per 1M characters

#### Setup
```bash
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

---

## 📊 Quality & Cost Comparison

| Provider | Cost (50k words) | Quality | Voices | Languages | Offline |
|----------|------------------|---------|---------|-----------|---------|
| **Piper TTS** | $0 | ⭐⭐⭐⭐ | 200+ | 50+ | ✅ |
| **Hugging Face** | $0 | ⭐⭐⭐⭐ | Many | 100+ | ❌ |
| **espeak-ng** | $0 | ⭐⭐ | Basic | 100+ | ✅ |
| **ElevenLabs** | $22/month | ⭐⭐⭐⭐⭐ | Premium | 25+ | ❌ |
| **Google Neural2** | $4.80 | ⭐⭐⭐⭐⭐ | 380+ | 40+ | ❌ |
| **Azure Neural** | $4.80 | ⭐⭐⭐⭐⭐ | 270+ | 110+ | ❌ |
| **Amazon Neural** | $4.80 | ⭐⭐⭐⭐ | 60+ | 30+ | ❌ |

---

## 🚀 Recommended Setups

### For Personal Use (Free)
```yaml
# config/default.yaml
tts:
  provider: "piper"  # or "huggingface"
  fallback: "espeak"
```

### For Professional Audiobooks
```yaml
# config/default.yaml
tts:
  provider: "elevenlabs"  # or "google_neural2"
  fallback: "piper"
```

### For Budget-Conscious Production
```yaml
# config/default.yaml
tts:
  provider: "google_standard"  # or "azure_standard"
  fallback: "piper"
```

---

## 🔧 Configuration Examples

### Hugging Face Free Setup
```bash
# .env
HUGGINGFACE_API_KEY=hf_your_free_token_here

# config/default.yaml
tts:
  provider: "huggingface"
  model: "microsoft/speecht5_tts"
  fallback: "piper"
```

### Multi-Provider Fallback
```yaml
# config/default.yaml
tts:
  provider: "elevenlabs"
  fallback_chain:
    - "google_neural2"
    - "piper"
    - "espeak"
```

---

## 💡 Tips for Cost Optimization

### 1. Use Free Tiers First
- Start with Hugging Face or Piper TTS
- Test with Azure/Google free monthly limits
- Only upgrade when quality requirements demand it

### 2. Hybrid Approach
- Use premium TTS for chapter titles and important sections
- Use free TTS for bulk content
- Combine multiple providers for cost efficiency

### 3. Content Optimization
- Remove unnecessary text before TTS processing
- Use abbreviations and contractions to reduce character count
- Pre-process mathematical expressions to shorter spoken forms

### 4. Batch Processing
- Process multiple documents together
- Take advantage of monthly free tiers across providers
- Use local TTS for drafts, premium for final production

---

## 🔍 Quality Testing

Test different providers with your content:

```bash
# Test with free provider
python scripts/process_audiobook.py documents/sample.md --tts-provider piper

# Test with premium provider  
python scripts/process_audiobook.py documents/sample.md --tts-provider elevenlabs

# Compare quality and choose best fit for your budget
```

---

## 📞 Support & Resources

- **Piper TTS**: [GitHub Repository](https://github.com/rhasspy/piper)
- **Hugging Face**: [TTS Models Hub](https://huggingface.co/models?pipeline_tag=text-to-speech)
- **ElevenLabs**: [Documentation](https://docs.elevenlabs.io/)
- **Google Cloud**: [TTS Documentation](https://cloud.google.com/text-to-speech/docs)
- **Azure Speech**: [Documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/speech-service/)

Choose the provider that best fits your quality requirements and budget!
