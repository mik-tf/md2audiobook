# TTS Provider Setup Guides

Step-by-step setup instructions for all supported TTS providers.

## 🆓 Free Providers Setup

### 1. Piper TTS (Recommended Free Option)

#### Ubuntu/Debian Installation
```bash
# Method 1: Package manager (if available)
sudo apt update
sudo apt install piper-tts

# Method 2: Download binary
wget https://github.com/rhasspy/piper/releases/latest/download/piper_amd64.tar.gz
tar -xzf piper_amd64.tar.gz
sudo mv piper /usr/local/bin/
```

#### Configuration
```yaml
# config/default.yaml
tts:
  provider: "piper"
  piper:
    model_path: "/usr/share/piper-voices/"  # or custom path
    voice: "en_US-lessac-medium"
```

**No API keys needed!** Works completely offline.

### 2. Hugging Face (Free API)

#### Step 1: Create Account
1. Go to [huggingface.co](https://huggingface.co)
2. Sign up for free account
3. Go to Settings → Access Tokens
4. Create new token with "Read" permissions

#### Step 2: Configure Environment
```bash
# .env
HUGGINGFACE_API_KEY=hf_your_token_here
```

#### Step 3: Choose Model
```yaml
# config/default.yaml
tts:
  provider: "huggingface"
  huggingface:
    model: "microsoft/speecht5_tts"  # High quality English
    # model: "suno/bark"  # Very natural, slower
    # model: "facebook/mms-tts"  # 1000+ languages
```

#### Available Models
- `microsoft/speecht5_tts` - Fast, high quality English
- `suno/bark` - Ultra-natural, supports emotions (slower)
- `facebook/mms-tts` - Massive multilingual support
- `espnet/kan-bayashi_ljspeech_vits` - Alternative English model

---

## 💰 Premium Providers Setup

### 1. ElevenLabs (Highest Quality)

#### Step 1: Create Account
1. Go to [elevenlabs.io](https://elevenlabs.io)
2. Sign up (free tier: 10k characters/month)
3. Go to Profile → API Key
4. Copy your API key

#### Step 2: Choose Voice
1. Go to VoiceLab
2. Browse available voices
3. Copy Voice ID (e.g., `21m00Tcm4TlvDq8ikWAM`)

#### Step 3: Configure
```bash
# .env
ELEVENLABS_API_KEY=your_api_key_here
ELEVENLABS_VOICE_ID=21m00Tcm4TlvDq8ikWAM
```

```yaml
# config/default.yaml
tts:
  provider: "elevenlabs"
  elevenlabs:
    voice_id: "21m00Tcm4TlvDq8ikWAM"
    model: "eleven_multilingual_v2"
    stability: 0.5
    similarity_boost: 0.5
```

#### Popular Voice IDs
- `21m00Tcm4TlvDq8ikWAM` - Rachel (Female, American)
- `AZnzlk1XvdvUeBnXmlld` - Domi (Female, American)
- `EXAVITQu4vr4xnSDxMaL` - Bella (Female, American)
- `ErXwobaYiN019PkySvjV` - Antoni (Male, American)
- `MF3mGyEYCl7XYWbV9V6O` - Elli (Female, American)

### 2. Google Cloud Text-to-Speech

#### Step 1: Create Google Cloud Project
1. Go to [console.cloud.google.com](https://console.cloud.google.com)
2. Create new project or select existing
3. Note your Project ID (not name)

#### Step 2: Enable API
1. Go to "APIs & Services" → "Library"
2. Search "Cloud Text-to-Speech API"
3. Click "Enable"

#### Step 3: Create Service Account
1. Go to "IAM & Admin" → "Service Accounts"
2. Click "Create Service Account"
3. Name: `md2audiobook-tts`
4. Role: "Cloud Text-to-Speech Client"
5. Create Key → JSON format
6. Download the JSON file

#### Step 4: Configure
```bash
# .env
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
GOOGLE_CLOUD_PROJECT=your-project-id
```

```yaml
# config/default.yaml
tts:
  provider: "google"
  google:
    voice_name: "en-US-Neural2-F"
    language_code: "en-US"
    audio_encoding: "MP3"
```

#### Popular Google Voices
- `en-US-Neural2-F` - Female, natural
- `en-US-Neural2-J` - Male, natural  
- `en-US-Wavenet-F` - Female, high quality
- `en-GB-Neural2-A` - British female
- `en-AU-Neural2-A` - Australian female

### 3. Azure Cognitive Services

#### Step 1: Create Azure Account
1. Go to [portal.azure.com](https://portal.azure.com)
2. Sign up (free tier available)

#### Step 2: Create Speech Resource
1. Click "Create a resource"
2. Search "Speech Services"
3. Create with these settings:
   - Pricing tier: F0 (free) or S0 (standard)
   - Region: Choose closest to you

#### Step 3: Get Credentials
1. Go to your Speech resource
2. Click "Keys and Endpoint"
3. Copy Key 1 and Region

#### Step 4: Configure
```bash
# .env
AZURE_SPEECH_KEY=your_key_here
AZURE_SPEECH_REGION=eastus
```

```yaml
# config/default.yaml
tts:
  provider: "azure"
  azure:
    voice_name: "en-US-AriaNeural"
    language: "en-US"
    output_format: "audio-24khz-48kbitrate-mono-mp3"
```

#### Popular Azure Voices
- `en-US-AriaNeural` - Female, conversational
- `en-US-DavisNeural` - Male, friendly
- `en-US-JennyNeural` - Female, assistant style
- `en-GB-SoniaNeural` - British female
- `en-AU-NatashaNeural` - Australian female

### 4. Amazon Polly

#### Step 1: Create AWS Account
1. Go to [aws.amazon.com](https://aws.amazon.com)
2. Sign up (free tier: 5M characters/month for 12 months)

#### Step 2: Create IAM User
1. Go to IAM Console
2. Create user with "Programmatic access"
3. Attach policy: "AmazonPollyFullAccess"
4. Save Access Key ID and Secret Access Key

#### Step 3: Configure
```bash
# .env
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_DEFAULT_REGION=us-east-1
```

```yaml
# config/default.yaml
tts:
  provider: "polly"
  polly:
    voice_id: "Joanna"
    engine: "neural"  # or "standard"
    output_format: "mp3"
```

#### Popular Polly Voices
- `Joanna` - Female, American (Neural)
- `Matthew` - Male, American (Neural)
- `Amy` - Female, British (Neural)
- `Brian` - Male, British (Neural)
- `Emma` - Female, British (Neural)

---

## 🔧 Testing Your Setup

### Test Individual Providers
```bash
# Test Piper (free)
echo "Hello world" | piper --model en_US-lessac-medium --output_file test.wav

# Test Hugging Face
python -c "
import requests
headers = {'Authorization': 'Bearer hf_your_token'}
response = requests.post('https://api-inference.huggingface.co/models/microsoft/speecht5_tts', 
                        headers=headers, json={'inputs': 'Hello world'})
print('Status:', response.status_code)
"

# Test with md2audiobook
python scripts/process_audiobook.py documents/example.md --tts-provider piper
```

### Verify Configuration
```bash
# Check if credentials are loaded
python -c "
import os
from dotenv import load_dotenv
load_dotenv()
print('Hugging Face:', 'HUGGINGFACE_API_KEY' in os.environ)
print('ElevenLabs:', 'ELEVENLABS_API_KEY' in os.environ)
print('Google:', 'GOOGLE_APPLICATION_CREDENTIALS' in os.environ)
print('Azure:', 'AZURE_SPEECH_KEY' in os.environ)
"
```

---

## 🚨 Troubleshooting

### Common Issues

#### Piper TTS Not Found
```bash
# Check if installed
which piper
# If not found, install:
sudo apt install piper-tts
# Or download binary from GitHub releases
```

#### Google Cloud Authentication Error
```bash
# Check credentials file exists
ls -la $GOOGLE_APPLICATION_CREDENTIALS
# Check project ID is correct
gcloud config get-value project
# Test authentication
gcloud auth application-default login
```

#### Azure Region Mismatch
```bash
# Common regions:
# eastus, westus2, northeurope, southeastasia
# Must match your Speech resource region exactly
```

#### ElevenLabs Rate Limit
```bash
# Free tier: 10k characters/month
# Check usage at: https://elevenlabs.io/usage
# Upgrade plan or wait for reset
```

#### Hugging Face Model Loading
```bash
# Some models need warm-up time
# First request may take 20+ seconds
# Subsequent requests are faster
```

---

## 💡 Pro Tips

### 1. Fallback Configuration
```yaml
# Always configure fallbacks
tts:
  provider: "elevenlabs"
  fallback_chain:
    - "google"
    - "piper"
    - "espeak"
```

### 2. Cost Monitoring
- Set up billing alerts in cloud providers
- Monitor usage in provider dashboards
- Test with small samples first

### 3. Quality Testing
```bash
# Create test samples with different providers
mkdir test_samples
python scripts/process_audiobook.py test.md --tts-provider piper --output test_samples/piper.m4b
python scripts/process_audiobook.py test.md --tts-provider elevenlabs --output test_samples/elevenlabs.m4b
# Compare quality and choose best fit
```

### 4. Voice Consistency
- Use same voice throughout entire audiobook
- Test voice with your specific content type
- Consider accent/language match for your audience

---

## 📋 Quick Setup Checklist

### Free Setup (Piper + Hugging Face)
- [ ] Install Piper TTS system package
- [ ] Create Hugging Face account
- [ ] Generate HF access token
- [ ] Add `HUGGINGFACE_API_KEY` to `.env`
- [ ] Test with sample document

### Premium Setup (ElevenLabs)
- [ ] Create ElevenLabs account
- [ ] Choose voice and copy Voice ID
- [ ] Add API key and Voice ID to `.env`
- [ ] Configure fallback to free provider
- [ ] Test with small sample first

### Enterprise Setup (Google Cloud)
- [ ] Create Google Cloud project
- [ ] Enable Text-to-Speech API
- [ ] Create service account
- [ ] Download JSON credentials
- [ ] Set environment variables
- [ ] Test authentication
- [ ] Configure billing alerts

Ready to create professional audiobooks! 🎧
