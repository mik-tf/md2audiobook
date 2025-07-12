"""
Audiobook Generator - Professional audio synthesis and audiobook creation
Part of md2audiobook pipeline
"""

import os
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import json
import tempfile
import shutil
from pydub import AudioSegment
from mutagen.mp4 import MP4, MP4Cover
import requests
from .text_enhancer import EnhancedText


@dataclass
class AudioChapter:
    """Audio chapter with metadata"""
    title: str
    audio_segment: AudioSegment
    start_time: float  # in seconds
    duration: float    # in seconds
    chapter_number: int


@dataclass
class AudiobookOutput:
    """Complete audiobook with metadata"""
    audio: AudioSegment
    chapters: List[AudioChapter]
    metadata: Dict[str, Any]
    file_path: Optional[Path] = None


class AudiobookGenerator:
    """
    Professional audiobook generation from enhanced text
    
    Supports multiple TTS providers and audio processing features:
    - Multi-voice synthesis with semantic awareness
    - Chapter markers and navigation
    - Professional audio post-processing
    - Multiple output formats (M4B, MP3, WAV)
    """
    
    def __init__(self, config: Dict[str, Any], processing_mode: str = 'basic'):
        self.config = config
        self.processing_mode = processing_mode
        self.audio_config = config.get('audio', {})
        self.voice_config = config.get('voices', {})
        self.tts_providers = config.get('tts_providers', {})
        
        # Audio settings
        self.sample_rate = self.audio_config.get('sample_rate', 44100)
        self.channels = self.audio_config.get('channels', 2)
        self.output_format = self.audio_config.get('output_format', 'm4b')
        
        # Initialize TTS providers
        self._init_tts_providers()
        
        # Temporary directory for audio processing
        self.temp_dir = Path(tempfile.mkdtemp(prefix='md2audiobook_'))
    
    def _init_tts_providers(self):
        """Initialize available TTS providers based on configuration"""
        self.available_providers = {}
        
        # Piper TTS (Local)
        piper_config = self.tts_providers.get('piper', {})
        if piper_config.get('enabled', True):
            self.available_providers['piper'] = {
                'type': 'local',
                'config': piper_config
            }
        
        # ElevenLabs (API)
        elevenlabs_config = self.tts_providers.get('elevenlabs', {})
        if elevenlabs_config.get('enabled', False) and elevenlabs_config.get('api_key'):
            self.available_providers['elevenlabs'] = {
                'type': 'api',
                'config': elevenlabs_config
            }
        
        # Azure Cognitive Services (API)
        azure_config = self.tts_providers.get('azure', {})
        if azure_config.get('enabled', False) and azure_config.get('api_key'):
            self.available_providers['azure'] = {
                'type': 'api',
                'config': azure_config
            }
        
        # Google Cloud TTS (API)
        google_config = self.tts_providers.get('google', {})
        if google_config.get('enabled', False) and google_config.get('credentials_file'):
            self.available_providers['google'] = {
                'type': 'api',
                'config': google_config
            }
        
        # Hugging Face TTS (API)
        huggingface_config = self.tts_providers.get('huggingface', {})
        if huggingface_config.get('enabled', False) and os.getenv('HUGGINGFACE_API_KEY'):
            self.available_providers['huggingface'] = {
                'type': 'api',
                'config': huggingface_config
            }
        
        # OpenAI TTS (API)
        openai_config = self.tts_providers.get('openai', {})
        if openai_config.get('enabled', False) and os.getenv('OPENAI_API_KEY'):
            self.available_providers['openai'] = {
                'type': 'api',
                'config': openai_config
            }
    
    def generate_audiobook(self, enhanced_text: EnhancedText, 
                          metadata: Dict[str, Any], 
                          output_path: Path) -> AudiobookOutput:
        """
        Generate complete audiobook from enhanced text
        
        Args:
            enhanced_text: Enhanced text with voice assignments
            metadata: Audiobook metadata
            output_path: Output file path
            
        Returns:
            AudiobookOutput: Complete audiobook with chapters
        """
        print(f"Generating audiobook: {metadata.get('title', 'Untitled')}")
        
        # Split text into chapters
        chapter_texts = self._split_into_chapters(enhanced_text)
        
        # Generate audio for each chapter
        audio_chapters = []
        combined_audio = AudioSegment.empty()
        current_time = 0.0
        
        for i, chapter_text in enumerate(chapter_texts):
            print(f"Processing chapter {i + 1}/{len(chapter_texts)}")
            
            # Generate chapter audio
            chapter_audio = self._generate_chapter_audio(chapter_text, i + 1)
            
            # Create chapter metadata
            audio_chapter = AudioChapter(
                title=chapter_text.get('title', f'Chapter {i + 1}'),
                audio_segment=chapter_audio,
                start_time=current_time,
                duration=len(chapter_audio) / 1000.0,  # Convert ms to seconds
                chapter_number=i + 1
            )
            
            audio_chapters.append(audio_chapter)
            combined_audio += chapter_audio
            current_time += audio_chapter.duration
        
        # Apply post-processing
        if self.audio_config.get('post_processing', {}).get('enabled', True):
            combined_audio = self._apply_post_processing(combined_audio)
        
        # Create audiobook output
        audiobook = AudiobookOutput(
            audio=combined_audio,
            chapters=audio_chapters,
            metadata=metadata,
            file_path=output_path
        )
        
        # Export audiobook
        self._export_audiobook(audiobook, output_path)
        
        return audiobook
    
    def _split_into_chapters(self, enhanced_text: EnhancedText) -> List[Dict[str, Any]]:
        """Split enhanced text into chapters"""
        content = enhanced_text.content
        chapter_breaks = enhanced_text.chapter_breaks
        
        if not chapter_breaks:
            # No chapter breaks, treat as single chapter
            return [{
                'title': 'Chapter 1',
                'content': content,
                'voice_assignments': enhanced_text.voice_assignments
            }]
        
        chapters = []
        for i, break_pos in enumerate(chapter_breaks):
            # Determine chapter boundaries
            start_pos = break_pos
            end_pos = chapter_breaks[i + 1] if i + 1 < len(chapter_breaks) else len(content)
            
            chapter_content = content[start_pos:end_pos].strip()
            
            # Use original chapter title from enhanced_text if available
            if hasattr(enhanced_text, 'chapter_titles') and i < len(enhanced_text.chapter_titles):
                title = enhanced_text.chapter_titles[i]
            else:
                # Fallback: extract from content (legacy behavior)
                lines = chapter_content.split('\n')
                raw_title = lines[0] if lines else f'Chapter {i + 1}'
                title = raw_title.replace('Chapter: ', '').strip()
                title = re.sub(r'^#+\s*', '', title)
                if not title or len(title.strip()) < 3:
                    title = f'Chapter {i + 1}'
            
            # Use all content for this chapter (don't skip first line)
            content_text = chapter_content
            
            # Filter voice assignments for this chapter
            chapter_voice_assignments = {}
            for range_key, voice in enhanced_text.voice_assignments.items():
                range_start, range_end = map(int, range_key.split(':'))
                if start_pos <= range_start < end_pos:
                    # Adjust positions relative to chapter start
                    new_key = f"{range_start - start_pos}:{range_end - start_pos}"
                    chapter_voice_assignments[new_key] = voice
            
            chapters.append({
                'title': title,
                'content': content_text,
                'voice_assignments': chapter_voice_assignments
            })
        
        return chapters
    
    def _generate_chapter_audio(self, chapter_data: Dict[str, Any], 
                               chapter_number: int) -> AudioSegment:
        """Generate audio for a single chapter"""
        content = chapter_data['content']
        voice_assignments = chapter_data.get('voice_assignments', {})
        
        # Split content into segments based on voice assignments
        segments = self._split_content_by_voice(content, voice_assignments)
        
        chapter_audio = AudioSegment.empty()
        
        for segment in segments:
            # Generate audio for segment
            segment_audio = self._synthesize_text_segment(
                segment['text'], 
                segment['voice']
            )
            
            # Add pauses if specified
            if segment.get('pause_before', 0) > 0:
                pause = AudioSegment.silent(duration=int(segment['pause_before'] * 1000))
                chapter_audio += pause
            
            chapter_audio += segment_audio
            
            if segment.get('pause_after', 0) > 0:
                pause = AudioSegment.silent(duration=int(segment['pause_after'] * 1000))
                chapter_audio += pause
        
        return chapter_audio
    
    def _split_content_by_voice(self, content: str, 
                               voice_assignments: Dict[str, str]) -> List[Dict[str, Any]]:
        """Split content into segments by voice assignment"""
        if not voice_assignments:
            # No voice assignments, use default voice
            return [{
                'text': content,
                'voice': 'main_narrator',
                'pause_before': 0,
                'pause_after': 0
            }]
        
        segments = []
        current_pos = 0
        
        # Sort voice assignments by position
        sorted_assignments = sorted(
            voice_assignments.items(),
            key=lambda x: int(x[0].split(':')[0])
        )
        
        for range_key, voice in sorted_assignments:
            start_pos, end_pos = map(int, range_key.split(':'))
            
            # Add content before this assignment (if any)
            if current_pos < start_pos:
                segments.append({
                    'text': content[current_pos:start_pos],
                    'voice': 'main_narrator',
                    'pause_before': 0,
                    'pause_after': 0
                })
            
            # Add the assigned segment
            segment_text = content[start_pos:end_pos]
            pause_before = 2.0 if voice == 'chapter_voice' else 0
            pause_after = 1.0 if voice == 'chapter_voice' else 0
            
            segments.append({
                'text': segment_text,
                'voice': voice,
                'pause_before': pause_before,
                'pause_after': pause_after
            })
            
            current_pos = end_pos
        
        # Add remaining content (if any)
        if current_pos < len(content):
            segments.append({
                'text': content[current_pos:],
                'voice': 'main_narrator',
                'pause_before': 0,
                'pause_after': 0
            })
        
        return [seg for seg in segments if seg['text'].strip()]
    
    def _synthesize_text_segment(self, text: str, voice_type: str) -> AudioSegment:
        """Synthesize audio for a text segment using specified voice"""
        # Clean text for TTS
        clean_text = self._clean_text_for_tts(text)
        
        if not clean_text.strip():
            return AudioSegment.silent(duration=100)  # 100ms silence
        
        # Get voice configuration
        voice_config = self.voice_config.get(voice_type, self.voice_config.get('main_narrator', {}))
        provider = voice_config.get('provider', 'piper')
        
        # Generate audio based on provider
        if provider == 'piper':
            return self._synthesize_with_piper(clean_text, voice_config)
        elif provider == 'elevenlabs':
            return self._synthesize_with_elevenlabs(clean_text, voice_config)
        elif provider == 'azure':
            return self._synthesize_with_azure(clean_text, voice_config)
        elif provider == 'google':
            return self._synthesize_with_google(clean_text, voice_config)
        elif provider == 'huggingface':
            return self._synthesize_with_huggingface(clean_text, voice_config)
        elif provider == 'openai':
            return self._synthesize_with_openai(clean_text, voice_config)
        else:
            # Fallback to piper
            return self._synthesize_with_piper(clean_text, voice_config)
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Clean text for TTS processing"""
        # Remove markdown formatting
        clean_text = text
        
        # Remove emphasis markers
        clean_text = clean_text.replace('[EMPHASIS]', '')
        clean_text = clean_text.replace('[/EMPHASIS]', '')
        clean_text = clean_text.replace('[SLIGHT_EMPHASIS]', '')
        clean_text = clean_text.replace('[/SLIGHT_EMPHASIS]', '')
        
        # Remove math markers
        clean_text = clean_text.replace('[MATH BLOCK START]', '')
        clean_text = clean_text.replace('[MATH BLOCK END]', '')
        
        # Replace pause markers with periods
        clean_text = clean_text.replace('[PAUSE]', '.')
        
        # Clean up extra whitespace
        clean_text = ' '.join(clean_text.split())
        
        return clean_text
    
    def _synthesize_with_piper(self, text: str, voice_config: Dict[str, Any]) -> AudioSegment:
        """Synthesize audio using Piper TTS"""
        # Check if piper is available
        if not self._is_piper_available():
            if not hasattr(self, '_piper_warning_shown'):
                print("Warning: Piper TTS not found. Using espeak fallback.")
                self._piper_warning_shown = True
            return self._generate_fallback_audio(text)
            
        try:
            voice_id = voice_config.get('voice_id', 'en_US-lessac-medium')
            speed = voice_config.get('speed', 1.0)
            
            # Create temporary files
            text_file = self.temp_dir / f"text_{hash(text)}.txt"
            audio_file = self.temp_dir / f"audio_{hash(text)}.wav"
            
            # Write text to file
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)
            
            # Run Piper TTS
            cmd = [
                'piper',
                '--model', voice_id,
                '--output_file', str(audio_file),
                '--input_file', str(text_file)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and audio_file.exists():
                # Load audio
                audio = AudioSegment.from_wav(str(audio_file))
                
                # Clean up temporary files
                text_file.unlink(missing_ok=True)
                audio_file.unlink(missing_ok=True)
                
                return audio
            else:
                if not hasattr(self, '_piper_error_shown'):
                    print(f"Piper TTS failed, using fallback. Error: {result.stderr}")
                    self._piper_error_shown = True
                return self._generate_fallback_audio(text)
                
        except Exception as e:
            if not hasattr(self, '_piper_exception_shown'):
                print(f"Piper synthesis error, using fallback: {e}")
                self._piper_exception_shown = True
            return self._generate_fallback_audio(text)
    
    def _is_piper_available(self) -> bool:
        """Check if Piper TTS is available"""
        try:
            subprocess.run(['piper', '--version'], capture_output=True, text=True)
            return True
        except FileNotFoundError:
            return False
    
    def _synthesize_with_elevenlabs(self, text: str, voice_config: Dict[str, Any]) -> AudioSegment:
        """Synthesize audio using ElevenLabs API"""
        try:
            elevenlabs_config = self.available_providers['elevenlabs']['config']
            api_key = elevenlabs_config['api_key']
            voice_id = voice_config.get('voice_id', elevenlabs_config['default_voice'])
            
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
            
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": api_key
            }
            
            data = {
                "text": text,
                "model_id": elevenlabs_config.get('model', 'eleven_monolingual_v1'),
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.5,
                    "style": 0.0,
                    "use_speaker_boost": True
                }
            }
            
            response = requests.post(url, json=data, headers=headers, timeout=30)
            
            if response.status_code == 200:
                # Save audio to temporary file
                audio_file = self.temp_dir / f"elevenlabs_{hash(text)}.mp3"
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                
                # Load audio
                audio = AudioSegment.from_mp3(str(audio_file))
                
                # Clean up
                audio_file.unlink(missing_ok=True)
                
                return audio
            else:
                print(f"ElevenLabs API error: {response.status_code}")
                return self._generate_fallback_audio(text)
                
        except Exception as e:
            print(f"ElevenLabs synthesis error: {e}")
            return self._generate_fallback_audio(text)
    
    def _synthesize_with_azure(self, text: str, voice_config: Dict[str, Any]) -> AudioSegment:
        """Synthesize audio using Azure Cognitive Services"""
        # Implementation would go here
        # For now, fallback to Piper
        return self._synthesize_with_piper(text, voice_config)
    
    def _synthesize_with_google(self, text: str, voice_config: Dict[str, Any]) -> AudioSegment:
        """Synthesize audio using Google Cloud TTS"""
        try:
            from google.cloud import texttospeech
            import io
            
            # Get Google TTS configuration
            google_config = self.available_providers.get('google', {}).get('config', {})
            
            # Initialize the client
            client = texttospeech.TextToSpeechClient()
            
            # Set up the synthesis input
            synthesis_input = texttospeech.SynthesisInput(text=text)
            
            # Build the voice request
            voice = texttospeech.VoiceSelectionParams(
                language_code=voice_config.get('language_code', google_config.get('language_code', 'en-US')),
                name=voice_config.get('voice_id', google_config.get('default_voice', 'en-US-Wavenet-D'))
            )
            
            # Select the type of audio file
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=voice_config.get('speed', 1.0),
                pitch=voice_config.get('pitch', 0.0)
            )
            
            # Perform the text-to-speech request
            response = client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            
            # Convert response to AudioSegment
            audio_data = io.BytesIO(response.audio_content)
            audio_segment = AudioSegment.from_mp3(audio_data)
            
            print(f"✓ Google TTS: Generated {len(audio_segment)/1000:.1f}s audio")
            return audio_segment
            
        except ImportError:
            print("Warning: Google Cloud TTS library not installed. Install with: pip install google-cloud-texttospeech")
            return self._generate_fallback_audio(text)
        except Exception as e:
            print(f"Google TTS error: {e}")
            return self._generate_fallback_audio(text)
    
    def _synthesize_with_huggingface(self, text: str, voice_config: Dict[str, Any]) -> AudioSegment:
        """Synthesize audio using Hugging Face TTS API"""
        try:
            api_key = os.getenv('HUGGINGFACE_API_KEY')
            if not api_key:
                print("Warning: HUGGINGFACE_API_KEY not found. Using fallback.")
                return self._generate_fallback_audio(text)
            
            # Get model configuration
            model = voice_config.get('model', 'microsoft/speecht5_tts')
            api_url = f"https://api-inference.huggingface.co/models/{model}"
            
            # Prepare headers
            headers = {
                'Authorization': f'Bearer {api_key}',
                'Content-Type': 'application/json'
            }
            
            # Prepare payload
            payload = {
                'inputs': text,
                'parameters': {
                    'speaker_embeddings': voice_config.get('speaker_embeddings', None)
                }
            }
            
            # Remove None values from parameters
            payload['parameters'] = {k: v for k, v in payload['parameters'].items() if v is not None}
            if not payload['parameters']:
                del payload['parameters']
            
            # Make API request
            response = requests.post(api_url, headers=headers, json=payload, timeout=60)
            
            if response.status_code == 200:
                # Save audio data to temporary file
                audio_file = self.temp_dir / f"hf_audio_{hash(text)}.wav"
                with open(audio_file, 'wb') as f:
                    f.write(response.content)
                
                # Load and return audio
                audio = AudioSegment.from_wav(str(audio_file))
                
                # Clean up temporary file
                audio_file.unlink(missing_ok=True)
                
                return audio
            
            elif response.status_code == 503:
                print(f"Hugging Face model loading (this may take a moment)...")
                # Model is loading, wait and retry once
                import time
                time.sleep(10)
                
                response = requests.post(api_url, headers=headers, json=payload, timeout=60)
                if response.status_code == 200:
                    audio_file = self.temp_dir / f"hf_audio_{hash(text)}.wav"
                    with open(audio_file, 'wb') as f:
                        f.write(response.content)
                    
                    audio = AudioSegment.from_wav(str(audio_file))
                    audio_file.unlink(missing_ok=True)
                    return audio
                else:
                    print(f"Hugging Face TTS error after retry: {response.status_code} - {response.text}")
                    return self._generate_fallback_audio(text)
            
            else:
                print(f"Hugging Face TTS error: {response.status_code} - {response.text}")
                return self._generate_fallback_audio(text)
                
        except Exception as e:
            print(f"Hugging Face TTS synthesis error: {e}")
            return self._generate_fallback_audio(text)
    
    def _generate_fallback_audio(self, text: str) -> AudioSegment:
        """Generate fallback audio using espeak"""
        try:
            audio_file = self.temp_dir / f"fallback_{hash(text)}.wav"
            
            cmd = [
                'espeak',
                '-w', str(audio_file),
                '-s', '150',  # Speed
                '-v', 'en',   # Voice
                text
            ]
            
            result = subprocess.run(cmd, capture_output=True)
            
            if result.returncode == 0 and audio_file.exists():
                audio = AudioSegment.from_wav(str(audio_file))
                audio_file.unlink(missing_ok=True)
                return audio
            else:
                # Ultimate fallback: silence
                return AudioSegment.silent(duration=len(text) * 50)  # ~50ms per character
                
        except Exception:
            return AudioSegment.silent(duration=len(text) * 50)
    
    def _apply_post_processing(self, audio: AudioSegment) -> AudioSegment:
        """Apply audio post-processing"""
        processed_audio = audio
        
        post_config = self.audio_config.get('post_processing', {})
        
        # Normalize audio levels
        if post_config.get('normalize', True):
            processed_audio = processed_audio.normalize()
        
        # Trim silence
        if post_config.get('trim_silence', True):
            processed_audio = processed_audio.strip_silence(silence_thresh=-40)
        
        # Add fade in/out
        fade_duration = int(post_config.get('fade_duration', 0.5) * 1000)  # Convert to ms
        if fade_duration > 0:
            processed_audio = processed_audio.fade_in(fade_duration).fade_out(fade_duration)
        
        return processed_audio
    
    def _export_audiobook(self, audiobook: AudiobookOutput, output_path: Path):
        """Export audiobook to specified format"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if self.output_format == 'm4b':
            self._export_m4b(audiobook, output_path)
        elif self.output_format == 'mp3':
            self._export_mp3(audiobook, output_path)
        elif self.output_format == 'wav':
            self._export_wav(audiobook, output_path)
        else:
            # Default to M4B
            self._export_m4b(audiobook, output_path)
    
    def _export_m4b(self, audiobook: AudiobookOutput, output_path: Path):
        """Export as M4B audiobook format"""
        # First export as M4A
        m4a_path = output_path.with_suffix('.m4a')
        
        # Export audio
        bitrate = self.audio_config.get('bitrate', '128k')
        if not bitrate.endswith('k'):
            bitrate = f"{bitrate}k"
        
        audiobook.audio.export(
            str(m4a_path),
            format="mp4",
            bitrate=bitrate,
            parameters=["-c:a", "aac"]
        )
        
        # Add metadata and chapter markers
        self._add_m4b_metadata(m4a_path, audiobook)
        
        # Rename to .m4b
        final_path = output_path.with_suffix('.m4b')
        shutil.move(str(m4a_path), str(final_path))
        
        audiobook.file_path = final_path
        print(f"Audiobook exported: {final_path}")
    
    def _add_m4b_metadata(self, file_path: Path, audiobook: AudiobookOutput):
        """Add metadata and chapter markers to M4B file"""
        try:
            audiofile = MP4(str(file_path))
            metadata = audiobook.metadata
            
            # Basic metadata
            audiofile['\xa9nam'] = [metadata.get('title', 'Untitled')]
            audiofile['\xa9ART'] = [metadata.get('author', 'Unknown Author')]
            audiofile['\xa9alb'] = [metadata.get('title', 'Untitled')]
            audiofile['\xa9day'] = [str(metadata.get('date', ''))]
            audiofile['\xa9gen'] = [metadata.get('genre', 'Educational')]
            audiofile['\xa9cmt'] = [metadata.get('description', '')]
            
            # Audiobook-specific metadata
            audiofile['stik'] = [2]  # Audiobook
            
            # Chapter markers
            chapters = []
            for chapter in audiobook.chapters:
                chapters.append((
                    int(chapter.start_time * 1000),  # Start time in milliseconds
                    chapter.title
                ))
            
            if chapters:
                audiofile['----:com.apple.iTunes:chapters'] = [
                    self._create_chapter_data(chapters)
                ]
            
            audiofile.save()
            
        except Exception as e:
            print(f"Failed to add M4B metadata: {e}")
    
    def _create_chapter_data(self, chapters: List[Tuple[int, str]]) -> bytes:
        """Create chapter data for M4B format"""
        # This is a simplified implementation
        # A full implementation would create proper QuickTime chapter atoms
        chapter_data = b""
        for start_time, title in chapters:
            chapter_data += f"{start_time}:{title}\n".encode('utf-8')
        return chapter_data
    
    def _export_mp3(self, audiobook: AudiobookOutput, output_path: Path):
        """Export as MP3 format"""
        bitrate = self.audio_config.get('bitrate', '128k')
        if not bitrate.endswith('k'):
            bitrate = f"{bitrate}k"
            
        audiobook.audio.export(
            str(output_path.with_suffix('.mp3')),
            format="mp3",
            bitrate=bitrate
        )
        audiobook.file_path = output_path.with_suffix('.mp3')
    
    def _export_wav(self, audiobook: AudiobookOutput, output_path: Path):
        """Export as WAV format"""
        audiobook.audio.export(
            str(output_path.with_suffix('.wav')),
            format="wav"
        )
        audiobook.file_path = output_path.with_suffix('.wav')
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def __del__(self):
        """Cleanup on destruction"""
        self.cleanup()
