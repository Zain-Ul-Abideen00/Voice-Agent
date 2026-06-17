import os
import sys
import httpx
import asyncio
import traceback
import litellm
from typing import AsyncIterator

from agents.extensions.models.litellm_model import LitellmModel
from agents.voice import (
    STTModel,
    STTModelSettings,
    AudioInput,
    StreamedAudioInput,
    StreamedTranscriptionSession,
    TTSModel,
    TTSModelSettings,
)

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# =============================================================================
# 1. Speech-to-Text (STT) Models
# =============================================================================

class GroqSTTModel(STTModel):
    def __init__(self, model_name: str = "groq/whisper-large-v3-turbo", api_key: str | None = None):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        filename, file_stream, content_type = input.to_audio_file()
        try:
            response = await asyncio.to_thread(
                litellm.transcription,
                model=self._model_name,
                file=(filename, file_stream, content_type),
                api_key=self.api_key
            )
            return response.get("text", "")
        except Exception as e:
            print(f"\n[Groq STT Error: {e}]")
            traceback.print_exc()
            return ""

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        raise NotImplementedError("Streaming session is not supported by Groq Whisper REST API.")


class ElevenLabsSTTModel(STTModel):
    def __init__(self, model_name: str = "elevenlabs/scribe_v2", api_key: str | None = None):
        self._model_name = model_name
        # Try both ELEVEN_LABS_API_KEY and ELEVENLABS_API_KEY
        self.api_key = api_key or os.getenv("ELEVEN_LABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def transcribe(
        self,
        input: AudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> str:
        filename, file_stream, content_type = input.to_audio_file()
        try:
            response = await asyncio.to_thread(
                litellm.transcription,
                model=self._model_name,
                file=(filename, file_stream, content_type),
                api_key=self.api_key
            )
            return response.get("text", "")
        except Exception as e:
            print(f"\n[ElevenLabs STT Error: {e}]")
            traceback.print_exc()
            return ""

    async def create_session(
        self,
        input: StreamedAudioInput,
        settings: STTModelSettings,
        trace_include_sensitive_data: bool,
        trace_include_sensitive_audio_data: bool,
    ) -> StreamedTranscriptionSession:
        raise NotImplementedError("Streaming session is not supported by ElevenLabs STT REST API.")


# =============================================================================
# 2. Text-to-Speech (TTS) Models
# =============================================================================

class DeepgramTTSModel(TTSModel):
    def __init__(self, model_name: str = "aura-2-asteria-en", api_key: str | None = None):
        self._model_name = model_name
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        if not text.strip():
            return

        # Deepgram TTS Speak endpoint requesting raw PCM at 24kHz
        url = f"https://api-alt.sac1.deepgram.com/v1/speak?model={self._model_name}&encoding=linear16&container=none&sample_rate=24000"
        headers = {
            "Authorization": f"Token {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {"text": text}

        try:
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, json=data, timeout=10.0) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        print(f"\n[Deepgram TTS API Error (Status {response.status_code}): {error_body.decode('utf-8', errors='ignore')}]")
                        response.raise_for_status()
                    async for chunk in response.aiter_bytes(chunk_size=2048):
                        yield chunk
        except Exception as e:
            print(f"\n[TTS Deepgram Generation Error: {e}]")
            traceback.print_exc()


class ElevenLabsTTSModel(TTSModel):
    _semaphore = None

    def __init__(
        self,
        model_name: str = "eleven_multilingual_v2",
        voice_id: str = "cgSgspJ2msm6clMCkdW9",  # Jessica (default free-tier friendly premade voice)
        api_key: str | None = None
    ):
        self._model_name = model_name
        self.voice_id = voice_id
        self.api_key = api_key or os.getenv("ELEVEN_LABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY")

    @property
    def model_name(self) -> str:
        return self._model_name

    async def run(self, text: str, settings: TTSModelSettings) -> AsyncIterator[bytes]:
        if not text.strip():
            return

        if ElevenLabsTTSModel._semaphore is None:
            ElevenLabsTTSModel._semaphore = asyncio.Semaphore(1)

        async with ElevenLabsTTSModel._semaphore:
            # Request raw PCM at 24kHz from ElevenLabs stream
            url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream?output_format=pcm_24000"
            headers = {
                "xi-api-key": self.api_key,
                "Content-Type": "application/json"
            }
            data = {
                "text": text,
                "model_id": self._model_name,
                "voice_settings": {
                    "stability": 0.5,
                    "similarity_boost": 0.75
                }
            }

            try:
                async with httpx.AsyncClient() as client:
                    async with client.stream("POST", url, headers=headers, json=data, timeout=15.0) as response:
                        if response.status_code != 200:
                            error_body = await response.aread()
                            print(f"\n[ElevenLabs TTS API Error (Status {response.status_code}): {error_body.decode('utf-8', errors='ignore')}]")
                            response.raise_for_status()
                        async for chunk in response.aiter_bytes(chunk_size=2048):
                            yield chunk
            except Exception as e:
                print(f"\n[TTS ElevenLabs Generation Error: {e}]")
                traceback.print_exc()


# =============================================================================
# 3. Model Helpers & Configuration Managers
# =============================================================================

def get_reasoning_model() -> LitellmModel:
    """Gets and configures the reasoning model based on environment variables."""
    model_name = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
    
    if "gemini" in model_name:
        api_key = os.getenv("GEMINI_API_KEY")
    elif "groq" in model_name:
        api_key = os.getenv("GROQ_API_KEY")
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        
    return LitellmModel(
        model=model_name,
        api_key=api_key
    )

def get_stt_model() -> STTModel:
    """Gets and configures the Speech-to-Text model based on environment variables."""
    has_eleven_key = bool(os.getenv("ELEVEN_LABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"))
    default_stt = "elevenlabs" if has_eleven_key else "groq"
    stt_provider = os.getenv("STT_PROVIDER", default_stt).lower()
    
    if stt_provider == "elevenlabs":
        stt_model_name = os.getenv("ELEVEN_LABS_STT_MODEL", "elevenlabs/scribe_v2")
        print(f"[Models] Using ElevenLabs STT ({stt_model_name})...")
        return ElevenLabsSTTModel(model_name=stt_model_name)
    else:
        stt_model_name = os.getenv("GROQ_STT_MODEL", "groq/whisper-large-v3-turbo")
        print(f"[Models] Using Groq STT ({stt_model_name})...")
        return GroqSTTModel(model_name=stt_model_name)

def get_tts_model() -> TTSModel:
    """Gets and configures the Text-to-Speech model based on environment variables."""
    has_eleven_key = bool(os.getenv("ELEVEN_LABS_API_KEY") or os.getenv("ELEVENLABS_API_KEY"))
    default_tts = "elevenlabs" if has_eleven_key else "deepgram"
    tts_provider = os.getenv("TTS_PROVIDER", default_tts).lower()
    
    if tts_provider == "elevenlabs":
        tts_model_name = os.getenv("ELEVEN_LABS_TTS_MODEL", "eleven_multilingual_v2")
        # Default voice: Sarah (EXAVITQu4vr4xnSDxMaL)
        voice_id = os.getenv("ELEVEN_LABS_VOICE_ID", "cgSgspJ2msm6clMCkdW9")
        print(f"[Models] Using ElevenLabs TTS ({tts_model_name}, voice_id={voice_id})...")
        return ElevenLabsTTSModel(model_name=tts_model_name, voice_id=voice_id)
    else:
        tts_model_name = os.getenv("DEEPGRAM_TTS_MODEL", "aura-2-asteria-en")
        print(f"[Models] Using Deepgram TTS ({tts_model_name})...")
        return DeepgramTTSModel(model_name=tts_model_name)
