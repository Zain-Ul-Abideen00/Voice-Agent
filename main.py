import os
import io
import time
import queue
import asyncio
import traceback
from typing import AsyncIterator
import numpy as np
import sounddevice as sd
import httpx
import litellm
from dotenv import load_dotenv

from agents import Agent, function_tool
from agents.extensions.models.litellm_model import LitellmModel
from agents.voice import (
    STTModel,
    STTModelSettings,
    AudioInput,
    StreamedAudioInput,
    StreamedTranscriptionSession,
    TTSModel,
    TTSModelSettings,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
)
from agents.voice.result import StreamedAudioResult

# Load environment variables
load_dotenv()

# Verify API Keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")

if not GEMINI_KEY:
    print("[WARNING] GEMINI_API_KEY is not set in environment!")
if not GROQ_KEY:
    print("[WARNING] GROQ_API_KEY is not set in environment!")
if not DEEPGRAM_KEY:
    print("[WARNING] DEEPGRAM_API_KEY is not set in environment!")


# =============================================================================
# Simple Time Check Tool
# =============================================================================
@function_tool
def get_current_time(timezone: str = "local") -> str:
    """Get the current time in the specified timezone.

    Args:
        timezone: The timezone to get the time for, e.g. 'local' or 'EST'. Defaults to 'local'.
    """
    from datetime import datetime
    current_time = datetime.now().strftime("%I:%M %p")
    print(f"\n[Tool Execution: get_current_time(timezone={timezone!r}) -> {current_time}]")
    return current_time


# =============================================================================
# Custom Speech-to-Text Model (Groq Whisper-large-v3-turbo)
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
        # Get filename, file stream, and content type from AudioInput
        filename, file_stream, content_type = input.to_audio_file()
        
        # Run LiteLLM transcription in a separate thread to keep it async
        try:
            response = await asyncio.to_thread(
                litellm.transcription,
                model=self._model_name,
                file=(filename, file_stream, content_type),
                api_key=self.api_key
            )
            return response.get("text", "")
        except Exception as e:
            print(f"\n[STT Transcription Client Error: {e}]")
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


# =============================================================================
# Custom Text-to-Speech Model (Deepgram Aura v2)
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


# =============================================================================
# Audio Calibration
# =============================================================================
def calibrate_threshold(samplerate=24000, duration_seconds=1.5) -> int:
    """
    Listens to ambient room noise for 1.5 seconds and calculates a dynamic
    silence threshold.
    """
    print("\nCalibrating microphone... Please remain silent.")
    import queue
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        q.put(indata.copy())

    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        callback=callback
    )

    rms_values = []
    start_time = time.time()

    with stream:
        # Wait a bit for device startup noise
        time.sleep(0.3)
        while time.time() - start_time < duration_seconds:
            try:
                block = q.get(timeout=0.1)
                rms = np.sqrt(np.mean(block.astype(np.float64)**2))
                rms_values.append(rms)
            except queue.Empty:
                continue

    if rms_values:
        avg_rms = np.mean(rms_values)
        max_rms = np.max(rms_values)
        # Recommended threshold: 50% above maximum observed noise
        # Lower bound to 300 to avoid trigger loops from low static hum
        recommended_threshold = int(max(max_rms * 1.5, avg_rms * 2.0, 300.0))
        print(f"Calibration complete: Avg Ambient RMS = {avg_rms:.0f}, Max Ambient RMS = {max_rms:.0f}.")
        print(f"Calibrated Silence Threshold set to: {recommended_threshold}")
        return recommended_threshold
    else:
        print("Calibration failed. Using default threshold of 450.")
        return 450


# =============================================================================
# Audio Capture and Volume Threshold Silence Detection
# =============================================================================
def record_phrase(samplerate=24000, threshold=400, silence_seconds=1.2, min_seconds=0.4) -> np.ndarray:
    """
    Captures mic input and returns a numpy array of int16 samples when silence is detected.
    Automatically handles speech threshold detection and duration check.
    """
    import queue
    q = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, flush=True)
        q.put(indata.copy())

    # Start sounddevice input stream
    stream = sd.InputStream(
        samplerate=samplerate,
        channels=1,
        dtype='int16',
        callback=callback
    )

    recorded_blocks = []
    speaking = False
    silence_start_time = None
    speech_start_time = None

    print(f"\nListening... (Threshold: {threshold}. Start speaking now!)")
    with stream:
        # Sleep for 300ms to discard transient initialization clicks/pops
        time.sleep(0.3)
        # Drain queue to start clean
        while not q.empty():
            q.get_nowait()

        while True:
            try:
                # Retrieve audio block from queue
                block = q.get(timeout=0.5)
            except queue.Empty:
                if speaking and silence_start_time:
                    if (time.time() - silence_start_time) >= silence_seconds:
                        break
                continue

            # Calculate Root Mean Square (RMS) for voice activity detection
            rms = np.sqrt(np.mean(block.astype(np.float64)**2))

            if not speaking:
                print(f"Volume: {rms:5.0f} / Threshold: {threshold}  ", end="\r", flush=True)
                if rms > threshold:
                    speaking = True
                    speech_start_time = time.time()
                    recorded_blocks.append(block)
                    silence_start_time = None
                    print(f"\n[Speech detected! (volume: {rms:.0f}) Recording...]")
            else:
                recorded_blocks.append(block)
                if rms < threshold:
                    if silence_start_time is None:
                        silence_start_time = time.time()
                    elif (time.time() - silence_start_time) >= silence_seconds:
                        duration = time.time() - speech_start_time
                        if duration >= min_seconds:
                            print("[Silence detected. Stop recording.]")
                            break
                        else:
                            # Phrase was too short, reset
                            speaking = False
                            recorded_blocks = []
                            silence_start_time = None
                            print("\n[Resetting: Phrase was too short]")
                else:
                    silence_start_time = None

    if recorded_blocks:
        return np.concatenate(recorded_blocks, axis=0).flatten()
    return np.array([], dtype=np.int16)


# =============================================================================
# Custom Voice Turn Runner (prints agent text and plays audio chunks)
# =============================================================================
async def run_voice_turn(pipeline: VoicePipeline, audio_data: np.ndarray):
    # Create static AudioInput
    audio_input = AudioInput(buffer=audio_data)

    # Transcribe input audio to text
    print("\nTranscribing...")
    try:
        input_text = await pipeline._process_audio_input(audio_input)
    except Exception as e:
        print(f"\n[STT Processing Error: {e}]")
        traceback.print_exc()
        return

    if not input_text.strip():
        print("No speech detected or transcription failed.")
        return

    print(f"You: {input_text}")

    # Set up StreamedAudioResult to capture synthesized audio chunks
    output = StreamedAudioResult(pipeline._get_tts_model(), pipeline.config.tts_settings, pipeline.config)

    async def process_workflow():
        try:
            print("Agent: ", end="", flush=True)
            # Execute agent workflow and yield text chunks
            async for text_event in pipeline.workflow.run(input_text):
                print(text_event, end="", flush=True)
                await output._add_text(text_event)
            print()
            await output._turn_done()
            await output._done()
        except Exception as e:
            print(f"\n[Workflow Run Error: {e}]")
            traceback.print_exc()
            await output._add_error(e)
            await output._done()

    # Start workflow and TTS generation task in the background
    output._set_task(asyncio.create_task(process_workflow()))

    # Concurrently play the output audio chunks as they are generated
    try:
        output_stream = sd.RawOutputStream(
            samplerate=24000,
            channels=1,
            dtype='int16'
        )
        with output_stream:
            async for event in output.stream():
                if event.type == "voice_stream_event_audio":
                    output_stream.write(event.data.tobytes())
                elif event.type == "voice_stream_event_error":
                    print(f"\n[TTS Stream Event Error: {event.error}]")
    except Exception as e:
        print(f"\n[Audio Playback Error: {e}]")


# =============================================================================
# Main Application Loop
# =============================================================================
async def main():
    print("=====================================================================")
    print("             REAL-TIME VOICE AGENT USING OPENAI AGENTS SDK           ")
    print("=====================================================================")

    # Display audio devices for debugging
    print("\nAvailable Audio Devices:")
    devices = sd.query_devices()
    print(devices)
    
    try:
        default_input = sd.query_devices(kind='input')
        default_output = sd.query_devices(kind='output')
        print(f"\nDefault Input Device: {default_input['name']}")
        print(f"Default Output Device: {default_output['name']}")
    except Exception as e:
        print(f"\n[Warning: Could not fetch default audio devices: {e}]")

    # Select the model dynamically
    model_name = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
    print(f"\nConfiguring reasoning model: {model_name}...")
    
    if "gemini" in model_name:
        api_key = GEMINI_KEY
        if not api_key:
            print("[ERROR] GEMINI_API_KEY is required but not set in environment.")
            return
    elif "groq" in model_name:
        api_key = GROQ_KEY
        if not api_key:
            print("[ERROR] GROQ_API_KEY is required but not set in environment.")
            return
    else:
        api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the reasoning model
    llm_model = LitellmModel(
        model=model_name,
        api_key=api_key
    )

    # Initialize the Agent
    agent = Agent(
        name="Voice Assistant",
        instructions="You are a helpful, brief, and concise voice assistant. Keep your responses short (1-2 sentences) since they will be read aloud.",
        model=llm_model,
        tools=[get_current_time]
    )

    # Calibrate ambient noise and threshold
    threshold = calibrate_threshold(24000, 1.5)

    # Configure the pipeline (disable tracing to avoid OpenAI trace errors)
    workflow = SingleAgentVoiceWorkflow(agent=agent)
    config = VoicePipelineConfig(tracing_disabled=True)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=GroqSTTModel(),
        tts_model=DeepgramTTSModel(),
        config=config
    )

    print("\nVoice Agent is ready! Press Ctrl+C to exit.")
    
    # Run the voice conversation loop
    try:
        while True:
            # 1. Record until silence is detected using the calibrated threshold
            audio_data = await asyncio.to_thread(record_phrase, 24000, threshold, 1.2, 0.4)
            
            if audio_data.size == 0:
                print("No audio captured.")
                continue

            # 2. Run the voice turn (transcribe, reason, speak)
            await run_voice_turn(pipeline, audio_data)
            
            # Simple pause before listening again
            await asyncio.sleep(0.5)

    except KeyboardInterrupt:
        print("\nExiting Voice Agent. Goodbye!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
