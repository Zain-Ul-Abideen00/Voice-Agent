import os
import io
import time
import asyncio
import traceback
import wave
from typing import AsyncIterator
import numpy as np
import sounddevice as sd
import httpx
import litellm
from dotenv import load_dotenv

from agents import Agent
from agents.extensions.models.litellm_model import LitellmModel
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
)
from agents.voice.result import StreamedAudioResult

# Import custom components from main.py
from main import GroqSTTModel, DeepgramTTSModel, get_current_time

# Load environment variables
load_dotenv()

# Verify API Keys
GEMINI_KEY = os.getenv("GEMINI_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY")


# =============================================================================
# Helper function to read a WAV file
# =============================================================================
def read_wav_file(filepath: str) -> tuple[np.ndarray, int, int]:
    """
    Reads a WAV file and returns the numpy int16 array, sample rate, and sample width.
    Handles mono/stereo conversion automatically.
    """
    print(f"Reading audio file: {filepath}")
    with wave.open(filepath, 'rb') as wf:
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        channels = wf.getnchannels()
        num_frames = wf.getnframes()
        
        raw_bytes = wf.readframes(num_frames)
        
        # Convert bytes to numpy int16
        if sample_width == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16)
        elif sample_width == 1:
            # 8-bit WAV is unsigned PCM, convert to signed 16-bit
            data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.int16) - 128) * 256
        else:
            raise ValueError(f"Unsupported sample width: {sample_width} bytes. Only 8-bit and 16-bit PCM WAV files are supported.")
        
        # Handle stereo conversion (average the channels)
        if channels == 2:
            print("Stereo audio detected. Converting to Mono...")
            data = data.reshape(-1, 2)
            data = data.mean(axis=1).astype(np.int16)
        elif channels > 2:
            raise ValueError(f"Unsupported number of channels: {channels}. Only Mono and Stereo are supported.")
            
        print(f"Loaded WAV: {len(data)} samples, Sample Rate = {sample_rate}Hz, Sample Width = {sample_width} bytes, Mono.")
        return data, sample_rate, sample_width


# =============================================================================
# Voice Turn Runner
# =============================================================================
async def run_voice_turn(pipeline: VoicePipeline, audio_data: np.ndarray, sample_rate: int, sample_width: int):
    # Create static AudioInput with custom sample rate and width matching the file
    audio_input = AudioInput(
        buffer=audio_data,
        frame_rate=sample_rate,
        sample_width=sample_width,
        channels=1
    )

    # Transcribe input audio to text
    print("\nTranscribing audio file...")
    try:
        input_text = await pipeline._process_audio_input(audio_input)
    except Exception as e:
        print(f"\n[STT Processing Error: {e}]")
        traceback.print_exc()
        return

    if not input_text.strip():
        print("No speech detected or transcription failed. Make sure the file has audible English speech.")
        return

    print(f"\nTranscribed User Speech: '{input_text}'")

    # Set up StreamedAudioResult to capture synthesized audio chunks
    output = StreamedAudioResult(pipeline._get_tts_model(), pipeline.config.tts_settings, pipeline.config)

    async def process_workflow():
        try:
            print("Agent response (text): ", end="", flush=True)
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
    print("Playing Agent response (audio)...")
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
    print("             WAV FILE VOICE AGENT TEST RUNNER                        ")
    print("=====================================================================")

    # Select the model dynamically
    model_name = os.getenv("AGENT_MODEL", "groq/openai/gpt-oss-120b")
    print(f"\nConfiguring reasoning model: {model_name}...")
    
    if "gemini" in model_name:
        api_key = GEMINI_KEY
    elif "groq" in model_name:
        api_key = GROQ_KEY
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

    # Configure the pipeline (disable tracing to avoid OpenAI trace errors)
    workflow = SingleAgentVoiceWorkflow(agent=agent)
    config = VoicePipelineConfig(tracing_disabled=True)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=GroqSTTModel(),
        tts_model=DeepgramTTSModel(),
        config=config
    )

    # Create test_audio folder if it doesn't exist
    folder_path = "test_audio"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"\nCreated folder: {folder_path}/")

    # Look for files in test_audio
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".wav")]

    if not files:
        print(f"\n[INFO] Please place a WAV audio file (e.g. 'input.wav') in the '{folder_path}' folder.")
        print(f"Once you place a file in '{folder_path}/', rerun this script.")
        return

    print(f"\nFound {len(files)} WAV files in '{folder_path}/':")
    for idx, f in enumerate(files):
        print(f" [{idx}] {f}")

    # Select the first file by default, or ask if multiple
    selected_file = files[0]
    filepath = os.path.join(folder_path, selected_file)

    try:
        # 1. Read the WAV file
        audio_data, sample_rate, sample_width = read_wav_file(filepath)
        
        # 2. Run the turn
        await run_voice_turn(pipeline, audio_data, sample_rate, sample_width)
        
    except Exception as e:
        print(f"\n[Test Runner Error: {e}]")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
