import os
import io
import sys
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

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from agents import Agent
from agents.voice import (
    AudioInput,
    SingleAgentVoiceWorkflow,
    VoicePipeline,
    VoicePipelineConfig,
)
from agents.voice.result import StreamedAudioResult

# Import custom components from main.py and models_config.py
from main import get_current_time
from rag_handler import query_knowledge_base, initialize_knowledge_base
from models_config import get_reasoning_model, get_stt_model, get_tts_model

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
            # Wait for remaining audio in sounddevice buffer to finish playing
            await asyncio.sleep(output_stream.latency + 0.2)
    except Exception as e:
        print(f"\n[Audio Playback Error: {e}]")


# =============================================================================
# Main Application Loop
# =============================================================================
async def main():
    print("=====================================================================")
    print("             WAV FILE VOICE AGENT TEST RUNNER                        ")
    print("=====================================================================")

    # Initialize the reasoning model dynamically from models_config
    llm_model = get_reasoning_model()

    # Initialize the RAG knowledge base
    initialize_knowledge_base()

    # Initialize the Agent with Urdu System Instructions
    agent = Agent(
        name="Valeria",
        instructions=(
            "You are Valeria, a warm, professional, and conversion-focused inbound call agent for Digital Graphiks, "
            "a leading custom website design, branding, and digital marketing agency based in the UAE (with over 15 years of experience).\n\n"
            "BILINGUAL LANGUAGE RULES:\n"
            "- Standard/First Priority Language: English. If the user initiates or speaks in English, respond in English.\n"
            "- Urdu Language Support: If the user speaks in Urdu (either in Urdu script or Roman Urdu), you MUST respond in Urdu (using standard Urdu script).\n"
            "- Always match the language the user speaks to you in.\n\n"
            "CRITICAL VOICE CALL RULES:\n"
            "- Always keep your responses very short and natural for a voice conversation (1 to 2 sentences, maximum 3).\n"
            "- Never output bullet points, markdown list markers, or asterisks (*). Present options or steps as running text.\n"
            "- NEVER say things like 'according to the document', 'based on the PDF', or 'in the knowledge base'. You are Valeria; "
            "speak in the first-person ('we', 'our team', 'at Digital Graphiks').\n"
            "- If the user asks about specific pricing, services (custom websites, branding, SEO, e-commerce, LMS, AI solutions), "
            "objections, or timelines, use the 'query_knowledge_base' tool to lookup the details. Do not guess.\n"
            "- Ballpark pricing guide:\n"
            "  * Basic Website starts around AED 2,000.\n"
            "  * Logo Design starts from AED 1,800.\n"
            "  * SEO Plans typically start from AED 3,500/month.\n"
            "  * Basic E-commerce Store (up to 20 products) starts around AED 7,500.\n"
            "  * Basic LMS starts at AED 15,000.\n"
            "  * Simple AI Tools (like chatbots) start from AED 15,000.\n"
            "- Your primary goal is to guide the user to schedule a discovery/strategy meeting (online or at the Dubai office). "
            "If they show interest, offer to schedule a meeting and send them the company profile and details."
        ),
        model=llm_model,
        tools=[get_current_time, query_knowledge_base]
    )

    # Configure the pipeline (disable tracing to avoid OpenAI trace errors)
    workflow = SingleAgentVoiceWorkflow(agent=agent)
    config = VoicePipelineConfig(tracing_disabled=True)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=get_stt_model(),
        tts_model=get_tts_model(),
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
