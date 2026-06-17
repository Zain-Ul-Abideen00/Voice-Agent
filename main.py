import os
import io
import sys
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

if sys.platform.startswith('win'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from agents import Agent, function_tool
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
from rag_handler import query_knowledge_base, initialize_knowledge_base
from models_config import get_reasoning_model, get_stt_model, get_tts_model

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
# Models are imported and managed from models_config.py
# =============================================================================


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
def record_phrase(samplerate=24000, threshold=400, silence_seconds=2.0, min_seconds=0.4) -> np.ndarray:
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
            # Wait for remaining audio in sounddevice buffer to finish playing
            await asyncio.sleep(output_stream.latency + 0.2)
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

    # Initialize the reasoning model dynamically from models_config
    llm_model = get_reasoning_model()

    # Initialize the RAG knowledge base
    initialize_knowledge_base()

    # Initialize the Agent with Bilingual (English/Urdu) System Instructions
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

    # Calibrate ambient noise and threshold
    threshold = calibrate_threshold(24000, 1.5)

    # Configure the pipeline (disable tracing to avoid OpenAI trace errors)
    workflow = SingleAgentVoiceWorkflow(agent=agent)
    config = VoicePipelineConfig(tracing_disabled=True)
    pipeline = VoicePipeline(
        workflow=workflow,
        stt_model=get_stt_model(),
        tts_model=get_tts_model(),
        config=config
    )

    print("\nVoice Agent is ready! Press Ctrl+C to exit.")
    
    # Run the voice conversation loop
    try:
        while True:
            # 1. Record until silence is detected using the calibrated threshold
            audio_data = await asyncio.to_thread(record_phrase, 24000, threshold, 2.0, 0.4)
            
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
