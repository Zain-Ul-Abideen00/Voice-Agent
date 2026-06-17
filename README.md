# Test Voice Agent

A real-time voice assistant built on the OpenAI Agents SDK and LiteLLM voice tooling.

This project captures microphone audio, transcribes speech with Groq Whisper, performs reasoning with a chosen LLM, and speaks back responses using Deepgram Aura TTS.

## Features

- live microphone recording with silence-based phrase detection
- transcription via `groq/whisper-large-v3-turbo`
- reasoning with `openai-agents` and LiteLLM model integration
- text-to-speech output using Deepgram Aura
- configurable model selection via environment variables
- basic utility tool for current time lookup

## Requirements

- Python 3.14 or newer
- microphone and audio output device
- Windows-compatible `sounddevice` and PortAudio

## Installation

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install httpx numpy openai-agents[litellm,voice] python-dotenv sounddevice
   ```

## Configuration

Create a `.env` file in the project root and add your API keys. The application uses the following variables:

```env
GEMINI_API_KEY=your_gemini_api_key
GROQ_API_KEY=your_groq_api_key
DEEPGRAM_API_KEY=your_deepgram_api_key
OPENAI_API_KEY=your_openai_api_key
AGENT_MODEL=groq/openai/gpt-oss-120b
```

Notes:

- `AGENT_MODEL` defaults to `groq/openai/gpt-oss-120b`.
- If you choose a Gemini model, set `GEMINI_API_KEY`.
- If you choose a Groq model, set `GROQ_API_KEY`.
- For other models, set `OPENAI_API_KEY`.

## Run the Voice Agent

From the project directory, run:

```powershell
python main.py
```

The agent will list available audio devices and then start listening for speech. Speak into the microphone, wait for silence, and the agent will transcribe, reason, and speak its response.

## Troubleshooting

- If audio capture fails, verify your microphone and output device are connected and available.
- On Windows, you may need to install the appropriate PortAudio dependencies for `sounddevice`.
- Ensure `.env` file keys are correct and available to the process.

## Project Files

- `main.py` — voice-agent application entrypoint
- `pyproject.toml` — package metadata and dependencies
- `.gitignore` — ignores generated files, virtual envs, and editor artifacts
