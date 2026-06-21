"""Simple voice agent that records or loads audio, transcribes, queries RAG, and prints the answer.

Usage examples:
  python voice_agent.py --duration 4
  python voice_agent.py --file example.wav

By default it prints the RAG answer. TTS can be added later (ElevenLabs) by implementing `speak_text`.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import tempfile
import sys
import time
from pathlib import Path

import stt
import rag_pipeline


def _configure_console() -> None:
    """Make stdout/stderr line-buffered so progress appears as it happens."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name)
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True, write_through=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple voice agent: record/transcribe -> RAG -> print")
    parser.add_argument("--file", help="Path to a WAV file to transcribe (skip recording)")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds (when not using --file)")
    parser.add_argument(
        "--countdown",
        type=int,
        default=0,
        help="Seconds to wait before recording starts. Use 0 to wait for Enter instead.",
    )
    parser.add_argument("--input-device", type=int, help="Optional sounddevice input device index")
    parser.add_argument("--model", default="small", help="Whisper model size (small, base, medium, etc.)")
    parser.add_argument("--device", default="cpu", help="Device for whisper (cpu, cuda)")
    parser.add_argument("--top-k", type=int, default=5, help="Number of context chunks to retrieve for RAG")
    parser.add_argument("--tts", action="store_true", help="(optional) speak the answer using ElevenLabs if configured")
    parser.add_argument("--list-devices", action="store_true", help="Show audio devices and exit")
    return parser.parse_args()


def synthesize_speech(text: str) -> bytes:
    """Call ElevenLabs TTS and return the raw audio bytes (WAV by default).

    Raises RuntimeError if no API key is configured or the request fails.
    Shared by both the CLI's speak_text (which plays audio locally) and the
    Streamlit app (which renders the bytes with st.audio).
    """
    api_key = os.getenv("ELEVENLABS_API_KEY") or os.getenv("ELEVEN_LABS_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVENLABS_API_KEY is not set; cannot synthesize speech.")

    voice_id = os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb")
    model_id = os.getenv("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")
    output_format = os.getenv("ELEVENLABS_OUTPUT_FORMAT", "wav_44100_16")

    import requests

    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
    params = {"output_format": output_format}
    payload = {
        "text": text,
        "model_id": model_id,
    }
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/wav",
    }

    response = requests.post(url, params=params, json=payload, headers=headers, timeout=60)
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise RuntimeError(f"ElevenLabs TTS failed: {response.status_code} {response.text}") from exc

    return response.content


def speak_text(text: str) -> None:
    """Speak text with ElevenLabs and play the returned WAV locally (CLI use only)."""
    try:
        audio_bytes = synthesize_speech(text)
    except RuntimeError as exc:
        print(f"{exc} Skipping TTS.", flush=True)
        return

    from scipy.io import wavfile
    import sounddevice as sd

    tmp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    try:
        tmp_audio.write(audio_bytes)
        tmp_audio.flush()
        tmp_audio.close()

        sample_rate, audio_data = wavfile.read(tmp_audio.name)
        sd.play(audio_data, sample_rate)
        sd.wait()
    finally:
        with contextlib.suppress(OSError):
            Path(tmp_audio.name).unlink()


def main() -> None:
    _configure_console()
    args = parse_args()
    cleanup_audio = False

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    if args.input_device is not None:
        import sounddevice as sd

        sd.default.device = (args.input_device, None)

    if args.file:
        audio_path = args.file
        if not Path(audio_path).exists():
            raise SystemExit(f"Audio file not found: {audio_path}")
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        if args.countdown > 0:
            prompt = f"Get ready. Recording starts in {args.countdown} seconds."
            print(prompt, flush=True)
            if args.tts:
                speak_text(prompt)
            for remaining in range(args.countdown, 0, -1):
                print(f"Starting in {remaining}...", flush=True)
                if args.tts:
                    speak_text(f"Starting in {remaining}")
                time.sleep(1)
            print("Recording now. Speak your question.", flush=True)
            if args.tts:
                speak_text("Recording now. Speak your question.")
        else:
            print("Press Enter to start recording, then speak your question.", flush=True)
            if args.tts:
                speak_text("Press Enter to start recording, then speak your question.")
            input()
            print("Recording now. Speak your question.", flush=True)
            if args.tts:
                speak_text("Recording now. Speak your question.")
        audio_path = stt.record_audio(tmp.name, duration=args.duration)
        cleanup_audio = True

    try:
        print("Transcribing...")
        if args.tts:
            speak_text("Transcribing.")
        transcript = stt.transcribe_audio(audio_path, model_size=args.model, device=args.device)
        print("Transcript:")
        print(transcript or "(no speech detected)")

        if not transcript:
            print("No speech was detected. Try a longer duration, a quieter room, or a different input device.", flush=True)
            return

        print("Querying RAG...")
        answer = rag_pipeline.answer_query(transcript, top_k=args.top_k)
        print("Answer:")
        print(answer)

        if args.tts:
            speak_text(answer)
    finally:
        if cleanup_audio:
            with contextlib.suppress(OSError):
                Path(audio_path).unlink()


if __name__ == "__main__":
    main()
