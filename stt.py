"""Simple local STT helpers using faster-whisper.

Functions:
- `record_audio(output_path, duration, sample_rate)` : record a short WAV file
- `transcribe_audio(path, model_size, device, language)` : transcribe audio to text

This module keeps dependencies minimal: `faster-whisper`, `sounddevice`, `scipy`, `numpy`.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import sounddevice as sd
from scipy.io import wavfile
from scipy import signal

_WHISPER_MODELS: dict[str, object] = {}


def _resolve_input_device(input_device: int | None = None) -> int:
    """Return a PortAudio input device index with at least one input channel."""
    devices = sd.query_devices()

    if input_device is not None:
        device = devices[input_device]
        if device["max_input_channels"] < 1:
            raise RuntimeError(
                f"Selected device {input_device} ({device['name']}) has no input channels. "
                "Pick a microphone device instead."
            )
        return input_device

    default_input = sd.default.device[0]
    if default_input is not None:
        device = devices[default_input]
        if device["max_input_channels"] >= 1:
            return int(default_input)

    for index, device in enumerate(devices):
        if device["max_input_channels"] >= 1:
            return index

    raise RuntimeError("No input-capable audio devices were found.")


def record_audio(
    output_path: str | Path = "recording.wav",
    duration: float = 5.0,
    sample_rate: int = 16000,
    input_device: int | None = None,
) -> str:
    """Record `duration` seconds from the default input device and save as WAV.

    Returns the path to the written WAV file.
    """
    output_path = str(output_path)
    try:
        frames = int(duration * sample_rate)
        device_index = _resolve_input_device(input_device)
        device = sd.query_devices(device_index)
        print(f"Recording {duration}s ({frames} frames) at {sample_rate} Hz...")
        print(f"Using input device {device_index}: {device['name']} ({device['max_input_channels']} input channels)")
        recording = sd.rec(
            frames,
            samplerate=sample_rate,
            channels=1,
            dtype="int16",
            blocking=True,
            device=device_index,
        )
        wavfile.write(output_path, sample_rate, recording)
        peak = int(np.max(np.abs(recording))) if recording.size else 0
        rms = float(np.sqrt(np.mean(np.square(recording.astype(np.float32))))) if recording.size else 0.0
        print(f"Saved recording to {output_path}")
        print(f"Audio level: peak={peak} rms={rms:.1f}")
        return output_path
    except Exception as exc:
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"Recording failed: {exc}") from exc


def _normalize_wav(input_path: str, target_rate: int = 16000) -> str:
    """Ensure audio is mono and at `target_rate`. Returns path to normalized WAV.

    If no change is needed, returns the original path.
    """
    sr, data = wavfile.read(input_path)
    if data.size == 0:
        return input_path

    if data.ndim > 1:
        data = data.mean(axis=1)

    if sr != target_rate:
        num_samples = round(len(data) * float(target_rate) / sr)
        data = signal.resample(data, num_samples)

    if data.dtype != np.int16:
        if np.issubdtype(data.dtype, np.floating):
            max_val = float(np.max(np.abs(data))) or 1.0
            data = np.clip(data / max_val, -1.0, 1.0) * 32767
        data = np.clip(data, -32768, 32767).astype(np.int16)

    if sr == target_rate and data.ndim == 1 and data.dtype == np.int16:
        return input_path

    tmp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_file.close()
    wavfile.write(tmp_file.name, target_rate, data)
    return tmp_file.name


def transcribe_audio(path: str, model_size: str = "small", device: str = "cpu", language: Optional[str] = None) -> str:
    """Transcribe an audio file using `faster-whisper`.

    Returns the transcription text.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # pragma: no cover - runtime dependency
        raise RuntimeError("faster-whisper is required for transcription") from exc

    norm_path = _normalize_wav(path, target_rate=16000)

    key = f"{model_size}:{device}"
    model = _WHISPER_MODELS.get(key)
    if model is None:
        # On CPU, float32 avoids repeated float16 fallback warnings.
        compute_type = "float32" if device == "cpu" else "float16"
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        _WHISPER_MODELS[key] = model

    segments, _ = model.transcribe(
        norm_path,
        language=language,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
    ) if language else model.transcribe(
        norm_path,
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
    )
    text = " ".join((segment.text or "").strip() for segment in segments).strip()

    if norm_path != path:
        with contextlib.suppress(OSError):
            os.remove(norm_path)

    return text.strip()


def transcribe_audio_bytes(
    audio_bytes: bytes,
    model_size: str = "small",
    device: str = "cpu",
    language: Optional[str] = None,
    suffix: str = ".wav",
) -> str:
    """Write raw audio bytes (e.g. from Streamlit's st.audio_input) to a temp file and
    transcribe them. Streamlit captures audio in-browser, so there's no local mic to
    record from with `sounddevice` like the CLI flow uses -- this just adapts the bytes
    Streamlit already gives us into something `transcribe_audio` can read."""
    tmp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp_file.write(audio_bytes)
        tmp_file.flush()
        tmp_file.close()
        return transcribe_audio(tmp_file.name, model_size=model_size, device=device, language=language)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp_file.name)
