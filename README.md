# RAG Voice Agent

This project combines retrieval-augmented generation with a voice interface. It can record speech, transcribe it with STT, retrieve relevant context from Pinecone, answer with the LLM, and can speak the final response back through ElevenLabs TTS.

It can be used either from the terminal (`voice_agent.py`) or from a Streamlit web app (`streamlit_app.py`) with an upload tab and a record button.

## What It Does

- STT records your microphone input and transcribes it to text.
- RAG retrieves the most relevant chunks from your indexed documents.
- LLM generates an answer from the retrieved context.
- TTS turns the final answer into a voiceover using ElevenLabs.
- Console output stays visible so you can follow recording, transcription, retrieval, and answer generation step by step.
- nginx - certbot - systemmd
## Setup

1. Create and activate a Python environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If you want to upload images (not just PDFs) in the Streamlit app, install the Tesseract OCR binary separately — it's a system program, not a Python package:
   - Windows: https://github.com/UB-Mannheim/tesseract/wiki
   - macOS: `brew install tesseract`
   - Linux: `sudo apt install tesseract-ocr`
4. Set environment variables locally:
   ```bash
   GROQ_API_KEY=your_groq_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=rag-minimal
   PINECONE_CLOUD=aws
   PINECONE_REGION=us-east-1
   GROQ_CHAT_MODEL=llama-3.3-70b-versatile
   EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
   ELEVENLABS_API_KEY=your_elevenlabs_key
      ELEVENLABS_VOICE_ID=JBFqnCBsd6RMkjVDRZzb
      ELEVENLABS_MODEL_ID=eleven_multilingual_v2
      ELEVENLABS_OUTPUT_FORMAT=wav_44100_16
   ```

## Usage

Index documents:

```bash
python rag_pipeline.py index ./docs
```

Ask a question:

```bash
python rag_pipeline.py query "What does the policy say about refunds?"
```

Run the voice agent with microphone capture:

```bash
python voice_agent.py --duration 5
```

If you launch through `conda run`, use `--no-capture-output` so the countdown, start prompt, and recording logs appear live:

```bash
conda run --no-capture-output -n rag python -u voice_agent.py --duration 5
```

The voice agent now pauses and waits for Enter before recording starts, so you can get ready and then speak immediately after the "Recording now" message.

To speak the final answer with ElevenLabs voiceover, add `--tts`:

```bash
python voice_agent.py --duration 5 --tts
```

If you want a different ElevenLabs voice, set `ELEVENLABS_VOICE_ID` before running the command.

If the wrong audio source is selected, list devices and choose a real input device:

```bash
python voice_agent.py --list-devices
python voice_agent.py --duration 5 --countdown 3 --input-device 1
```

Use a microphone input for your voice. Use `Stereo Mix` only if you want to capture system audio coming from your speakers.

## Streamlit App

Instead of the terminal flow above, you can run everything from a browser:

```bash
streamlit run streamlit_app.py
```

This opens a two-tab interface:

1. **Upload tab** — upload PDF or image files. Each file is chunked and indexed into Pinecone (the same index used by the CLI's `index` command, under the `default` namespace), so it becomes searchable by the voice agent immediately. Image text is pulled out with OCR (Tesseract), so a photo of a page or a scanned document works the same as a real PDF, quality permitting.
2. **Ask the agent tab** — choose one of three input modes:
   - **Voice**: capture a question with Streamlit's built-in browser microphone control, then transcribe and answer on the server.
   - **Audio upload**: upload a WAV, MP3, M4A, OGG, WEBM, or FLAC file and let the server transcribe it.
   - **Text**: type a question directly and skip STT entirely.

All three modes share the same retrieval and answer pipeline, so this works well when the browser microphone is unavailable or blocked in a hosted AWS deployment.

## Output Flow

When you run the voice agent, the output is intended to feel like this:

1. The terminal prints: `Press Enter to start recording, then speak your question.`
2. After you press Enter, the terminal prints: `Recording now. Speak your question.`
3. Recording starts and stops after the configured duration.
4. Transcription appears in the terminal.
5. The RAG answer is printed.
6. TTS speaks the final answer if enabled.

If no speech is detected, the app prints a clear message and stops before querying RAG.

For debugging, you can print the retrieved chunks before the answer:

```bash
python rag_pipeline.py query "What does the paper say about hybrid RAG?" --show-context
```

## Supported files

- `.txt`
- `.md`
- `.pdf`
- `.docx`
- `.csv`
- `.png`, `.jpg`, `.jpeg` (Streamlit upload tab only, via OCR — requires the Tesseract binary, see Setup)
