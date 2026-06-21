"""Streamlit frontend for the RAG Voice Agent.

Two tabs:
  1. Upload   - upload PDFs/images, chunk them, and index them into Pinecone.
  2. Voice Agent - record a question with the browser mic, transcribe it,
     retrieve context from Pinecone, answer with the LLM, and speak the
     answer back using ElevenLabs.

Run with:
    streamlit run streamlit_app.py
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

import rag_pipeline
import stt
import voice_agent

st.set_page_config(page_title="RAG Voice Agent", page_icon="🎙️", layout="wide")

NAMESPACE = "default"  # keep in sync with the CLI's default namespace


def _save_upload_to_temp(uploaded_file) -> Path:
    """Write a Streamlit UploadedFile to a temp path on disk with the original
    suffix preserved, since rag_pipeline reads files by extension."""
    suffix = Path(uploaded_file.name).suffix.lower()
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(uploaded_file.getbuffer())
    tmp.close()
    return Path(tmp.name)


def render_upload_tab() -> None:
    st.header("Upload documents")
    st.write(
        "Upload PDFs or images. Each file is chunked and stored in Pinecone, "
        "so the voice agent can retrieve relevant passages when you ask a question."
    )

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = []

    uploaded_files = st.file_uploader(
        "Choose PDF or image files",
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
    )

    if st.button("Upload and index", type="primary", disabled=not uploaded_files):
        progress = st.progress(0.0, text="Starting...")
        total = len(uploaded_files)
        results = []

        for i, uploaded_file in enumerate(uploaded_files):
            progress.progress(i / total, text=f"Indexing {uploaded_file.name}...")
            tmp_path = _save_upload_to_temp(uploaded_file)
            try:
                chunk_count = rag_pipeline.index_uploaded_file(
                    tmp_path, source_name=uploaded_file.name, namespace=NAMESPACE
                )
                results.append((uploaded_file.name, chunk_count, None))
            except Exception as exc:  # surface the error per-file instead of failing the whole batch
                results.append((uploaded_file.name, 0, str(exc)))
            finally:
                tmp_path.unlink(missing_ok=True)

        progress.progress(1.0, text="Done.")

        for name, chunk_count, error in results:
            if error:
                st.error(f"**{name}** failed: {error}")
            else:
                st.success(f"**{name}** indexed ({chunk_count} chunks).")
                st.session_state.indexed_files.append(name)

    if st.session_state.indexed_files:
        st.divider()
        st.subheader("Indexed this session")
        for name in st.session_state.indexed_files:
            st.write(f"- {name}")

    with st.expander("Notes"):
        st.markdown(
            "- Image text is extracted with OCR (Tesseract), so scanned text and "
            "photos of documents work, but results depend on image quality.\n"
            "- Image OCR requires the Tesseract binary installed on this machine "
            "(separate from the `pytesseract` pip package) — see the README.\n"
            "- Re-uploading a file with the same name will add new vectors rather "
            "than overwrite the old ones unless the chunk IDs match exactly."
        )


def render_voice_tab() -> None:
    st.header("Ask the agent")
    st.write(
        "Record your question, and the agent will transcribe it, retrieve the most "
        "relevant chunks from your uploaded documents, and answer with the LLM."
    )

    col1, col2 = st.columns(2)
    with col1:
        top_k = st.slider("Chunks to retrieve", min_value=1, max_value=10, value=5)
    with col2:
        use_tts = st.checkbox("Speak the answer (ElevenLabs)", value=True)

    audio_value = st.audio_input("Click to record your question")

    if audio_value is not None:
        st.audio(audio_value)

        if st.button("Transcribe and answer", type="primary"):
            with st.spinner("Transcribing..."):
                try:
                    transcript = stt.transcribe_audio_bytes(audio_value.getvalue())
                except Exception as exc:
                    st.error(f"Transcription failed: {exc}")
                    return

            if not transcript:
                st.warning(
                    "No speech was detected. Try recording again, closer to the mic "
                    "or in a quieter room."
                )
                return

            st.markdown(f"**You asked:** {transcript}")

            with st.spinner("Retrieving context and generating an answer..."):
                try:
                    contexts = rag_pipeline.retrieve_context(
                        transcript, top_k=top_k, namespace=NAMESPACE
                    )
                    answer = rag_pipeline.answer_query(
                        transcript, top_k=top_k, namespace=NAMESPACE, contexts=contexts
                    )
                except Exception as exc:
                    st.error(f"Could not get an answer: {exc}")
                    return

            st.markdown("**Answer:**")
            st.write(answer)

            with st.expander("Retrieved chunks (for debugging)"):
                if contexts:
                    for item in contexts:
                        st.caption(f"{item['source']} · chunk {item['chunk_index']} · score {item['score']:.4f}")
                        st.text(item["text"])
                else:
                    st.write("No chunks were retrieved. Have you indexed any documents yet?")

            if use_tts:
                with st.spinner("Generating voice answer..."):
                    try:
                        audio_bytes = voice_agent.synthesize_speech(answer)
                        st.audio(audio_bytes, format="audio/wav")
                    except RuntimeError as exc:
                        st.warning(str(exc))


def main() -> None:
    st.title("🎙️ RAG Voice Agent")
    upload_tab, voice_tab = st.tabs(["📤 Upload", "🎤 Voice Agent"])

    with upload_tab:
        render_upload_tab()

    with voice_tab:
        render_voice_tab()


if __name__ == "__main__":
    main()
