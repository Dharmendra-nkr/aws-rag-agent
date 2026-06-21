from __future__ import annotations

import argparse
import csv
import os
from urllib3.exceptions import MaxRetryError
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from groq import Groq
from pinecone import Pinecone, ServerlessSpec

try:
    from docx import Document as DocxDocument
except ImportError:
    DocxDocument = None

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from PIL import Image
    import pytesseract
except ImportError:
    Image = None
    pytesseract = None


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
ALL_SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS | IMAGE_EXTENSIONS
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150

load_dotenv()

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_CHAT_MODEL = os.getenv("GROQ_CHAT_MODEL", "llama-3.3-70b-versatile")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-minimal")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    chunk_index: int


def load_text_from_file(path: Path) -> str:
    suffix = path.suffix.lower()

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")

    if suffix == ".pdf":
        if PdfReader is None:
            raise RuntimeError("pypdf is required to read PDF files.")
        reader = PdfReader(str(path))
        pages: list[str] = []
        for page in reader.pages:
            pages.append(page.extract_text() or "")
        return "\n".join(pages)

    if suffix == ".docx":
        if DocxDocument is None:
            raise RuntimeError("python-docx is required to read DOCX files.")
        document = DocxDocument(str(path))
        return "\n".join(paragraph.text for paragraph in document.paragraphs)

    if suffix == ".csv":
        rows: list[str] = []
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as handle:
            reader = csv.reader(handle)
            for row in reader:
                rows.append(", ".join(cell.strip() for cell in row if cell.strip()))
        return "\n".join(rows)

    if suffix in IMAGE_EXTENSIONS:
        if Image is None or pytesseract is None:
            raise RuntimeError(
                "pytesseract and Pillow are required for image OCR. "
                "Install them with 'pip install pytesseract pillow', and make sure the "
                "Tesseract OCR binary is installed on your system (not just the pip package)."
            )
        try:
            with Image.open(path) as img:
                return pytesseract.image_to_string(img)
        except pytesseract.TesseractNotFoundError as exc:
            raise RuntimeError(
                "Tesseract OCR binary not found on this system. Install it separately: "
                "Windows -> https://github.com/UB-Mannheim/tesseract/wiki, "
                "macOS -> 'brew install tesseract', Linux -> 'sudo apt install tesseract-ocr'."
            ) from exc

    raise ValueError(f"Unsupported file type: {path.suffix}")


def iter_documents(documents_dir: Path) -> Iterable[Path]:
    for path in documents_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
            yield path


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= text_length:
            break
        start = max(0, end - overlap)

    return chunks


def build_chunks_for_file(document_path: Path, source_name: str | None = None) -> list[Chunk]:
    """Build chunks for a single file. `source_name` overrides the stored source label
    (useful for Streamlit uploads where the on-disk temp path shouldn't leak into metadata)."""
    text = load_text_from_file(document_path)
    label = source_name or document_path.name
    return [
        Chunk(
            id=f"{document_path.stem}-{chunk_index}",
            text=chunk_text_value,
            source=label,
            chunk_index=chunk_index,
        )
        for chunk_index, chunk_text_value in enumerate(chunk_text(text))
    ]


def build_chunks(documents_dir: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for document_path in iter_documents(documents_dir):
        chunks.extend(
            build_chunks_for_file(document_path, source_name=str(document_path.relative_to(documents_dir)))
        )
    return chunks


def get_groq_client() -> Groq:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is required.")
    return Groq(api_key=GROQ_API_KEY)


def get_pinecone_index() -> tuple[Pinecone, str]:
    if not PINECONE_API_KEY:
        raise RuntimeError("PINECONE_API_KEY is required.")
    return Pinecone(api_key=PINECONE_API_KEY), PINECONE_INDEX_NAME


def _list_index_names(pc: Pinecone) -> list[str]:
    indexes = pc.list_indexes()
    if hasattr(indexes, "names"):
        return list(indexes.names())

    names: list[str] = []
    for index in indexes:
        if isinstance(index, str):
            names.append(index)
        elif isinstance(index, dict):
            name = index.get("name")
            if name:
                names.append(name)
        else:
            name = getattr(index, "name", None)
            if name:
                names.append(name)
    return names


def ensure_index(pc: Pinecone, index_name: str, dimension: int) -> None:
    existing_indexes = _list_index_names(pc)
    if index_name in existing_indexes:
        return

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
    )


_embedding_model = None


def get_embedding_model(model_name: str | None = None):
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(model_name or DEFAULT_EMBEDDING_MODEL)
    return _embedding_model


def embed_texts(texts: list[str], model_name: str | None = None) -> list[list[float]]:
    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [embedding.tolist() for embedding in embeddings]


def index_chunks(chunks: list[Chunk], namespace: str = "default") -> int:
    """Embed and upsert a list of already-built chunks into Pinecone. Shared by both
    the CLI directory-indexing flow and the Streamlit single-file upload flow."""
    if not chunks:
        return 0

    pc, index_name = get_pinecone_index()
    sample_embedding = embed_texts([chunks[0].text])[0]
    ensure_index(pc, index_name, len(sample_embedding))
    index = pc.Index(index_name)

    batch_size = int(os.getenv("PINECONE_BATCH_SIZE", "32"))
    upserted = 0

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        embeddings = embed_texts([chunk.text for chunk in batch])
        vectors = []
        for chunk, embedding in zip(batch, embeddings, strict=True):
            vectors.append(
                {
                    "id": f"{chunk.source}:{chunk.chunk_index}",
                    "values": embedding,
                    "metadata": {
                        "text": chunk.text,
                        "source": chunk.source,
                        "chunk_index": chunk.chunk_index,
                    },
                }
            )
        index.upsert(vectors=vectors, namespace=namespace)
        upserted += len(vectors)

    return upserted


def index_documents(documents_dir: Path, namespace: str = "default") -> int:
    chunks = build_chunks(documents_dir)
    return index_chunks(chunks, namespace=namespace)


def index_uploaded_file(file_path: Path, source_name: str, namespace: str = "default") -> int:
    """Index a single uploaded file (e.g. from Streamlit's file_uploader, saved to a
    temp path on disk). `source_name` is the human-readable original filename to store
    in Pinecone metadata, since the temp path itself isn't meaningful to the user."""
    chunks = build_chunks_for_file(file_path, source_name=source_name)
    return index_chunks(chunks, namespace=namespace)


def retrieve_context(query: str, top_k: int = 5, namespace: str = "default") -> list[dict[str, object]]:
    pc, index_name = get_pinecone_index()
    index = pc.Index(index_name)

    query_embedding = embed_texts([query])[0]
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )
    matches = result.matches if hasattr(result, "matches") else result.get("matches", [])

    contexts: list[dict[str, object]] = []
    for match in matches:
        metadata = match.get("metadata", {}) if isinstance(match, dict) else getattr(match, "metadata", {})
        score = match.get("score", 0.0) if isinstance(match, dict) else getattr(match, "score", 0.0)
        contexts.append(
            {
                "text": metadata.get("text", ""),
                "source": metadata.get("source", "unknown"),
                "chunk_index": metadata.get("chunk_index", -1),
                "score": score,
            }
        )
    return contexts


def build_context_block(contexts: list[dict[str, object]]) -> str:
    return "\n\n".join(
        f"Source: {item['source']}\nChunk: {item['chunk_index']}\nScore: {item['score']:.4f}\nText: {item['text']}"
        for item in contexts
        if item["text"]
    )


def answer_query(
    query: str,
    top_k: int = 5,
    namespace: str = "default",
    contexts: list[dict[str, object]] | None = None,
) -> str:
    client = get_groq_client()
    if contexts is None:
        contexts = retrieve_context(query=query, top_k=top_k, namespace=namespace)
    context_block = build_context_block(contexts)

    messages = [
        {
            "role": "system",
            "content": (
                "You answer questions using only the provided context. "
                "If the context is insufficient, say exactly what is missing instead of guessing. "
                "Keep the answer grounded and concise."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question: {query}\n\n"
                f"Context:\n{context_block or 'No relevant context found.'}\n\n"
                "If you use a fact from the context, mention the source in parentheses."
            ),
        },
    ]

    response = client.chat.completions.create(
        model=DEFAULT_CHAT_MODEL,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message.content or ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal RAG pipeline over local documents.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Index documents into Pinecone.")
    index_parser.add_argument("documents_dir", type=Path, help="Directory containing source documents.")
    index_parser.add_argument("--namespace", default="default", help="Pinecone namespace to use.")

    query_parser = subparsers.add_parser("query", help="Ask a question against the indexed documents.")
    query_parser.add_argument("question", help="Question to ask.")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve.")
    query_parser.add_argument("--namespace", default="default", help="Pinecone namespace to use.")
    query_parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved context before the model answer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        if args.command == "index":
            indexed = index_documents(args.documents_dir, namespace=args.namespace)
            print(f"Indexed {indexed} chunks.")
            return

        if args.command == "query":
            contexts = retrieve_context(args.question, top_k=args.top_k, namespace=args.namespace)
            if args.show_context:
                for item in contexts:
                    print(f"[{item['score']:.4f}] {item['source']}#{item['chunk_index']}")
                    print(item["text"])
                    print()
            answer = answer_query(
                args.question,
                top_k=args.top_k,
                namespace=args.namespace,
                contexts=contexts,
            )
            print(answer)
            return
    except (PermissionError, OSError, MaxRetryError) as exc:
        print(
            "Remote service access failed. "
            "Pinecone and Groq queries require outbound network access plus valid API credentials."
        )
        print(f"Details: {exc}")
        raise SystemExit(1) from exc

    raise RuntimeError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
