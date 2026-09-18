"""
RAG pipeline: load docs -> chunk -> embed -> store in Chroma -> retrieve.

Built without heavy LangChain abstraction first (deliberately) so you can
explain every step in an interview, then wired into LangChain's Chroma
wrapper for convenience.
"""
import os
import glob
import chromadb
from chromadb.utils import embedding_functions

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "docs")
CHUNK_SIZE = 500  # characters
CHUNK_OVERLAP = 50


def load_docs() -> list[dict]:
    """Load all markdown docs from data/docs/."""
    docs = []
    for path in glob.glob(os.path.join(DOCS_DIR, "*.md")):
        with open(path, "r", encoding="utf-8") as f:
            docs.append({"source": os.path.basename(path), "text": f.read()})
    return docs


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple sliding-window chunker. Good enough for short policy docs;
    for longer/real documents, swap in a recursive/semantic chunker."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if c]


class KnowledgeBase:
    """Wraps a local Chroma collection for retrieval."""

    def __init__(self, collection_name: str = "support_kb"):
        self.client = chromadb.Client()
        # Free local embedding model, no API key needed for the retrieval step
        self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name, embedding_function=self.embed_fn
        )
        self._indexed = False

    def index(self):
        """Chunk and embed all docs into the vector store."""
        docs = load_docs()
        ids, texts, metadatas = [], [], []
        for doc in docs:
            for i, chunk in enumerate(chunk_text(doc["text"])):
                ids.append(f"{doc['source']}-{i}")
                texts.append(chunk)
                metadatas.append({"source": doc["source"]})
        if texts:
            self.collection.add(ids=ids, documents=texts, metadatas=metadatas)
        self._indexed = True

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return top-k relevant chunks with their source doc."""
        if not self._indexed:
            self.index()
        results = self.collection.query(query_texts=[query], n_results=k)
        chunks = []
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            chunks.append({"text": text, "source": meta["source"]})
        return chunks
