import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

MODEL_EMBEDDING_DIMS = {
    "sentence-transformers/all-MiniLM-L12-v2": 384,
    "sentence-transformers/all-mpnet-base-v2": 768,
    "nomic-ai/nomic-embed-text-v1": 768,
    "sentence-transformers/msmarco-bert-base-dot-v5": 768,
    "BAAI/bge-base-en": 768,
    "BAAI/bge-large-en": 1024,
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/e5-large-v2": 1024,
    "thenlper/gte-large": 1024
}

class HelpRetriever:
    def __init__(self, help_text, embedding_model, index_file="vtr_faiss.index", metadata_file="metadata.json"):
        self.help_text = help_text
        self.index_file = index_file
        self.metadata_file = metadata_file
        self.embedder = SentenceTransformer(embedding_model, trust_remote_code=True)
        self.index = None
        self.metadata = None
        self.embedding_dim = MODEL_EMBEDDING_DIMS[embedding_model]  # Adjust based on the embedding model

        # Load index and metadata if they exist, otherwise create them
        try:
            self.index, self.metadata = self.load_index_and_metadata()
        except:
            self.create_index()

    def get_embedding(self, text):
        """Generate an embedding for a given text."""
        return self.embedder.encode(text, convert_to_numpy=True).astype(np.float32)

    def create_index(self):
        """Read the help text file, create embeddings, and store them in FAISS."""
        help_text = self.help_text

        # Split text into manageable chunks
        chunks = help_text.split("\n\n")  # Splitting by double newline
        self.metadata = []

        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        for i, chunk in enumerate(chunks):
            embedding = self.get_embedding(chunk)
            self.index.add(np.array([embedding]))  # Add to FAISS index
            self.metadata.append({"id": i, "text": chunk})

        # Save FAISS index and metadata
        faiss.write_index(self.index, self.index_file)
        with open(self.metadata_file, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)

        print("VTR Help Documentation Indexed Successfully!")

    def load_index_and_metadata(self):
        """Load the FAISS index and metadata from disk."""
        index = faiss.read_index(self.index_file)
        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        return index, metadata

    def retrieve_help_text(self, query, top_k=3):
        """Retrieve the top-k most relevant chunks for a given query."""
        query_embedding = self.get_embedding(query)
        _, idxs = self.index.search(np.array([query_embedding]), top_k)
        retrieved_chunks = [self.metadata[i]["text"] for i in idxs[0] if i < len(self.metadata)]
        return " ".join(retrieved_chunks)


