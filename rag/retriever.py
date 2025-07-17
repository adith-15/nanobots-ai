import pickle
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def load_index():
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "rb") as f:
        metadata = pickle.load(f)
    return index, metadata

def query_index(query, top_k=3):
    index, metadata = load_index()
    query_embedding = embedder.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)

    D, I = index.search(query_embedding, top_k)
    results = []
    for score, idx in zip(D[0], I[0]):
        meta = metadata[idx]
        results.append({
            "score": float(score),
            "text_id": idx,
            "source": meta["source"],
            "text": get_chunk_text(idx)
        })
    return results

def get_chunk_text(idx):
  with open("chunks.pkl", "rb") as f:
        chunks = pickle.load(f)
  return chunks[idx]
  raise NotImplementedError("Storing raw text chunks is recommended for retrieval.")

if __name__ == "__main__":
    query = input("Enter your question: ")
    results = query_index(query, top_k=3)
    for i, r in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Source: {r['source']}")
        print(f"Score: {r['score']:.4f}")
        print("----")
