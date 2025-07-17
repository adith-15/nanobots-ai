import os
import glob
import pickle

from PyPDF2 import PdfReader

from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np

DATA_DIR = "/Users/amitshrote/Documents/voice-salesbot/data"
INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "metadata.pkl"

# Initialize embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")  # Small and fast

def load_documents():
    texts = []
    filenames = []
    files = glob.glob(os.path.join(DATA_DIR, "*"))
    for f in files:
        if f.lower().endswith(".pdf"):
            reader = PdfReader(f)
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        elif f.lower().endswith(".txt"):
            with open(f, "r", encoding="utf-8") as infile:
                text = infile.read()
        else:
            continue
        texts.append(text)
        filenames.append(os.path.basename(f))
    return texts, filenames

def split_texts(texts, filenames):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = []
    meta = []
    for text, fname in zip(texts, filenames):
        splits = splitter.split_text(text)
        chunks.extend(splits)
        meta.extend([{"source": fname}] * len(splits))
        with open("chunks.pkl", "wb") as f:
            pickle.dump(chunks, f)
    return chunks, meta

def build_index(chunks):
    embeddings = embedder.encode(chunks, show_progress_bar=True, normalize_embeddings=True)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(np.array(embeddings, dtype=np.float32))
    return index, embeddings

def save_index(index, meta):
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "wb") as f:
        pickle.dump(meta, f)

if __name__ == "__main__":
    texts, filenames = load_documents()
    if not texts:
        print("No documents found in data/. Add some PDFs or TXT files first.")
        exit()
    chunks, meta = split_texts(texts, filenames)
    index, _ = build_index(chunks)
    save_index(index, meta)
    print(f"Indexed {len(chunks)} text chunks from {len(filenames)} files.")
