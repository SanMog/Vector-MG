import os, sys, glob, json
import fitz
import chromadb
import ollama
import numpy as np
from chromadb.config import Settings

DOCS_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "docs")
CHROMA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
VECTOR_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vectors.npz")
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR  = os.path.dirname(PROJECT_DIR)
EMBED_MODEL = "nomic-embed-text"
CHUNK_SIZE  = 800
OVERLAP     = 150
TEXT_PAGE_SIZE = 3500

def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for num, page in enumerate(doc, start=1):
        text = page.get_text("text")
        if text.strip():
            pages.append({"page": num, "text": text.strip()})
    return pages

def extract_text_pages(txt_path):
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    pages = []
    start = 0
    page = 1
    while start < len(text):
        part = text[start:start + TEXT_PAGE_SIZE].strip()
        if part:
            pages.append({"page": page, "text": part})
            page += 1
        start += TEXT_PAGE_SIZE
    return pages

def make_chunks(page_text, page_num, filename):
    chunks = []
    text = page_text
    start = 0
    idx = 0
    while start < len(text):
        chunk = text[start:start + CHUNK_SIZE].strip()
        if chunk:
            chunks.append({
                "text": chunk,
                "page": page_num,
                "source": filename,
                "idx": idx,
            })
            idx += 1
        start += CHUNK_SIZE - OVERLAP
    return chunks

def embed(text):
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]

def main():
    print("=" * 55)
    print("Mango Office RAG - Indexing")
    print("=" * 55)

    print("\n[1/4] Testing embedding model...")
    try:
        v = embed("test")
        print("  OK dim=" + str(len(v)))
    except Exception as e:
        print("  ERROR: " + str(e))
        print("  Run: ollama pull " + EMBED_MODEL)
        sys.exit(1)

    print("\n[2/4] Init ChromaDB at " + CHROMA_DIR)
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )
    try:
        client.delete_collection("mango_docs")
        print("  Dropped old collection (rebuilding with page numbers)")
    except Exception:
        pass
    col = client.get_or_create_collection(
        name="mango_docs",
        metadata={"hnsw:space": "cosine"}
    )
    print("  OK")

    pdfs = glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    txts = glob.glob(os.path.join(DOCS_DIR, "*.txt"))
    parent_training = os.path.join(PARENT_DIR, "данные для тренеровки.txt")
    if os.path.exists(parent_training):
        txts.append(parent_training)

    if not pdfs and not txts:
        print("ERROR: no documents in " + DOCS_DIR)
        sys.exit(1)

    print("\n[3/4] Found " + str(len(pdfs)) + " PDF files and " + str(len(txts)) + " TXT files:")
    for f in pdfs:
        print("  - " + os.path.basename(f))
    for f in txts:
        print("  - " + os.path.basename(f))

    print("\n[4/4] Indexing...")
    total = 0

    for pdf_path in pdfs:
        fname = os.path.basename(pdf_path)
        print("\n  [" + fname + "]")
        try:
            pages = extract_pages(pdf_path)
        except Exception as e:
            print("  ERROR reading: " + str(e))
            continue

        all_chunks = []
        for p in pages:
            all_chunks.extend(make_chunks(p["text"], p["page"], fname))

        print("  pages=" + str(len(pages)) + " chunks=" + str(len(all_chunks)))

        for i, c in enumerate(all_chunks):
            doc_id = fname + "::p" + str(c["page"]) + "::c" + str(c["idx"])
            emb = embed(c["text"])
            col.add(
                ids=[doc_id],
                embeddings=[emb],
                documents=[c["text"]],
                metadatas=[{
                    "source": fname,
                    "page": c["page"],
                    "chunk_index": c["idx"],
                }]
            )
            total += 1
            if (i + 1) % 20 == 0:
                print("  progress: " + str(i+1) + "/" + str(len(all_chunks)), end="\r")

        print("  done (" + str(len(all_chunks)) + " chunks)          ")

    for txt_path in txts:
        fname = os.path.basename(txt_path)
        print("\n  [" + fname + "]")
        try:
            pages = extract_text_pages(txt_path)
        except Exception as e:
            print("  ERROR reading: " + str(e))
            continue

        all_chunks = []
        for p in pages:
            all_chunks.extend(make_chunks(p["text"], p["page"], fname))

        print("  pseudo_pages=" + str(len(pages)) + " chunks=" + str(len(all_chunks)))

        for i, c in enumerate(all_chunks):
            doc_id = fname + "::p" + str(c["page"]) + "::c" + str(c["idx"])
            emb = embed(c["text"])
            col.add(
                ids=[doc_id],
                embeddings=[emb],
                documents=[c["text"]],
                metadatas=[{
                    "source": fname,
                    "page": c["page"],
                    "chunk_index": c["idx"],
                }]
            )
            total += 1
            if (i + 1) % 20 == 0:
                print("  progress: " + str(i+1) + "/" + str(len(all_chunks)), end="\r")

        print("  done (" + str(len(all_chunks)) + " chunks)          ")

    # Save our own vector index (reliable alternative to ChromaDB HNSW)
    print("\n  Saving vector index to vectors.npz...")
    # Re-read all chunks from ChromaDB and re-embed to build numpy index
    all_vecs = []
    all_meta = []
    res_all = col.get(include=["embeddings", "documents", "metadatas"])
    for emb, doc, meta in zip(
            res_all["embeddings"], res_all["documents"], res_all["metadatas"]):
        all_vecs.append(emb)
        all_meta.append({
            "text": doc,
            "source": meta.get("source", "?"),
            "page": meta.get("page", 0),
        })
    vecs_np = np.array(all_vecs, dtype=np.float32)
    np.savez_compressed(VECTOR_FILE,
        vectors=vecs_np,
        meta=np.array([json.dumps(m, ensure_ascii=False) for m in all_meta]))
    print(f"  Saved {len(all_vecs)} vectors -> {VECTOR_FILE}")

    # Warm-up: query each file with where-filter to force per-segment HNSW build
    print("\n  Warming up HNSW index (per-file)...")
    test_emb = embed("test query for index warmup")
    all_files = list(dict.fromkeys(
        [os.path.basename(p) for p in pdfs] +
        [os.path.basename(t) for t in txts]
    ))
    for fname in all_files:
        try:
            col.query(query_embeddings=[test_emb], n_results=1,
                      where={"source": {"$eq": fname}},
                      include=["documents"])
            print("    OK: " + fname)
        except Exception as e:
            print("    WARN: " + fname + " -> " + str(e))
    print("  HNSW warmup done")

    print("\n" + "=" * 55)
    print("Finished! Total chunks: " + str(total))
    print("=" * 55)
    print("\nNow run: python -m streamlit run app.py")

if __name__ == "__main__":
    main()
