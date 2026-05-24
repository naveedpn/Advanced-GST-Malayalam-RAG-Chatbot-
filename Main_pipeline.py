import os
import re
import glob
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
import chromadb

from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# CONFIGURATION — edit these paths
# ─────────────────────────────────────────────
 
PDF_FILES = [
    "1-53.pdf",      # ← change to your actual filenames
    "(54-275).pdf",
    "276-412.pdf",
]
 
CHROMA_DB_PATH = "./gst_chromadb"       # folder where vectors will be saved
COLLECTION_NAME = "gst_malayalam"
OCR_LANG = "mal+eng"                        # Malayalam; use "mal+eng" if mixed
CHUNK_SIZE = 400                        # words per chunk
CHUNK_OVERLAP = 50                      # word overlap between chunks
DPI = 300                               # higher = better OCR, slower

# STEP 1 + 2: PDF → Images → OCR text
# ─────────────────────────────────────────────
 
def pdf_to_text_pages(pdf_path: str, page_offset: int = 0) -> list[dict]:
    """Convert each page of a PDF to an image and OCR it."""
    print(f"\nProcessing: {pdf_path}")
    pages_data = []
 
    images = convert_from_path(pdf_path, dpi=DPI, poppler_path=r"c:\Users\navee\OneDrive\Desktop\Rag_GST\poppler-extracted\poppler-24.08.0\Library\bin")
    total = len(images)
 
    for i, image in enumerate(images):
        page_num = page_offset + i + 1
        print(f"  OCR page {page_num} ({i+1}/{total})...", end="\r")
 
        text = pytesseract.image_to_string(image, lang=OCR_LANG)
        text = text.strip()
 
        if text:  # skip blank pages
            pages_data.append({
                "page": page_num,
                "text": text,
                "source": os.path.basename(pdf_path)
            })
 
    print(f"  Done — {len(pages_data)} pages extracted from {pdf_path}")
    return pages_data
 
 
def ocr_all_pdfs(pdf_files: list[str]) -> list[dict]:
    """OCR all PDFs in order, tracking global page numbers."""
    all_pages = []
    page_offset = 0
 
    for pdf_path in pdf_files:
        if not os.path.exists(pdf_path):
            print(f"  File not found: {pdf_path} — skipping")
            continue
        pages = pdf_to_text_pages(pdf_path, page_offset=page_offset)
        all_pages.extend(pages)
        page_offset += len(pages)
 
    print(f"\nTotal pages OCR'd: {len(all_pages)}")
    return all_pages

# STEP 3: Chunk text
# ─────────────────────────────────────────────
 
def clean_text(text: str) -> str:
    """Remove excessive whitespace and noise from OCR output."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()
 
 
def chunk_page(page_data: dict, chunk_size: int, overlap: int) -> list[dict]:
    """Split a single page's text into overlapping chunks."""
    text = clean_text(page_data["text"])
    words = text.split()
    chunks = []
 
    if len(words) <= chunk_size:
        # Page is short enough to be a single chunk
        chunks.append({
            "text": text,
            "page": page_data["page"],
            "source": page_data["source"],
            "chunk_index": 0
        })
    else:
        step = chunk_size - overlap
        for i, start in enumerate(range(0, len(words), step)):
            chunk_words = words[start:start + chunk_size]
            if len(chunk_words) < 30:   # skip tiny tail chunks
                continue
            chunks.append({
                "text": " ".join(chunk_words),
                "page": page_data["page"],
                "source": page_data["source"],
                "chunk_index": i
            })
 
    return chunks
 
 
def chunk_all_pages(pages: list[dict]) -> list[dict]:
    """Chunk all pages into vector-ready pieces."""
    all_chunks = []
    for page in pages:
        all_chunks.extend(chunk_page(page, CHUNK_SIZE, CHUNK_OVERLAP))
 
    print(f"Total chunks created: {len(all_chunks)}")
    return all_chunks
 
#  STEP 4 + 5: Embed and store in ChromaDB
# ─────────────────────────────────────────────
 
def build_vector_db(chunks: list[dict]):
    """Embed chunks and store in a persistent ChromaDB collection."""
    print(f"\nLoading embedding model...")
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name="intfloat/multilingual-e5-large"
        # Good alternatives if this is too slow on your machine:
        # "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  (faster, slightly less accurate)
        # "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  (fastest)
    )
 
    print(f"Initializing ChromaDB at: {CHROMA_DB_PATH}")
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
 
    # Delete existing collection if re-running
    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"  Replaced existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass
 
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )
 
    # Insert in batches of 100 to avoid memory issues
    batch_size = 100
    total = len(chunks)
 
    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        print(f"  Embedding batch {i//batch_size + 1}/{(total//batch_size) + 1}...", end="\r")
 
        collection.add(
            ids=[f"chunk_{i+j}" for j in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[{
                "page": c["page"],
                "source": c["source"],
                "chunk_index": c["chunk_index"]
            } for c in batch]
        )
 
    print(f"\nVector DB built! {total} chunks stored in '{COLLECTION_NAME}'")
    print(f"Saved to: {os.path.abspath(CHROMA_DB_PATH)}")
    return collection

# QUICK TEST: Query the DB after building
# ─────────────────────────────────────────────
 
def test_query(collection, question: str, top_k: int = 3):
    """Test a query against the vector DB."""
    print(f"\nTest query: '{question}'")
    results = collection.query(
        query_texts=[question],
        n_results=top_k
    )
 
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"\n  Result {i+1} — Page {meta['page']} ({meta['source']})")
        print(f"  {doc[:200]}...")
 
 # MAIN
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
    print("=" * 50)
    print("  Malayalam GST Book — RAG Pipeline")
    print("=" * 50)
 
    # 1. OCR all PDFs
    pages = ocr_all_pdfs(PDF_FILES)
 
    if not pages:
        print("No pages extracted. Check your PDF file paths.")
        exit(1)
 
    # 2. Chunk
    chunks = chunk_all_pages(pages)
 
    # 3. Embed + Store
    collection = build_vector_db(chunks)
 
    # 4. Quick test
    test_query(collection, "GST registration rules")      # replace with a Malayalam query for better results
    test_query(collection, "ജിഎസ്ടി രജിസ്ട്രേഷൻ")       # same in Malayalam
 
    print("\nPipeline complete! Your vector DB is ready for the chatbot.")
 
