import os
import re
import streamlit as st
import pytesseract
from pdf2image import convert_from_path
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Import config from Main_pipeline
from Main_pipeline import (
    PDF_FILES, CHROMA_DB_PATH, COLLECTION_NAME, OCR_LANG, 
    DPI, clean_text
)

# Overrides for structural context-aware RAG
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

st.set_page_config(page_title="GST RAG Assistant", layout="wide")

st.title("Malayalam GST Book — RAG Assistant")

# Use tabs to separate Chatbot and Builder
tab1, tab2 = st.tabs(["Chatbot", "Database Builder"])

# Helper functions with caching to avoid reloading models on every interaction
@st.cache_resource
def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name="intfloat/multilingual-e5-large")

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Chunker that preserves rule hierarchy and metadata (Slide 2: Proxy-Pointer Architecture)
def chunk_all_pages_structurally(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    all_chunks = []
    
    current_chapter = "General"
    current_rule = "General"
    
    # Flexible matching patterns for Chapters and Rules/Sections (Malayalam + English)
    chapter_pattern = re.compile(r"(അദ്ധ്യായം|അധ്യായം|Chapter)\s+([IVXLCDM\d]+|[-്അ-ഹ]+)", re.IGNORECASE)
    rule_pattern = re.compile(r"(വകുപ്പ്|ചട്ടം|Rule|Section)\s+(\d+)", re.IGNORECASE)
    
    for page in pages:
        text = clean_text(page["text"])
        lines = text.split('\n')
        
        # Check first few lines of each page to identify page-level context headers
        for line in lines[:5]:
            line_strip = line.strip()
            chap_match = chapter_pattern.search(line_strip)
            if chap_match:
                current_chapter = chap_match.group(0).strip()
            rule_match = rule_pattern.search(line_strip)
            if rule_match:
                current_rule = rule_match.group(0).strip()
                
        words = text.split()
        if len(words) <= chunk_size:
            context_header = f"Context: Chapter: {current_chapter} | Rule/Section: {current_rule}\n"
            enriched_text = context_header + text
            
            all_chunks.append({
                "text": enriched_text,
                "raw_text": text,
                "page": page["page"],
                "source": page["source"],
                "chunk_index": 0,
                "chapter": current_chapter,
                "rule": current_rule
            })
        else:
            step = chunk_size - overlap
            for i, start in enumerate(range(0, len(words), step)):
                chunk_words = words[start:start + chunk_size]
                if len(chunk_words) < 30:
                    continue
                
                chunk_text = " ".join(chunk_words)
                
                # Check if this specific chunk contains a new chapter or rule definition to update local context
                chunk_chap_match = chapter_pattern.search(chunk_text)
                chunk_rule_match = rule_pattern.search(chunk_text)
                
                local_chapter = chunk_chap_match.group(0).strip() if chunk_chap_match else current_chapter
                local_rule = chunk_rule_match.group(0).strip() if chunk_rule_match else current_rule
                
                # Update global state if a new section begins
                if chunk_chap_match:
                    current_chapter = local_chapter
                if chunk_rule_match:
                    current_rule = local_rule
                    
                context_header = f"Context: Chapter: {local_chapter} | Rule/Section: {local_rule}\n"
                enriched_text = context_header + chunk_text
                
                all_chunks.append({
                    "text": enriched_text,
                    "raw_text": chunk_text,
                    "page": page["page"],
                    "source": page["source"],
                    "chunk_index": i,
                    "chapter": local_chapter,
                    "rule": local_rule
                })
                
    return all_chunks

# ==================================================================
# TAB 1: Chatbot (Upgraded with Proxy-Pointer RAG / Structural Grouping)
# ==================================================================
with tab1:
    st.header("Ask Questions")
    st.write("Query the loaded GST book database using Gemini 2.5 Flash.")
    
    # 1. Initialize Conversation History in Streamlit Session State
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
        
    embedding_fn = get_embedding_function()
    client = get_chroma_client()
    
    # Try to load the collection
    try:
        collection = client.get_collection(name=COLLECTION_NAME, embedding_function=embedding_fn)
        db_ready = True
    except Exception:
        st.warning("Database collection not found. Please build the database first in the 'Database Builder' tab.")
        db_ready = False
        
    if db_ready:
        api_key = st.text_input("Enter your Gemini API Key", type="password")
        
        # Add a clear history button if chat exists
        if st.session_state.chat_history:
            if st.button("Clear Chat History", type="secondary"):
                st.session_state.chat_history = []
                st.rerun()
        
        # 2. Display previous chat messages in a beautiful chat UI
        for role, text in st.session_state.chat_history:
            with st.chat_message(role):
                st.write(text)
                
        # 3. Use st.chat_input for a premium ChatGPT-style experience
        if user_query := st.chat_input("Ask a question about GST (e.g., 'വകുപ്പ് 4' or 'Explain it further'):"):
            if not api_key:
                st.error("Please enter your Gemini API Key first.")
            else:
                # Display user's question immediately
                with st.chat_message("user"):
                    st.write(user_query)
                st.session_state.chat_history.append(("user", user_query))
                
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel('gemini-2.5-flash')
                
                # Step 1: Condense/Rewrite follow-up question based on past context
                search_query = user_query
                if len(st.session_state.chat_history) > 1:
                    with st.spinner("Rewriting query based on chat history..."):
                        history_str = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat_history[:-1]])
                        rewrite_prompt = f"""
                        Given the following conversation history and a follow-up question, rewrite the follow-up question to be a standalone, search-friendly query in the same language. Include any relevant context from the history. Do not change the original language.
                        
                        History:
                        {history_str}
                        
                        Follow-up Question: {user_query}
                        Standalone Query:
                        """
                        try:
                            rewrite_response = model.generate_content(rewrite_prompt)
                            search_query = rewrite_response.text.strip()
                        except Exception:
                            search_query = user_query
                
                # Step 2: Direct Vector Query using the search_query
                with st.spinner("Searching database for relevant context (Initial Vector Match)..."):
                    results = collection.query(
                        query_texts=[search_query],
                        n_results=5
                    )
                
                # Step 3: Implement Proxy-Pointer Logic (Reconstruct full parent sections)
                matched_rules = []
                for meta in results["metadatas"][0]:
                    rule = meta.get("rule")
                    if rule and rule != "General" and rule not in matched_rules:
                        matched_rules.append(rule)
                
                context_blocks = []
                source_information = []
                
                # If we mapped specific sections/rules, fetch 100% of their child chunks to prevent fragmentation
                if matched_rules:
                    with st.spinner(f"Proxy-Pointer RAG: Reconstructing complete sections for {', '.join(matched_rules)}..."):
                        for rule in matched_rules:
                            rule_chunks = collection.get(
                                where={"rule": rule}
                            )
                            # Sort chunks by chunk_index to reconstruct correct reading order
                            sorted_chunks = sorted(
                                zip(rule_chunks["documents"], rule_chunks["metadatas"]),
                                key=lambda x: (x[1]["page"], x[1]["chunk_index"])
                            )
                            
                            full_rule_text = "\n".join([doc for doc, meta in sorted_chunks])
                            context_blocks.append(f"--- Full Section Context: {rule} ---\n{full_rule_text}")
                            
                            # Add references for this rule
                            for doc, meta in sorted_chunks:
                                source_information.append({
                                    "text": doc,
                                    "page": meta["page"],
                                    "source": meta["source"],
                                    "rule": rule
                                })
                
                # Fallback to standard top chunks if no specific section structure is detected
                if not context_blocks:
                    context_blocks = results["documents"][0]
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        source_information.append({
                            "text": doc,
                            "page": meta.get("page", "Unknown"),
                            "source": meta.get("source", "Unknown"),
                            "rule": meta.get("rule", "General")
                        })
                
                context = "\n\n".join(context_blocks)
                
                # Step 4: Generate Response using context and conversation history
                with st.chat_message("assistant"):
                    with st.spinner("Generating answer with Gemini..."):
                        try:
                            chat_context = "\n".join([f"{r}: {m}" for r, m in st.session_state.chat_history])
                            prompt = f"""
                            You are a professional assistant answering questions about the Malayalam GST book.
                            Provide a highly detailed, comprehensive, and complete answer based on the provided context and the conversation history.
                            Do not summarize or provide short answers. Include all relevant points, sub-points, and details found in the context.
                            
                            Use the following retrieved context and chat history to answer the user's question accurately.
                            If the answer cannot be found in the context, state that the information is not available in the provided documents.
                            
                            Context:
                            {context}
                            
                            Chat History:
                            {chat_context}
                            
                            Question: {user_query}
                            
                            Answer:
                            """
                            
                            response = model.generate_content(prompt)
                            ans_text = response.text
                            st.write(ans_text)
                            
                            # Append assistant response to history
                            st.session_state.chat_history.append(("assistant", ans_text))
                            
                            # Display structural references in expandable sections
                            st.subheader("Source References (Structural Context Loaded)")
                            
                            if matched_rules:
                                st.info(f"Proxy-Pointer RAG resolved {len(matched_rules)} complete section(s). Click below to view the loaded text in order.")
                                
                            # Unique references to keep display clean
                            displayed_refs = set()
                            for i, ref in enumerate(source_information):
                                ref_key = f"{ref['source']}_P{ref['page']}_{ref['rule']}_{ref['text'][:50]}"
                                if ref_key not in displayed_refs:
                                    displayed_refs.add(ref_key)
                                    with st.expander(f"Reference {i+1} | {ref['rule']} | Page {ref['page']} | {ref['source']}"):
                                        st.write(ref["text"])
                                    
                        except ImportError:
                            st.error("The 'google-generativeai' package is not installed. Please run 'pip install google-generativeai' in your terminal.")
                        except Exception as e:
                            st.error(f"Error generating answer: {e}")

# ==================================================================
# TAB 2: Database Builder (With Hierarchical Context Enrichment)
# ==================================================================
with tab2:
    st.header("Database Builder")
    st.write("Run the OCR and Vector DB build process.")
    
    if st.button("Start Pipeline", type="primary"):
        all_pages = []
        page_offset = 0
        
        for pdf_path in PDF_FILES:
            if not os.path.exists(pdf_path):
                st.warning(f"File not found: {pdf_path} — skipping")
                continue
                
            st.write(f"**Processing:** {pdf_path}")
            
            # Convert PDF to images
            with st.spinner(f"Converting {pdf_path} to images..."):
                images = convert_from_path(pdf_path, dpi=DPI, poppler_path=r"c:\Users\navee\OneDrive\Desktop\Rag_GST\poppler-extracted\poppler-24.08.0\Library\bin")
            
            total = len(images)
            st.write(f"Found {total} pages. Starting OCR...")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            pages_data = []
            for i, image in enumerate(images):
                page_num = page_offset + i + 1
                status_text.text(f"OCR page {page_num} ({i+1}/{total})...")
                
                text = pytesseract.image_to_string(image, lang=OCR_LANG)
                text = text.strip()
                
                if text:
                    pages_data.append({
                        "page": page_num,
                        "text": text,
                        "source": os.path.basename(pdf_path)
                    })
                
                progress_bar.progress((i + 1) / total)
                
            all_pages.extend(pages_data)
            page_offset += len(pages_data)
            st.success(f"Done — {len(pages_data)} pages extracted from {pdf_path}")
    
        st.info(f"Total pages OCR'd: {len(all_pages)}")
    
        # Chunking
        st.subheader("Text Chunking (Structural Splitter)")
        with st.spinner("Executing context-aware chunking..."):
            all_chunks = chunk_all_pages_structurally(all_pages)
                
        st.success(f"Total chunks created: {len(all_chunks)}")
        
        st.subheader("Chunk Data Sample")
        for i, chunk in enumerate(all_chunks[:5]):
            with st.expander(f"Chunk {i+1} | {chunk['rule']} | Page {chunk['page']} | {chunk['source']}"):
                st.write(f"**Metadata:** Chapter: `{chunk['chapter']}` | Rule: `{chunk['rule']}`")
                st.write(chunk['text'])
    
        # Embed and store
        st.subheader("Vector DB Storage")
        
        embedding_fn = get_embedding_function()
        client = get_chroma_client()
        
        # Delete existing collection if re-running
        try:
            client.delete_collection(COLLECTION_NAME)
            st.write(f"Replaced existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass
            
        collection = client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        
        batch_size = 100
        total_chunks = len(all_chunks)
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(0, total_chunks, batch_size):
            batch = all_chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (total_chunks // batch_size) + 1
            
            status_text.text(f"Embedding batch {batch_num}/{total_batches}...")
            
            collection.add(
                ids=[f"chunk_{i+j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[{
                    "page": c["page"],
                    "source": c["source"],
                    "chunk_index": c["chunk_index"],
                    "chapter": c["chapter"],
                    "rule": c["rule"]
                } for c in batch]
            )
            
            progress_bar.progress(min((i + batch_size) / total_chunks, 1.0))
            
        st.success(f"Vector DB built successfully! {total_chunks} chunks stored in '{COLLECTION_NAME}'")
