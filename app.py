import os
from pathlib import Path
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Configure Streamlit page layout first
st.set_page_config(
    page_title="Sleek Resume Reader RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

import importlib
from config import Config
from utils.logger import setup_logger
from utils.timer import Timer

# Dynamically import pipeline stages due to leading numbers in filenames
doc_loader_mod = importlib.import_module("pipeline.1_document_loader")
load_pdf = doc_loader_mod.load_pdf

chunking_mod = importlib.import_module("pipeline.2_chunking")
chunk_documents = chunking_mod.chunk_documents

retriever_mod = importlib.import_module("pipeline.5_retriever")
get_retriever = retriever_mod.get_retriever

prompt_mod = importlib.import_module("pipeline.6_prompt")
get_rag_prompt = prompt_mod.get_rag_prompt

llm_mod = importlib.import_module("pipeline.7_llm")
get_llm = llm_mod.get_llm
strip_thinking_tags = llm_mod.strip_thinking_tags

rag_chain_mod = importlib.import_module("pipeline.8_rag_chain")
create_rag_chain = rag_chain_mod.create_rag_chain

logger = setup_logger("streamlit_app")

# Ensure required directories exist
Config.get_absolute_path(Config.UPLOADS_DIR).mkdir(parents=True, exist_ok=True)

# Inject custom premium styling
css_path = Config.get_absolute_path("assets/style.css")
if css_path.exists():
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        logger.warning(f"Could not load custom CSS: {e}")

# Validate API Configuration — keys are loaded from .env only, never shown in the UI
if not Config.validate():
    st.error("⚠️ Groq API Key is missing!")
    st.info("Please configure `GROQ_API_KEY` in your `.env` file and restart the app.")
    st.stop()


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None
if "pages" not in st.session_state:
    st.session_state.pages = 0
if "chunks" not in st.session_state:
    st.session_state.chunks = 0
if "elapsed" not in st.session_state:
    st.session_state.elapsed = 0.0
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

def handle_clear_chat():
    """Resets the conversation messages."""
    st.session_state.messages = []
    logger.info("Chat history cleared by user.")

def handle_clear_all():
    """Resets conversation and clears upload folders."""
    st.session_state.messages = []
    st.session_state.processed_filename = None
    st.session_state.pages = 0
    st.session_state.chunks = 0
    st.session_state.elapsed = 0.0
    st.session_state.rag_chain = None
    
    # Clear local file uploads
    uploads_path = Config.get_absolute_path(Config.UPLOADS_DIR)
    for file in uploads_path.glob("*"):
        try:
            if file.is_file():
                file.unlink()
        except Exception as e:
            logger.warning(f"Could not delete uploaded file {file}: {e}")
            
    st.success("Session, metrics, and local uploads cleared successfully!")
    logger.info("Full reset triggered by user.")

# ----------------- SIDEBAR PANEL -----------------
with st.sidebar:
    st.markdown(
        "<div class='sidebar-header'><div class='sidebar-title'>📄 Resume RAG Chatbot</div></div>",
        unsafe_allow_html=True
    )
    
    # 1. File Upload
    uploaded_file = st.file_uploader(
        "Upload Candidate Resume (PDF)",
        type=["pdf"],
        help="Upload a single PDF resume. Uploading a new PDF will overwrite the database and start a new session."
    )
    
    # Pipeline processor execution
    if uploaded_file is not None:
        if st.session_state.processed_filename != uploaded_file.name:
            with st.spinner("Processing resume..."):
                try:
                    # Save PDF
                    uploads_dir = Config.get_absolute_path(Config.UPLOADS_DIR)
                    file_path = uploads_dir / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    # Run RAG indexing pipeline
                    with Timer() as timer:
                        # 1. Loader
                        docs = load_pdf(file_path)
                        num_pages = len(docs)
                        
                        # 2. Chunking
                        chunks = chunk_documents(
                            docs, 
                            chunk_size=Config.CHUNK_SIZE, 
                            chunk_overlap=Config.CHUNK_OVERLAP
                        )
                        num_chunks = len(chunks)
                        
                        # 3. Hybrid Retriever (BM25 keyword + Semantic vector search)
                        retriever = get_retriever(chunks, k=Config.RETRIEVAL_K)
                        
                        # 4. LLM & Chain
                        llm = get_llm()
                        prompt = get_rag_prompt()
                        rag_chain = create_rag_chain(retriever, prompt, llm)
                        
                    # Save successfully created pipeline states
                    st.session_state.rag_chain = rag_chain
                    st.session_state.processed_filename = uploaded_file.name
                    st.session_state.pages = num_pages
                    st.session_state.chunks = num_chunks
                    st.session_state.elapsed = timer.elapsed_time
                    st.session_state.messages = [] # reset chat for new resume
                    
                    st.toast("Resume parsed successfully!", icon="✅")
                except Exception as e:
                    logger.exception(f"Pipeline error: {e}")
                    st.error(f"❌ Failed to process resume. Error details: {str(e)}")
                    # Reset state on failure
                    st.session_state.processed_filename = None
                    st.session_state.rag_chain = None
                    st.session_state.pages = 0
                    st.session_state.chunks = 0
                    st.session_state.elapsed = 0.0

    # 2. Pipeline Metadata Metrics Cards
    if st.session_state.processed_filename:
        st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
        st.markdown("### 📊 Status & Metrics")
        
        # Display custom styled card metrics
        st.markdown(
            f"<div class='metric-card'>"
            f"<div class='metric-title'>Parsed Resume</div>"
            f"<div class='metric-value' style='font-size: 1rem; word-break: break-all;'>{st.session_state.processed_filename}</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>Pages</div>"
                f"<div class='metric-value'>{st.session_state.pages}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>Chunks</div>"
                f"<div class='metric-value'>{st.session_state.chunks}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            
        col3, col4 = st.columns(2)
        with col3:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>Model</div>"
                f"<div class='metric-value' style='font-size: 0.95rem; color: #a855f7;'>{Config.GROQ_MODEL_NAME}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        with col4:
            st.markdown(
                f"<div class='metric-card'>"
                f"<div class='metric-title'>Index Time</div>"
                f"<div class='metric-value'>{st.session_state.elapsed:.2f}s</div>"
                f"</div>",
                unsafe_allow_html=True
            )

    # 3. Debug Section & Action Buttons
    st.markdown("<hr style='margin: 15px 0; opacity: 0.1;'>", unsafe_allow_html=True)
    st.markdown("### ⚙️ Controls")
    
    show_debug = st.checkbox(
        "Show Retrieved Chunks (Debug)", 
        value=False,
        help="Display the exact resume chunks retrieved by ChromaDB under the assistant response."
    )
    st.session_state.show_debug = show_debug
    
    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)
    st.button("Clear Chat History", on_click=handle_clear_chat, use_container_width=True)
    st.button("Reset App & Clear DB", on_click=handle_clear_all, use_container_width=True, type="secondary")

# ----------------- MAIN CHAT PANEL -----------------
# Header Section
st.markdown("<div class='gradient-text'>Modular Resume Reader RAG</div>", unsafe_allow_html=True)
st.markdown("<div class='gradient-subtext'>Upload a candidate PDF resume to ask questions and inspect source documents in real time.</div>", unsafe_allow_html=True)

# Fallback: prompt for file upload if none is active
if not st.session_state.processed_filename or not st.session_state.rag_chain:
    st.info("👈 Please upload a PDF resume in the sidebar to begin chatting.")
    st.stop()

# Print conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Source rendering for assistant answers
        if msg["role"] == "assistant":
            if msg.get("sources"):
                st.markdown("<div class='source-container'>", unsafe_allow_html=True)
                for src in msg["sources"]:
                    st.markdown(f"<span class='source-tag'>📄 {src}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # If debug is enabled and retrieve chunks exist
            if st.session_state.show_debug and msg.get("chunks"):
                with st.expander("🔍 Retrieved Source Chunks"):
                    for idx, chunk in enumerate(msg["chunks"]):
                        st.markdown(
                            f"<div class='chunk-box'>"
                            f"<div class='chunk-metadata'>"
                            f"<span>Chunk {idx+1}</span>"
                            f"<span>Source: {chunk['source']}</span>"
                            f"<span>Page: {chunk['page']}</span>"
                            f"</div>"
                            f"<div>{chunk['content']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )

# Pre-prepared interactive query suggestions
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("<div class='suggestion-title'>💡 Suggested Questions</div>", unsafe_allow_html=True)
cols = st.columns(4)
suggestions = [
    "Summarize professional experience",
    "What are the technical skills?",
    "List education & credentials",
    "What is the contact information?"
]

selected_suggestion = None
for i, suggestion in enumerate(suggestions):
    with cols[i]:
        if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
            selected_suggestion = suggestion

# Capture inputs
chat_input = st.chat_input("Ask a question about the candidate's resume...")
user_query = chat_input or selected_suggestion

# RAG Chain logic
if user_query:
    # Display user query
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)
        
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        try:
            # 1. Fetch current conversation history in LangChain format (excluding current message)
            langchain_history = []
            for msg in st.session_state.messages[:-1]:
                if msg["role"] == "user":
                    langchain_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_history.append(AIMessage(content=msg["content"]))
            
            # 2. Run the chain with progress indicator
            with st.spinner("Analyzing resume (hybrid keyword + semantic search)..."):
                chain = st.session_state.rag_chain
                
                # Execute retrieval + response generation
                result = chain.invoke({
                    "question": user_query,
                    "chat_history": langchain_history
                })
                
                response_text = strip_thinking_tags(result["answer"])
                retrieved_docs = result.get("context", [])
                
            # Formatting sources
            unique_sources = []
            serialized_chunks = []
            for doc in retrieved_docs:
                # Skip the synthetic full-document chunk from source display
                if doc.metadata.get("chunk_type") == "full_document":
                    continue
                filename = Path(doc.metadata.get("source", "resume")).name
                page = doc.metadata.get("page", 0) + 1
                src_str = f"{filename} (Page {page})"
                if src_str not in unique_sources:
                    unique_sources.append(src_str)
                    
                serialized_chunks.append({
                    "content": doc.page_content,
                    "source": filename,
                    "page": page
                })
            
            # Post-processing: enforce the fallback message ONLY when retrieval
            # returned zero documents (the LLM had nothing to reason over).
            # When docs ARE retrieved, we trust the LLM's reasoning — its prompt
            # already instructs it to emit the sentinel phrase itself when the
            # information is truly absent, while still synthesising partial answers.
            NOT_FOUND_SENTINEL = "i couldn't find this information in the uploaded resume"
            
            is_not_found = False
            if not retrieved_docs:
                # Zero chunks retrieved — nothing to reason over
                response_text = "I couldn't find this information in the uploaded resume."
                is_not_found = True
                unique_sources = []
            elif response_text.lower().strip().startswith(NOT_FOUND_SENTINEL):
                # LLM itself decided the context contains nothing relevant
                response_text = "I couldn't find this information in the uploaded resume."
                is_not_found = True
                unique_sources = []
                
            # Display response
            message_placeholder.markdown(response_text)
            
            if unique_sources:
                st.markdown("<div class='source-container'>", unsafe_allow_html=True)
                for src in unique_sources:
                    st.markdown(f"<span class='source-tag'>📄 {src}</span>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            # If debug mode is active
            if st.session_state.show_debug and serialized_chunks and not is_not_found:
                with st.expander("🔍 Retrieved Source Chunks"):
                    for idx, chunk in enumerate(serialized_chunks):
                        st.markdown(
                            f"<div class='chunk-box'>"
                            f"<div class='chunk-metadata'>"
                            f"<span>Chunk {idx+1}</span>"
                            f"<span>Source: {chunk['source']}</span>"
                            f"<span>Page: {chunk['page']}</span>"
                            f"</div>"
                            f"<div>{chunk['content']}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                        
            # Store in session state chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response_text,
                "sources": unique_sources if not is_not_found else [],
                "chunks": serialized_chunks if not is_not_found else []
            })
            
        except Exception as e:
            logger.exception(f"Inference error: {e}")
            error_msg = f"Sorry, I encountered an error while processing your request: {e}"
            message_placeholder.error(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I encountered an error processing your query. Please check your connection or API key.",
                "sources": [],
                "chunks": []
            })
            
    # Trigger script rerun to align layout state
    st.rerun()
