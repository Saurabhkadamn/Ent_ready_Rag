# app.py - Working Open-Source Multi-Modal RAG System
# Simple, clean implementation that actually works

import streamlit as st
import os
import fitz  # PyMuPDF
import requests
import json
import base64
import hashlib
from datetime import datetime
import logging
from typing import List, Dict, Any, Optional
import time
from pathlib import Path
import subprocess
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🤖 Multi-Modal RAG",
    page_icon="🤖",
    layout="wide"
)

# CSS
st.markdown("""
<style>
    .header { font-size: 2rem; color: #2E8B57; text-align: center; margin-bottom: 1rem; }
    .success { background: #d4edda; padding: 0.5rem; border-radius: 0.3rem; color: #155724; }
    .error { background: #f8d7da; padding: 0.5rem; border-radius: 0.3rem; color: #721c24; }
    .warning { background: #fff3cd; padding: 0.5rem; border-radius: 0.3rem; color: #856404; }
</style>
""", unsafe_allow_html=True)

def create_directories():
    """Create required directories"""
    dirs = ["data/uploads", "data/processed", "data/cache", "logs"]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

def check_ollama():
    """Check if Ollama is available"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_ollama_models():
    """Get available Ollama models"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            return [model["name"] for model in models]
        return []
    except:
        return []

class SimpleVisionProcessor:
    """Simple vision processor using Ollama LLaVA"""
    
    def __init__(self, model_name="llava:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_image(self, image_path: str, context: str = "") -> Dict[str, Any]:
        """Analyze image with LLaVA"""
        try:
            base64_image = self.encode_image(image_path)
            
            prompt = f"""Describe this image in detail. Include any text, numbers, charts, diagrams, or technical elements you can see. 
Context: {context if context else 'general document'}
Description:"""
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 300}
            }
            
            response = requests.post(self.api_url, json=payload, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()
                return {
                    "description": description,
                    "success": True,
                    "confidence": len(description) / 200 if description else 0.1
                }
            else:
                return {"description": "Analysis failed", "success": False, "confidence": 0.0}
                
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {"description": f"Error: {str(e)}", "success": False, "confidence": 0.0}

class SimpleEmbeddings:
    """Simple embeddings using sentence transformers"""
    
    def __init__(self):
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load embedding model"""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            st.error("❌ Please install: pip install sentence-transformers")
    
    def encode(self, texts):
        """Encode texts to embeddings"""
        if self.model is None:
            return None
        return self.model.encode(texts)

class SimpleVectorStore:
    """Simple in-memory vector store"""
    
    def __init__(self):
        self.documents = {}  # doc_id -> list of chunks
        self.embeddings_processor = SimpleEmbeddings()
    
    def store_document(self, doc_id: str, chunks: List[Dict]):
        """Store document chunks"""
        try:
            # Extract text content for embedding
            texts = [chunk["content"] for chunk in chunks]
            
            if self.embeddings_processor.model is None:
                st.error("❌ Embeddings not available")
                return False
            
            # Generate embeddings
            embeddings = self.embeddings_processor.encode(texts)
            
            # Store with embeddings
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
            
            self.documents[doc_id] = chunks
            logger.info(f"Stored {len(chunks)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def search(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for relevant chunks"""
        try:
            if doc_id not in self.documents:
                return []
            
            if self.embeddings_processor.model is None:
                return []
            
            # Get query embedding
            query_embedding = self.embeddings_processor.encode([query])[0]
            
            # Calculate similarities
            chunks = self.documents[doc_id]
            similarities = []
            
            for i, chunk in enumerate(chunks):
                chunk_embedding = chunk.get("embedding", [])
                if chunk_embedding:
                    # Simple cosine similarity
                    import numpy as np
                    sim = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    similarities.append((i, sim))
            
            # Sort by similarity
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top results
            results = []
            for i, sim in similarities[:top_k]:
                chunk = chunks[i].copy()
                chunk["relevance_score"] = float(sim)
                results.append(chunk)
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class SimpleLLM:
    """Simple LLM using Ollama"""
    
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt: str, context: str = "") -> str:
        """Generate response"""
        try:
            if context:
                full_prompt = f"""Context: {context}

Question: {prompt}

Answer based on the context provided:"""
            else:
                full_prompt = prompt
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1000}
            }
            
            response = requests.post(self.api_url, json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: API returned {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class DocumentProcessor:
    """Document processor"""
    
    def __init__(self, vision_processor):
        self.vision_processor = vision_processor
    
    def process_pdf(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        """Process PDF document"""
        try:
            doc = fitz.open(pdf_path)
            text_chunks = []
            image_chunks = []
            
            total_pages = len(doc)
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for page_num in range(total_pages):
                progress = (page_num + 1) / total_pages
                progress_bar.progress(progress)
                status_text.text(f"Processing page {page_num + 1}/{total_pages}...")
                
                page = doc[page_num]
                
                # Extract text
                text = page.get_text()
                if text.strip():
                    # Simple chunking by paragraphs
                    paragraphs = text.split('\n\n')
                    for para in paragraphs:
                        if len(para.strip()) > 50:  # Skip very short paragraphs
                            text_chunks.append({
                                "content": para.strip(),
                                "page": page_num + 1,
                                "type": "text"
                            })
                
                # Extract images
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Skip small images
                        if pix.width < 100 or pix.height < 100:
                            pix = None
                            continue
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_path = f"data/processed/{doc_id}_p{page_num}_{img_index}.png"
                            pix.save(img_path)
                            
                            # Analyze image
                            analysis = self.vision_processor.analyze_image(
                                img_path, 
                                context=f"document page {page_num + 1}"
                            )
                            
                            image_chunks.append({
                                "content": analysis["description"],
                                "page": page_num + 1,
                                "type": "image",
                                "path": img_path,
                                "confidence": analysis["confidence"]
                            })
                        
                        pix = None
                        
                    except Exception as e:
                        logger.warning(f"Error processing image {img_index} on page {page_num}: {e}")
            
            doc.close()
            progress_bar.empty()
            status_text.empty()
            
            return {
                "text_chunks": text_chunks,
                "image_chunks": image_chunks,
                "total_pages": total_pages
            }
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

class MultiModalRAG:
    """Main RAG system"""
    
    def __init__(self):
        self.vision_processor = None
        self.vector_store = SimpleVectorStore()
        self.llm = None
        self.document_processor = None
        self.processed_docs = {}
        
        create_directories()
        self.load_registry()
    
    def load_registry(self):
        """Load document registry"""
        registry_path = "data/cache/registry.json"
        if os.path.exists(registry_path):
            try:
                with open(registry_path, 'r') as f:
                    self.processed_docs = json.load(f)
            except:
                self.processed_docs = {}
    
    def save_registry(self):
        """Save document registry"""
        try:
            registry_path = "data/cache/registry.json"
            with open(registry_path, 'w') as f:
                json.dump(self.processed_docs, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving registry: {e}")
    
    def initialize_components(self, vision_model="llava:7b", llm_model="llama3.1:8b"):
        """Initialize all components"""
        try:
            # Initialize vision processor
            self.vision_processor = SimpleVisionProcessor(vision_model)
            
            # Initialize LLM
            self.llm = SimpleLLM(llm_model)
            
            # Initialize document processor
            self.document_processor = DocumentProcessor(self.vision_processor)
            
            return True
            
        except Exception as e:
            logger.error(f"Initialization error: {e}")
            return False
    
    def process_document(self, uploaded_file) -> str:
        """Process uploaded document"""
        try:
            # Generate document ID
            file_content = uploaded_file.read()
            uploaded_file.seek(0)
            doc_id = hashlib.md5(file_content).hexdigest()[:8]
            
            # Check if already processed
            if doc_id in self.processed_docs:
                st.warning(f"⚠️ Document already processed! ID: {doc_id}")
                return doc_id
            
            # Save file
            file_path = f"data/uploads/{doc_id}_{uploaded_file.name}"
            with open(file_path, "wb") as f:
                f.write(file_content)
            
            # Process document
            st.info(f"📄 Processing: {uploaded_file.name}")
            result = self.document_processor.process_pdf(file_path, doc_id)
            
            # Store in vector database
            all_chunks = result["text_chunks"] + result["image_chunks"]
            success = self.vector_store.store_document(doc_id, all_chunks)
            
            if success:
                # Update registry
                self.processed_docs[doc_id] = {
                    "filename": uploaded_file.name,
                    "processed_at": datetime.now().isoformat(),
                    "total_pages": result["total_pages"],
                    "text_chunks": len(result["text_chunks"]),
                    "image_chunks": len(result["image_chunks"])
                }
                self.save_registry()
                
                return doc_id
            else:
                raise Exception("Failed to store document in vector database")
            
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise
    
    def query_document(self, query: str, doc_id: str) -> Dict[str, Any]:
        """Query document"""
        try:
            # Search for relevant content
            results = self.vector_store.search(doc_id, query, top_k=5)
            
            if not results:
                return {"error": "No relevant content found"}
            
            # Prepare context
            context_parts = []
            for result in results:
                context_parts.append(
                    f"[Page {result['page']} - {result['type'].title()}]: {result['content']}"
                )
            context = "\n\n".join(context_parts)
            
            # Generate response
            response = self.llm.generate(query, context)
            
            return {
                "response": response,
                "sources": [
                    {
                        "page": r["page"],
                        "type": r["type"],
                        "relevance": round(r["relevance_score"], 3),
                        "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                        "image_path": r.get("path", "") if r["type"] == "image" else ""
                    }
                    for r in results
                ]
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": f"Error processing query: {str(e)}"}

def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="header">🤖 Open-Source Multi-Modal RAG</div>', unsafe_allow_html=True)
    st.markdown("**LLaVA + Ollama + Local Embeddings - Completely Private**")
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = MultiModalRAG()
    
    rag = st.session_state.rag_system
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ System Setup")
        
        # Check Ollama
        if not check_ollama():
            st.markdown('<div class="error">❌ Ollama not running!</div>', unsafe_allow_html=True)
            st.markdown("""
            **Start Ollama:**
            ```bash
            ollama serve
            ```
            
            **Install models:**
            ```bash
            ollama pull llama3.1:8b
            ollama pull llava:7b
            ```
            """)
            st.stop()
        
        # Show available models
        models = get_ollama_models()
        st.markdown('<div class="success">✅ Ollama connected</div>', unsafe_allow_html=True)
        st.write(f"**Available models:** {len(models)}")
        
        # Model selection
        vision_models = [m for m in models if "llava" in m.lower()]
        llm_models = [m for m in models if any(x in m.lower() for x in ["llama", "mistral", "gemma"])]
        
        if vision_models and llm_models:
            vision_model = st.selectbox("Vision Model", vision_models, index=0)
            llm_model = st.selectbox("LLM Model", llm_models, index=0)
            
            # Initialize components
            components_ready = (rag.vision_processor is not None and rag.llm is not None)
            
            if not components_ready:
                if st.button("🚀 Initialize System", type="primary"):
                    with st.spinner("Loading models..."):
                        success = rag.initialize_components(vision_model, llm_model)
                    if success:
                        st.success("✅ System ready!")
                        st.rerun()
                    else:
                        st.error("❌ Initialization failed")
            else:
                st.markdown('<div class="success">✅ System ready</div>', unsafe_allow_html=True)
        else:
            st.error("❌ Required models not found. Install llava:7b and llama3.1:8b")
        
        # Document status
        st.divider()
        st.header("📚 Documents")
        st.metric("Processed", len(rag.processed_docs))
        
        if rag.processed_docs:
            # Auto-select latest document
            latest_doc = max(rag.processed_docs.keys(), 
                           key=lambda x: rag.processed_docs[x]['processed_at'])
            doc_info = rag.processed_docs[latest_doc]
            st.success(f"📄 {doc_info['filename']}")
            st.session_state.active_doc = latest_doc
        else:
            st.session_state.active_doc = None
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["📄 Upload", "💬 Chat", "📊 Documents"])
    
    # Tab 1: Upload
    with tab1:
        st.header("📄 Upload Document")
        
        if not (rag.vision_processor and rag.llm):
            st.warning("⚠️ Please initialize the system first")
            st.stop()
        
        uploaded_file = st.file_uploader("Choose PDF", type="pdf")
        
        if uploaded_file:
            file_size = uploaded_file.size / (1024 * 1024)
            st.info(f"📁 {uploaded_file.name} ({file_size:.1f} MB)")
            
            if st.button("🔄 Process Document", type="primary"):
                try:
                    doc_id = rag.process_document(uploaded_file)
                    st.success(f"✅ Document processed! ID: {doc_id}")
                    
                    doc_info = rag.processed_docs[doc_id]
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Pages", doc_info["total_pages"])
                    with col2:
                        st.metric("Text Chunks", doc_info["text_chunks"])
                    with col3:
                        st.metric("Images", doc_info["image_chunks"])
                    
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Processing failed: {str(e)}")
    
    # Tab 2: Chat
    with tab2:
        st.header("💬 Chat with Document")
        
        if not (rag.vision_processor and rag.llm):
            st.warning("⚠️ System not initialized")
            st.stop()
        
        if not st.session_state.get('active_doc'):
            st.warning("⚠️ No document selected")
            st.stop()
        
        active_doc = st.session_state.active_doc
        doc_info = rag.processed_docs[active_doc]
        st.info(f"📖 Chatting with: {doc_info['filename']}")
        
        # Chat interface
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask about your document..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = rag.query_document(prompt, active_doc)
                
                if "error" in result:
                    response = f"❌ {result['error']}"
                else:
                    response = result["response"]
                    
                    # Show sources
                    if "sources" in result:
                        response += "\n\n**Sources:**\n"
                        for i, source in enumerate(result["sources"]):
                            response += f"- Page {source['page']} ({source['type']}) - Relevance: {source['relevance']:.3f}\n"
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Tab 3: Documents
    with tab3:
        st.header("📊 Document Management")
        
        if not rag.processed_docs:
            st.info("📭 No documents processed yet")
            st.stop()
        
        for doc_id, info in rag.processed_docs.items():
            with st.expander(f"📄 {info['filename']}", expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**ID:** {doc_id}")
                    st.write(f"**Processed:** {info['processed_at'][:19]}")
                
                with col2:
                    st.metric("Pages", info["total_pages"])
                    st.metric("Chunks", info["text_chunks"] + info["image_chunks"])
                
                if st.button(f"🗑️ Delete {doc_id}", key=f"del_{doc_id}"):
                    try:
                        # Remove from memory
                        if doc_id in rag.vector_store.documents:
                            del rag.vector_store.documents[doc_id]
                        del rag.processed_docs[doc_id]
                        rag.save_registry()
                        st.success("✅ Document deleted")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Delete failed: {e}")

if __name__ == "__main__":
    main()