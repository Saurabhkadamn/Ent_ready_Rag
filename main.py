# app.py - Enhanced Multi-Modal RAG System with Text + Image Output
# Returns both text responses and relevant images from documents

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
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="🤖 Multi-Modal RAG",
    page_icon="🤖",
    layout="wide"
)

# Enhanced CSS
st.markdown("""
<style>
    .header { font-size: 2rem; color: #2E8B57; text-align: center; margin-bottom: 1rem; }
    .success { background: #d4edda; padding: 0.5rem; border-radius: 0.3rem; color: #155724; }
    .error { background: #f8d7da; padding: 0.5rem; border-radius: 0.3rem; color: #721c24; }
    .warning { background: #fff3cd; padding: 0.5rem; border-radius: 0.3rem; color: #856404; }
    .source-card { 
        background: #f8f9fa; 
        padding: 1rem; 
        border-radius: 0.5rem; 
        border-left: 4px solid #2E8B57; 
        margin: 0.5rem 0; 
    }
    .image-container { 
        background: white; 
        padding: 0.5rem; 
        border-radius: 0.3rem; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.1); 
    }
    .relevance-score {
        background: #e3f2fd;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        font-size: 0.8rem;
        font-weight: bold;
        color: #1976d2;
    }
</style>
""", unsafe_allow_html=True)

def create_directories():
    """Create required directories"""
    dirs = ["data/uploads", "data/processed", "data/cache", "logs", "data/images"]
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

class EnhancedVisionProcessor:
    """Enhanced vision processor with better image analysis"""
    
    def __init__(self, model_name="llava:7b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def analyze_image(self, image_path: str, context: str = "", query: str = "") -> Dict[str, Any]:
        """Enhanced image analysis with query-specific focus"""
        try:
            base64_image = self.encode_image(image_path)
            
            # Enhanced prompt based on query
            if query:
                prompt = f"""Analyze this image in detail, focusing on elements related to: "{query}"

Context: {context if context else 'document analysis'}

Please describe:
1. What you see in the image
2. Any text, numbers, charts, diagrams, or technical elements
3. How this image relates to the query: "{query}"
4. Key visual elements that might be relevant

Detailed Description:"""
            else:
                prompt = f"""Describe this image in comprehensive detail. Include any text, numbers, charts, diagrams, tables, graphs, or technical elements you can see.

Context: {context if context else 'general document'}

Focus on:
- All visible text and numbers
- Charts, graphs, and diagrams
- Technical or scientific content
- Visual relationships and layouts

Description:"""
            
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 500}
            }
            
            response = requests.post(self.api_url, json=payload, timeout=180)
            
            if response.status_code == 200:
                result = response.json()
                description = result.get("response", "").strip()
                
                # Calculate relevance based on query keywords
                relevance = self._calculate_image_relevance(description, query) if query else 0.5
                
                return {
                    "description": description,
                    "success": True,
                    "confidence": len(description) / 300 if description else 0.1,
                    "relevance": relevance
                }
            else:
                return {"description": "Analysis failed", "success": False, "confidence": 0.0, "relevance": 0.0}
                
        except Exception as e:
            logger.error(f"Vision analysis error: {e}")
            return {"description": f"Error: {str(e)}", "success": False, "confidence": 0.0, "relevance": 0.0}
    
    def _calculate_image_relevance(self, description: str, query: str) -> float:
        """Calculate how relevant the image is to the query"""
        if not query or not description:
            return 0.5
        
        query_words = set(query.lower().split())
        description_words = set(description.lower().split())
        
        # Calculate word overlap
        overlap = len(query_words.intersection(description_words))
        relevance = min(overlap / len(query_words), 1.0) if query_words else 0.0
        
        return max(relevance, 0.1)  # Minimum relevance

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

class EnhancedVectorStore:
    """Enhanced vector store with image relevance scoring"""
    
    def __init__(self):
        self.documents = {}  # doc_id -> list of chunks
        self.embeddings_processor = SimpleEmbeddings()
    
    def store_document(self, doc_id: str, chunks: List[Dict]):
        """Store document chunks with enhanced metadata"""
        try:
            # Extract text content for embedding
            texts = [chunk["content"] for chunk in chunks]
            
            if self.embeddings_processor.model is None:
                st.error("❌ Embeddings not available")
                return False
            
            # Generate embeddings
            embeddings = self.embeddings_processor.encode(texts)
            
            # Store with embeddings and enhanced metadata
            for i, chunk in enumerate(chunks):
                chunk["embedding"] = embeddings[i].tolist()
                chunk["chunk_id"] = f"{doc_id}_{i}"
                # Ensure image path is preserved
                if chunk["type"] == "image" and "path" in chunk:
                    # Copy image to permanent location
                    permanent_path = f"data/images/{chunk['chunk_id']}.png"
                    if os.path.exists(chunk["path"]) and not os.path.exists(permanent_path):
                        import shutil
                        shutil.copy2(chunk["path"], permanent_path)
                    chunk["permanent_path"] = permanent_path
            
            self.documents[doc_id] = chunks
            logger.info(f"Stored {len(chunks)} chunks for document {doc_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing document: {e}")
            return False
    
    def search(self, doc_id: str, query: str, top_k: int = 8) -> List[Dict]:
        """Enhanced search with separate text and image results"""
        try:
            if doc_id not in self.documents:
                return []
            
            if self.embeddings_processor.model is None:
                return []
            
            # Get query embedding
            query_embedding = self.embeddings_processor.encode([query])[0]
            
            # Calculate similarities
            chunks = self.documents[doc_id]
            text_similarities = []
            image_similarities = []
            
            for i, chunk in enumerate(chunks):
                chunk_embedding = chunk.get("embedding", [])
                if chunk_embedding:
                    # Calculate cosine similarity
                    import numpy as np
                    sim = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    
                    # Separate text and image results
                    if chunk["type"] == "text":
                        text_similarities.append((i, sim))
                    else:  # image
                        # Boost image relevance if it has high visual relevance
                        visual_relevance = chunk.get("relevance", 0.5)
                        boosted_sim = sim * 0.7 + visual_relevance * 0.3
                        image_similarities.append((i, boosted_sim))
            
            # Sort both lists
            text_similarities.sort(key=lambda x: x[1], reverse=True)
            image_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Combine results - take top text and top images
            results = []
            
            # Add top text results (60% of results)
            text_count = max(1, int(top_k * 0.6))
            for i, sim in text_similarities[:text_count]:
                chunk = chunks[i].copy()
                chunk["relevance_score"] = float(sim)
                results.append(chunk)
            
            # Add top image results (40% of results)
            image_count = top_k - len(results)
            for i, sim in image_similarities[:image_count]:
                chunk = chunks[i].copy()
                chunk["relevance_score"] = float(sim)
                results.append(chunk)
            
            # Sort final results by relevance
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return []

class SimpleLLM:
    """Enhanced LLM with image-aware responses"""
    
    def __init__(self, model_name="llama3.1:8b"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
    
    def generate(self, prompt: str, context: str = "", image_descriptions: List[str] = None) -> str:
        """Generate response with image awareness"""
        try:
            # Build enhanced prompt
            full_prompt = f"""You are an AI assistant analyzing a document. Answer the user's question based on the provided context.

Context from document:
{context}
"""
            
            if image_descriptions:
                full_prompt += f"""

Visual content found in the document:
{chr(10).join([f"- {desc}" for desc in image_descriptions])}
"""
            
            full_prompt += f"""

User Question: {prompt}

Instructions:
- Provide a comprehensive answer based on both text and visual content
- Reference specific visual elements when relevant
- If images contain important information, mention what you can see
- Be specific about charts, graphs, diagrams, or technical content in images
- Cite page numbers when possible

Answer:"""
            
            payload = {
                "model": self.model_name,
                "prompt": full_prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 1500}
            }
            
            response = requests.post(self.api_url, json=payload, timeout=240)
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                return f"Error: API returned {response.status_code}"
                
        except Exception as e:
            return f"Error generating response: {str(e)}"

class DocumentProcessor:
    """Enhanced document processor with better image handling"""
    
    def __init__(self, vision_processor):
        self.vision_processor = vision_processor
    
    def process_pdf(self, pdf_path: str, doc_id: str) -> Dict[str, Any]:
        """Process PDF with enhanced image extraction"""
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
                
                # Extract text with better chunking
                text = page.get_text()
                if text.strip():
                    # Smart chunking by sentences and paragraphs
                    sentences = text.replace('\n', ' ').split('. ')
                    current_chunk = ""
                    
                    for sentence in sentences:
                        if len(current_chunk + sentence) < 500:  # Optimal chunk size
                            current_chunk += sentence + ". "
                        else:
                            if current_chunk.strip():
                                text_chunks.append({
                                    "content": current_chunk.strip(),
                                    "page": page_num + 1,
                                    "type": "text"
                                })
                            current_chunk = sentence + ". "
                    
                    # Add remaining text
                    if current_chunk.strip():
                        text_chunks.append({
                            "content": current_chunk.strip(),
                            "page": page_num + 1,
                            "type": "text"
                        })
                
                # Enhanced image extraction
                image_list = page.get_images()
                for img_index, img in enumerate(image_list):
                    try:
                        xref = img[0]
                        pix = fitz.Pixmap(doc, xref)
                        
                        # Better image filtering
                        if pix.width < 80 or pix.height < 80:
                            pix = None
                            continue
                        
                        if pix.n - pix.alpha < 4:  # GRAY or RGB
                            img_path = f"data/processed/{doc_id}_p{page_num}_{img_index}.png"
                            pix.save(img_path)
                            
                            # Enhanced image analysis
                            analysis = self.vision_processor.analyze_image(
                                img_path, 
                                context=f"document page {page_num + 1}",
                                query=""  # Will be filled during search
                            )
                            
                            if analysis["success"] and len(analysis["description"]) > 50:
                                image_chunks.append({
                                    "content": analysis["description"],
                                    "page": page_num + 1,
                                    "type": "image",
                                    "path": img_path,
                                    "confidence": analysis["confidence"],
                                    "width": pix.width,
                                    "height": pix.height
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

class EnhancedMultiModalRAG:
    """Enhanced RAG system with text + image output"""
    
    def __init__(self):
        self.vision_processor = None
        self.vector_store = EnhancedVectorStore()
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
            # Initialize enhanced vision processor
            self.vision_processor = EnhancedVisionProcessor(vision_model)
            
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
        """Enhanced query with text + image results"""
        try:
            # Search for relevant content
            results = self.vector_store.search(doc_id, query, top_k=8)
            
            if not results:
                return {"error": "No relevant content found"}
            
            # Separate text and image results
            text_results = [r for r in results if r["type"] == "text"]
            image_results = [r for r in results if r["type"] == "image"]
            
            # Prepare context for LLM
            context_parts = []
            image_descriptions = []
            
            for result in text_results:
                context_parts.append(
                    f"[Page {result['page']} - Text]: {result['content']}"
                )
            
            for result in image_results:
                context_parts.append(
                    f"[Page {result['page']} - Image]: {result['content']}"
                )
                image_descriptions.append(f"Page {result['page']}: {result['content']}")
            
            context = "\n\n".join(context_parts)
            
            # Generate enhanced response
            response = self.llm.generate(query, context, image_descriptions)
            
            return {
                "response": response,
                "text_sources": [
                    {
                        "page": r["page"],
                        "type": r["type"],
                        "relevance": round(r["relevance_score"], 3),
                        "content_preview": r["content"][:300] + "..." if len(r["content"]) > 300 else r["content"]
                    }
                    for r in text_results
                ],
                "image_sources": [
                    {
                        "page": r["page"],
                        "type": r["type"],
                        "relevance": round(r["relevance_score"], 3),
                        "content_preview": r["content"][:200] + "..." if len(r["content"]) > 200 else r["content"],
                        "image_path": r.get("permanent_path", r.get("path", "")),
                        "width": r.get("width", 0),
                        "height": r.get("height", 0)
                    }
                    for r in image_results if os.path.exists(r.get("permanent_path", r.get("path", "")))
                ],
                "total_sources": len(results)
            }
            
        except Exception as e:
            logger.error(f"Query error: {e}")
            return {"error": f"Error processing query: {str(e)}"}

def display_sources_with_images(result):
    """Display sources with images in an organized way"""
    
    # Text Sources
    if result.get("text_sources"):
        st.subheader("📝 Text Sources")
        for i, source in enumerate(result["text_sources"]):
            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <strong>📄 Page {source['page']}</strong> 
                    <span class="relevance-score">Relevance: {source['relevance']:.3f}</span>
                    <br><br>
                    <em>{source['content_preview']}</em>
                </div>
                """, unsafe_allow_html=True)
    
    # Image Sources
    if result.get("image_sources"):
        st.subheader("🖼️ Visual Sources")
        
        # Create columns for images
        num_images = len(result["image_sources"])
        if num_images > 0:
            cols = st.columns(min(num_images, 3))
            
            for i, img_source in enumerate(result["image_sources"]):
                col_idx = i % 3
                
                with cols[col_idx]:
                    st.markdown(f"""
                    <div class="image-container">
                        <strong>🖼️ Page {img_source['page']}</strong><br>
                        <span class="relevance-score">Relevance: {img_source['relevance']:.3f}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Display image
                    img_path = img_source.get('image_path', '')
                    if img_path and os.path.exists(img_path):
                        try:
                            # Open and display image
                            img = Image.open(img_path)
                            st.image(img, 
                                   caption=f"Page {img_source['page']}", 
                                   use_column_width=True)
                            
                            # Show image description in expander
                            with st.expander(f"📝 Description"):
                                st.write(img_source['content_preview'])
                                
                        except Exception as e:
                            st.error(f"Error displaying image: {e}")
                    else:
                        st.warning("Image not found")
                        st.write(img_source['content_preview'])

def main():
    """Enhanced main application"""
    
    # Header
    st.markdown('<div class="header">🤖 Enhanced Multi-Modal RAG</div>', unsafe_allow_html=True)
    st.markdown("**LLaVA + Ollama + Local Embeddings - Text + Image Responses**")
    
    # Initialize system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = EnhancedMultiModalRAG()
    
    rag = st.session_state.rag_system
    
    # Sidebar (same as before)
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
    
    # Tab 1: Upload (same as before)
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
    
    # Tab 2: Enhanced Chat with Images
    with tab2:
        st.header("💬 Chat with Document (Text + Images)")
        
        if not (rag.vision_processor and rag.llm):
            st.warning("⚠️ System not initialized")
            st.stop()
        
        if not st.session_state.get('active_doc'):
            st.warning("⚠️ No document selected")
            st.stop()
        
        active_doc = st.session_state.active_doc
        doc_info = rag.processed_docs[active_doc]
        st.info(f"📖 Chatting with: {doc_info['filename']}")
        
        # Enhanced chat interface with image support
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display images if present in assistant messages
                if message["role"] == "assistant" and "images" in message:
                    for img_data in message["images"]:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            if os.path.exists(img_data["path"]):
                                img = Image.open(img_data["path"])
                                st.image(img, caption=f"Page {img_data['page']}", width=200)
                        with col2:
                            st.markdown(f"**Page {img_data['page']} - Relevance: {img_data['relevance']:.3f}**")
                            st.markdown(f"*{img_data['description'][:150]}...*")
        
        # Chat input
        if prompt := st.chat_input("Ask about your document (text and images will be shown)..."):
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("Analyzing document (text + images)..."):
                    result = rag.query_document(prompt, active_doc)
                
                if "error" in result:
                    response = f"❌ {result['error']}"
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                else:
                    # Display main response
                    response = result["response"]
                    st.markdown(response)
                    
                    # Display sources with images
                    st.markdown("---")
                    display_sources_with_images(result)
                    
                    # Prepare message with images for history
                    message_data = {
                        "role": "assistant", 
                        "content": response,
                        "images": [
                            {
                                "path": img["image_path"],
                                "page": img["page"],
                                "relevance": img["relevance"],
                                "description": img["content_preview"]
                            }
                            for img in result.get("image_sources", [])
                        ]
                    }
                    st.session_state.messages.append(message_data)
    
    # Tab 3: Enhanced Documents Management
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
                    st.metric("Text Chunks", info["text_chunks"])
                    st.metric("Image Chunks", info["image_chunks"])
                
                # Show sample images from this document
                if info["image_chunks"] > 0:
                    st.subheader("Sample Images")
                    image_dir = "data/images"
                    sample_images = []
                    
                    # Find images for this document
                    if os.path.exists(image_dir):
                        for file in os.listdir(image_dir):
                            if file.startswith(doc_id):
                                sample_images.append(os.path.join(image_dir, file))
                    
                    if sample_images:
                        # Show first 3 images
                        cols = st.columns(min(len(sample_images), 3))
                        for i, img_path in enumerate(sample_images[:3]):
                            with cols[i]:
                                try:
                                    img = Image.open(img_path)
                                    st.image(img, caption=f"Sample {i+1}", width=150)
                                except:
                                    st.text("Image not available")
                
                # Action buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button(f"🔍 Analyze {doc_id}", key=f"analyze_{doc_id}"):
                        st.session_state.active_doc = doc_id
                        st.success(f"✅ Switched to {info['filename']}")
                        st.rerun()
                
                with col2:
                    if st.button(f"🗑️ Delete {doc_id}", key=f"del_{doc_id}"):
                        try:
                            # Remove from memory
                            if doc_id in rag.vector_store.documents:
                                del rag.vector_store.documents[doc_id]
                            
                            # Clean up image files
                            image_dir = "data/images"
                            if os.path.exists(image_dir):
                                for file in os.listdir(image_dir):
                                    if file.startswith(doc_id):
                                        try:
                                            os.remove(os.path.join(image_dir, file))
                                        except:
                                            pass
                            
                            del rag.processed_docs[doc_id]
                            rag.save_registry()
                            st.success("✅ Document deleted")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Delete failed: {e}")

# Additional utility functions
def create_sample_queries_sidebar():
    """Create a sidebar with sample queries"""
    st.sidebar.markdown("---")
    st.sidebar.header("💡 Sample Queries")
    
    sample_queries = [
        "What charts and graphs are shown in this document?",
        "Summarize the key visual elements",
        "What text appears in the images?",
        "Describe any technical diagrams",
        "What are the main data points shown?",
        "Explain the visual relationships in the document",
        "What tables or structured data do you see?",
        "Describe any flowcharts or process diagrams"
    ]
    
    for query in sample_queries:
        if st.sidebar.button(f"💬 {query}", key=f"sample_{hash(query)}"):
            st.session_state.sample_query = query

def handle_sample_query():
    """Handle sample query selection"""
    if hasattr(st.session_state, 'sample_query'):
        # Auto-fill the chat input with sample query
        st.session_state.chat_input = st.session_state.sample_query
        del st.session_state.sample_query

# Enhanced error handling and logging
def setup_error_handling():
    """Setup comprehensive error handling"""
    import traceback
    import sys
    
    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        st.error(f"An unexpected error occurred: {exc_value}")
    
    sys.excepthook = exception_handler

# Performance monitoring
def monitor_performance():
    """Monitor system performance"""
    import psutil
    import time
    
    # Memory usage
    memory = psutil.virtual_memory()
    cpu_percent = psutil.cpu_percent()
    
    with st.sidebar:
        st.markdown("---")
        st.header("📊 System Status")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Memory", f"{memory.percent:.1f}%")
        with col2:
            st.metric("CPU", f"{cpu_percent:.1f}%")
        
        if memory.percent > 80:
            st.warning("⚠️ High memory usage")
        if cpu_percent > 80:
            st.warning("⚠️ High CPU usage")

if __name__ == "__main__":
    # Setup error handling and performance monitoring
    setup_error_handling()
    
    # Add sample queries to sidebar
    create_sample_queries_sidebar()
    
    # Monitor performance
    monitor_performance()
    
    # Handle sample queries
    handle_sample_query()
    
    # Run main application
    main()