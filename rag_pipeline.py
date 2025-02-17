import sqlite3
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import requests
from datetime import datetime
from typing import List, Dict, Optional, Union
import logging
from pathlib import Path
import boto3
import json
import psutil
import gc
from functools import lru_cache
import nvidia_smi
from contextlib import contextmanager
from sqlalchemy import create_engine, text
from sqlalchemy.pool import QueuePool
import tempfile
from torch.cuda import empty_cache
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    CUDA_VERSION = torch.version.cuda
    if CUDA_VERSION < "11.8":
        raise RuntimeError(f"CUDA version {CUDA_VERSION} is not supported. Please use CUDA 11.8 or higher.")

class GPUMemoryManager:
    def __init__(self):
        try:
            nvidia_smi.nvmlInit()
            self.handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
            self.enabled = True
        except Exception as e:
            logger.warning(f"Failed to initialize GPU monitoring: {e}")
            self.enabled = False
        
    def get_memory_info(self):
        if not self.enabled:
            return None
        try:
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(self.handle)
            return {
                'total': info.total / 1024**2,  # MB
                'used': info.used / 1024**2,
                'free': info.free / 1024**2
            }
        except Exception as e:
            logger.warning(f"Failed to get GPU memory info: {e}")
            return None
        
    @contextmanager
    def track_memory_usage(self, operation_name: str):
        """Context manager for tracking GPU memory usage with error handling."""
        try:
            before = self.get_memory_info()
            yield
            if before:  # Only log if we successfully got initial memory info
                after = self.get_memory_info()
                if after:
                    delta = after['used'] - before['used']
                    logger.info(f"{operation_name} memory delta: {delta:.2f}MB")
        except Exception as e:
            logger.warning(f"Failed to track memory for {operation_name}: {e}")
            yield

class DataportalRAG:
    def __init__(
        self,
        db_path: Union[str, Path] = "data/rag_database.db",
        index_path: Union[str, Path] = "data/document_index.faiss",
        hf_token: Optional[str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        base_url: str = "https://dataportal.se",
        s3_backup: Optional[Dict] = None,
        cache_size: int = 1000,
        batch_size: int = 32,
        db_pool_size: int = 5,
        embedding_dim: int = 768
    ):
        """Initialize optimized RAG system with proper directory setup."""
        # Create data directories
        self.data_dir = Path("data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = Path(db_path)
        self.index_path = Path(index_path)
        
        # Create parent directories
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize GPU manager first
        try:
            if device == "cuda":
                self.gpu_manager = GPUMemoryManager()
                logger.info("GPU memory manager initialized successfully")
            else:
                self.gpu_manager = None
                logger.info("Running in CPU mode - GPU manager not initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU manager: {e}. Falling back to CPU mode")
            self.gpu_manager = None
            device = "cpu"
        
        self.device = device
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.s3_backup = s3_backup
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.base_url = base_url
        
        # Add session with proper URL handling
        self.session = requests.Session()
        self.session.mount("https://", requests.adapters.HTTPAdapter(max_retries=3))
        
        if not self.hf_token:
            raise ValueError("HF_TOKEN must be provided or set as environment variable")
        
        # Initialize components
        self._init_database(db_pool_size)
        self._init_models()
        self._init_faiss_index()
        self._init_s3()
        
        # Configure query cache
        self.query = lru_cache(maxsize=cache_size)(self._query_implementation)
        
        logger.info(f"Initialized DataportalRAG with device: {device}")
        
    def _init_database(self, pool_size: int) -> None:
        """Initialize SQLite database with SQLAlchemy connection pooling."""
        db_url = f"sqlite:///{self.db_path}"
        
        self.engine = create_engine(
            db_url,
            poolclass=QueuePool,
            pool_size=pool_size,
            max_overflow=10,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False}
        )
        
        # Create tables
        with self.engine.connect() as conn:
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    text TEXT NOT NULL,
                    embedding BLOB NOT NULL,
                    source_id TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            
            conn.execute(text('''
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id INTEGER PRIMARY KEY,
                    operation TEXT,
                    duration FLOAT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            '''))
            
            conn.execute(text("PRAGMA journal_mode=WAL"))
            conn.commit()
            
    def _init_models(self) -> None:
        """Load models with caching for faster startup."""
        logger.info("Starting model initialization...")
        
        cache_dir = Path("model_cache")
        cache_dir.mkdir(exist_ok=True)
        
        try:
            # Step 1: Load BERT (smaller model) first with caching
            logger.info("Loading Swedish BERT model...")
            with self.gpu_manager.track_memory_usage("bert_loading"):
                self.embed_model = SentenceTransformer(
                    'KBLab/sentence-bert-swedish-cased',
                    device=self.device,
                    cache_folder=str(cache_dir)  # Add caching
                )
            
            # Clear cache between models
            empty_cache()
            gc.collect()
            
            # Step 2: Load Mistral with caching
            logger.info("Loading Mistral model...")
            model_name = "mistralai/Mistral-7B-Instruct-v0.1"
            
            # Initialize tokenizer first with caching
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=self.hf_token,
                cache_dir=str(cache_dir)  # Add caching
            )
            
            # Configure memory settings
            max_memory = {0: "18GB", "cpu": "24GB"} if self.device == "cuda" else None
            
            with self.gpu_manager.track_memory_usage("mistral_loading"):
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto" if self.device == "cuda" else None,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    max_memory=max_memory,
                    token=self.hf_token,
                    cache_dir=str(cache_dir),  # Add caching
                    offload_folder="data/model_offload"
                )
                
                # Compile model if on GPU
                if self.device == "cuda" and hasattr(torch, 'compile'):
                    logger.info("Compiling model for optimized performance...")
                    self.model = torch.compile(self.model)
            
            logger.info("Model initialization complete")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            empty_cache()
            raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _init_faiss_index(self) -> None:
        """Initialize FAISS index with memory tracking."""
        logger.info("Starting FAISS index initialization...")
        
        try:
            if self.gpu_manager and self.device == "cuda":
                with self.gpu_manager.track_memory_usage("faiss_initialization"):
                    self._create_or_load_index()
            else:
                self._create_or_load_index()
            
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {str(e)}")
            empty_cache()  # Ensure GPU memory is cleared on failure
            raise RuntimeError(f"FAISS index initialization failed: {str(e)}")

    def _create_or_load_index(self) -> None:
        """Create new or load existing FAISS index with proper error handling."""
        try:
            if self.index_path.exists():
                logger.info(f"Loading existing index from {self.index_path}")
                try:
                    self.index = faiss.read_index(str(self.index_path))
                    logger.info(f"Successfully loaded index with {self.index.ntotal} vectors")
                except Exception as e:
                    logger.error(f"Failed to load existing index: {e}")
                    logger.info("Creating new index due to load failure")
                    self._create_new_index()
            else:
                logger.info("No existing index found, creating new one")
                self._create_new_index()
            
        except Exception as e:
            logger.error(f"Error in index creation/loading: {e}")
            raise

    def _create_new_index(self) -> None:
        """Create new FAISS index with optimized memory management."""
        logger.info(f"Creating new FAISS index with dimension {self.embedding_dim}")
        gpu_resource = None
        
        try:
            # Clear GPU memory before FAISS initialization
            empty_cache()
            gc.collect()
            
            # Create CPU index first
            cpu_index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Created CPU index")
            
            # Move to GPU with memory management
            if torch.cuda.is_available() and self.device == "cuda":
                logger.info("Moving index to GPU...")
                try:
                    # Allocate GPU resources with specific memory limits
                    gpu_resource = faiss.StandardGpuResources()
                    gpu_resource.setTempMemory(256 * 1024 * 1024)  # 256MB temp memory
                    
                    # Configure GPU index
                    gpu_config = faiss.GpuIndexFlatConfig()
                    gpu_config.device = 0
                    gpu_config.useFloat16 = True  # Use FP16 for memory efficiency
                    
                    # Create GPU index
                    with self.gpu_manager.track_memory_usage("faiss_gpu_transfer"):
                        self.index = faiss.GpuIndexFlatL2(gpu_resource, self.embedding_dim, gpu_config)
                        logger.info("Successfully created GPU index")
                    
                except Exception as e:
                    logger.warning(f"Failed to create GPU index, falling back to CPU: {e}")
                    self.index = cpu_index
                    if gpu_resource:
                        del gpu_resource
                        empty_cache()
            else:
                self.index = cpu_index
                logger.info("Using CPU index (GPU not available)")
            
            # Save index with memory cleanup
            logger.info(f"Saving empty index to {self.index_path}")
            with self.gpu_manager.track_memory_usage("index_saving"):
                index_to_save = faiss.index_gpu_to_cpu(self.index) if isinstance(self.index, faiss.GpuIndex) else self.index
                faiss.write_index(index_to_save, str(self.index_path))
            logger.info("Successfully saved empty index")
            
        except Exception as e:
            logger.error(f"Failed to create new index: {e}")
            if gpu_resource:
                del gpu_resource
            empty_cache()
            raise RuntimeError(f"FAISS index creation failed: {str(e)}")
        finally:
            # Ensure GPU memory is cleaned up
            empty_cache()
            gc.collect()

    def _init_s3(self) -> None:
        """Initialize S3 client if backup is configured."""
        if self.s3_backup:
            self.s3 = boto3.client('s3')
            
    def backup_to_s3(self) -> None:
        """Backup database and index to S3."""
        if not self.s3_backup:
            return
            
        try:
            bucket = self.s3_backup['bucket']
            prefix = self.s3_backup['prefix']
            
            # Backup database
            self.s3.upload_file(
                str(self.db_path),
                bucket,
                f"{prefix}/database.db"
            )
            
            # Backup FAISS index
            self.s3.upload_file(
                str(self.index_path),
                bucket,
                f"{prefix}/index.faiss"
            )
            
            logger.info("Successfully backed up to S3")
            
        except Exception as e:
            logger.error(f"Error backing up to S3: {e}")
            
    def process_batch(self, documents: List[Dict]) -> List[np.ndarray]:
        """Process a batch of documents efficiently."""
        with self.gpu_manager.track_memory_usage("batch_processing"):
            texts = [doc['text'] for doc in documents]
            embeddings = self.embed_model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_numpy=True
            )
            return embeddings
            
    @contextmanager
    def get_db_connection(self):
        """Context manager for database connections."""
        conn = self.engine.connect()
        try:
            yield conn
        finally:
            conn.close()
            
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def fetch_datasets(self, page: int = 1, page_size: int = 100) -> Optional[Dict]:
        """Fetch datasets from dataportal.se API."""
        base_url = f"{self.base_url}/api/1/datasets"
        
        params = {
            "page": page,
            "size": page_size
        }
        
        headers = {
            "Accept": "application/json"
        }
        
        try:
            response = self.session.get(
                base_url,
                params=params,
                headers=headers,
                timeout=30
            )
            response.raise_for_status()
            
            data = response.json()
            if not data.get('result'):
                logger.warning(f"No datasets found for page {page}")
                return None
            
            return {
                'hits': data['result'],
                'total': data['total'],
                'page': page,
                'size': page_size
            }
        except Exception as e:
            logger.error(f"Error fetching data from dataportal.se: {e}")
            raise

    def update_knowledge_base(self, limit: int = 100) -> None:
        """Update database with proper index management."""
        start_time = time.time()
        
        with self.get_db_connection() as conn:
            try:
                page = 1
                total_stored = 0
                data_embeddings = []
                
                while total_stored < limit:
                    api_response = self.fetch_datasets(page=page)
                    if not api_response:
                        break
                        
                    documents = self._process_datasets(api_response)
                    
                    # Process in batches
                    for i in range(0, len(documents), self.batch_size):
                        batch = documents[i:i + self.batch_size]
                        embeddings = self.process_batch(batch)
                        
                        for doc, embedding in zip(batch, embeddings):
                            data_embeddings.append(embedding)
                            conn.execute(
                                text("""
                                INSERT INTO documents (text, embedding, source_id, last_updated)
                                VALUES (:text, :embedding, :source_id, :last_updated)
                                """),
                                {
                                    "text": doc['text'],
                                    "embedding": pickle.dumps(embedding),
                                    "source_id": doc['source_id'],
                                    "last_updated": doc['modified']
                                }
                            )
                            
                            total_stored += 1
                            if total_stored >= limit:
                                break
                                
                        conn.commit()
                        empty_cache()
                        
                    page += 1
                    
                # Update FAISS index
                if data_embeddings:
                    self.index = faiss.IndexFlatL2(self.embedding_dim)
                    if torch.cuda.is_available():
                        res = faiss.StandardGpuResources()
                        self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    self.index.add(np.array(data_embeddings))
                    faiss.write_index(faiss.index_gpu_to_cpu(self.index) if torch.cuda.is_available() else self.index, 
                                    str(self.index_path))
                    
                # Log performance
                duration = time.time() - start_time
                conn.execute(
                    text("INSERT INTO performance_metrics (operation, duration) VALUES (:op, :dur)"),
                    {"op": "update_knowledge_base", "dur": duration}
                )
                conn.commit()
                
                # Backup after successful update
                self.backup_to_s3()
                
            except Exception as e:
                logger.error(f"Error updating knowledge base: {e}")
                raise
                
    def _query_implementation(self, query_text: str, top_k: int = 3) -> str:
        """Internal query implementation with SQLAlchemy connection pooling."""
        start_time = time.time()
        
        try:
            with self.gpu_manager.track_memory_usage("query_processing"):
                # Get relevant documents
                query_embedding = self.embed_model.encode(query_text).reshape(1, -1)
                
                # Memory-mapped FAISS index loading
                index = faiss.read_index(str(self.index_path))
                distances, indices = index.search(query_embedding, top_k)
                
                # Retrieve documents using SQLAlchemy
                with self.get_db_connection() as conn:
                    retrieved_docs = []
                    for idx in indices[0]:
                        result = conn.execute(
                            text("SELECT text FROM documents WHERE id = :id"),
                            {"id": idx + 1}
                        ).fetchone()
                        if result:
                            retrieved_docs.append(result[0])
                            
                # Generate response
                context = "\n".join(retrieved_docs)
                prompt = f"Context:\n{context}\n\nQuestion: {query_text}\nAnswer:"
                
                inputs = self.tokenizer(prompt, return_tensors="pt")
                inputs.pop("token_type_ids", None)
                
                if torch.cuda.is_available():
                    inputs = {k: v.to("cuda") for k, v in inputs.items()}
                    
                with torch.no_grad(), torch.cuda.amp.autocast():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=300,
                        temperature=0.7,
                        top_p=0.9
                    )
                    
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Log performance with SQLAlchemy
                duration = time.time() - start_time
                with self.get_db_connection() as conn:
                    conn.execute(
                        text("INSERT INTO performance_metrics (operation, duration) VALUES (:op, :dur)"),
                        {"op": "query", "dur": duration}
                    )
                    conn.commit()
                    
                return response
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise
            
    def cleanup(self):
        """Clean up resources including database connections and session."""
        if hasattr(self, 'session'):
            self.session.close()
        empty_cache()
        gc.collect()
        if hasattr(self, 'engine'):
            self.engine.dispose()

    def _process_datasets(self, api_response: Dict) -> List[Dict]:
        """Process API response into structured documents."""
        if not api_response or 'result' not in api_response:
            return []
        
        documents = []
        for dataset in api_response['result']:
            try:
                # Extract fields from actual API format
                doc = {
                    'title': dataset.get('title', ''),
                    'description': dataset.get('description', ''),
                    'organization': dataset.get('publisher', {}).get('name', ''),
                    'source_id': dataset.get('id', ''),
                    'modified': dataset.get('modified', datetime.now().isoformat()),
                    'keywords': ', '.join(dataset.get('keywords', [])),
                    'theme': dataset.get('theme', '')
                }
                
                if doc['title'] and doc['description']:
                    doc['text'] = (
                        f"Titel: {doc['title']}\n"
                        f"Beskrivning: {doc['description']}\n"
                        f"Organisation: {doc['organization']}\n"
                        f"Tema: {doc['theme']}\n"
                        f"Nyckelord: {doc['keywords']}\n"
                        f"Uppdaterad: {doc['modified']}"
                    )
                    documents.append(doc)
                    
            except Exception as e:
                logger.error(f"Error processing dataset: {e}")
                continue
            
        return documents

if __name__ == "__main__":
    # Example usage with optimizations
    s3_config = {
        'bucket': 'your-bucket',
        'prefix': 'rag-backups'
    }
    
    try:
        with DataportalRAG(
            s3_backup=s3_config,
            cache_size=1000,
            batch_size=32
        ) as rag:
            # Update knowledge base
            logger.info("Updating knowledge base...")
            rag.update_knowledge_base(limit=100)
            
            # Example query
            query = "Vilka öppna datakällor finns tillgängliga från svenska kommuner?"
            logger.info(f"Processing query: {query}")
            
            start_time = time.time()
            response = rag.query(query)
            duration = time.time() - start_time
            
            print(f"\nQuery: {query}")
            print(f"Response: {response}")
            print(f"Query duration: {duration:.2f}s")
            
            # Print memory usage
            memory_info = rag.gpu_manager.get_memory_info()
            print(f"\nGPU Memory Usage:")
            print(f"Total: {memory_info['total']:.2f}MB")
            print(f"Used: {memory_info['used']:.2f}MB")
            print(f"Free: {memory_info['free']:.2f}MB")
            
    except Exception as e:
        logger.error(f"Error in main: {e}")
