import json
import uuid
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class JSONLQdrantProcessor:
    def __init__(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "document_chunks",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        """
        Initialize the processor with Qdrant client and text splitter.
        
        Args:
            qdrant_host: Qdrant server host
            qdrant_port: Qdrant server port
            collection_name: Name of the Qdrant collection
            embedding_model: Sentence transformer model for embeddings
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
        """
        # Initialize Qdrant client
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        
        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Get vector size from the model
        sample_embedding = self.embedding_model.encode("sample")
        self.vector_size = len(sample_embedding)
        
        logger.info(f"Initialized with vector size: {self.vector_size}")

    def create_collection(self, recreate: bool = False):
        """Create or recreate the Qdrant collection."""
        try:
            if recreate:
                self.client.delete_collection(self.collection_name)
                logger.info(f"Deleted existing collection: {self.collection_name}")
        except Exception as e:
            logger.info(f"Collection {self.collection_name} doesn't exist or couldn't be deleted: {e}")
        
        # Create collection with vector configuration
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE
            )
        )
        logger.info(f"Created collection: {self.collection_name}")

    def read_jsonl(self, file_path: str) -> List[Dict[str, Any]]:
        """Read and parse JSONL file."""
        data = []
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for line_num, line in enumerate(file, 1):
                    line = line.strip()
                    if line:  # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing line {line_num}: {e}")
                            continue
            
            logger.info(f"Successfully read {len(data)} records from {file_path}")
            return data
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            raise

    def chunk_and_embed(self, records: List[Dict[str, Any]]) -> List[PointStruct]:
        """Chunk content and create embeddings for Qdrant points."""
        points = []
        
        for record_idx, record in enumerate(records):
            # Split content into chunks
            content = record['description']
            code = record['code']
            chunks = self.text_splitter.split_text(content)
            chunks = [f'{code} {record["name"]} - {c}' for c in chunks]
            logger.info(f"Section '{code}': Split into {len(chunks)} chunks")
            
            # Create embeddings and points for each chunk
            for chunk_idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embedding_model.encode(chunk).tolist()
                
                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "chunk_index": chunk_idx,
                        "total_chunks": len(chunks),
                        # Include any additional fields from the original record
                        **{k: v for k, v in record.items() if k not in ['section', 'content']}
                    }
                )
                points.append(point)
        
        logger.info(f"Created {len(points)} total chunks across all sections")
        return points

    def upload_to_qdrant(self, points: List[PointStruct], batch_size: int = 100):
        """Upload points to Qdrant in batches."""
        total_points = len(points)
        
        for i in range(0, total_points, batch_size):
            batch = points[i:i + batch_size]
            
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch
                )
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(total_points + batch_size - 1)//batch_size} "
                           f"({len(batch)} points)")
            except Exception as e:
                logger.error(f"Error uploading batch starting at index {i}: {e}")
                raise

    def process_file(self, jsonl_file_path: str, recreate_collection: bool = False, batch_size: int = 100):
        """Complete processing pipeline."""
        logger.info("Starting JSONL processing pipeline")
        
        # Create collection
        self.create_collection(recreate=recreate_collection)
        
        # Read JSONL file
        records = self.read_jsonl(jsonl_file_path)
        
        if not records:
            logger.warning("No records found in the file")
            return
        
        # Chunk and embed
        points = self.chunk_and_embed(records)
        
        if not points:
            logger.warning("No valid chunks created")
            return
        
        # Upload to Qdrant
        self.upload_to_qdrant(points, batch_size=batch_size)
        
        logger.info(f"Processing complete! {len(points)} chunks uploaded to Qdrant collection '{self.collection_name}'")

def main():
    parser = argparse.ArgumentParser(description="Process JSONL file and upload to Qdrant")
    parser.add_argument("jsonl_file", help="Path to the JSONL file")
    parser.add_argument("--qdrant-host", default="localhost", help="Qdrant host (default: localhost)")
    parser.add_argument("--qdrant-port", type=int, default=6333, help="Qdrant port (default: 6333)")
    parser.add_argument("--collection-name", default="document_chunks", help="Qdrant collection name")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size (default: 1000)")
    parser.add_argument("--chunk-overlap", type=int, default=200, help="Chunk overlap (default: 200)")
    parser.add_argument("--recreate-collection", action="store_true", help="Recreate the collection if it exists")
    parser.add_argument("--batch-size", type=int, default=100, help="Upload batch size (default: 100)")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = JSONLQdrantProcessor(
        qdrant_host=args.qdrant_host,
        qdrant_port=args.qdrant_port,
        collection_name=args.collection_name,
        embedding_model=args.embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Process file
    processor.process_file(
        jsonl_file_path=args.jsonl_file,
        recreate_collection=args.recreate_collection,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()