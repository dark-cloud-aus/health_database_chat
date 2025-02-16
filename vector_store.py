from pinecone import Pinecone, PodSpec
from openai import OpenAI
from decouple import config
import logging
from typing import Dict, List
import tiktoken
import time
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter
import requests
import ssl

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        self.pinecone_api_key = config('PINECONE_API_KEY')
        self.openai_client = OpenAI(api_key=config('OPENAI_API_KEY'))
        self.index_name = ""  # Add your index name here from Pinecone
        self.host = "https://"  # Your specific host URL from pinecone
        self.max_retries = 3
        self.initialize_pinecone()
        
    def initialize_pinecone(self):
        """Initialize Pinecone and create index if it doesn't exist"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                logger.info(f"Attempt {retry_count + 1} to initialize Pinecone...")
                # Initialize Pinecone with new API
                self.pc = Pinecone(
                    api_key=self.pinecone_api_key
                )
                logger.info("Pinecone client created")
                
                # List existing indexes with timeout
                logger.info("Checking existing indexes...")
                existing_indexes = self.pc.list_indexes().names()
                logger.info(f"Found existing indexes: {existing_indexes}")
                
                # Connect to the index with retries
                logger.info("Connecting to index...")
                
                # Use the exact host from your Pinecone console
                logger.info(f"Using host: {self.host}")
                
                # Initialize the index with the host
                self.index = self.pc.Index(
                    name=self.index_name,
                    host=self.host
                )
                
                # Simple test query
                test_vector = [0.0] * 1536
                self.index.query(vector=test_vector, top_k=1)
                logger.info("Test query successful")
                
                logger.info("Pinecone initialization successful")
                return
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Attempt {retry_count} failed: {str(e)}")
                if retry_count == self.max_retries:
                    logger.error("Max retries reached. Initialization failed.")
                    raise
                time.sleep(10)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    
    def upsert_records(self, records: List[Dict]):
        """Insert or update records in Pinecone with retry logic"""
        retry_count = 0
        while retry_count < self.max_retries:
            try:
                vectors = []
                for i, record in enumerate(records):
                    text = f"Patient record: {str(record)}"
                    embedding = self.get_embedding(text)
                    
                    vectors.append({
                        'id': f'rec_{i}',
                        'values': embedding,
                        'metadata': record
                    })
                    
                    # Batch upsert in smaller groups
                    if len(vectors) >= 50:  # Reduced batch size
                        self.index.upsert(vectors=vectors)
                        vectors = []
                        time.sleep(0.5)  # Add delay between batches
                
                # Upsert any remaining vectors
                if vectors:
                    self.index.upsert(vectors=vectors)
                    
                logger.info(f"Successfully upserted {len(records)} records to Pinecone")
                return
                
            except Exception as e:
                retry_count += 1
                logger.error(f"Upsert attempt {retry_count} failed: {str(e)}")
                if retry_count == self.max_retries:
                    logger.error("Max retries reached. Upsert failed.")
                    raise
                time.sleep(5)  # Wait before retrying
    
    def query_similar(self, query: str, top_k: int = 5) -> List[Dict]:
        """Query similar records from Pinecone"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            return [match.metadata for match in results.matches]
            
        except Exception as e:
            logger.error(f"Error querying Pinecone: {str(e)}")
            raise 
