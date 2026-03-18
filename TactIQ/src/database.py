"""
ChromaDB Vector Database Manager
Handles vector storage, retrieval, and persistence for TactIQ RAG system
"""

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from typing import List, Dict, Optional, Any
from pathlib import Path
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class VectorDatabase:
    """Manages ChromaDB vector store for football scouting data"""
    
    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_function=None
    ):
        """
        Initialize ChromaDB vector database
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the collection
            embedding_function: Custom embedding function (optional)
        """
        self.persist_directory = persist_directory or os.getenv("CHROMA_DB_PATH", "./db/chroma")
        self.collection_name = collection_name or os.getenv("CHROMA_COLLECTION_NAME", "tactiq_football")
        
        # Create persist directory
        Path(self.persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=self.persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Set up embedding function
        if embedding_function is None:
            # Use default sentence-transformers embedding
            embedding_model = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=embedding_model
            )
        else:
            self.embedding_function = embedding_function
        
        # Get or create collection
        self.collection = self._get_or_create_collection()
        
        logger.info(f"Initialized ChromaDB at {self.persist_directory}")
        logger.info(f"Collection: {self.collection_name} with {self.collection.count()} documents")
    
    def _get_or_create_collection(self):
        """Get existing collection or create new one"""
        try:
            collection = self.client.get_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function
            )
            logger.info(f"Loaded existing collection: {self.collection_name}")
        except:
            collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "TactIQ football scouting vector store"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
        
        return collection
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Add documents to the vector store
        
        Args:
            documents: List of text documents to embed
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
        """
        try:
            # Ensure all metadata values are strings (ChromaDB requirement)
            clean_metadatas = []
            for meta in metadatas:
                clean_meta = {k: str(v) if v is not None else "" for k, v in meta.items()}
                clean_metadatas.append(clean_meta)
            
            self.collection.add(
                documents=documents,
                metadatas=clean_metadatas,
                ids=ids
            )
            logger.info(f"Added {len(documents)} documents to collection")
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def add_documents_batch(
        self,
        documents: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str],
        batch_size: int = 100
    ) -> None:
        """
        Add documents in batches for better performance
        
        Args:
            documents: List of text documents
            metadatas: List of metadata dictionaries
            ids: List of unique document IDs
            batch_size: Number of documents per batch
        """
        total = len(documents)
        
        for i in range(0, total, batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_meta = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            
            self.add_documents(batch_docs, batch_meta, batch_ids)
            logger.info(f"Progress: {min(i + batch_size, total)}/{total} documents")
    
    def query(
        self,
        query_text: str = None,
        query_texts: List[str] = None,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None,
        where_document: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query the vector store (supports both single query_text and list query_texts)
        
        Args:
            query_text: Single query string (converts to list internally)
            query_texts: List of query strings
            n_results: Number of results to return per query
            where: Metadata filter conditions
            where_document: Document content filter conditions
            
        Returns:
            Query results with documents, metadatas, and distances
        """
        try:
            # Support both single and multiple queries
            if query_text:
                query_texts = [query_text]
            elif not query_texts:
                raise ValueError("Must provide either query_text or query_texts")
            
            results = self.collection.query(
                query_texts=query_texts,
                n_results=n_results,
                where=where,
                where_document=where_document
            )
            return results
        except Exception as e:
            logger.error(f"Error querying collection: {e}")
            raise
    
    def query_with_scores(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Query and return results with similarity scores
        
        Args:
            query_text: Single query string
            n_results: Number of results
            where: Metadata filter
            
        Returns:
            List of result dictionaries with scores
        """
        results = self.query([query_text], n_results=n_results, where=where)
        
        formatted_results = []
        
        for i in range(len(results['ids'][0])):
            formatted_results.append({
                'id': results['ids'][0][i],
                'document': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i],
                'similarity': 1 - results['distances'][0][i]  # Convert distance to similarity
            })
        
        return formatted_results
    
    def delete_collection(self) -> None:
        """Delete the current collection"""
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
    
    def reset_collection(self) -> None:
        """Reset collection by deleting and recreating"""
        self.delete_collection()
        self.collection = self._get_or_create_collection()
        logger.info(f"Reset collection: {self.collection_name}")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection"""
        count = self.collection.count()
        
        # Sample some documents to understand metadata structure
        sample = self.collection.peek(limit=5)
        
        return {
            "name": self.collection_name,
            "document_count": count,
            "persist_directory": self.persist_directory,
            "sample_metadata": sample['metadatas'][:3] if sample['metadatas'] else []
        }
    
    def filter_by_metadata(
        self,
        query_text: str,
        filters: Dict[str, Any],
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query with metadata filters (e.g., filter by team, position, league)
        
        Args:
            query_text: Query string
            filters: Metadata filters (e.g., {"team": "Liverpool", "position": "FW"})
            n_results: Number of results
            
        Returns:
            Filtered query results
        """
        return self.query_with_scores(query_text, n_results=n_results, where=filters)
    
    def get_by_ids(self, ids: List[str]) -> Dict[str, Any]:
        """
        Retrieve specific documents by their IDs
        
        Args:
            ids: List of document IDs
            
        Returns:
            Documents and metadata
        """
        try:
            results = self.collection.get(ids=ids)
            return results
        except Exception as e:
            logger.error(f"Error retrieving documents by IDs: {e}")
            return {"ids": [], "documents": [], "metadatas": []}


def main():
    """Test database functionality"""
    # Initialize database
    db = VectorDatabase()
    
    # Get stats
    stats = db.get_collection_stats()
    logger.info(f"Collection stats: {stats}")
    
    # Test query if collection has data
    if stats['document_count'] > 0:
        test_query = "strikers with high goal scoring"
        results = db.query_with_scores(test_query, n_results=3)
        
        logger.info(f"\nTest query: '{test_query}'")
        for i, result in enumerate(results, 1):
            logger.info(f"\nResult {i}:")
            logger.info(f"Similarity: {result['similarity']:.3f}")
            logger.info(f"Document: {result['document'][:200]}...")
            logger.info(f"Metadata: {result['metadata']}")


if __name__ == "__main__":
    main()
