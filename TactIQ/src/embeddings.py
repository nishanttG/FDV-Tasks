"""
Embedding Pipeline for TactIQ
Handles text embedding using sentence-transformers all-MiniLM-L6-v2
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import numpy as np
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()


class EmbeddingPipeline:
    """Generates embeddings for text documents using sentence-transformers"""
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize embedding model
        
        Args:
            model_name: Sentence-transformers model name
        """
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", 
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Remove 'sentence-transformers/' prefix if present for loading
        model_id = self.model_name.replace("sentence-transformers/", "")
        
        logger.info(f"Loading embedding model: {model_id}")
        self.model = SentenceTransformer(model_id)
        logger.info(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error embedding text: {e}")
            raise
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for multiple texts in batches
        
        Args:
            texts: List of text strings
            batch_size: Batch size for encoding
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embedding vectors
        """
        try:
            logger.info(f"Embedding {len(texts)} texts with batch size {batch_size}")
            
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error in batch embedding: {e}")
            raise
    
    def embed_documents(
        self,
        documents: List[str],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Embed documents and return as list of arrays
        
        Args:
            documents: List of document texts
            batch_size: Batch size for encoding
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.embed_batch(documents, batch_size=batch_size)
        return [emb for emb in embeddings]
    
    def compute_similarity(
        self,
        text1: Union[str, np.ndarray],
        text2: Union[str, np.ndarray]
    ) -> float:
        """
        Compute cosine similarity between two texts or embeddings
        
        Args:
            text1: First text string or embedding
            text2: Second text string or embedding
            
        Returns:
            Cosine similarity score
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Get embeddings if inputs are strings
        if isinstance(text1, str):
            text1 = self.embed_text(text1)
        if isinstance(text2, str):
            text2 = self.embed_text(text2)
        
        # Reshape for sklearn
        text1 = text1.reshape(1, -1)
        text2 = text2.reshape(1, -1)
        
        similarity = cosine_similarity(text1, text2)[0][0]
        return float(similarity)
    
    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        Find most similar candidates to query
        
        Args:
            query: Query text
            candidates: List of candidate texts
            top_k: Number of top results to return
            
        Returns:
            List of (index, similarity_score, text) tuples
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Embed query
        query_emb = self.embed_text(query).reshape(1, -1)
        
        # Embed candidates
        candidate_embs = self.embed_batch(candidates, show_progress=False)
        
        # Compute similarities
        similarities = cosine_similarity(query_emb, candidate_embs)[0]
        
        # Get top k
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = [
            (idx, float(similarities[idx]), candidates[idx])
            for idx in top_indices
        ]
        
        return results
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings produced by the model"""
        return self.model.get_sentence_embedding_dimension()


class ChunkingStrategy:
    """Handles text chunking for optimal embedding"""
    
    @staticmethod
    def chunk_by_sentences(
        text: str,
        max_chunk_size: int = 500,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into chunks by sentences with overlap
        
        Args:
            text: Input text
            max_chunk_size: Maximum characters per chunk
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        import re
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if overlap > 0 and current_chunk:
                    overlap_text = current_chunk[-overlap:]
                    current_chunk = overlap_text + sentence + " "
                else:
                    current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    @staticmethod
    def chunk_by_tokens(
        text: str,
        max_tokens: int = 256,
        overlap_tokens: int = 32
    ) -> List[str]:
        """
        Split text into chunks by approximate token count
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk (approximate)
            overlap_tokens: Overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        # Approximate: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        overlap_chars = overlap_tokens * 4
        
        return ChunkingStrategy.chunk_by_sentences(text, max_chars, overlap_chars)


def main():
    """Test embedding pipeline"""
    # Initialize pipeline
    pipeline = EmbeddingPipeline()
    
    # Test single embedding
    test_text = "Mohamed Salah is a prolific striker who scores many goals for Liverpool."
    embedding = pipeline.embed_text(test_text)
    logger.info(f"Single embedding shape: {embedding.shape}")
    
    # Test batch embedding
    test_texts = [
        "Erling Haaland is a powerful striker with excellent positioning.",
        "Kevin De Bruyne is a creative midfielder known for his passing.",
        "Mohamed Salah excels at cutting inside from the right wing.",
    ]
    
    embeddings = pipeline.embed_batch(test_texts)
    logger.info(f"Batch embeddings shape: {embeddings.shape}")
    
    # Test similarity
    query = "Find a prolific goal scorer"
    results = pipeline.find_most_similar(query, test_texts, top_k=3)
    
    logger.info(f"\nQuery: '{query}'")
    for idx, score, text in results:
        logger.info(f"Score {score:.3f}: {text}")
    
    # Test chunking
    long_text = """
    Manchester City dominated the match with 70% possession. 
    Kevin De Bruyne orchestrated attacks from midfield. 
    Erling Haaland scored a clinical hat-trick. 
    The defense remained solid throughout the game.
    Pep Guardiola's tactical adjustments in the second half were crucial.
    """
    
    chunks = ChunkingStrategy.chunk_by_sentences(long_text, max_chunk_size=100)
    logger.info(f"\nChunked into {len(chunks)} pieces:")
    for i, chunk in enumerate(chunks, 1):
        logger.info(f"Chunk {i}: {chunk}")


if __name__ == "__main__":
    main()
