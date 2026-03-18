"""
day5_6_demo.py
===============

Demo for REFRAG + Self-Check testing using real CRAG retrieval with local Ollama Qwen2.5-1.5B.
Supports:
1. REFRAG Reasoning Queries (with real CRAG context)
2. Self-Check Verification
3. Combined REFRAG + Self-Check
"""

import os
import sys
from loguru import logger
from dotenv import load_dotenv
import chromadb

# --- Add project root to path FIRST ---
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)
load_dotenv()

# --- Import agents from src ---
from src.agents.enhanced_crag_agent import EnhancedCRAGAgent
from src.agents.refrag_agent import REFRAGAgent


def run_refrag_demo(agent: EnhancedCRAGAgent):
    """
    Run example reasoning queries using EnhancedCRAGAgent with real CRAG retrieval
    """
    example_queries = [
        "Compare Erling Haaland and Mohamed Salah's scoring performance this season",
        "Why is Alexander Isak performing well at Newcastle?",
        "Who is the best young striker under 23 in Europe?"
    ]
    
    logger.info("=== REFRAG REASONING DEMO (with real CRAG retrieval) ===\n")
    
    for query in example_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}")
        
        # Use EnhancedCRAGAgent which connects REFRAG to real CRAG.query()
        result = agent.query(query, force_reasoning=True, skip_verification=True)
        
        logger.info(f"\n✅ Answer: {result['answer']}\n")
        logger.info(f"Grade: {result.get('grade', 'N/A')}")
        logger.info(f"Confidence: {result.get('confidence', 0.0):.2%}")
        logger.info(f"Used Web Search: {'Yes' if result.get('used_web_search') else 'No'}")
        
        if result.get('reasoning_trace'):
            logger.info("\n🔍 Reasoning trace:")
            for i, step in enumerate(result['reasoning_trace'], 1):
                logger.info(f"  {i}. {step}")
        
        if result.get('sources'):
            logger.info(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:5], 1):
                logger.info(f"  {i}. {source}")


def run_self_check_demo(agent: EnhancedCRAGAgent):
    """
    Run Self-Check verification demo with real CRAG retrieval
    """
    logger.info("\n=== SELF-CHECK VERIFICATION DEMO ===\n")
    
    example_queries = [
        "How many goals did Mohamed Salah score this season?",
        "Find young strikers under 23 with high potential"
    ]
    
    for query in example_queries:
        logger.info(f"\n{'='*80}")
        logger.info(f"Query: {query}")
        logger.info(f"{'='*80}")
        
        # Use EnhancedCRAGAgent with Self-Check enabled
        result = agent.query(query, force_reasoning=False, skip_verification=False)
        
        logger.info(f"\n✅ Answer: {result['answer']}\n")
        logger.info(f"Regenerated: {'Yes' if result.get('regenerated') else 'No'}")
        
        if result.get('verification'):
            ver = result['verification']
            logger.info(f"\n🔍 Verification:")
            logger.info(f"  Passed: {'✅' if ver['passed'] else '❌'}")
            logger.info(f"  Confidence: {ver['confidence']:.2%}")
            logger.info(f"  Verdict: {ver.get('verdict', 'N/A')}")
            if ver.get('issues'):
                logger.info(f"  Issues: {', '.join(ver['issues'][:3])}")


def main():
    """
    Main entry point for Day 5-6 demo
    """
    print("╭──────────────────────────────────────────────────────────────────╮")
    print("│ Day 5-6 Demo: REFRAG + Self-Check with Real CRAG Retrieval      │")
    print("│ Using local Ollama Qwen2.5:1.5b model                            │")
    print("╰──────────────────────────────────────────────────────────────────╯\n")
    
    logger.info("Initializing ChromaDB and CRAG Agent...")
    
    try:
        # Temporarily disable offline mode to allow embedding model download
        # (Only needed once - embedding model is separate from Ollama LLM)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        os.environ.pop("HF_HUB_OFFLINE", None)
        
        chroma_client = chromadb.PersistentClient(path="./db/chroma")
        
        # Use sentence-transformers embedding (will download ~90MB if not cached)
        # This is DIFFERENT from Ollama - it's for vector similarity search
        from chromadb.utils import embedding_functions
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        collection = chroma_client.get_collection(
            name="player_stats",
            embedding_function=embedding_fn
        )
        logger.info(f"✅ ChromaDB loaded ({collection.count()} documents)")
        
        # Initialize EnhancedCRAGAgent with local Ollama REFRAG
        logger.info("\nInitializing EnhancedCRAGAgent (REFRAG + Self-Check)...")
        agent = EnhancedCRAGAgent(
            vector_db=collection,
            enable_refrag=True,
            enable_selfcheck=True,
            refrag_model_path="ollama:qwen2.5:1.5b"  # Use local Ollama, no downloads
        )
        logger.info("✅ EnhancedCRAGAgent ready\n")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Menu
    print("🎯 Select test:")
    print("1. REFRAG Reasoning Queries (real CRAG context)")
    print("2. Self-Check Verification")
    print("3. Combined REFRAG + Self-Check")
    print("4. Run All Tests")
    
    try:
        choice = int(input("\nEnter choice (1-4): ").strip())
    except ValueError:
        print("Invalid input. Exiting.")
        sys.exit(1)

    if choice == 1:
        run_refrag_demo(agent)
    elif choice == 2:
        run_self_check_demo(agent)
    elif choice == 3:
        run_self_check_demo(agent)
        print("\n")
        run_refrag_demo(agent)
    elif choice == 4:
        run_self_check_demo(agent)
        print("\n")
        run_refrag_demo(agent)
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    logger.info("\n✅ Demo complete!")


if __name__ == "__main__":
    main()
