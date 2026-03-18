"""
Day 3 Production CRAG System
============================

Production-ready CRAG (Corrective RAG) query system with:
- Player name detection
- LangGraph workflow orchestration
- LLM-based retrieval grading
- Web search fallback
- Interactive CLI interface
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

import chromadb
from loguru import logger
from src.agents.crag_agent import CRAGAgent
from datetime import datetime


class CRAGQuerySystem:
    """Production CRAG query system"""
    
    def __init__(self, chroma_path: str = "./db/chroma", collection_name: str = "player_stats"):
        """
        Initialize CRAG system
        
        Args:
            chroma_path: Path to ChromaDB
            collection_name: Collection name
        """
        logger.info("Initializing CRAG Query System...")
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma_client.get_collection(collection_name)
        
        # Get API keys
        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        
        # Check API keys
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY not set - using heuristic grading")
        if not self.tavily_api_key:
            logger.warning("TAVILY_API_KEY not set - web search unavailable")
        
        # Initialize CRAG agent
        try:
            self.crag_agent = CRAGAgent(
                vector_db=self.collection,
                groq_api_key=self.groq_api_key,
                tavily_api_key=self.tavily_api_key
            )
            logger.info("✓ CRAG system ready")
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            logger.error("Install with: pip install langgraph langchain-groq tavily-python")
            raise
    
    def query(self, user_query: str, verbose: bool = True) -> dict:
        """
        Process a query through CRAG system
        
        Args:
            user_query: User's natural language query
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with answer, sources, confidence, metadata
        """
        if verbose:
            print("\n" + "="*70)
            print(f"Query: {user_query}")
            print("="*70)
        
        start_time = datetime.now()
        
        # Process query
        result = self.crag_agent.query(user_query)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        if verbose:
            print(f"\n📊 Grade: {result['grade']}")
            print(f"🌐 Used web search: {result['used_web_search']}")
            print(f"🎯 Confidence: {result['confidence']:.2f}")
            print(f"⏱️  Time: {elapsed:.2f}s")
            print(f"\n💬 Answer:")
            print(result['answer'])
            print(f"\n📚 Sources ({len(result['sources'])}):")
            for i, source in enumerate(result['sources'][:5], 1):
                print(f"  {i}. {source}")
            if len(result['sources']) > 5:
                print(f"  ... and {len(result['sources']) - 5} more")
        
        result['query_time'] = elapsed
        return result
    
    def interactive_mode(self):
        """Run interactive query session"""
        print("\n" + "="*70)
        print("🏟️  CRAG FOOTBALL SCOUT ASSISTANT")
        print("="*70)
        print("\nAsk questions about:")
        print("  • Player performance and statistics")
        print("  • Tactical analysis and formations")
        print("  • Current season updates")
        print("  • Transfer market insights")
        print("\nType 'quit' or 'exit' to stop\n")
        
        query_count = 0
        
        while True:
            try:
                user_input = input("📝 Your question: ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print(f"\n✓ Processed {query_count} queries. Goodbye!")
                    break
                
                # Process query
                self.query(user_input, verbose=True)
                query_count += 1
                
            except KeyboardInterrupt:
                print(f"\n\n✓ Processed {query_count} queries. Goodbye!")
                break
            except Exception as e:
                logger.error(f"Query error: {e}")
                print(f"\n❌ Error: {str(e)}\n")
    
    def batch_query(self, queries: list, output_file: str = None):
        """
        Process multiple queries in batch
        
        Args:
            queries: List of query strings
            output_file: Optional file to save results
        """
        results = []
        
        print(f"\n📋 Processing {len(queries)} queries...")
        
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] Processing...")
            result = self.query(query, verbose=True)
            results.append({
                'query': query,
                'answer': result['answer'],
                'sources': result['sources'],
                'confidence': result['confidence'],
                'grade': result['grade'],
                'used_web_search': result['used_web_search'],
                'query_time': result['query_time']
            })
        
        # Save results if requested
        if output_file:
            import json
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results saved to: {output_file}")
        
        # Summary
        avg_time = sum(r['query_time'] for r in results) / len(results)
        avg_confidence = sum(r['confidence'] for r in results) / len(results)
        web_search_count = sum(1 for r in results if r['used_web_search'])
        
        print("\n" + "="*70)
        print("📊 BATCH SUMMARY")
        print("="*70)
        print(f"Total queries: {len(queries)}")
        print(f"Average time: {avg_time:.2f}s")
        print(f"Average confidence: {avg_confidence:.2f}")
        print(f"Web search used: {web_search_count}/{len(queries)}")
        
        return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="CRAG Football Scout Assistant")
    parser.add_argument('--query', '-q', type=str, help="Single query to process")
    parser.add_argument('--batch', '-b', type=str, help="File with queries (one per line)")
    parser.add_argument('--output', '-o', type=str, help="Output file for batch results")
    parser.add_argument('--interactive', '-i', action='store_true', help="Interactive mode")
    parser.add_argument('--quiet', action='store_true', help="Minimal output")
    
    args = parser.parse_args()
    
    # Configure logging
    logger.remove()
    if not args.quiet:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="WARNING")
    
    try:
        # Initialize system
        system = CRAGQuerySystem()
        
        if args.query:
            # Single query mode
            system.query(args.query, verbose=not args.quiet)
        
        elif args.batch:
            # Batch mode
            with open(args.batch, 'r', encoding='utf-8') as f:
                queries = [line.strip() for line in f if line.strip()]
            system.batch_query(queries, output_file=args.output)
        
        else:
            # Interactive mode (default)
            system.interactive_mode()
    
    except Exception as e:
        logger.error(f"System error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
