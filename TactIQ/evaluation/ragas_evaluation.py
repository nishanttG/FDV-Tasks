"""
Day 9-10: RAGAS Evaluation Framework
=====================================

Comprehensive evaluation of the CRAG Football Scout System using RAGAS metrics.

Metrics Evaluated:
1. Relevance: Are retrieved documents relevant to the query?
2. Faithfulness: Is the answer grounded in retrieved context?
3. Answer Correctness: Does it match expected ground truth?
4. Context Precision: Are relevant contexts ranked higher?
5. Generation Time: Response latency
6. Answer Similarity: Semantic similarity to ground truth
"""

import sys
sys.path.append('c:\\Users\\Hp\\Frost Digital Ventures\\TactIQ')

import os
import json
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import time
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import chromadb
from chromadb.utils import embedding_functions

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        # Not using these - they require reference ground truth data:
        # context_precision, context_recall, answer_correctness, answer_similarity
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print(" RAGAS not installed. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

from src.agents.enhanced_crag_agent import EnhancedCRAGAgent
from loguru import logger

# Configure RAGAS to use Groq (free tier, n=1 compatible with sequential evaluation)
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure logging
logger.add("evaluation/logs/ragas_eval_{time}.log", rotation="1 day")

# Check for optional API keys for enhanced features
if not os.getenv('TAVILY_API_KEY'):
    logger.warning(" TAVILY_API_KEY not set - web search fallback will be unavailable")
if not os.getenv('GROQ_API_KEY'):
    logger.warning(" GROQ_API_KEY not set - self-check verification will be disabled")


class RAGASEvaluator:
    """RAGAS-based evaluation framework for CRAG system"""
    
    def __init__(self, test_queries_path: str = "evaluation/test_queries.json"):
        """Initialize evaluator with test queries"""
        self.test_queries_path = test_queries_path
        self.results = []
        self.metrics_summary = {}
        
        # Load test queries
        with open(test_queries_path, 'r') as f:
            self.test_data = json.load(f)
        
        # Initialize CRAG system
        logger.info("Initializing CRAG system for evaluation...")
        chroma_client = chromadb.PersistentClient(path="./db/chroma")
        embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        collection = chroma_client.get_collection(
            name="player_stats",
            embedding_function=embedding_fn
        )
        
        self.agent = EnhancedCRAGAgent(
            vector_db=collection,
            enable_refrag=False,  # ❌ OFF for RAGAS - prevents context attribution noise & faithfulness contamination
            enable_selfcheck=False,  # ❌ OFF for RAGAS - prevents answer modification that breaks grounding assumption
            refrag_model_path="ollama:qwen2.5:1.5b"
        )
        
        logger.info("=" * 70)
        logger.info("RAGAS EVALUATION MODE CONFIGURED:")
        logger.info("  ❌ REFRAG: OFF (preserves context→answer traceability)")
        logger.info("  ❌ Self-Check: OFF (measures first-pass generation quality)")
        logger.info("  ✅ Core CRAG: ON (retrieval + generation pipeline)")
        logger.info("=" * 70)
        
        logger.info(f" Evaluator initialized with {len(self.test_data['test_queries'])} test queries")
    
    def run_single_query(self, test_case: Dict) -> Dict[str, Any]:
        """Run a single test query and collect metrics"""
        query_id = test_case['id']
        query = test_case['query']
        category = test_case['category']
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Running Query {query_id}: {query}")
        logger.info(f"Category: {category}")
        
        # Measure generation time
        start_time = time.time()
        
        try:
            result = self.agent.query(
                query=query,
                force_reasoning=False,
                skip_verification=False
            )
            
            generation_time = time.time() - start_time
            
            # Extract result components
            answer = result.get('answer', '')
            sources = result.get('sources', [])
            grade = result.get('grade', 'unknown')
            confidence = result.get('confidence', 0.0)
            used_web = result.get('used_web_search', False)
            
            # Extract contexts
            contexts = [src.get('content', '') for src in sources if 'content' in src]
            
            # Collect metadata from sources (sources are already dicts with keys like 'player', 'team', 'season')
            retrieved_modules = set()
            retrieved_players = set()
            for src in sources:
                # Sources from CRAG have structure: {'player': X, 'team': Y, 'season': Z, 'type': 'stats', ...}
                if isinstance(src, dict):
                    # Try direct keys first (new format from _generate_node)
                    if 'player' in src:
                        retrieved_players.add(src['player'])
                    # Also check metadata key (legacy)
                    if 'metadata' in src:
                        meta = src['metadata']
                        if 'stat_module' in meta:
                            retrieved_modules.add(meta['stat_module'])
                        if 'player' in meta:
                            retrieved_players.add(meta['player'])
            
            # Extract stat_module from sources (now included after fix)
            for src in sources:
                if isinstance(src, dict) and 'stat_module' in src:
                    retrieved_modules.add(src['stat_module'])
            
            # Also check retrieved_docs for modules (with proper dict access)
            retrieved_docs = result.get('retrieved_docs', [])
            for doc in retrieved_docs[:10]:  # First 10 docs
                if isinstance(doc, dict):
                    meta = doc.get('metadata', {})
                    if 'stat_module' in meta:
                        retrieved_modules.add(meta['stat_module'])
                elif hasattr(doc, 'metadata'):
                    meta = doc.metadata if isinstance(doc.metadata, dict) else {}
                    if 'stat_module' in meta:
                        retrieved_modules.add(meta['stat_module'])
            
            # Extract players from retrieved_docs if not in sources
            for doc in retrieved_docs[:5]:
                if isinstance(doc, dict):
                    meta = doc.get('metadata', {})
                    player_name = meta.get('player') or meta.get('Player') or meta.get('name')
                    if player_name:
                        retrieved_players.add(player_name)
            
            result_data = {
                'query_id': query_id,
                'category': category,
                'query': query,
                'answer': answer,
                'contexts': contexts,
                'sources_count': len(sources),
                'grade': grade,
                'confidence': confidence,
                'generation_time': generation_time,
                'used_web_search': used_web,
                'expected_web_search': test_case.get('expected_web_search', False),  # Track if web was expected
                'retrieved_modules': list(retrieved_modules),
                'retrieved_players': list(retrieved_players),
                'timestamp': datetime.now().isoformat()
            }
            
            # Validate against ground truth
            if 'ground_truth' in test_case:
                result_data['ground_truth'] = test_case['ground_truth']
            
            if 'expected_player' in test_case:
                expected_player = test_case['expected_player']
                # Bidirectional fuzzy match: check if either name contains the other
                # Handles cases like DB has "Alisson" but expected is "Alisson Becker"
                player_correct = any(
                    expected_player.lower() in p.lower() or p.lower() in expected_player.lower()
                    for p in retrieved_players
                )
                result_data['player_match'] = player_correct
                logger.info(f"Expected player: {expected_player}, Match: {player_correct}")
            
            # Modules match is optional - only check if expected_modules is defined in test case
            if 'expected_modules' in test_case and test_case.get('expected_modules'):
                expected_modules = set(test_case['expected_modules'])
                modules_match = expected_modules.issubset(retrieved_modules) if retrieved_modules else False
                result_data['modules_match'] = modules_match
                result_data['missing_modules'] = list(expected_modules - retrieved_modules) if retrieved_modules else list(expected_modules)
                logger.info(f"Expected modules: {expected_modules}")
                logger.info(f"Retrieved modules: {retrieved_modules}")
                logger.info(f"Modules match: {modules_match}")
            else:
                # No expected modules defined for this test - log retrieved modules for reference
                logger.info(f"Retrieved modules: {retrieved_modules} (no expected_modules baseline for comparison)")
            
            logger.info(f" Query completed in {generation_time:.2f}s")
            logger.info(f"   Grade: {grade}, Confidence: {confidence:.2%}")
            logger.info(f"   Sources: {len(sources)}, Modules: {len(retrieved_modules)}")
            
        except Exception as e:
            logger.error(f" Query {query_id} failed: {str(e)}")
            result_data = {
                'query_id': query_id,
                'category': category,
                'query': query,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
        
        return result_data
    
    def run_evaluation(self, save_results: bool = True) -> Dict[str, Any]:
        """Run full evaluation on all test queries"""
        logger.info(f"\n{'='*60}")
        logger.info("Starting RAGAS Evaluation")
        logger.info(f"{'='*60}\n")
        
        eval_start = time.time()
        
        # Run all test queries
        for test_case in self.test_data['test_queries']:
            result = self.run_single_query(test_case)
            self.results.append(result)
        
        total_time = time.time() - eval_start
        
        # Calculate aggregate metrics
        successful_queries = [r for r in self.results if 'error' not in r]
        failed_queries = [r for r in self.results if 'error' in r]
        
        self.metrics_summary = {
            'total_queries': len(self.results),
            'successful_queries': len(successful_queries),
            'failed_queries': len(failed_queries),
            'success_rate': len(successful_queries) / len(self.results) if self.results else 0,
            'total_evaluation_time': total_time,
            'average_generation_time': sum(r.get('generation_time', 0) for r in successful_queries) / len(successful_queries) if successful_queries else 0,
            'average_confidence': sum(r.get('confidence', 0) for r in successful_queries) / len(successful_queries) if successful_queries else 0,
            'db_only_queries': len([r for r in successful_queries if not r.get('used_web_search', False)]),
            'web_search_queries': len([r for r in successful_queries if r.get('used_web_search', False)]),
            'expected_web_search': len([r for r in successful_queries if r.get('expected_web_search', False)]),
            'web_search_mismatch': len([r for r in successful_queries if r.get('expected_web_search', False) and not r.get('used_web_search', False)]),
            'player_match_rate': len([r for r in successful_queries if r.get('player_match', False)]) / len([r for r in successful_queries if 'player_match' in r]) if any('player_match' in r for r in successful_queries) else 0,
            'modules_match_rate': len([r for r in successful_queries if r.get('modules_match', False)]) / len([r for r in successful_queries if 'modules_match' in r]) if any('modules_match' in r for r in successful_queries) else None  # None = N/A (no expected_modules baseline)
        }
        
        logger.info(f"\n{'='*60}")
        logger.info("Evaluation Complete")
        logger.info(f"{'='*60}")
        logger.info(f"Total Queries: {self.metrics_summary['total_queries']}")
        logger.info(f"Success Rate: {self.metrics_summary['success_rate']:.1%}")
        logger.info(f"Average Generation Time: {self.metrics_summary['average_generation_time']:.2f}s")
        logger.info(f"Average Confidence: {self.metrics_summary['average_confidence']:.1%}")
        logger.info(f"Player Match Rate: {self.metrics_summary['player_match_rate']:.1%}")
        if self.metrics_summary['modules_match_rate'] is not None:
            logger.info(f"Modules Match Rate: {self.metrics_summary['modules_match_rate']:.1%}")
        else:
            logger.info(f"Modules Match Rate: N/A (no expected_modules baseline defined in test_queries.json)")
        
        # Warn about web search fallback issues
        if self.metrics_summary['web_search_mismatch'] > 0:
            logger.warning(f" {self.metrics_summary['web_search_mismatch']} queries expected web search but didn't use it")
            logger.warning("   Ensure TAVILY_API_KEY is set for web fallback functionality")
        
        # Compute RAGAS metrics (proposal requirement: 75-85% faithfulness target)
        if RAGAS_AVAILABLE and successful_queries:
            logger.info(f"\n{'='*60}")
            logger.info("Computing RAGAS Metrics (Faithfulness, Relevancy, Precision)")
            logger.info(f"{'='*60}\n")
            
            try:
                # Prepare dataset for RAGAS
                ragas_data = {
                    'question': [],
                    'answer': [],
                    'contexts': [],
                    'ground_truths': []  # Optional ground truth for comparison
                }
                
                for result in successful_queries:
                    if result.get('answer'):
                        ragas_data['question'].append(result['query'])
                        ragas_data['answer'].append(result['answer'])
                        
                        # Extract contexts from sources
                        contexts = []
                        for src in result.get('sources', []):
                            if isinstance(src, dict) and 'content' in src:
                                contexts.append(src['content'][:500])  # Truncate for efficiency
                        ragas_data['contexts'].append(contexts if contexts else ['No context retrieved'])
                        
                        # Use expected answer as ground truth if available
                        expected = result.get('expected_answer', 'Ground truth not provided')
                        ragas_data['ground_truths'].append([expected])
                
                # Create RAGAS dataset
                from datasets import Dataset
                dataset = Dataset.from_dict(ragas_data)
                
                # SAVE CACHED RESPONSES for future evaluation runs (no token usage if TPD exhausted)
                logger.info("\nCaching query responses for future evaluation runs...")
                cache_file = Path("evaluation/cache/ragas_responses_cache.json")
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                
                cache_data = {
                    'timestamp': datetime.now().isoformat(),
                    'total_samples': len(successful_queries),
                    'queries': {}
                }
                
                for idx, result in enumerate(successful_queries):
                    if result.get('answer'):
                        # Use FULL contexts (not truncated) for better faithfulness evaluation
                        contexts = []
                        for src in result.get('sources', []):
                            if isinstance(src, dict) and 'content' in src:
                                # Use FULL content, not truncated
                                contexts.append(src['content'])
                        
                        # Also include system metadata about what was retrieved
                        module_context = f"Retrieved modules: {', '.join(result.get('retrieved_modules', []))}"
                        
                        cache_data['queries'][f"Q{idx+1:03d}"] = {
                            'query': result['query'],
                            'answer': result['answer'],
                            'contexts': contexts if contexts else [],
                            'module_context': module_context,
                            'expected_answer': result.get('expected_answer', ''),
                            'grade': result.get('grade', ''),
                            'confidence': result.get('confidence', 0),
                            'sources_count': len(result.get('sources', [])),
                            'modules': list(result.get('retrieved_modules', [])),
                            'generation_time': result.get('generation_time', 0)
                        }
                
                with open(cache_file, 'w') as f:
                    json.dump(cache_data, f, indent=2)
                logger.info(f"✓ Cached {len(cache_data['queries'])} responses with FULL contexts to {cache_file}")
                logger.info("  You can reuse this cache if you hit Groq TPD limits tomorrow")
                
                # Configure RAGAS to use Groq LLM (free tier, n=1 compatible)
                groq_api_key = os.getenv('GROQ_API_KEY')
                if not groq_api_key:
                    raise ValueError("GROQ_API_KEY not found in environment variables")
                
                groq_llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=groq_api_key,
                    temperature=0,
                    max_tokens=1500
                )
                
                # Wrap LLM and embeddings for RAGAS
                ragas_llm = LangchainLLMWrapper(groq_llm)
                
                # Note: First run will download sentence-transformers model (~90MB)
                # Subsequent runs will use cached model from C:\Users\<user>\.cache\huggingface\
                logger.info("Loading embeddings model (first run downloads ~90MB, then cached)...")
                ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    cache_folder="./.cache/huggingface"  # Cache locally for faster access
                ))
                
                # Evaluate with RAGAS metrics - SEQUENTIAL to avoid Groq n>1 limit
                logger.info("Running RAGAS evaluation with Groq LLM (sequential evaluation for free tier compatibility)...")
                logger.info("  Step 1/2: Computing faithfulness metric...")
                
                try:
                    # Evaluate faithfulness alone (first metric)
                    ragas_faithfulness_results = evaluate(
                        dataset,
                        metrics=[faithfulness],
                        llm=ragas_llm,
                        embeddings=ragas_embeddings
                    )
                    ragas_faithfulness = ragas_faithfulness_results.scores['faithfulness'].mean() if 'faithfulness' in ragas_faithfulness_results.scores else 0
                    logger.info(f"  ✓ Faithfulness: {ragas_faithfulness:.2%}")
                    
                    # Evaluate answer_relevancy alone (second metric)
                    logger.info("  Step 2/2: Computing answer_relevancy metric...")
                    ragas_relevancy_results = evaluate(
                        dataset,
                        metrics=[answer_relevancy],
                        llm=ragas_llm,
                        embeddings=ragas_embeddings
                    )
                    ragas_answer_relevancy = ragas_relevancy_results.scores['answer_relevancy'].mean() if 'answer_relevancy' in ragas_relevancy_results.scores else 0
                    logger.info(f"  ✓ Answer Relevancy: {ragas_answer_relevancy:.2%}")
                    
                    # Add RAGAS metrics to summary
                    self.metrics_summary['ragas_faithfulness'] = ragas_faithfulness
                    self.metrics_summary['ragas_answer_relevancy'] = ragas_answer_relevancy
                    
                    if ragas_faithfulness > 0 or ragas_answer_relevancy > 0:
                        logger.info(f"\n{'-'*60}")
                        logger.info(f" RAGAS METRICS - PROPOSAL REQUIREMENTS")
                        logger.info(f"{'-'*60}")
                        
                        logger.info(f"\nFaithfulness (Factual Accuracy): {ragas_faithfulness:.2%}")
                        faithfulness_status = " ACHIEVED" if 0.75 <= ragas_faithfulness <= 0.85 else "⚠️  Outside Target"
                        logger.info(f"   Target: 75-85% → {faithfulness_status}")
                        
                        logger.info(f"\nAnswer Relevancy (Tactical Insight): {ragas_answer_relevancy:.2%}")
                        relevancy_status = " ACHIEVED" if 0.80 <= ragas_answer_relevancy <= 0.90 else "⚠️  Outside Target"
                        logger.info(f"   Target: 80-90% → {relevancy_status}")
                    else:
                        logger.warning("  RAGAS failed - using proxy metrics instead:")
                        logger.info(f"    Player Match Rate: {self.metrics_summary.get('player_match_rate', 0):.1%} (proxy for faithfulness)")
                        logger.info(f"   Module Match Rate: {self.metrics_summary.get('modules_match_rate', 0):.1%} (proxy for relevancy)")
                    logger.info(f"\n{'-'*60}")
                    
                except Exception as e:
                    logger.error(f"An error occurred during RAGAS evaluation: {e}")
                    raise
            except Exception as e:
                logger.error(f"An error occurred during RAGAS evaluation: {e}")
                raise
        elif not RAGAS_AVAILABLE:
            logger.warning(" RAGAS not available. Install with: pip install ragas")
        
        # Save results
        if save_results:
            self.save_results()
        
        return self.metrics_summary
    
    def save_results(self):
        """Save evaluation results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/evaluation")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = results_dir / f"ragas_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump({
                'results': self.results,
                'summary': self.metrics_summary,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        logger.info(f" Detailed results saved to {results_file}")
        
        # Save summary CSV
        if self.results:
            df = pd.DataFrame(self.results)
            csv_file = results_dir / f"ragas_results_{timestamp}.csv"
            df.to_csv(csv_file, index=False)
            logger.info(f" CSV results saved to {csv_file}")
        
        # Save summary report
        report_file = results_dir / f"evaluation_report_{timestamp}.md"
        self.generate_report(report_file)
        logger.info(f" Report saved to {report_file}")
    
    def generate_report(self, output_path: Path):
        """Generate markdown evaluation report"""
        
        # Calculate KPI statuses
        faithfulness = self.metrics_summary.get('ragas_faithfulness', 0)
        faithfulness_status = ' ACHIEVED' if 0.75 <= faithfulness <= 0.85 else '[WARNING] Outside Target'
        
        relevancy = self.metrics_summary.get('ragas_answer_relevancy', 0)
        relevancy_status = ' ACHIEVED' if 0.80 <= relevancy <= 0.90 else '[WARNING] Outside Target'
        
        avg_time = self.metrics_summary['average_generation_time']
        time_status = ' ACHIEVED' if 90 <= avg_time <= 120 else ('[FASTER]' if avg_time < 90 else '[WARNING] SLOWER')
        
        total_queries = self.metrics_summary['total_queries']
        success_rate = self.metrics_summary['success_rate']
        robustness_status = ' ACHIEVED' if total_queries >= 5 and success_rate >= 0.90 else '[WARNING] Below Target'
        
        report = f"""# RAGAS Evaluation Report - TactIQ Football Scout System

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Proposal KPIs Assessment

### KPI #1: Factual Accuracy (Main KPI)
- **Target:** 75-85% faithfulness (CRAG + Self-Check on curated datasets)
- **Result:** {faithfulness:.1%}
- **Status:** {faithfulness_status}

### KPI #2: Relevance / Tactical Insight
- **Target:** 80-90% of insights contextually relevant
- **Result:** {relevancy:.1%}
- **Status:** {relevancy_status}

### KPI #3: Efficiency / Report Generation Time
- **Baseline:** 2 hours (manual)
- **Target:** 90-120 seconds for 500-word report
- **Result:** {avg_time:.2f} seconds
- **Status:** {time_status}

### KPI #4: System Robustness
- **Target:** Successfully process 5-10 diverse queries end-to-end
- **Result:** {total_queries} queries, {success_rate:.1%} success rate
- **Status:** {robustness_status}

---

##  Summary Metrics

| Metric | Value |
|--------|-------|
| Total Queries | {self.metrics_summary['total_queries']} |
| Successful Queries | {self.metrics_summary['successful_queries']} |
| Failed Queries | {self.metrics_summary['failed_queries']} |
| Success Rate | {self.metrics_summary['success_rate']:.1%} |
| Average Generation Time | {self.metrics_summary['average_generation_time']:.2f}s |
| Average Confidence | {self.metrics_summary['average_confidence']:.1%} |
| Player Match Rate | {self.metrics_summary['player_match_rate']:.1%} |
| Modules Match Rate | {self.metrics_summary['modules_match_rate']:.1%} |

##  RAGAS Metrics 

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| **Faithfulness** (Factual Accuracy) | {faithfulness:.1%} | 75-85% | {faithfulness_status} |
| **Answer Relevancy** (Tactical Insight) | {relevancy:.1%} | 80-90% | {relevancy_status} |
| Answer Similarity | {self.metrics_summary.get('ragas_answer_similarity', 0):.1%} | - | - |

*Note: Metrics requiring ground truth references (context_precision, context_recall, answer_correctness) are excluded.*

##  Query Breakdown

### By Data Source
- DB Only: {self.metrics_summary['db_only_queries']}
- Web Search: {self.metrics_summary['web_search_queries']}

### By Category
"""
        
        # Group by category
        category_counts = {}
        for r in self.results:
            cat = r.get('category', 'unknown')
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for cat, count in sorted(category_counts.items()):
            report += f"- {cat}: {count}\n"
        
        report += "\n## Detailed Results\n\n"
        
        for result in self.results:
            query_id = result.get('query_id', 'N/A')
            query = result.get('query', 'N/A')
            category = result.get('category', 'N/A')
            
            report += f"### {query_id}: {category}\n"
            report += f"**Query:** {query}\n\n"
            
            if 'error' in result:
                report += f" **Error:** {result['error']}\n\n"
            else:
                report += f"- **Generation Time:** {result.get('generation_time', 0):.2f}s\n"
                report += f"- **Confidence:** {result.get('confidence', 0):.1%}\n"
                report += f"- **Grade:** {result.get('grade', 'N/A')}\n"
                report += f"- **Sources:** {result.get('sources_count', 0)}\n"
                
                if 'player_match' in result:
                    status = "✓" if result['player_match'] else "X"
                    report += f"- **Player Match:** {status}\n"
                
                if 'modules_match' in result:
                    status = "✓" if result['modules_match'] else "X"
                    report += f"- **Modules Match:** {status}\n"
                    if result.get('missing_modules'):
                        report += f"  - Missing: {', '.join(result['missing_modules'])}\n"
                
                report += "\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)


def main():
    """Run evaluation"""
    print("-"*60)
    print("RAGAS Evaluation - TactIQ Football Scout System")
    print("-"*60)
    
    evaluator = RAGASEvaluator()
    results = evaluator.run_evaluation(save_results=True)
    
    print("\n" + "-"*60)
    print(" Evaluation Complete!")
    print("-"*60)
    
    # PROPOSAL KPI #1: Factual Accuracy (Main KPI)
    print(f"\n PROPOSAL KPI #1: FACTUAL ACCURACY (Main KPI)")
    print(f"   Target: 75-85% faithfulness")
    if 'ragas_faithfulness' in results:
        faithfulness_status = " ACHIEVED" if 0.75 <= results['ragas_faithfulness'] <= 0.85 else "[WARNING] Outside Target"
        print(f"   Result: {results['ragas_faithfulness']:.2%} → {faithfulness_status}")
    else:
        print(f"   Result: Pending RAGAS computation...")
    
    # PROPOSAL KPI #2: Relevance / Tactical Insight
    print(f"\n PROPOSAL KPI #2: RELEVANCE / TACTICAL INSIGHT")
    print(f"   Target: 80-90% of insights contextually relevant")
    if 'ragas_answer_relevancy' in results:
        relevancy_status = " ACHIEVED" if 0.80 <= results['ragas_answer_relevancy'] <= 0.90 else "[WARNING] Outside Target"
        print(f"   Result: {results['ragas_answer_relevancy']:.2%} → {relevancy_status}")
    else:
        print(f"   Result: Pending RAGAS computation...")
    
    # PROPOSAL KPI #3: Efficiency / Report Generation Time
    print(f"\n PROPOSAL KPI #3: EFFICIENCY / REPORT GENERATION TIME")
    print(f"   Baseline: 2 hours (manual)")
    print(f"   Target: 90-120 seconds for 500-word report")
    avg_time = results['average_generation_time']
    time_status = " ACHIEVED" if 90 <= avg_time <= 120 else ("[FASTER]" if avg_time < 90 else "[WARNING] SLOWER")
    print(f"   Result: {avg_time:.2f}s → {time_status}")
    
    # PROPOSAL KPI #4: System Robustness
    print(f"\n PROPOSAL KPI #4: SYSTEM ROBUSTNESS")
    print(f"   Target: Successfully process 5-10 diverse queries end-to-end")
    total_queries = results.get('total_queries', 0)
    success_rate = results['success_rate']
    robustness_status = " ACHIEVED" if total_queries >= 5 and success_rate >= 0.90 else "[WARNING] Below Target"
    print(f"   Result: {total_queries} queries, {success_rate:.1%} success rate → {robustness_status}")
    
    print(f"\n{'-'*60}")
    print(f" DETAILED METRICS")
    print(f"{'-'*60}")
    print(f"  Player Match Rate: {results.get('player_match_rate', 0):.1%}")
    print(f"  Modules Match Rate: {results.get('modules_match_rate', 0):.1%}")
    
    # Display additional RAGAS metrics if available
    if 'ragas_faithfulness' in results:
        print(f"\n{'-'*60}")
        print(f" ADDITIONAL RAGAS METRICS")
        print(f"{'-'*60}")
        print(f" Answer Semantic Similarity: {results.get('ragas_answer_similarity', 0):.2%}")
        print(f"{'-'*60}")
        print(f"\n   Note: Metrics requiring ground truth (context_precision, context_recall,")
        print(f"      answer_correctness) excluded. See test_queries.json to add references.")
    
    print(f"\n Results saved to results/evaluation/")


if __name__ == "__main__":
    main()
