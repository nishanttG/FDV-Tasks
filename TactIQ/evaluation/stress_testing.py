"""
Stress Testing Framework
========================

Tests system behavior under load:
1. Concurrent queries
2. Rapid succession
3. Cache effectiveness
4. Memory usage
5. Response time degradation
"""

import sys
sys.path.append('c:\\Users\\Hp\\Frost Digital Ventures\\TactIQ')

import time
import json
import psutil
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions

from src.agents.enhanced_crag_agent import EnhancedCRAGAgent
from loguru import logger


class StressTester:
    """Stress testing framework for CRAG system"""
    
    def __init__(self):
        """Initialize stress tester"""
        logger.info("Initializing stress tester...")
        
        # Initialize CRAG system
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
            enable_refrag=True,
            enable_selfcheck=True
        )
        
        self.results = []
        self.process = psutil.Process()
    
    def get_system_metrics(self) -> Dict:
        """Get current system metrics"""
        return {
            'cpu_percent': self.process.cpu_percent(),
            'memory_mb': self.process.memory_info().rss / 1024 / 1024,
            'timestamp': datetime.now().isoformat()
        }
    
    def run_single_query(self, query: str, query_id: str) -> Dict:
        """Run a single query and collect metrics"""
        start_metrics = self.get_system_metrics()
        start_time = time.time()
        
        try:
            result = self.agent.query(query, skip_verification=True)
            generation_time = time.time() - start_time
            end_metrics = self.get_system_metrics()
            
            return {
                'query_id': query_id,
                'query': query,
                'generation_time': generation_time,
                'success': True,
                'confidence': result.get('confidence', 0),
                'cpu_start': start_metrics['cpu_percent'],
                'cpu_end': end_metrics['cpu_percent'],
                'memory_start_mb': start_metrics['memory_mb'],
                'memory_end_mb': end_metrics['memory_mb'],
                'timestamp': start_metrics['timestamp']
            }
        except Exception as e:
            return {
                'query_id': query_id,
                'query': query,
                'generation_time': time.time() - start_time,
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def test_concurrent_load(self, queries: List[str], n_threads: int = 5) -> List[Dict]:
        """Test concurrent query execution"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Concurrent Load Test: {n_threads} threads")
        logger.info(f"{'='*60}")
        
        results = []
        threads = []
        
        def worker(query, query_id):
            result = self.run_single_query(query, query_id)
            results.append(result)
        
        start_time = time.time()
        
        for i, query in enumerate(queries[:n_threads]):
            thread = threading.Thread(target=worker, args=(query, f"C{i+1}"))
            threads.append(thread)
            thread.start()
        
        for thread in threads:
            thread.join()
        
        total_time = time.time() - start_time
        
        logger.info(f"Completed {len(results)} concurrent queries in {total_time:.2f}s")
        logger.info(f"Average time per query: {total_time/len(results):.2f}s")
        
        return results
    
    def test_rapid_succession(self, queries: List[str], delay: float = 1.0) -> List[Dict]:
        """Test rapid successive queries"""
        logger.info(f"\n{'='*60}")
        logger.info(f"Rapid Succession Test: {len(queries)} queries")
        logger.info(f"{'='*60}")
        
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Query {i+1}/{len(queries)}: {query}")
            result = self.run_single_query(query, f"R{i+1}")
            results.append(result)
            
            if i < len(queries) - 1:
                time.sleep(delay)
        
        # Analyze performance degradation
        times = [r['generation_time'] for r in results if r['success']]
        if len(times) > 1:
            first_half = times[:len(times)//2]
            second_half = times[len(times)//2:]
            
            logger.info(f"First half avg: {sum(first_half)/len(first_half):.2f}s")
            logger.info(f"Second half avg: {sum(second_half)/len(second_half):.2f}s")
            
            degradation = (sum(second_half)/len(second_half)) / (sum(first_half)/len(first_half))
            logger.info(f"Performance degradation: {degradation:.2%}")
        
        return results
    
    def test_cache_effectiveness(self) -> Dict:
        """Test cache effectiveness with repeated queries"""
        logger.info(f"\n{'='*60}")
        logger.info("Cache Effectiveness Test")
        logger.info(f"{'='*60}")
        
        test_query = "Florian Wirtz scout report 2025-2026"
        
        # First run (cache miss)
        logger.info("Run 1 (cache miss):")
        result1 = self.run_single_query(test_query, "CACHE1")
        time1 = result1['generation_time']
        logger.info(f"  Time: {time1:.2f}s")
        
        # Second run (cache hit?)
        logger.info("Run 2 (potential cache hit):")
        result2 = self.run_single_query(test_query, "CACHE2")
        time2 = result2['generation_time']
        logger.info(f"  Time: {time2:.2f}s")
        
        # Third run
        logger.info("Run 3:")
        result3 = self.run_single_query(test_query, "CACHE3")
        time3 = result3['generation_time']
        logger.info(f"  Time: {time3:.2f}s")
        
        speedup = time1 / time2 if time2 > 0 else 1.0
        logger.info(f"\nSpeedup (run 1 vs run 2): {speedup:.2f}x")
        
        return {
            'query': test_query,
            'run1_time': time1,
            'run2_time': time2,
            'run3_time': time3,
            'speedup': speedup
        }
    
    def run_full_stress_test(self) -> Dict:
        """Run full stress test suite"""
        logger.info(f"\n{'='*60}")
        logger.info("Starting Full Stress Test Suite")
        logger.info(f"{'='*60}\n")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {}
        }
        
        # Test 1: Concurrent load
        concurrent_queries = [
            "Florian Wirtz scout report",
            "Mohamed Salah performance",
            "Erling Haaland goals",
            "Kevin De Bruyne analysis",
            "Virgil van Dijk defensive stats"
        ]
        test_results['tests']['concurrent'] = self.test_concurrent_load(concurrent_queries)
        
        # Test 2: Rapid succession
        rapid_queries = [
            "Salah 2025-2026",
            "Salah 2024-2025",
            "Salah 2023-2024",
            "Salah 2022-2023"
        ]
        test_results['tests']['rapid_succession'] = self.test_rapid_succession(rapid_queries, delay=1.0)
        
        # Test 3: Cache effectiveness
        test_results['tests']['cache'] = self.test_cache_effectiveness()
        
        # Save results
        self.save_results(test_results)
        
        return test_results
    
    def save_results(self, results: Dict):
        """Save stress test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = Path("results/stress_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        json_file = results_dir / f"stress_test_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"\n Stress test results saved to {json_file}")


def main():
    """Run stress tests"""
    print("Stress Testing - TactIQ Football Scout System")
    
    tester = StressTester()
    results = tester.run_full_stress_test()
    
    print("\n Stress testing complete!")
    print("Results saved to results/stress_tests/")


if __name__ == "__main__":
    main()
