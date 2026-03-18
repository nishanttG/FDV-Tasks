import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
import time
import json
from datetime import datetime
from typing import Dict, List, Any
from loguru import logger

from src.agents.crag_agent import CRAGAgent

# Configure logging
logger.remove()
logger.add(sys.stderr, level="WARNING")  # Reduce noise


class CRAGValidationSuite:
    """Comprehensive validation suite for CRAG system"""
    
    def __init__(self):
        """Initialize validation suite"""
        
        print("DAY 4: CRAG VALIDATION SUITE")
       
        
        # Initialize CRAG
        print("\n Initializing CRAG system...")
        chroma_client = chromadb.PersistentClient(path="./db/chroma")
        collection = chroma_client.get_collection("player_stats")
        
        self.crag = CRAGAgent(
            vector_db=collection,
            groq_api_key=None,  # Load from .env
            tavily_api_key=None
        )
        
        self.results = []
        self.passed = 0
        self.failed = 0
        print(" CRAG system ready\n")
    
    def test_player_name_detection(self):
        """Test 1: Player name detection"""
        print(" Testing player name detection...")
        
        test_cases = [
            {
                "query": "How good is Mo. Salah this season",
                "should_detect": True,
                "expected_player": "Mo. Salah"
            },
            {
                "query": "Mohamed Salah performance",
                "should_detect": True,
                "expected_player": "Mohamed Salah"
            },
            {
                "query": "Cristiano Ronaldo stats",
                "should_detect": True,
                "expected_player": "Cristiano Ronaldo"
            },
            {
                "query": "what are the best tactics",
                "should_detect": False,
                "expected_player": None
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = self.crag.query(test["query"])
                
                # Check if player was retrieved
                has_player_source = any('(' in s and ')' in s for s in result['sources'])
                
                passed = has_player_source == test["should_detect"]
                
                self.results.append({
                    "test": f"Player Detection {i}",
                    "query": test["query"],
                    "expected": f"Detect: {test['should_detect']}",
                    "actual": f"Detected: {has_player_source}",
                    "passed": passed
                })
                
                if passed:
                    self.passed += 1
                    print(f" Test {i}/4 passed")
                else:
                    self.failed += 1
                    print(f" Test {i}/4 failed")
                    
            except Exception as e:
                self.failed += 1
                self.results.append({
                    "test": f"Player Detection {i}",
                    "query": test["query"],
                    "expected": "No error",
                    "actual": f"Error: {str(e)}",
                    "passed": False
                })
                print(f"Test {i}/4 failed with error")
        
        print()
    
    def test_grade_accuracy(self):
        """Test 2: Grading accuracy"""
        print("Testing grading accuracy...")
        
        test_cases = [
            {
                "query": "Tell me about Mohamed Salah 2024-2025 season",
                "expected_grade": ["context_sufficient", "context_outdated"],
                "description": "Should find Salah data"
            },
            {
                "query": "Who is the best player born in 1875",
                "expected_grade": ["context_missing_facts"],
                "description": "Should not find historical player"
            },
            {
                "query": "What are high pressing tactics",
                "expected_grade": ["context_sufficient"],
                "description": "Should find tactical articles"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = self.crag.query(test["query"])
                grade = result['grade']
                
                passed = grade in test["expected_grade"]
                
                self.results.append({
                    "test": f"Grading {i}",
                    "query": test["query"],
                    "expected": f"Grade: {test['expected_grade']}",
                    "actual": f"Grade: {grade}",
                    "passed": passed
                })
                
                if passed:
                    self.passed += 1
                    print(f" Test {i}/3 passed ({grade})")
                else:
                    self.failed += 1
                    print(f" Test {i}/3 failed ({grade})")
                    
            except Exception as e:
                self.failed += 1
                self.results.append({
                    "test": f"Grading {i}",
                    "query": test["query"],
                    "expected": "No error",
                    "actual": f"Error: {str(e)}",
                    "passed": False
                })
                print(f" Test {i}/3 failed with error")
        
        print()
    
    def test_web_search_fallback(self):
        """Test 3: Web search fallback logic"""
        print("Testing web search fallback...")
        
        test_cases = [
            {
                "query": "Top young players in 2025 World Cup",
                "should_use_web": True,
                "reason": "Future event not in DB"
            },
            {
                "query": "Mohamed Salah 2022-2023 stats",
                "should_use_web": False,
                "reason": "Historical data in DB"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = self.crag.query(test["query"])
                used_web = result['used_web_search']
                
                passed = used_web == test["should_use_web"]
                
                self.results.append({
                    "test": f"Web Fallback {i}",
                    "query": test["query"],
                    "expected": f"Web: {test['should_use_web']}",
                    "actual": f"Web: {used_web}",
                    "passed": passed
                })
                
                if passed:
                    self.passed += 1
                    print(f" Test {i}/2 passed")
                else:
                    self.failed += 1
                    print(f" Test {i}/2 failed")
                    
            except Exception as e:
                self.failed += 1
                self.results.append({
                    "test": f"Web Fallback {i}",
                    "query": test["query"],
                    "expected": "No error",
                    "actual": f"Error: {str(e)}",
                    "passed": False
                })
                print(f" Test {i}/2 failed with error")
        
        print()
    
    def test_edge_cases(self):
        """Test 4: Edge cases and error handling"""
        print(" Testing edge cases...")
        
        test_cases = [
            {
                "query": "",
                "description": "Empty query"
            },
            {
                "query": "a",
                "description": "Single character"
            },
            {
                "query": "compare player1 and player2 and player3 and player4",
                "description": "Multiple comparisons"
            },
            {
                "query": "find players under -5 years old",
                "description": "Invalid age"
            }
        ]
        
        for i, test in enumerate(test_cases, 1):
            try:
                result = self.crag.query(test["query"])
                
                # Should not crash
                has_answer = len(result.get('answer', '')) > 0
                
                passed = True  # As long as it doesn't crash
                
                self.results.append({
                    "test": f"Edge Case {i}",
                    "query": test["query"],
                    "expected": "No crash",
                    "actual": f"Handled gracefully (answer: {has_answer})",
                    "passed": passed
                })
                
                self.passed += 1
                print(f" Test {i}/4 passed")
                    
            except Exception as e:
                self.failed += 1
                self.results.append({
                    "test": f"Edge Case {i}",
                    "query": test["query"],
                    "expected": "No crash",
                    "actual": f"Crashed: {str(e)}",
                    "passed": False
                })
                print(f" Test {i}/4 failed (crashed)")
        
        print()
    
    def test_performance(self):
        """Test 5: Performance benchmarks"""
        print(" Testing performance...")
        
        test_cases = [
            "How good is Mo. Salah",
            "Find top strikers under 25",
            "What are the latest tactics for high pressing"
        ]
        
        times = []
        
        for i, query in enumerate(test_cases, 1):
            try:
                start = time.time()
                result = self.crag.query(query)
                elapsed = time.time() - start
                
                times.append(elapsed)
                
                # Performance should be under 10s
                passed = elapsed < 10.0
                
                self.results.append({
                    "test": f"Performance {i}",
                    "query": query,
                    "expected": "< 10s",
                    "actual": f"{elapsed:.2f}s",
                    "passed": passed
                })
                
                if passed:
                    self.passed += 1
                    print(f"  Test {i}/3 passed ({elapsed:.2f}s)")
                else:
                    self.failed += 1
                    print(f"  Test {i}/3 failed ({elapsed:.2f}s)")
                    
            except Exception as e:
                self.failed += 1
                self.results.append({
                    "test": f"Performance {i}",
                    "query": query,
                    "expected": "< 10s",
                    "actual": f"Error: {str(e)}",
                    "passed": False
                })
                print(f"  Test {i}/3 failed with error")
        
        if times:
            avg_time = sum(times) / len(times)
            print(f"\n  Average query time: {avg_time:.2f}s")
        
        print()
    
    def generate_report(self):
        """Generate validation report"""
        print("VALIDATION REPORT")
    
        
        total = self.passed + self.failed
        pass_rate = (self.passed / total * 100) if total > 0 else 0
        
        print(f"\nTotal Tests: {total}")
        print(f"Passed: {self.passed} ({pass_rate:.1f}%)")
        print(f"Failed: {self.failed}")
        
        if self.failed > 0:
            print(f"\n FAILED TESTS:")
            for result in self.results:
                if not result['passed']:
                    print(f"\n  Test: {result['test']}")
                    print(f"  Query: {result['query']}")
                    print(f"  Expected: {result['expected']}")
                    print(f"  Actual: {result['actual']}")
        
        # Save report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total": total,
                "passed": self.passed,
                "failed": self.failed,
                "pass_rate": pass_rate
            },
            "results": self.results
        }
        
        report_path = Path("results/day4_validation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\nReport saved to: {report_path}")
        
        # Overall status
        if pass_rate >= 80:
            print(" VALIDATION PASSED (>= 80%)")
        elif pass_rate >= 60:
            print("  VALIDATION WARNING (60-79%)")
            print("Some issues found - review failed tests")
        else:
            print("VALIDATION FAILED (< 60%)")
            print("Critical issues found - fix before proceeding")
    
        
        return pass_rate >= 80


def main():
    """Run validation suite"""
    suite = CRAGValidationSuite()
    
    # Run all tests
    suite.test_player_name_detection()
    suite.test_grade_accuracy()
    suite.test_web_search_fallback()
    suite.test_edge_cases()
    suite.test_performance()
    
    # Generate report
    passed = suite.generate_report()
    
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
