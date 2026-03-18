"""
Self-Check Verification Agent
==============================

Validates generated answers for accuracy, consistency, and hallucination detection.
Provides confidence scores and triggers regeneration if quality is insufficient.
"""

from typing import Dict, Any, List, Optional
from langchain_groq import ChatGroq
from loguru import logger
import re
import os


class SelfCheckAgent:
    """
    Self-Check Verification Agent
    
    Validates answers through:
    1. Factual Grounding: Does answer match retrieved context?
    2. Hallucination Detection: Any unsupported claims?
    3. Consistency Check: Internal contradictions?
    4. Completeness: Does it answer the query?
    5. Confidence Scoring: Overall quality assessment
    """
    
    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        model_name: str = "llama-3.1-8b-instant"  # LLaMA-3.1-8B (replaces decommissioned llama3-8b-8192)
    ):
        """
        Initialize Self-Check agent
        
        Args:
            groq_api_key: Groq API key (loads from env if None)
            model_name: LLM model for verification
        """
        groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(
            api_key=groq_api_key,
            model_name=model_name,
            temperature=0.0,  # Deterministic for verification
            max_tokens=1000
        )
        logger.info(f"Self-Check initialized with model: {model_name}")
        
        # Verification thresholds
        self.confidence_threshold = 0.7  # Minimum acceptable confidence
        self.max_retries = 2  # Maximum regeneration attempts
    
    def verify_answer(
        self,
        query: str,
        answer: str,
        sources: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify answer quality and detect issues
        
        Args:
            query: Original user query
            answer: Generated answer to verify
            sources: List of source descriptions
            context: Retrieved context (optional, for deeper verification)
            
        Returns:
            Dict with verification results
        """
        # Build verification prompt
        source_text = '\n'.join([f"- {s}" for s in sources[:5]]) if sources else "No sources"
        context_snippet = context[:500] if context else "Not provided"
        
        prompt = f"""You are a fact-checking expert. Verify this football analytics answer for accuracy and quality.

Query: {query}

Generated Answer:
{answer}

Available Sources:
{source_text}

Context Snippet:
{context_snippet}

Evaluate on these criteria and provide scores (0-10):

1. FACTUAL GROUNDING: Does the answer match the sources/context?
   - 10: Fully grounded in sources
   - 5: Partially grounded, some unsupported claims
   - 0: Contradicts sources or fully hallucinated

2. HALLUCINATION: Any fabricated information?
   - 10: No hallucinations detected
   - 5: Minor unsupported details
   - 0: Major fabrications (fake stats, names, etc.)

3. COMPLETENESS: Does it answer the query?
   - 10: Fully answers the query
   - 5: Partial answer
   - 0: Doesn't address query

4. CONSISTENCY: Internal contradictions?
   - 10: Fully consistent
   - 5: Minor inconsistencies
   - 0: Major contradictions

Output format (EXACTLY):
GROUNDING: <score>
HALLUCINATION: <score>
COMPLETENESS: <score>
CONSISTENCY: <score>
ISSUES: <list any specific problems found, or "None">
VERDICT: <PASS or FAIL>
"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Parse scores
            scores = self._parse_verification_scores(content)
            
            # Calculate confidence
            confidence = self._calculate_confidence(scores)
            
            # Extract issues
            issues = self._extract_issues(content)
            
            # Determine if passed
            verdict = self._extract_verdict(content)
            passed = verdict == "PASS" and confidence >= self.confidence_threshold
            
            return {
                'passed': passed,
                'confidence': confidence,
                'scores': scores,
                'issues': issues,
                'verdict': verdict,
                'raw_verification': content
            }
            
        except Exception as e:
            logger.error(f"Verification failed: {e}")
            return {
                'passed': False,
                'confidence': 0.0,
                'scores': {},
                'issues': [f"Verification error: {str(e)}"],
                'verdict': "ERROR",
                'raw_verification': ""
            }
    
    def _parse_verification_scores(self, content: str) -> Dict[str, float]:
        """Parse verification scores from LLM output"""
        scores = {}
        
        patterns = {
            'grounding': r'GROUNDING:\s*(\d+)',
            'hallucination': r'HALLUCINATION:\s*(\d+)',
            'completeness': r'COMPLETENESS:\s*(\d+)',
            'consistency': r'CONSISTENCY:\s*(\d+)'
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                scores[key] = float(match.group(1)) / 10.0  # Normalize to 0-1
            else:
                scores[key] = 0.5  # Default to neutral
        
        return scores
    
    def _extract_issues(self, content: str) -> List[str]:
        """Extract issues from verification output"""
        issues = []
        
        # Find ISSUES section
        match = re.search(r'ISSUES:\s*(.+?)(?:VERDICT:|$)', content, re.DOTALL | re.IGNORECASE)
        if match:
            issues_text = match.group(1).strip()
            if issues_text.lower() != "none":
                # Split by newlines or commas
                for line in issues_text.split('\n'):
                    line = line.strip().lstrip('-•*')
                    if line and line.lower() != "none":
                        issues.append(line)
        
        return issues
    
    def _extract_verdict(self, content: str) -> str:
        """Extract verdict from verification output"""
        match = re.search(r'VERDICT:\s*(PASS|FAIL)', content, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return "UNKNOWN"
    
    def _calculate_confidence(self, scores: Dict[str, float]) -> float:
        """
        Calculate overall confidence score
        
        Weighted average:
        - Grounding: 35%
        - Hallucination: 35%
        - Completeness: 20%
        - Consistency: 10%
        """
        weights = {
            'grounding': 0.35,
            'hallucination': 0.35,
            'completeness': 0.20,
            'consistency': 0.10
        }
        
        confidence = 0.0
        for key, weight in weights.items():
            confidence += scores.get(key, 0.5) * weight
        
        return min(max(confidence, 0.0), 1.0)
    
    def check_hallucination(
        self,
        answer: str,
        sources: List[str]
    ) -> Dict[str, Any]:
        """
        Quick hallucination check
        
        Args:
            answer: Generated answer
            sources: Available sources
            
        Returns:
            Dict with hallucination indicators
        """
        # Simple heuristics for quick checks
        hallucination_indicators = []
        
        # 1. Check if answer is too generic
        generic_phrases = [
            "it depends", "varies", "generally", "typically",
            "may vary", "could be", "might be"
        ]
        if any(phrase in answer.lower() for phrase in generic_phrases):
            if len(sources) == 0:
                hallucination_indicators.append("Generic answer with no sources")
        
        # 2. Check if answer mentions specific data but no sources
        has_numbers = bool(re.search(r'\d+', answer))
        has_names = bool(re.search(r'[A-Z][a-z]+\s+[A-Z][a-z]+', answer))
        
        if (has_numbers or has_names) and len(sources) == 0:
            hallucination_indicators.append("Specific claims without sources")
        
        # 3. Check answer length vs source availability
        if len(answer) > 200 and len(sources) < 2:
            hallucination_indicators.append("Detailed answer with minimal sources")
        
        # 4. Check for contradictory statements
        if "however" in answer.lower() and "but" in answer.lower():
            hallucination_indicators.append("Potential contradictions detected")
        
        has_hallucination = len(hallucination_indicators) > 0
        confidence = 0.3 if has_hallucination else 0.8
        
        return {
            'has_hallucination': has_hallucination,
            'indicators': hallucination_indicators,
            'confidence': confidence
        }
    
    def should_regenerate(
        self,
        verification_result: Dict[str, Any],
        retry_count: int
    ) -> Dict[str, Any]:
        """
        Determine if answer should be regenerated
        
        Args:
            verification_result: Result from verify_answer()
            retry_count: Current retry attempt
            
        Returns:
            Dict with regeneration decision and guidance
        """
        # Don't regenerate if max retries reached
        if retry_count >= self.max_retries:
            return {
                'should_regenerate': False,
                'reason': f"Max retries ({self.max_retries}) reached",
                'guidance': None
            }
        
        # Don't regenerate if passed
        if verification_result['passed']:
            return {
                'should_regenerate': False,
                'reason': "Verification passed",
                'guidance': None
            }
        
        # Regenerate if failed
        issues = verification_result.get('issues', [])
        scores = verification_result.get('scores', {})
        
        # Generate improvement guidance
        guidance = []
        
        if scores.get('grounding', 1.0) < 0.6:
            guidance.append("Ground answer more closely in provided sources")
        
        if scores.get('hallucination', 1.0) < 0.6:
            guidance.append("Remove unsupported claims and fabricated information")
        
        if scores.get('completeness', 1.0) < 0.6:
            guidance.append("Provide a more complete answer to the query")
        
        if scores.get('consistency', 1.0) < 0.6:
            guidance.append("Resolve internal contradictions")
        
        return {
            'should_regenerate': True,
            'reason': f"Confidence {verification_result['confidence']:.2f} below threshold {self.confidence_threshold}",
            'guidance': guidance,
            'issues': issues
        }
    
    def verify_with_retry(
        self,
        query: str,
        generate_fn: callable,
        sources: List[str],
        context: Optional[str] = None,
        initial_answer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Verify answer with automatic regeneration
        
        Args:
            query: User query
            generate_fn: Function to generate/regenerate answer
            sources: Available sources
            context: Retrieved context
            initial_answer: Initial answer (if available)
            
        Returns:
            Dict with final answer and verification history
        """
        verification_history = []
        retry_count = 0
        
        # Use initial answer or generate first one
        current_answer = initial_answer or generate_fn(query, sources, context)
        
        while retry_count <= self.max_retries:
            # Verify current answer
            verification = self.verify_answer(query, current_answer, sources, context)
            verification_history.append({
                'attempt': retry_count + 1,
                'answer': current_answer,
                'verification': verification
            })
            
            # Check if regeneration needed
            regenerate_decision = self.should_regenerate(verification, retry_count)
            
            if not regenerate_decision['should_regenerate']:
                # Verification passed or max retries reached
                return {
                    'final_answer': current_answer,
                    'verification': verification,
                    'regenerated': retry_count > 0,
                    'attempts': retry_count + 1,
                    'history': verification_history
                }
            
            # Regenerate with guidance
            logger.info(f"Regenerating answer (attempt {retry_count + 1})")
            guidance = regenerate_decision.get('guidance', [])
            
            try:
                current_answer = generate_fn(
                    query,
                    sources,
                    context,
                    improvement_guidance=guidance
                )
            except Exception as e:
                logger.error(f"Regeneration failed: {e}")
                # Return best attempt so far
                best_attempt = max(verification_history, key=lambda x: x['verification']['confidence'])
                return {
                    'final_answer': best_attempt['answer'],
                    'verification': best_attempt['verification'],
                    'regenerated': retry_count > 0,
                    'attempts': retry_count + 1,
                    'history': verification_history,
                    'error': str(e)
                }
            
            retry_count += 1
        
        # Shouldn't reach here, but return last answer
        return {
            'final_answer': current_answer,
            'verification': verification_history[-1]['verification'],
            'regenerated': True,
            'attempts': len(verification_history),
            'history': verification_history
        }
