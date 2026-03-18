"""
Enhanced CRAG Agent with REFRAG + Self-Check
============================================

Extended CRAG workflow with:
- REFRAG multi-hop reasoning
- Self-Check verification
- Regeneration loop
"""

from typing import Dict, List, Any, Optional
from loguru import logger
import os

from src.agents.crag_agent import CRAGAgent
from src.agents.refrag_agent import REFRAGAgent
from src.agents.selfcheck_agent import SelfCheckAgent
from src.agents.intent_classifier import IntentClassifier, QueryIntent

class EnhancedCRAGAgent:
    """
    Enhanced CRAG with REFRAG reasoning and Self-Check verification
    
    Workflow:
    1. Query Analysis: Determine if reasoning needed
    2. If Reasoning Needed:
        a. REFRAG decomposition
        b. Sub-query CRAG retrieval
        c. Synthesis
    3. Else: Standard CRAG
    4. Self-Check verification
    5. Regeneration if needed
    6. Return verified answer
    """
    
    def __init__(
        self,
        vector_db,
        groq_api_key: Optional[str] = None,
        tavily_api_key: Optional[str] = None,
        enable_refrag: bool = True,
        enable_selfcheck: bool = True,
        refrag_model_path: str = "ollama:qwen2.5:1.5b"
    ):
        """
        Initialize Enhanced CRAG
        
        Args:
            vector_db: ChromaDB collection
            groq_api_key: Groq API key
            tavily_api_key: Tavily API key
            enable_refrag: Enable REFRAG reasoning
            enable_selfcheck: Enable Self-Check verification
        """
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        # Initialize base CRAG
        self.crag = CRAGAgent(
            vector_db=vector_db,
            groq_api_key=self.groq_api_key,
            tavily_api_key=self.tavily_api_key
        )
        
        # Initialize REFRAG
        self.enable_refrag = enable_refrag
        if enable_refrag:
            # Initialize REFRAG with local Ollama model to avoid HF downloads
            self.refrag = REFRAGAgent(model_path=refrag_model_path)
            logger.info(f"REFRAG enabled with model: {refrag_model_path}")
        else:
            self.refrag = None
            logger.info("REFRAG disabled")
        
        # Initialize Self-Check
        self.enable_selfcheck = enable_selfcheck
        if enable_selfcheck and self.groq_api_key:
            self.selfcheck = SelfCheckAgent(groq_api_key=self.groq_api_key)
            logger.info("Self-Check enabled")
        else:
            self.selfcheck = None
            if enable_selfcheck and not self.groq_api_key:
                logger.warning("Self-Check disabled: GROQ_API_KEY not set")
            else:
                logger.info("Self-Check disabled")
        
        # Initialize intent classifier
        self.intent_classifier = IntentClassifier()
        
        logger.info("Enhanced CRAG initialized")
    
    def query(
        self,
        query: str,
        intent: Optional[Any] = None,
        intent_metadata: Optional[Dict] = None,
        force_reasoning: bool = False,
        skip_verification: bool = False,
        detect_intent: bool = True
    ) -> Dict[str, Any]:
        """
        Query with enhanced workflow and intent detection
        
        Args:
            query: User query
            intent: Pre-classified intent (optional, will auto-detect if not provided)
            intent_metadata: Intent metadata dict (optional)
            force_reasoning: Force REFRAG reasoning
            skip_verification: Skip Self-Check (for speed)
            detect_intent: Classify query intent if intent not provided (default True)
            
        Returns:
            Enhanced result with reasoning, verification, and intent
        """
        logger.info(f"Enhanced CRAG query: {query}")
        
        # Use provided intent or detect it
        if intent is None and detect_intent:
            intent, intent_confidence, intent_metadata = self.intent_classifier.classify(query)
            logger.info(f"Detected intent: {intent.value} (confidence: {intent_confidence:.2f})")
            # Ensure confidence stored in metadata for downstream components/UI
            intent_metadata = intent_metadata or {}
            intent_metadata['confidence'] = float(intent_confidence)
        elif intent is not None:
            # Intent provided from app
            intent_confidence = intent_metadata.get('confidence', 0.0) if intent_metadata else 0.0
            intent_metadata = intent_metadata or {}
            logger.info(f"Using provided intent: {intent.value if hasattr(intent, 'value') else intent}")
        else:
            # No intent detection requested
            intent = QueryIntent.UNKNOWN
            intent_confidence = 0.0
            intent_metadata = {}
        
        # Extract intent value as string
        intent_str = intent.value if hasattr(intent, 'value') else (str(intent) if intent else "unknown")
        
        result = {
            'query': query,
            'answer': '',
            'sources': [],
            'grade': '',
            'confidence': 0.0,
            'used_web_search': False,
            'reasoning_trace': [],
            'intent': intent_str,
            'intent_confidence': intent_confidence,
            'intent_metadata': intent_metadata or {},
            'verification': None,
            'regenerated': False
        }
        
        # Step 1: Check if REFRAG reasoning is needed
        needs_reasoning = False
        if self.enable_refrag and self.refrag:
            needs_reasoning = force_reasoning or self.refrag.requires_reasoning(query)
        
        # Step 2: Execute appropriate workflow (with intent)
        if needs_reasoning:
            logger.info("Using REFRAG reasoning workflow")
            result = self._reasoning_workflow(query, result, intent, intent_metadata)
        else:
            logger.info("Using standard CRAG workflow")
            result = self._standard_workflow(query, result, intent, intent_metadata)
        
        # Step 3: Self-Check verification
        if self.enable_selfcheck and self.selfcheck and not skip_verification:
            logger.info("Running Self-Check verification")
            result = self._verification_workflow(query, result)
        
        return result
    
    def _standard_workflow(self, query: str, result: Dict[str, Any], intent: QueryIntent, intent_metadata: Dict) -> Dict[str, Any]:
        """Standard CRAG workflow with intent"""
        try:
            crag_result = self.crag.query(query, intent=intent, intent_metadata=intent_metadata)
            
            result['answer'] = crag_result.get('answer', '')
            result['sources'] = crag_result.get('sources', [])
            result['grade'] = crag_result.get('grade', '')
            result['confidence'] = crag_result.get('confidence', 0.0)
            result['used_web_search'] = crag_result.get('used_web_search', False)
            result['retrieved_docs'] = crag_result.get('retrieved_docs', [])
            result['data_source'] = crag_result.get('data_source', 'Database')  # Pass through data source
            result['reasoning_trace'] = crag_result.get('reasoning_trace', '')  # Pass through reasoning trace
            
            return result
            
        except Exception as e:
            logger.error(f"Standard workflow failed: {e}")
            result['answer'] = f"Error: {str(e)}"
            return result
    
    def _reasoning_workflow(self, query: str, result: Dict[str, Any], intent: QueryIntent, intent_metadata: Dict) -> Dict[str, Any]:
        """REFRAG reasoning workflow with intent"""
        try:
            # Define retrieval function for REFRAG (with intent)
            def retrieve_fn(sub_query: str) -> Dict[str, Any]:
                """Retrieve answer for sub-query using CRAG"""
                return self.crag.query(sub_query, intent=intent, intent_metadata=intent_metadata)
            
            # Run REFRAG reasoning (let it decide if decomposition needed)
            refrag_result = self.refrag.reason(
                query=query,
                retrieve_fn=retrieve_fn,
                force_reasoning=False  # REFRAG will decompose only if needed
            )
            
            # Use REFRAG result (it passes through CRAG for simple queries)
            if refrag_result.get('answer'):
                result['answer'] = refrag_result['answer']
                result['reasoning_trace'] = refrag_result.get('reasoning_trace', [])
                result['confidence'] = refrag_result.get('confidence', 0.0)
                result['sources'] = refrag_result.get('sources', [])
                result['grade'] = refrag_result.get('grade', 'context_sufficient')
                result['used_web_search'] = refrag_result.get('used_web_search', False)
                result['retrieved_docs'] = refrag_result.get('retrieved_docs', [])  # Pass through for evaluation
                
            else:
                # Fallback to standard CRAG (shouldn't happen)
                logger.warning("REFRAG didn't provide answer, falling back to CRAG")
                result = self._standard_workflow(query, result)
            
            return result
            
        except Exception as e:
            logger.error(f"Reasoning workflow failed: {e}")
            # Fallback to standard CRAG
            return self._standard_workflow(query, result)
    
    def _verification_workflow(self, query: str, result: Dict[str, Any]) -> Dict[str, Any]:
        """Self-Check verification workflow"""
        try:
            # FIRST: If CRAG graded retrieval as missing facts, block generation
            if result.get('grade') == 'context_missing_facts' or ('Season data not available' in (result.get('answer') or '')):
                logger.warning("Blocking output: retrieval graded as missing facts; aborting with explanation")
                result['verification'] = {
                    'passed': False,
                    'confidence': 0.0,
                    'scores': {'grounding': 0.0, 'hallucination': 0.0, 'completeness': 0.0, 'consistency': 0.0},
                    'issues': ['missing_facts'],
                    'verdict': 'FAIL',
                    'raw_verification': 'Blocked by Self-Check: required DB facts missing (season/row/columns)'
                }
                result['confidence'] = 0.0
                # Ensure we do not regenerate or produce a hallucinated report
                return result

            # Verify current answer
            verification = self.selfcheck.verify_answer(
                query=query,
                answer=result['answer'],
                sources=result['sources']
            )

            result['verification'] = verification
            
            # Update confidence based on verification
            if verification['passed']:
                # Verification passed, use verification confidence
                result['confidence'] = max(
                    result.get('confidence', 0.0),
                    verification['confidence']
                )
            else:
                # Verification failed
                logger.warning(f"Verification failed: {verification.get('issues', [])}")
                
                # Check if verification indicates missing-data related issues
                issues = verification.get('issues', [])
                lower_issues = ' '.join(issues).lower() if issues else ''

                if 'missing' in lower_issues or 'data not available' in lower_issues or 'missing_facts' in lower_issues:
                    # Block output and return explanatory message rather than regenerating
                    logger.warning("Verification indicates missing data; blocking final report")
                    result['answer'] = (
                        "Data required to produce a full scout report is missing from the database. "
                        "Please request a season for which data exists or enable DB updates."
                    )
                    result['confidence'] = 0.0
                    return result

                # Otherwise follow normal regeneration flow
                regenerate_decision = self.selfcheck.should_regenerate(
                    verification,
                    retry_count=0
                )

                if regenerate_decision['should_regenerate']:
                    logger.info("Regenerating answer due to verification failure")

                    # Regenerate with guidance
                    guidance = regenerate_decision.get('guidance', [])
                    regenerated_result = self._regenerate_answer(
                        query,
                        result,
                        guidance
                    )

                    if regenerated_result:
                        result = regenerated_result
                        result['regenerated'] = True

                # Use verification confidence
                result['confidence'] = verification['confidence']
            
            return result
            
        except Exception as e:
            logger.error(f"Verification workflow failed: {e}")
            # Return original result
            return result
    
    def _regenerate_answer(
        self,
        query: str,
        original_result: Dict[str, Any],
        guidance: List[str]
    ) -> Optional[Dict[str, Any]]:
        """
        Regenerate answer with improvement guidance
        
        Args:
            query: Original query
            original_result: Original result to improve
            guidance: Improvement guidance from Self-Check
            
        Returns:
            Regenerated result or None if failed
        """
        try:
            # Add guidance to query
            enhanced_query = query
            if guidance:
                guidance_text = " Focus on: " + "; ".join(guidance)
                enhanced_query = query + guidance_text
            
            # Re-run workflow (without verification to avoid loop)
            logger.info(f"Regenerating with guidance: {guidance}")
            
            # Use standard workflow (no reasoning for regeneration)
            result = {
                'query': query,
                'answer': '',
                'sources': [],
                'grade': '',
                'confidence': 0.0,
                'used_web_search': False,
                'reasoning_trace': original_result.get('reasoning_trace', []),
                'verification': None,
                'regenerated': True
            }
            
            # Query again
            crag_result = self.crag.query(enhanced_query)
            
            result['answer'] = crag_result.get('answer', '')
            result['sources'] = crag_result.get('sources', [])
            result['grade'] = crag_result.get('grade', '')
            result['confidence'] = crag_result.get('confidence', 0.0)
            result['used_web_search'] = crag_result.get('used_web_search', False)
            
            # Verify regenerated answer
            verification = self.selfcheck.verify_answer(
                query=query,
                answer=result['answer'],
                sources=result['sources']
            )
            
            result['verification'] = verification
            result['confidence'] = verification['confidence']
            
            # Only use regenerated if it's better
            if verification['confidence'] > original_result.get('confidence', 0.0):
                logger.info("Regeneration improved answer")
                return result
            else:
                logger.info("Regeneration didn't improve, keeping original")
                return original_result
                
        except Exception as e:
            logger.error(f"Regeneration failed: {e}")
            return None
    
    def batch_query(
        self,
        queries: List[str],
        enable_reasoning: bool = True,
        enable_verification: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Process multiple queries
        
        Args:
            queries: List of queries
            enable_reasoning: Enable REFRAG for all queries
            enable_verification: Enable Self-Check for all queries
            
        Returns:
            List of results
        """
        results = []
        
        for i, query in enumerate(queries, 1):
            logger.info(f"Processing query {i}/{len(queries)}")
            
            result = self.query(
                query=query,
                force_reasoning=enable_reasoning,
                skip_verification=not enable_verification
            )
            
            results.append(result)
        
        return results
