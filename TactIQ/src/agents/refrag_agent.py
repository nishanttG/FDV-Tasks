"""
REFRAG (Reasoning-Enhanced RAG) Agent
======================================

Adds multi-hop reasoning capabilities to the CRAG system.
Decomposes complex queries into sub-questions, retrieves context for each,
and synthesizes a comprehensive answer with reasoning traces.

Uses LOCAL Ollama Qwen2.5-1.5B-Instruct for unlimited reasoning (no API costs/limits).
"""

from typing import List, Dict, Any
from loguru import logger
import re

# Local model clients
# try:
#     from src.models.qwen_gguf import QwenGGUF
#     QWEN_GGUF_AVAILABLE = True
# except Exception:
#     QWEN_GGUF_AVAILABLE = False

try:
    from src.models.ollama_client import OllamaClient
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False
    logger.warning("Ollama client not available. Install for local Qwen support.")

class REFRAGAgent:
    """
    Reasoning-Enhanced RAG Agent using local Ollama or GGUF models
    """

    def __init__(
        self,
        model_path: str = "ollama:qwen2.5:1.5b",
        max_tokens: int = 512
    ):
        """
        Initialize REFRAG agent with a local Ollama or GGUF model.
        """
        self.max_tokens = max_tokens
        self._use_ollama = False
        self._use_gguf = False

        # GGUF support
        if model_path.lower().endswith(".gguf") and QWEN_GGUF_AVAILABLE:
            logger.info(f"GGUF model detected at {model_path}; using llama.cpp loader.")
            self.gguf = QwenGGUF(model_path)
            self._use_gguf = True
            logger.info(f"✅ REFRAG initialized with GGUF model on CPU")
            return

        # Ollama local support
        if model_path.startswith("ollama:") or model_path.startswith("ollama://"):
            model_name = model_path.split(":", 1)[1].lstrip("/")
            if OLLAMA_AVAILABLE:
                logger.info(f"Using Ollama model '{model_name}' via local Ollama API")
                self.ollama = OllamaClient(model=model_name)
                self._use_ollama = True
                logger.info(f"✅ REFRAG initialized with Ollama model {model_name}")
                return
            else:
                raise RuntimeError("Ollama client not available for local model loading")

        raise RuntimeError("No valid local model available. Provide GGUF or Ollama model path.")

    def _generate_text(self, prompt: str, max_new_tokens: int = None) -> str:
        """
        Generate text using the local model
        """
        max_tokens = max_new_tokens or self.max_tokens

        if getattr(self, "_use_gguf", False):
            return self.gguf.generate(prompt, max_tokens=max_tokens)

        if getattr(self, "_use_ollama", False):
            return self.ollama.generate(prompt, max_tokens=max_tokens)

        return "ERROR: No local model initialized"

    def decompose_query(self, query: str) -> List[str]:
        """
        Break a complex query into simpler sub-questions
        """
        prompt = f"""You are a football analytics expert. Break down this query into 2 simpler sub-questions.

Query: {query}

Rules:
1. Each sub-question should be independently answerable
2. Sub-questions should build towards answering the main query
3. Focus on player stats, comparisons, tactical analysis, recent form
4. Do NOT include the original query
5. Output numbered questions (1., 2.)
"""
        try:
            response = self._generate_text(prompt, max_new_tokens=300)
            sub_questions = []
            for line in response.strip().split("\n"):
                match = re.match(r"^\d+[\.\)]\s*(.+)$", line.strip())
                if match:
                    sub_questions.append(match.group(1).strip())

            if not sub_questions:
                sub_questions = [query]  # fallback
            return sub_questions[:2]  # limit to 2 for efficiency
        except Exception as e:
            logger.error(f"Query decomposition failed: {e}")
            return [query]

    def synthesize_answer(
        self,
        query: str,
        sub_questions: List[str],
        sub_answers: List[Dict[str, Any]],
        reasoning_required: bool = True
    ) -> Dict[str, Any]:
        """
        Combine sub-answers into a final answer
        """
        context_parts = []
        for i, (q, a) in enumerate(zip(sub_questions, sub_answers), 1):
            answer_text = a.get("answer", "No answer available")
            sources = a.get("sources", [])
            # Fix: sources are dicts, extract string representation
            source_strs = []
            for src in sources[:3]:
                if isinstance(src, dict):
                    player = src.get('player', '')
                    team = src.get('team', '')
                    season = src.get('season', '')
                    if player:
                        source_strs.append(f"{player} ({team}, {season})" if team and season else player)
                    else:
                        source_strs.append(src.get('title', str(src))[:50])
                else:
                    source_strs.append(str(src)[:50])
            source_text = ", ".join(source_strs) if source_strs else "No sources"
            context_parts.append(
                f"Sub-Question {i}: {q}\nAnswer: {answer_text}\nSources: {source_text}\n"
            )
        context = "\n".join(context_parts)

        prompt = f"""You are a football analytics expert. Synthesize a comprehensive answer.

Original Query: {query}

Sub-Question Answers:
{context}

Instructions:
1. Provide clear, comprehensive answer
2. Combine insights logically
3. Include specific stats and names from sub-answers
4. Be concise but thorough
5. {"Include reasoning trace" if reasoning_required else "Provide final answer only"}
"""
        try:
            response = self._generate_text(prompt, max_new_tokens=500)
            answer = response.strip()
            reasoning_trace = []

            if "REASONING:" in answer:
                parts = answer.split("REASONING:", 1)
                answer = parts[0].strip()
                reasoning_trace = [line.strip("-• ").strip() for line in parts[1].split("\n") if line.strip()]

            return {
                "answer": answer,
                "reasoning_trace": reasoning_trace,
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
                "confidence": 1.0  # default full confidence for local generation
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                "answer": "Unable to synthesize answer",
                "reasoning_trace": [],
                "sub_questions": sub_questions,
                "sub_answers": sub_answers,
                "confidence": 0.0
            }

    def requires_reasoning(self, query: str) -> bool:
        """
        Determine if query needs multi-hop reasoning
        """
        keywords = [
            "compare", "vs", "versus", "why", "how", "best", "worst",
            "analyze", "should", "different", "similar", "impact", "effect"
        ]
        return any(kw in query.lower() for kw in keywords) or len(query.split()) > 10

    def reason(self, query: str, retrieve_fn: callable, force_reasoning: bool = False) -> Dict[str, Any]:
        """
        Main reasoning method
        """
        if not force_reasoning and not self.requires_reasoning(query):
            # Simple query - just use CRAG directly and pass through
            crag_result = retrieve_fn(query)
            return {
                "needs_reasoning": False,
                "answer": crag_result.get("answer", ""),
                "sources": crag_result.get("sources", []),
                "grade": crag_result.get("grade", ""),
                "used_web_search": crag_result.get("used_web_search", False),
                "confidence": crag_result.get("confidence", 0.0),
                "reasoning_trace": [],
                "sub_questions": [],
                "sub_answers": [],
                "retrieved_docs": crag_result.get("retrieved_docs", [])  # Pass through
            }

        logger.info(f"REFRAG reasoning for query: {query}")
        sub_questions = self.decompose_query(query)
        sub_answers = []
        for sub_q in sub_questions:
            try:
                result = retrieve_fn(sub_q)
                sub_answers.append(result)
            except Exception as e:
                logger.error(f"Sub-query retrieval failed: {e}")
                sub_answers.append({"question": sub_q, "answer": "Retrieval failed", "sources": [], "grade": "error"})

        synthesis = self.synthesize_answer(query, sub_questions, sub_answers)
        return {"needs_reasoning": True, **synthesis}
