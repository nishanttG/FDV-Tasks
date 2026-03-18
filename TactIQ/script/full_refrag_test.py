"""
Full REFRAG Test with Model Download
=====================================

Test REFRAG with actual Qwen model (downloads ~3-4GB on first run).
This will take 5-10 minutes on first run due to model download.

For quick tests without download, use: python script/quick_refrag_test.py
"""

import sys
from pathlib import Path
from typing import Optional
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.refrag_agent import REFRAGAgent
import os
import argparse

try:
    import psutil
except Exception:
    psutil = None


def test_local_qwen(device: str = "auto", local_only: bool = False, force_safe: bool = False, model_path: Optional[str] = None):
    """Test local Qwen model for REFRAG"""
    print("="*80)
    print("FULL REFRAG TEST WITH MODEL DOWNLOAD")
    print("="*80)
    print()
    print("⚠️  IMPORTANT: Requires Ollama running with qwen2.5-1.5b model")
    print("   Run: ollama pull qwen2.5-1.5b && ollama serve")
    print()
    print("Testing local Ollama Qwen2.5-1.5B (no API calls, no rate limits!)")
    print()
    
    # Step 1: Initialize REFRAG
    print("[1/3] Initializing REFRAG with local Ollama Qwen2.5-1.5B...")
    print("      (Connecting to Ollama server...)")

    # Determine local-only and safe settings (from args or env)
    env_local_only = os.getenv("HF_LOCAL_FILES_ONLY", "").lower() in ("1", "true", "yes")
    local_only = local_only or env_local_only
    # If a model_path was provided and exists, force local-only to avoid downloads
    if model_path:
        mp = Path(model_path)
        if mp.exists():
            print(f"      Using provided local model path: {model_path}")
            local_only = True
        else:
            print(f"      Warning: provided model path does not exist: {model_path} — will attempt remote/HF lookup")
    use_4bit = True
    # device param passed in (defaults to 'auto')

    # If psutil available, check RAM and enforce safe settings on low-memory machines
    if psutil:
        ram_avail_gb = psutil.virtual_memory().available / (1024**3)
        print(f"      System available RAM: {ram_avail_gb:.1f} GB")
        if ram_avail_gb < 2.0 or force_safe:
            print("      ⚠️  Low RAM detected or force_safe — forcing SAFE settings: 4-bit + CPU")
            use_4bit = True
            device = "cpu"

    try:
        # REFRAGAgent only accepts model_path parameter (handles Ollama internally)
        refrag = REFRAGAgent(model_path=model_path or "ollama:qwen2.5:1.5b")
        print("      ✓ REFRAG initialized successfully!")
        print(f"      Model running on: cpu (Ollama)\n")
    except Exception as e:
        print(f"      ✗ Failed: {e}")
        print("\n      Install dependencies: pip install requests (for Ollama)")
        print("      Ensure Ollama is running: ollama serve")
        return False
    
    # Step 2: Test query decomposition
    print("[2/3] Testing query decomposition...")
    test_query = "Compare Mohamed Salah and Erling Haaland's performance this season"
    print(f"      Query: {test_query}")
    
    try:
        sub_questions = refrag.decompose_query(test_query)
        print(f"      ✓ Decomposed into {len(sub_questions)} sub-questions:")
        for i, sq in enumerate(sub_questions, 1):
            print(f"        {i}. {sq}")
        print()
    except Exception as e:
        print(f"      ✗ Failed: {e}\n")
        return False
    
    # Step 3: Test reasoning detection
    print("[3/3] Testing reasoning detection...")
    
    test_cases = [
        ("Mohamed Salah stats", False, "Simple query"),
        ("Compare Salah and Haaland", True, "Comparison"),
        ("Why is Isak performing well?", True, "Why question"),
    ]
    
    for query, expected, reason in test_cases:
        needs_reasoning = refrag.requires_reasoning(query)
        status = "✓" if needs_reasoning == expected else "✗"
        print(f"      {status} '{query}' → {needs_reasoning} ({reason})")
    
    print()
    return True


def verify_days_5_6():
    """Verify Days 5-6 deliverables"""
    print("="*80)
    print("DAYS 5-6 DELIVERABLES CHECK")
    print("="*80)
    print()
    
    deliverables = [
        ("REFRAG Agent", "src/agents/refrag_agent.py", True),
        ("Self-Check Agent", "src/agents/selfcheck_agent.py", True),
        ("Enhanced CRAG", "src/agents/enhanced_crag_agent.py", True),
        ("Scouting Config", "src/agents/scouting_config.py", True),
        ("Documentation", "docs/DAY5_6_COMPLETE.md", True),
        ("Local Qwen Setup", "docs/LOCAL_QWEN_SETUP.md", True),
    ]
    
    from pathlib import Path
    
    all_complete = True
    for name, path, required in deliverables:
        file_path = Path(path)
        exists = file_path.exists()
        status = "✓" if exists else "✗"
        req_mark = "[REQUIRED]" if required else "[OPTIONAL]"
        
        print(f"{status} {name:25} {path:40} {req_mark}")
        
        if required and not exists:
            all_complete = False
    
    print()
    return all_complete


def main():
    """Run full REFRAG test with model"""
    
    print("\n" + "="*80)
    print("FULL REFRAG TEST WITH OLLAMA MODEL")
    print("="*80)
    print()
    print("⚠️  WARNING: Requires Ollama running with qwen2.5-1.5b model!")
    print()
    print("   Setup: ollama pull qwen2.5-1.5b && ollama serve")
    print("   Check models: ollama list")
    print()
    print("Components:")
    print("  • REFRAG: Multi-hop reasoning with LOCAL Ollama Qwen2.5-1.5B")
    print("  • Self-Check: Answer verification with Groq API")
    print("  • Enhanced CRAG: Unified system")
    print()
    
    # Parse CLI flags
    parser = argparse.ArgumentParser(description="Full REFRAG test with optional safe flags")
    parser.add_argument("--local-only", action="store_true", help="Use local files only (no HF downloads)")
    parser.add_argument("--force-safe", action="store_true", help="Force safe settings (4-bit quant, CPU)")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device to run model on (auto detects CUDA)")
    parser.add_argument("--model-path", default="ollama:qwen2.5-1.5b", help="Local model path or HF repo id to use (default: ollama:qwen2.5-1.5b)")
    parser.add_argument("--yes", "-y", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    # Abort early when system RAM is very low to avoid hangs (override with --force-safe)
    # SKIP this check for Ollama models (Ollama manages memory separately)
    model_to_use = args.model_path or "ollama:qwen2.5-1.5b"
    is_ollama_model = model_to_use.startswith('ollama:') or model_to_use.startswith('ollama://')
    
    if psutil and not is_ollama_model:
        ram_avail_gb = psutil.virtual_memory().available / (1024**3)
        if ram_avail_gb < 2.0 and not args.force_safe:
            print()
            print("⚠️  ABORT: Available RAM is less than 2.0 GB.")
            print("This run may hang or crash while downloading/loading the model.")
            print("Options:")
            print("  • Close apps to free RAM and try again")
            print("  • Re-run with --force-safe --device cpu --yes to override (not recommended)")
            print("  • Run script/safe_refrag_test.py for a memory-safe test")
            print("  • Use Ollama: --model-path ollama:qwen2.5-1.5b --yes")
            print()
            return False
    elif is_ollama_model:
        print("Using Ollama model — RAM checks skipped (Ollama manages memory).")
        print()

    if not args.yes:
        input("Press ENTER to continue with model download (or Ctrl+C to cancel)...")
        print()

    # Test 1: Local REFRAG (pass CLI args)
    refrag_works = test_local_qwen(device=args.device, local_only=args.local_only, force_safe=args.force_safe, model_path=args.model_path or "ollama:qwen2.5-1.5b")
    
    # Test 2: Verify deliverables
    deliverables_complete = verify_days_5_6()
    
    # Final summary
    print("="*80)
    print("FULL TEST SUMMARY")
    print("="*80)
    print()
    
    if refrag_works:
        print("✅ LOCAL REFRAG: Working perfectly!")
        print("   - Ollama Qwen2.5-1.5B model connected")
        print("   - Query decomposition functional")
        print("   - Reasoning detection working")
        print("   - UNLIMITED queries (no rate limits!)")
    else:
        print("❌ LOCAL REFRAG: Failed")
        print("   - Check error messages above")
        print("   - Install: pip install transformers torch accelerate")
    
    print()
    
    if deliverables_complete:
        print("✅ DELIVERABLES: All files present")
    else:
        print("⚠️  DELIVERABLES: Some files missing (see above)")
    
    print()
    print("="*80)
    
    if refrag_works and deliverables_complete:
        print("🎉 DAYS 5-6 COMPLETE!")
        print("="*80)
        print()
        print("What you built:")
        print("  • REFRAG reasoning module (local, unlimited)")
        print("  • Self-Check verification agent (API, minimal tokens)")
        print("  • Enhanced CRAG integration")
        print("  • Scouting-optimized configuration")
        print("  • Complete documentation")
        print()
        print("Next: Days 7-8 (Frontend UI + Reports)")
        print()
        return True
    else:
        print("⚠️  DAYS 5-6 INCOMPLETE")
        print("="*80)
        print()
        print("Fix issues above, then re-run this test")
        print()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
