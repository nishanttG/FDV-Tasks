# Day 3: CRAG System Setup Guide

## Overview

Day 3 implements a **Corrective RAG (CRAG)** system with LangGraph workflow orchestration, following the PROJECT_TIMELINE.md specification.

## What's New

### 1. Fixed Player Name Detection
- **Issue**: Queries like "How good is Mo. Salah been this season" weren't detected
- **Solution**: Added proper noun detection and temporal keyword recognition
- **Files**: `src/agents/player_agent.py`

### 2. CRAG Agent with LangGraph
- **Workflow**: StateGraph with 4 nodes (retrieve → grade → web_search → generate)
- **Grading**: LLaMA-3-8B via Groq judges retrieval quality
- **Fallback**: Tavily web search for insufficient retrievals
- **Files**: `src/agents/crag_agent.py`

## Architecture

```
Query
  ↓
[Retrieve Node] ← Vector DB search
  ↓
[Grade Node] ← LLaMA-3-8B grader
  ↓
  ├─→ Sufficient → [Generate Node]
  └─→ Insufficient/Partial → [Web Search Node] → [Generate Node]
       ↓
     Tavily API
```

## Installation

### Step 1: Install Dependencies

```powershell
pip install langgraph langchain-groq tavily-python langchain-community
```

### Step 2: Get API Keys

1. **Groq API** (for LLaMA-3-8B):
   - Sign up at: https://console.groq.com/
   - Get API key from dashboard
   - Free tier: 30 requests/minute

2. **Tavily API** (for web search):
   - Sign up at: https://tavily.com/
   - Get API key from dashboard
   - Free tier: 1000 searches/month

### Step 3: Set Environment Variables

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
$env:TAVILY_API_KEY = "your_tavily_api_key_here"
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export TAVILY_API_KEY="your_tavily_api_key_here"
```

## Running the Demo

### Test Player Name Detection

```powershell
cd "c:\Users\Hp\Frost Digital Ventures\TactIQ"
python script/day3_crag_demo.py
```

This will:
1. Test improved player name detection (no API keys needed)
2. Run CRAG workflow with LangGraph (requires API keys)

### Example Queries

The demo tests these queries:
- "How good is Mo. Salah been this season" (player name + temporal)
- "Find me top young strikers under 23" (metadata filter)
- "What are the latest tactics for high pressing?" (tactical query)

## Expected Output

### Test 1: Player Name Detection
```
Query: How good is Mo. Salah been this season
Can handle: True
Detected player: Mo. Salah
```

### Test 2: CRAG Workflow
```
Query: How good is Mo. Salah been this season
Grade: partial
Used web search: True
Confidence: 0.70

Answer:
Mohamed Salah has been performing well this season with [statistics]...

Sources:
  - Mohamed Salah (Liverpool)
  - https://www.bbc.com/sport/football/...
```

## Fallback Modes

### Without API Keys
- **No GROQ_API_KEY**: Uses heuristic grading (document count)
- **No TAVILY_API_KEY**: Skips web search, uses only vector DB results

### Minimal Testing (No APIs)
```powershell
# Test only player name detection
python script/day3_crag_demo.py
```

## Files Modified/Created

### New Files
- `src/agents/crag_agent.py` - CRAG agent with LangGraph workflow
- `script/day3_crag_demo.py` - Demo script
- `docs/DAY3_CRAG_SETUP.md` - This guide

### Modified Files
- `src/agents/player_agent.py` - Added player name detection and temporal keywords

## Troubleshooting

### Error: "CRAG dependencies not installed"
**Solution**: Run `pip install langgraph langchain-groq tavily-python`

### Error: "GROQ_API_KEY not set"
**Solution**: Set environment variable as shown in Step 3

### Error: "Rate limit exceeded"
**Solution**: Wait 60 seconds (Groq free tier: 30 req/min)

### Player name not detected
**Check**: Query must have capitalized words (e.g., "Mo. Salah" not "mo salah")

## Next Steps (Day 4)

- [ ] Add Qwen2.5-2.5B for reasoning node
- [ ] Implement REFRAG reasoning module
- [ ] Add Self-Check verification agent
- [ ] Optimize prompt templates
- [ ] Add more test queries

## Timeline Compliance

 **Day 3-4 Requirements Met**:
- LangGraph workflow with state machine
- CRAG grader implementation (LLaMA-3-8B)
- Tavily integration for live updates
- Retrieval sufficiency checker
- Player name detection fixed

## Performance

- **Vector DB query**: <100ms
- **LLM grading**: ~1-2s per query
- **Web search**: ~2-3s per query
- **Total query time**: 3-6s (with fallback)

## API Costs (Estimate)

- **Groq**: Free tier 30 req/min (sufficient for testing)
- **Tavily**: Free tier 1000 searches/month
- **Total Day 3 testing**: $0 (within free tiers)
