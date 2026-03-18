# TactIQ Football Scout System - Complete Technical Guide

> **Your AI-powered football scouting system using Advanced RAG techniques**
> 
> **Last Updated**: January 17, 2026

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Project Architecture](#project-architecture)
3. [File Structure & Reading Order](#file-structure--reading-order)
4. [Data Flow & Process](#data-flow--process)
5. [RAG Concepts Explained](#rag-concepts-explained)
6. [CRAG (Corrective RAG)](#crag-corrective-rag)
7. [REFRAG (Reasoning + Factual RAG)](#refrag-reasoning--factual-rag)
8. [Self-RAG (Self-Reflective RAG)](#self-rag-self-reflective-rag)
9. [RAGAS (RAG Assessment)](#ragas-rag-assessment)
10. [Implementation Details](#implementation-details)
11. [Code Walkthrough](#code-walkthrough)
12. [Debugging & Troubleshooting](#debugging--troubleshooting)

---

## 1. System Overview

### What is TactIQ?

TactIQ is an **AI-powered football scouting system** that generates professional scout reports for football players using:
- **Statistical data** from FBref (70,904+ player records)
- **Advanced RAG** (Retrieval-Augmented Generation) techniques
- **LangGraph** for orchestrating complex workflows
- **Multiple LLM models**: LLaMA-3.1-8B (Groq) + Qwen2.5:1.5b (Ollama)

### Key Capabilities

 **Position-specific scout reports** (GK, DF, MF, FW)  
 **Multi-season analysis** (2021-2026)  
 **Confidence scoring** based on data quality  
 **Self-verification** to prevent hallucinations  
 **Reasoning traces** for transparency  
 **PDF export** with professional formatting

---

## 2. Project Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        STREAMLIT UI (app_old_backup.py)         │
│                     User Query → Intent Classification           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              ENHANCED CRAG AGENT (enhanced_crag_agent.py)       │
│         ┌──────────────┬──────────────┬──────────────┐          │
│         │ CRAG Core    │  REFRAG      │ Self-Check   │          │
│         │ (Retrieve +  │  (Reasoning  │ (Verify)     │          │
│         │  Generate)   │   Layer)     │              │          │
│         └──────┬───────┴──────┬───────┴──────┬───────┘          │
└────────────────┼──────────────┼──────────────┼──────────────────┘
                 │              │              │
                 ▼              ▼              ▼
┌────────────────────────┐  ┌──────────────┐  ┌──────────────────┐
│   VECTOR DATABASE      │  │  LLM Models  │  │  Position-Aware  │
│   (ChromaDB)           │  │  - LLaMA 3.1 │  │  Templates       │
│   - 70,904 docs        │  │  - Qwen2.5   │  │  (4 positions)   │
│   - Embedded stats     │  └──────────────┘  └──────────────────┘
│   - Metadata-rich      │
└────────────────────────┘
```

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI Framework** | Streamlit | Interactive web interface |
| **Orchestration** | LangGraph | State machine for RAG workflow |
| **Vector DB** | ChromaDB | Similarity search on player stats |
| **LLM (Main)** | LLaMA-3.1-8B via Groq | Report generation |
| **LLM (Reasoning)** | Qwen2.5:1.5b via Ollama | Local reasoning layer |
| **Data Source** | FBref CSV files | Player statistics |
| **Embeddings** | sentence-transformers | Text vectorization |

---

## 3. File Structure & Reading Order

###  **START HERE**: Entry Point Files

1. **`app_old_backup.py`**  **(MAIN APP - READ FIRST)**
   - Streamlit UI interface
   - Query handling and intent classification
   - Result rendering and PDF export
   - **Lines to focus**: 
     - 196-220: Query processing
     - 490-495: Rendering logic

2. **`src/agents/enhanced_crag_agent.py`**  **(CORE LOGIC - READ SECOND)**
   - Orchestrates CRAG + REFRAG + Self-Check
   - **Lines to focus**:
     - 115-180: Main query method
     - 160-168: REFRAG invocation

3. **`src/agents/crag_agent.py`** **(RAG ENGINE - READ THIRD)**
   - LangGraph state machine
   - Retrieval, grading, generation nodes
   - **Lines to focus**:
     - 268-916: Retrieval logic
     - 1260-1650: Metadata merging + CSV fallback
     - 1415-1485: Position-specific priority stats

###  Supporting Files (Read After Core)

4. **`src/agents/position_prompts.py`**
   - GK, DF, MF, FW scout report templates
   - Position-specific instructions for LLM

5. **`src/agents/refrag_agent.py`**
   - Reasoning layer implementation
   - Uses local Ollama model for efficiency

6. **`src/agents/selfcheck_agent.py`**
   - Verification logic to prevent hallucinations
   - Confidence scoring

7. **`src/ui/intent_renderers.py`**
   - Intent-based rendering (scout_report, comparison, etc.)

8. **`src/agents/intent_classifier.py`**
   - Classifies user query into intent types

###  Data Files

9. **`data/processed/player_stats_unified_FINAL_DEDUPED.csv`**
   - Main database with all player stats
   - 13,888 records, 100+ columns per player

10. **`db/chroma/`**
    - ChromaDB vector database storage
    - Embedded player descriptions + metadata

---

## 4. Data Flow & Process

### Complete Request Flow

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. USER INPUT                                                    │
│    Query: "Rayan Cherki scout report"                           │
│    Season: 2025-2026                                             │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. INTENT CLASSIFICATION (intent_classifier.py)                 │
│    → Detected Intent: "scout_report"                            │
│    → Confidence: 90%                                             │
│    → Metadata: {player: "Rayan Cherki", season: "2025-2026"}   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ENHANCED CRAG AGENT (enhanced_crag_agent.py)                │
│    → Decides to use REFRAG reasoning (force_reasoning=True)     │
│    → Calls crag_agent.query()                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. CRAG WORKFLOW (crag_agent.py - LangGraph State Machine)     │
│                                                                  │
│  ┌─────────────┐                                                │
│  │  RETRIEVE   │ ← Query ChromaDB for relevant player docs     │
│  │   NODE      │   (identity, shooting, passing, defensive)    │
│  └──────┬──────┘                                                │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │   GRADE     │ ← Check if retrieved docs are sufficient      │
│  │   NODE      │   Grade: "context_sufficient" or              │
│  └──────┬──────┘   "context_missing_facts"                     │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐                                                │
│  │  GENERATE   │ ← Merge metadata from ALL docs                │
│  │   NODE      │   + CSV fallback for missing columns          │
│  └──────┬──────┘   + Apply position-specific template          │
│         │            + Generate scout report with LLaMA         │
│         ▼                                                        │
│    [Report Generated]                                           │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. REFRAG REASONING (refrag_agent.py)                          │
│    → Uses Qwen2.5 to validate reasoning                         │
│    → Checks for logical consistency                             │
│    → Returns reasoning trace                                    │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. SELF-CHECK VERIFICATION (selfcheck_agent.py)                │
│    → Verifies no hallucinations                                 │
│    → Checks stat accuracy                                       │
│    → Returns confidence score                                   │
└────────────────────────┬────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. RESULT RENDERING (intent_renderers.py)                      │
│    → Position-aware formatting                                  │
│    → Adds tactical view                                         │
│    → Confidence indicators                                      │
│    → PDF export option                                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. RAG Concepts Explained

### What is RAG? (Retrieval-Augmented Generation)

**Problem**: Large Language Models (LLMs) have limited knowledge:
- Trained on static data (cutoff date)
- No access to private/domain-specific data
- Can hallucinate facts

**Solution**: RAG = Retrieval + Generation

```
Traditional LLM:
Question → LLM → Answer (may hallucinate)

RAG:
Question → Retrieve Relevant Docs → LLM + Context → Accurate Answer
```

### Basic RAG Pipeline

```python
# 1. User asks question
query = "What is Rayan Cherki's xG?"

# 2. Retrieve relevant documents from vector database
docs = vector_db.similarity_search(query, k=5)

# 3. Create prompt with retrieved context
prompt = f"Context: {docs}\n\nQuestion: {query}"

# 4. Generate answer using LLM + context
answer = llm.generate(prompt)
```

### Why Basic RAG Isn't Enough

 **Problems with Basic RAG**:
1. **Retrieval failures**: May not find the right documents
2. **Hallucinations**: LLM may ignore context and make up facts
3. **No verification**: Can't tell if answer is accurate
4. **Context confusion**: Too much irrelevant context
5. **No reasoning transparency**: Black box decision-making

 **Solution**: Advanced RAG techniques (CRAG, REFRAG, Self-RAG)

---

## 6. CRAG (Corrective RAG)

### What is CRAG?

**CRAG = Corrective Retrieval-Augmented Generation**

CRAG adds a **self-correction mechanism** to RAG:
- **Grades** the retrieved documents
- **Corrects** retrieval if documents are insufficient
- **Routes** to web search if database is missing facts

### CRAG Workflow in TactIQ

```python
# File: src/agents/crag_agent.py

class CRAGWorkflow:
    def __init__(self):
        # Build state machine with LangGraph
        workflow = StateGraph(CRAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("grade", self._grade_node)
        workflow.add_node("generate", self._generate_node)
        workflow.add_node("web_search", self._web_search_node)
        
        # Add edges (routing logic)
        workflow.add_edge("retrieve", "grade")
        workflow.add_conditional_edges(
            "grade",
            self._route_after_grade,  # Decides next step
            {
                "generate": "generate",
                "web_search": "web_search"
            }
        )
```

### CRAG Nodes Explained

#### 1️⃣ **RETRIEVE Node** (`_retrieve_node`, lines 268-916)

**Purpose**: Find relevant player documents from ChromaDB

**Process**:
```python
def _retrieve_node(self, state):
    query = state['question']
    
    # 1. Detect player name and season
    player_name = extract_player_name(query)  # "Rayan Cherki"
    season = extract_season(query)            # "2025-2026"
    
    # 2. Detect required stat modules
    modules = ['identity', 'shooting', 'passing', 'defensive', 'progression']
    
    # 3. Query ChromaDB with filters
    docs = self.vector_db.query(
        query_texts=[f"{player_name} {season}"],
        n_results=50,
        where={'season': season}  # Filter by season
    )
    
    # 4. Ensure all modules are present
    # If missing, do targeted retrieval for each module
    for module in missing_modules:
        module_docs = self.vector_db.query(
            query_texts=[f"{player_name} {module}"],
            where={'season': season, 'stat_module': module}
        )
        docs.extend(module_docs)
    
    return {'documents': docs, 'player_name': player_name}
```

**Key Innovation in TactIQ**:
- **Module-aware retrieval**: Ensures all stat types (shooting, passing, etc.) are retrieved
- **Fuzzy name matching**: Handles "Salah" vs "Mohamed Salah"
- **Fallback seasons**: If 2025-2026 has no data, tries 2024-2025

#### 2️⃣ **GRADE Node** (`_grade_node`, lines 933-975)

**Purpose**: Evaluate if retrieved documents are sufficient

**Grading Logic**:
```python
def _grade_node(self, state):
    docs = state['documents']
    question = state['question']
    
    # Check sufficiency
    if len(docs) == 0:
        return {'grade': 'context_missing_facts'}
    
    # Check if player data is present
    player_found = any(
        doc.get('metadata', {}).get('player') == player_name
        for doc in docs
    )
    
    if player_found and len(docs) >= 5:
        return {'grade': 'context_sufficient'}
    else:
        return {'grade': 'context_missing_facts'}
```

**Grades**:
-  `context_sufficient`: Good documents, proceed to generation
-  `context_missing_facts`: Poor documents, trigger web search or refine retrieval

#### 3️⃣ **ROUTING** (`_route_after_grade`, lines 977-1010)

**Purpose**: Decide next step based on grade

```python
def _route_after_grade(self, state):
    grade = state['grade']
    has_docs = len(state['documents']) > 0
    
    if grade == 'context_sufficient' and has_docs:
        return 'generate'  # Use database data
    else:
        return 'web_search'  # Search Tavily for recent facts
```

#### 4️⃣ **GENERATE Node** (`_generate_node`, lines 1260-2000)

**Purpose**: Create scout report using LLM + retrieved context

**Critical Innovation - CSV Fallback** (lines 1295-1315):
```python
# FALLBACK: Load player data from CSV to fill missing columns
# This ensures all stats are available even if not in Chroma documents
try:
    if player_name and season:
        csv_path = 'data/processed/player_stats_unified_FINAL_DEDUPED.csv'
        df = pd.read_csv(csv_path)
        
        # Find matching player record
        player_mask = (df['player'] == player_name) & (df['season'] == season)
        player_row = df[player_mask].iloc[0]
        
        # Fill in missing columns from CSV
        for col in player_row.index:
            if col not in merged_meta and pd.notna(player_row[col]):
                merged_meta[col] = player_row[col]
                
        logger.info(f"✅ Filled missing columns from CSV")
```

**Why This is Critical**:
- ChromaDB documents only store **embedded text + limited metadata**
- **CSV has ALL 100+ columns** (Expected_xG_std, Standard_SoT%_shoot, etc.)
- **Fallback ensures completeness**: Even if ChromaDB is missing stats, we get them from CSV

**Position-Specific Generation** (lines 1415-1485):
```python
PRIORITY_BY_POSITION = {
    'GK': ['saves', 'save%', 'PSxG-GA', 'clean_sheets', 'pass_completion'],
    'DF': ['tackles', 'interceptions', 'blocks', 'clearances', 'pass%'],
    'MF': ['goals', 'assists', 'xG', 'xAG', 'progressive_passes', 'key_passes'],
    'FW': ['goals', 'xG', 'shots', 'conversion%', 'key_passes']
}

# Show priority stats first, then remaining stats
detected_position = extract_position(merged_meta)
priority_list = PRIORITY_BY_POSITION[detected_position]

for stat in priority_list:
    if stat in merged_meta:
        stat_dict[stat] = merged_meta[stat]
```

---

## 7. REFRAG (Reasoning + Factual RAG)

### What is REFRAG?

**REFRAG = REasoning + Factual RAG**

REFRAG adds a **reasoning layer** on top of CRAG:
- **Pre-generates reasoning** before final answer
- **Validates logical consistency**
- **Provides transparency** into decision-making

### Why REFRAG?

**Problem**: Standard RAG can produce answers without clear reasoning:
```
Question: "Is Rayan Cherki worth €50M?"
Standard RAG: "Yes, he's worth €50M."
```

 **What's wrong?**:
- No explanation of how we reached this conclusion
- No breakdown of evaluation criteria
- Can't audit the decision

**REFRAG Solution**:
```
Question: "Is Rayan Cherki worth €50M?"

REFRAG Reasoning:
1. Market context: Mid-tier attacking midfielders €30-70M
2. Output: 2G + 7A in 642 minutes = 0.73 G+A/90 (above average)
3. Age: 22 years old = prime development window
4. Sample size: Limited minutes (642) = risk factor
5. Conclusion: Fair valuation given output + age + risk

REFRAG Answer: "Yes, €50M is fair given..."
```

### REFRAG Implementation in TactIQ

**File**: `src/agents/refrag_agent.py`

```python
class REFRAGAgent:
    def __init__(self):
        # Use local Ollama for efficiency
        self.llm = ChatOllama(
            model="qwen2.5:1.5b",  # Small, fast model
            temperature=0.1
        )
    
    def generate_reasoning(self, question, context, retrieved_docs):
        # Step 1: Extract key facts from context
        prompt = f"""
        Context: {context}
        Question: {question}
        
        Generate step-by-step reasoning:
        1. What data do we have?
        2. What patterns do we see?
        3. What conclusions can we draw?
        4. What are the limitations?
        """
        
        reasoning = self.llm.invoke(prompt)
        
        # Step 2: Validate reasoning against facts
        validation_prompt = f"""
        Reasoning: {reasoning}
        Facts: {retrieved_docs}
        
        Check:
        - Are all claims supported by facts?
        - Are there logical inconsistencies?
        - What's the confidence level?
        """
        
        validation = self.llm.invoke(validation_prompt)
        
        return {
            'reasoning': reasoning,
            'validation': validation,
            'confidence': extract_confidence(validation)
        }
```

### REFRAG in Enhanced CRAG Agent

**File**: `src/agents/enhanced_crag_agent.py`, lines 160-168

```python
def query(self, question, intent=None, force_reasoning=True):
    # Run standard CRAG workflow
    result = self.crag.query(question, intent=intent)
    
    # If REFRAG enabled, add reasoning layer
    if self.use_refrag and force_reasoning:
        reasoning_result = self.refrag.generate_reasoning(
            question=question,
            context=result.get('answer'),
            retrieved_docs=result.get('documents')
        )
        
        result['reasoning'] = reasoning_result['reasoning']
        result['reasoning_confidence'] = reasoning_result['confidence']
    
    return result
```

**When REFRAG Activates**:
 Always for scout reports (default)  
 User can toggle in sidebar  
 Skipped for simple lookups (saves compute)

---

## 8. Self-RAG (Self-Reflective RAG)

### What is Self-RAG?

**Self-RAG = Self-Reflective RAG**

Self-RAG adds **self-verification** after generation:
- **Checks for hallucinations**
- **Validates stat accuracy**
- **Flags inconsistencies**
- **Scores confidence**

### Why Self-Check?

**Problem**: LLMs can hallucinate even with context:
```
Context: "Cherki: 2 goals, 7 assists"
LLM Output: "Cherki scored 15 goals this season"  ❌ HALLUCINATION
```

**Self-RAG Solution**:
```python
# After generation, verify the answer
def self_check(answer, context):
    # Extract all factual claims from answer
    claims = extract_claims(answer)  # ["2 goals", "7 assists", "48 progressive passes"]
    
    # Check each claim against context
    for claim in claims:
        if not verify_claim_in_context(claim, context):
            return {
                'status': 'hallucination_detected',
                'issue': f"Claim '{claim}' not found in context"
            }
    
    return {'status': 'verified', 'confidence': 0.95}
```

### Self-Check Implementation in TactIQ

**File**: `src/agents/selfcheck_agent.py`

```python
class SelfCheckAgent:
    def verify(self, answer, context, documents):
        # 1. Extract claims from answer
        claims = self._extract_claims(answer)
        
        # 2. Cross-reference with database
        issues = []
        for claim in claims:
            # Check if stat is in retrieved documents
            if not self._verify_stat(claim, documents):
                issues.append(f"Unverified stat: {claim}")
        
        # 3. Check for logical consistency
        if self._has_contradictions(answer):
            issues.append("Logical contradictions detected")
        
        # 4. Calculate confidence
        confidence = 1.0 - (len(issues) * 0.1)  # -10% per issue
        
        return {
            'verified': len(issues) == 0,
            'confidence': max(0.0, confidence),
            'issues': issues
        }
```

**Self-Check in Enhanced CRAG** (lines 168-178):

```python
def query(self, question, intent=None, skip_verification=False):
    # Run CRAG + REFRAG
    result = self.crag.query(question)
    
    # Self-check verification
    if self.use_selfcheck and not skip_verification:
        verification = self.selfcheck.verify(
            answer=result['answer'],
            context=result['context'],
            documents=result['documents']
        )
        
        result['verification'] = verification
        
        # Adjust confidence if issues found
        if verification['confidence'] < 0.7:
            logger.warning(f"Low confidence: {verification['issues']}")
    
    return result
```

**Confidence Scoring** (lines 1961-1991):

```python
# Calculate confidence based on multiple factors
def calculate_confidence(minutes, num_stats, sources):
    base_confidence = 0.5
    
    # Factor 1: Minutes played (sample size)
    if minutes >= 900:
        minutes_score = 0.3  # 🟢 Strong
    elif minutes >= 450:
        minutes_score = 0.2  # 🟡 Limited
    else:
        minutes_score = 0.1  # 🔴 Sparse
    
    # Factor 2: Metric coverage
    if num_stats >= 30:
        metrics_score = 0.15  # 🟢 Comprehensive
    elif num_stats >= 15:
        metrics_score = 0.1   # 🟡 Partial
    else:
        metrics_score = 0.05  # 🔴 Sparse
    
    # Factor 3: Data source
    if sources == 'DB':
        source_score = 0.05  # 🟢 Strong
    else:
        source_score = 0.02  # 🟡 Limited
    
    total_confidence = base_confidence + minutes_score + metrics_score + source_score
    
    return min(1.0, total_confidence)
```

---

## 9. RAGAS (RAG Assessment)

### What is RAGAS?

**RAGAS = Retrieval-Augmented Generation Assessment**

RAGAS is a **framework for evaluating RAG systems**, not part of the generation pipeline.

**Metrics**:
1. **Context Precision**: Are retrieved docs relevant?
2. **Context Recall**: Did we retrieve all relevant docs?
3. **Faithfulness**: Is answer grounded in context?
4. **Answer Relevancy**: Does answer match question?

### RAGAS in TactIQ (Evaluation, not Production)

**File**: `evaluation/` (not used in production)

```python
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy
)

# Evaluate system performance
dataset = [
    {
        'question': 'What is Salah\'s xG?',
        'contexts': retrieved_docs,
        'answer': generated_answer,
        'ground_truth': '22.4 xG'
    }
]

results = evaluate(
    dataset,
    metrics=[context_precision, faithfulness, answer_relevancy]
)

print(results)
# Output:
# context_precision: 0.92
# faithfulness: 0.88
# answer_relevancy: 0.95
```

**When to Use RAGAS**:
-  Development/testing phase
-  System performance evaluation
-  Comparing RAG configurations
-  NOT used in production (adds latency)

---

## 10. Implementation Details

### Complete Query Flow with Code References

```python
# ============================================================
# STEP 1: USER SUBMITS QUERY (app_old_backup.py, line 196)
# ============================================================
query = "Rayan Cherki scout report"
selected_season = "2025-2026"

# ============================================================
# STEP 2: INTENT CLASSIFICATION (line 198)
# ============================================================
intent_classifier = IntentClassifier()
intent, confidence, metadata = intent_classifier.classify(query)
# Result: intent='scout_report', confidence=0.9

# ============================================================
# STEP 3: ENHANCED CRAG QUERY (line 207)
# ============================================================
result = agent.query(
    query,
    intent=intent,
    intent_metadata=metadata,
    force_reasoning=True,  # Enable REFRAG
    skip_verification=False  # Enable Self-Check
)

# ============================================================
# STEP 4: CRAG RETRIEVAL (crag_agent.py, line 268)
# ============================================================
def _retrieve_node(state):
    # Extract player name
    player_name = "Rayan Cherki"
    season = "2025-2026"
    
    # Query ChromaDB
    docs = vector_db.query(
        query_texts=[f"{player_name} midfielder stats"],
        n_results=50,
        where={'season': season}
    )
    
    # Ensure all stat modules present
    required_modules = ['identity', 'shooting', 'passing', 'defensive', 'progression']
    for module in required_modules:
        if module not in retrieved_modules:
            # Targeted retrieval
            module_docs = vector_db.query(
                query_texts=[f"{player_name} {module}"],
                where={'season': season, 'stat_module': module}
            )
            docs.extend(module_docs)
    
    return {'documents': docs}

# ============================================================
# STEP 5: METADATA MERGING + CSV FALLBACK (line 1262)
# ============================================================
merged_meta = {}

# Merge from ChromaDB docs
for doc in docs:
    for key, val in doc['metadata'].items():
        if key not in merged_meta:
            merged_meta[key] = val

# CSV FALLBACK - CRITICAL INNOVATION
csv_path = 'data/processed/player_stats_unified_FINAL_DEDUPED.csv'
df = pd.read_csv(csv_path)
player_row = df[(df['player'] == player_name) & (df['season'] == season)].iloc[0]

for col in player_row.index:
    if col not in merged_meta and pd.notna(player_row[col]):
        merged_meta[col] = player_row[col]
        
# Result: merged_meta now has ALL stats (Expected_xG_std, Standard_SoT%_shoot, etc.)

# ============================================================
# STEP 6: POSITION-SPECIFIC STATS (line 1415)
# ============================================================
detected_position = 'MF'  # From merged_meta['pos']

PRIORITY_BY_POSITION = {
    'MF': [
        'Performance_Gls_std', 'Performance_Ast_std',
        'Expected_xG_std', 'Expected_xAG_std',
        'KP_pass', 'PrgP_pass', 'Progression_PrgP_std',
        'Standard_Sh_shoot', 'Standard_SoT%_shoot'
    ]
}

priority_stats = PRIORITY_BY_POSITION['MF']

# Build stat display
stat_lines = []
for stat in priority_stats:
    if stat in merged_meta:
        stat_lines.append(f"{stat} = {merged_meta[stat]}")

# Result:
# Performance_Gls_std = 2
# Performance_Ast_std = 7
# Expected_xG_std = 1.2
# Expected_xAG_std = 4.8
# KP_pass = 27
# PrgP_pass = 48.0

# ============================================================
# STEP 7: GENERATE SCOUT REPORT (line 1916)
# ============================================================
from src.agents.position_prompts import MIDFIELDER_SCOUT_REPORT_PROMPT

prompt = MIDFIELDER_SCOUT_REPORT_PROMPT.format(
    question=query,
    context=stat_lines + retrieved_docs_text
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
answer = llm.invoke(prompt)

# ============================================================
# STEP 8: REFRAG REASONING (enhanced_crag_agent.py, line 160)
# ============================================================
reasoning_result = refrag.generate_reasoning(
    question=query,
    context=answer,
    retrieved_docs=docs
)

# Result:
# reasoning: "Cherki operates as interior 10. High progressive passes (48)..."
# confidence: 0.85

# ============================================================
# STEP 9: SELF-CHECK VERIFICATION (line 168)
# ============================================================
verification = selfcheck.verify(
    answer=answer,
    context=stat_lines,
    documents=docs
)

# Result:
# verified: True
# confidence: 0.95
# issues: []

# ============================================================
# STEP 10: RENDER RESULT (app_old_backup.py, line 490)
# ============================================================
render_result(result, query)
```

---

## 11. Code Walkthrough

### Key Functions Deep Dive

#### 1. CSV Fallback Logic (CRITICAL!)

**File**: `src/agents/crag_agent.py`, lines 1295-1315

```python
# FALLBACK: Load player data from CSV to fill in missing columns
try:
    if player_name and season:
        import pandas as pd
        import os
        
        csv_path = os.path.join(
            os.path.dirname(__file__), '..', '..', 
            'data', 'processed', 
            'player_stats_unified_FINAL_DEDUPED.csv'
        )
        
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Find matching player record
            player_mask = (
                df['player'].str.contains(player_name, na=False, case=False, regex=False)
            ) & (df['season'] == season)
            
            if len(player_mask) > 0 and player_mask.sum() > 0:
                player_row = df[player_mask].iloc[0]
                
                # Fill in missing columns from CSV
                for col in player_row.index:
                    if col not in merged_meta and pd.notna(player_row[col]) and player_row[col] != '':
                        merged_meta[col] = player_row[col]
                
                logger.info(
                    f" Filled {len(merged_meta)} columns total after CSV fallback"
                )
except Exception as e:
    logger.debug(f"CSV fallback lookup failed: {e}")  # Silent fail
```

**Why This is Critical**:
1. **ChromaDB limitation**: Vector databases store embedded text + limited metadata
2. **CSV has everything**: All 100+ stat columns are in the CSV
3. **Ensures completeness**: Even if ChromaDB is incomplete, we get full stats
4. **Silent fallback**: If CSV fails, system continues (graceful degradation)

**Example**:
- ChromaDB has: `player, pos, age, team, season, Tackles_Tkl_def=10`
- CSV has: `Expected_xG_std=1.2, Standard_SoT%_shoot=37.5, KP_pass=27` ← MISSING IN CHROMA
- After fallback: `merged_meta` has ALL columns 

#### 2. Position-Specific Filtering

**File**: `src/agents/crag_agent.py`, lines 1540-1560

```python
# POSITION-SPECIFIC FILTERING: Skip columns inappropriate for detected position
POSITION_SKIP_PATTERNS = {
    'GK': [
        '_def', '_std', 'Tackles', 'Interceptions', 'Int_def', 
        'Blocks_', 'Clr', 'Fls', 'Fld', 'Pressures', 'Pressure', 
        'Duels', 'Dribble', 'Challenges'
    ],  # Skip defender/field player stats for GK
    'DF': ['_gk', 'Performance_Saves', 'PSxG', 'Save%', 'Launch%'],
    'MF': ['_gk', 'Performance_Saves', 'Save%', 'Launch%'],
    'FW': ['_gk', 'Performance_Saves', 'Save%', 'Launch%', 'Clearance', 'Blocks_']
}

skip_patterns = POSITION_SKIP_PATTERNS.get(detected_position, [])

# Filter out inappropriate stats
for key, val in sorted(merged_meta.items()):
    # Skip if already in priority list
    if key in stat_dict or key in SKIP_KEYS:
        continue
    
    # Skip position-inappropriate columns
    skip_this = False
    for pattern in skip_patterns:
        if pattern.lower() in key.lower():
            skip_this = True
            break
    
    if skip_this:
        continue  # Don't show this column for this position
    
    # Add to display
    stat_dict[key] = val
```

**Why This Matters**:
- **GK shouldn't show tackles**: Goalkeepers don't make tackles
- **FW shouldn't show saves**: Forwards don't save goals
- **Prevents confusion**: Only shows relevant stats per position

#### 3. Intent Classification

**File**: `src/agents/intent_classifier.py`

```python
class IntentClassifier:
    def classify(self, query: str):
        query_lower = query.lower()
        
        # Scout report intent
        if any(kw in query_lower for kw in ['report', 'scout', 'analysis']):
            # Extract player name
            player_name = self._extract_player_name(query)
            return 'scout_report', 0.9, {'player': player_name}
        
        # Comparison intent
        if any(kw in query_lower for kw in ['compare', 'vs', 'versus']):
            players = self._extract_multiple_players(query)
            return 'comparison', 0.85, {'players': players}
        
        # Stats lookup intent
        if any(kw in query_lower for kw in ['what is', 'how many', 'stats']):
            return 'stat_lookup', 0.8, {}
        
        # Default
        return 'general', 0.5, {}
```

**Intent Types**:
- `scout_report`: Full position-specific report
- `comparison`: Side-by-side player comparison
- `stat_lookup`: Quick stat retrieval
- `tactical_analysis`: Deep tactical breakdown
- `general`: Open-ended query

---

## 12. Debugging & Troubleshooting

### Common Issues & Solutions

#### Issue 1: "Data not available" for stats that exist

**Symptom**: Report shows "xG: Data not available" even though player has xG data

**Root Causes**:
1. **Column name mismatch**:
   - Template asks for `xG`
   - Database has `Expected_xG_std`
   
2. **ChromaDB incomplete**:
   - Vector DB only has partial metadata
   - CSV fallback not working

**Solution**:
```python
# Check priority list uses actual DB column names
PRIORITY_BY_POSITION = {
    'MF': [
        'Expected_xG_std',  # Correct
        # NOT 'xG'           #  Wrong
    ]
}

# Verify CSV fallback is running
logger.info(f" Filled {len(merged_meta)} columns after CSV fallback")
# Should see this in logs
```

**Debug Steps**:
1. Check logs: `2026-01-16 18:06:17.050 | INFO | Filled 117 columns total after CSV fallback`
2. If not seeing CSV fallback, check file path
3. Verify column exists in CSV: `pd.read_csv('data/processed/player_stats_unified_FINAL_DEDUPED.csv').columns`

#### Issue 2: Goalkeeper showing defender stats

**Symptom**: GK report shows "Tackles: 0, Interceptions: 0"

**Root Cause**: Position filtering not aggressive enough

**Solution**:
```python
# Enhanced skip patterns for GK
POSITION_SKIP_PATTERNS = {
    'GK': [
        '_def',       # All defender columns
        '_std',       # Standard field player stats
        'Tackles',    # Explicit tackle columns
        'Int_def',    # Interceptions
        'Blocks_',    # Blocks
        'Pressures',  # Pressing stats
        'Dribble'     # Dribbling stats
    ]
}
```

#### Issue 3: Low confidence scores

**Symptom**: Confidence always 50-60%

**Root Cause**: Insufficient minutes or sparse metrics

**Solution**:
```python
# Check confidence calculation
def calculate_confidence(minutes, num_stats):
    base = 0.5
    
    # Minutes factor (0-0.3)
    if minutes >= 900:
        base += 0.3
    elif minutes >= 450:
        base += 0.2
    
    # Metrics factor (0-0.15)
    if num_stats >= 30:
        base += 0.15
    elif num_stats >= 15:
        base += 0.1
    
    return min(1.0, base)

# For 642 minutes + 32 stats → 81% confidence
```

#### Issue 4: ChromaDB "where clause" error

**Symptom**: `Expected where to have exactly one operator`

**Root Cause**: Incorrect where filter syntax

**Solution**:
```python
#  WRONG: Multiple keys in where dict
results = vector_db.query(
    where={'season': '2025-2026', 'player': 'Salah'}
)

#  CORRECT: Use $and operator
results = vector_db.query(
    where={
        '$and': [
            {'season': {'$eq': '2025-2026'}},
            {'player': {'$eq': 'Salah'}}
        ]
    }
)

#  SIMPLER: Filter client-side
results = vector_db.query(where={'season': '2025-2026'})
filtered = [r for r in results if r['metadata']['player'] == 'Salah']
```

### Logging Best Practices

**Enable debug logging**:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or in Streamlit:
streamlit run app_old_backup.py --logger.level=debug
```

**Key log messages to look for**:
```
 Position from merged_meta 'pos' field: MF -> primary: MF -> template: MF
 Filled 104 columns total after CSV fallback (added 18 new columns from CSV)
 Showing 37 available stats (37 priority + 0 other)
 Final Confidence: 81% (minutes: 642.0, metrics: 114, sources: DB)
```

---

## Summary: Key Takeaways

### 1. **TactIQ Architecture**
- **Streamlit UI** → **Enhanced CRAG Agent** → **CRAG + REFRAG + Self-Check** → **ChromaDB + CSV**

### 2. **CRAG (Corrective RAG)**
- Retrieves docs → Grades quality → Routes to generation or web search
- **Critical innovation**: CSV fallback ensures completeness

### 3. **REFRAG (Reasoning RAG)**
- Adds reasoning layer before final answer
- Uses local Ollama for efficiency
- Provides transparency

### 4. **Self-RAG (Self-Check)**
- Verifies answer after generation
- Prevents hallucinations
- Scores confidence

### 5. **Key Files to Master**
1. `app_old_backup.py` - UI and orchestration
2. `enhanced_crag_agent.py` - Main agent logic
3. `crag_agent.py` - RAG engine with CSV fallback
4. `position_prompts.py` - Position-specific templates

### 6. **Most Important Code**
- **CSV Fallback** (lines 1295-1315): Ensures all stats available
- **Position Filtering** (lines 1540-1560): Shows only relevant stats
- **Confidence Calculation** (lines 1961-1991): Quality scoring

### 7. **Debugging Checklist**
 Check logs for CSV fallback message  
 Verify column names match database  
 Ensure position filtering is correct  
 Validate confidence calculation  
 Test with multiple players/positions

---

## Next Steps: Learning Path

### Week 1: Core Concepts
- [ ] Read this guide thoroughly
- [ ] Understand basic RAG (retrieval + generation)
- [ ] Study CRAG workflow diagram
- [ ] Run system with debug logging

### Week 2: Code Deep Dive
- [ ] Read `crag_agent.py` retrieve node (lines 268-916)
- [ ] Study CSV fallback logic (lines 1295-1315)
- [ ] Understand position-specific generation (lines 1415-1485)
- [ ] Trace a complete query from UI to response

### Week 3: Advanced Topics
- [ ] Study REFRAG reasoning layer
- [ ] Understand Self-Check verification
- [ ] Learn LangGraph state machine
- [ ] Experiment with prompt engineering

### Week 4: Customization
- [ ] Add new position (e.g., Wing-Back)
- [ ] Create custom stat priority lists
- [ ] Modify confidence calculation
- [ ] Improve prompt templates

---

## Resources

### Official Documentation
- **LangGraph**: https://langchain-ai.github.io/langgraph/
- **ChromaDB**: https://docs.trychroma.com/
- **Streamlit**: https://docs.streamlit.io/

### Research Papers
- **CRAG**: "Corrective Retrieval Augmented Generation" (2024)
- **Self-RAG**: "Self-Reflective RAG with Self-Assessment" (2023)
- **RAGAS**: "RAG Assessment Framework" (2024)


---

**Created with 💙 for TactIQ Football Scout System**

**Last Updated**: January 17, 2026  
**Version**: 1.0.0  
**Author**: TactIQ Development Team

---

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                    TACTIQ CHEAT SHEET                        │
├─────────────────────────────────────────────────────────────┤
│ START APP                                                    │
│   streamlit run app_old_backup.py                           │
│                                                              │
│ DEBUG MODE                                                   │
│   streamlit run app_old_backup.py --logger.level=debug      │
│                                                              │
│ KEY FILES                                                    │
│   1. app_old_backup.py           - Main UI                  │
│   2. enhanced_crag_agent.py      - Orchestration            │
│   3. crag_agent.py               - RAG engine               │
│   4. position_prompts.py         - Templates                │
│                                                              │
│ CRITICAL CODE                                                │
│   CSV Fallback:       crag_agent.py:1295-1315              │
│   Position Filter:    crag_agent.py:1540-1560              │
│   Confidence Score:   crag_agent.py:1961-1991              │
│                                                              │
│ DATA LOCATION                                                │
│   CSV: data/processed/player_stats_unified_FINAL_DEDUPED.csv│
│   Vector DB: db/chroma/                                      │
│                                                              │
│ POSITIONS                                                    │
│   GK  - Goalkeeper    (saves, clean sheets)                 │
│   DF  - Defender      (tackles, clearances)                 │
│   MF  - Midfielder    (passes, progression)                 │
│   FW  - Forward       (goals, shots, xG)                    │
└─────────────────────────────────────────────────────────────┘
```
