"""
Streamlit App for CRAG Football Scout System
Interactive web interface for querying the CRAG system 

"""

import streamlit as st
import sys
from pathlib import Path
from datetime import datetime
from loguru import logger
import json
import re
import pandas as pd
import unicodedata
import difflib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

import chromadb
from src.agents.intent_classifier import IntentClassifier
from src.agents.position_prompts import (
    detect_position_from_query,
    detect_position_from_metadata,
    get_prompt_for_position
)
from src.ui.intent_renderers import render_result, show_query_insights
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Page config
st.set_page_config(
    page_title="Football Scout Assistant",
    page_icon="",
    layout="wide"
)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []

# Initialize Enhanced CRAG system (cached)
@st.cache_resource
def initialize_crag_system():
    """Initialize Enhanced CRAG agent with REFRAG + Self-Check (cached)"""
    from src.agents.enhanced_crag_agent import EnhancedCRAGAgent
    from chromadb.utils import embedding_functions
    
    chroma_client = chromadb.PersistentClient(path="./db/chroma")
    
    # Get collection with embedding function
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    collection = chroma_client.get_collection(
        name="player_stats",
        embedding_function=embedding_fn
    )
    
    # Initialize with REFRAG and Self-Check enabled
    agent = EnhancedCRAGAgent(
        vector_db=collection,
        groq_api_key=None,  # Will load from .env
        tavily_api_key=None,
        enable_refrag=True,
        enable_selfcheck=True,
        refrag_model_path="ollama:qwen2.5:1.5b"  # Use local Ollama
    )
    
    return agent, collection


def _normalize_player_name(name: str) -> str:
    """Normalize player names for matching: remove diacritics, punctuation, collapse spaces, lowercase."""
    if not name:
        return ""
    # Normalize unicode and remove combining marks
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(ch for ch in name if not unicodedata.combining(ch))
    # Remove punctuation except hyphen and spaces, collapse whitespace
    name = re.sub(r"[^A-Za-z0-9\s\-]", "", name)
    name = re.sub(r"\s+", " ", name).strip().lower()
    return name


def _name_similarity(a: str, b: str) -> float:
    """Return similarity ratio between two normalized names."""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

# UI
st.title("Football Scout CRAG Assistant")
st.markdown("Ask questions about player performance, tactics, and transfers")

# Sidebar
with st.sidebar:
    st.header("System Info")
    
    try:
        agent, collection = initialize_crag_system()
        st.success("Enhanced CRAG System Ready")
        
        # System stats
        doc_count = collection.count()
        st.metric("Documents in DB", f"{doc_count:,}")
        
        # Show enabled features
        st.info(" **REFRAG Reasoning**: Enabled\n\n **Self-Check**: Enabled\n\n **Local Ollama**: qwen2.5:1.5b")
    except Exception as e:
        st.error(f" System Error: {str(e)}")
        agent = None
        collection = None
    
    st.divider()
    
    # Example queries
    st.header(" Example Queries")
    examples = [
        "How good is Mo. Salah this season",
        "Find me top young strikers under 23",
        "What are the latest tactics for high pressing?",
        "Best defenders in Premier League",
        "Who are affordable midfielders under 30M?"
    ]
    
    for example in examples:
        if st.button(example, key=example):
            st.session_state.current_query = example
    
    st.divider()
    
    # Query history
    st.header(" Recent Queries")
    if st.session_state.query_history:
        for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
            st.text(f"{i+1}. {item['query'][:40]}...")
    else:
        st.text("No queries yet")

    st.divider()

    # Advanced options
    st.header(" Query Options")
    use_refrag = st.checkbox("Use REFRAG Reasoning", value=True, help="Enable multi-hop reasoning for complex queries")
    use_selfcheck = st.checkbox("Use Self-Check", value=True, help="Verify answer for hallucinations")
    
    st.divider()

# Season selector
st.header(" Season Selection")
seasons = ["2025-2026", "2024-2025", "2023-2024", "2022-2023", "2021-2022"]
selected_season = st.radio(
    "Select Season:",
    seasons,
    horizontal=True,
    index=0,
    help="Choose which season to analyze (data scraped up to Dec 30, 2025)"
)

st.divider()

# Main query interface
col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Ask your question:",
        value=st.session_state.get('current_query', ''),
        placeholder=f"e.g., How good is Mo. Salah in {selected_season}?",
        key='query_input'
    )

with col2:
    st.write("")  # Spacing
    search_button = st.button(" Search", type="primary", use_container_width=True)

# Process query
if (search_button or st.session_state.get('current_query')) and query:
    if not agent:
        st.error("CRAG system not initialized. Check API keys in .env file.")
    else:
        # Add season to query if not already specified
        if not any(season in query for season in seasons):
            enhanced_query = f"{query} {selected_season}"
        else:
            enhanced_query = query
            
        with st.spinner("Thinking..."):
            start_time = datetime.now()
            
            try:
                # Classify query intent
                intent_classifier = IntentClassifier()
                intent, intent_confidence, intent_metadata = intent_classifier.classify(query)
                
                # Get query options from sidebar
                force_reasoning = st.session_state.get('use_refrag', True)
                skip_verification = not st.session_state.get('use_selfcheck', True)
                
                # Query agent with options and intent
                result = agent.query(
                    enhanced_query,
                    intent=intent,
                    intent_metadata=intent_metadata,
                    force_reasoning=force_reasoning if force_reasoning else None,
                    skip_verification=skip_verification
                )
                elapsed = (datetime.now() - start_time).total_seconds()
                
                # Add to history
                st.session_state.query_history.append({
                    'query': query,
                    'timestamp': datetime.now().strftime("%H:%M:%S"),
                    'result': result
                })
                
                # Clear current_query
                if 'current_query' in st.session_state:
                    del st.session_state.current_query
                
                # Display results
                st.divider()
                
                # Metrics row
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    grade_emoji = {
                        'context_sufficient': '✓',
                        'context_outdated': 'Outd.',
                        'context_missing_facts': 'X'
                    }.get(result['grade'], '?')
                    st.metric("Grade", f"{grade_emoji} {result['grade'].replace('context_', '')}")
                
                with col2:
                    st.metric("Confidence", f"{result['confidence']*100:.0f}%")
                
                with col3:
                    # Improved data source detection - use result's data_source or detect from sources
                    data_source = result.get('data_source', '')
                    if not data_source or data_source == 'Unknown':
                        # Fallback: detect from sources
                        if result.get('sources'):
                            has_db = any(s.get('type') in ['stats', 'player'] for s in result['sources'])
                            has_web = any(s.get('type') == 'web' for s in result['sources'])
                            has_blog = any(s.get('type') == 'blog' for s in result['sources'])
                            
                            if has_db and has_web:
                                data_source = "DB + Web"
                            elif has_db or has_blog:
                                data_source = "Database"
                            elif has_web:
                                data_source = "Web"
                            else:
                                data_source = "Database"
                        else:
                            data_source = "Unknown"
                    
                    # Clean up emoji markers if present
                    data_source = data_source.replace('🌐 ', '').replace('💾 ', '')
                    web_icon = "🌐" if 'Web' in data_source else "💾"
                    st.metric("Data Source", f"{web_icon} {data_source}")
                
                with col4:
                    reasoning_used = "🧠" if result.get('reasoning_trace') else "⚡"
                    st.metric("Mode", f"{reasoning_used} {'REFRAG' if result.get('reasoning_trace') else 'CRAG'}")
                
                with col5:
                    st.metric("Time", f"{elapsed:.2f}s")
                
                st.divider()
                
                # ===== REASONING TRACE / LLM THINKING PROCESS =====
                reasoning_trace = result.get('reasoning_trace')
                if reasoning_trace and reasoning_trace != '':
                    # Show REFRAG reasoning if available
                    with st.expander(" Reasoning Trace (REFRAG Multi-Step Process)", expanded=False):
                        st.markdown("**How the system decomposed and analyzed your query:**")
                        if isinstance(reasoning_trace, list):
                            for i, step in enumerate(reasoning_trace, 1):
                                if isinstance(step, dict):
                                    st.markdown(f"** Step {i}:** {step.get('sub_question', 'Unknown')}")
                                    if step.get('answer'):
                                        st.info(f" **Analysis:** {step['answer'][:400]}..." if len(step.get('answer', '')) > 400 else step.get('answer', ''))
                                else:
                                    st.markdown(f"** Step {i}:** {str(step)}")
                        else:
                            st.success(f"✓ {reasoning_trace}")
                else:
                    # Show simple CRAG workflow (LLM thinking process)
                    with st.expander(" LLM Thinking Process (Standard CRAG)", expanded=False):
                        st.markdown("**How the system processed your query:**")
                        st.markdown(f"1.  **Document Retrieval:** Found {len(result.get('sources', []))} relevant documents")
                        st.markdown(f"2.  **Context Grading:** Assessed as '{result.get('grade', 'unknown').replace('context_', '')}'")
                        st.markdown(f"3.  **Intent Detection:** Classified as '{result.get('intent', 'unknown')}' ({result.get('intent_confidence', 0)*100:.0f}% confidence)")
                        st.markdown(f"4.  **Answer Generation:** Used {result.get('intent', 'unknown')} template")
                        if result.get('verification'):
                            st.markdown(f"5.  **Self-Check Verification:** {result.get('verification', {}).get('confidence', 0)*100:.0f}% confidence")
                        st.markdown("")
                        # Use the cleaned data_source from col3 calculation above
                        actual_data_source = data_source if 'data_source' in locals() else result.get('data_source', 'Database')
                        if actual_data_source == 'Unknown' and result.get('sources'):
                            actual_data_source = 'Database'  # If sources exist, it's from DB
                        st.markdown(f"** Data Source:** {actual_data_source}")
                        st.markdown(f"** Final Confidence:** {result.get('confidence', 0)*100:.0f}%")
                
                st.divider()
                
                # Top exports
                exp_col1, exp_col2, exp_col3 = st.columns([1,1,1])
                with exp_col1:
                    # Clean result for JSON serialization - recursively convert all enums
                    def clean_for_json(obj):
                        """Recursively convert enums and non-serializable objects to strings"""
                        if hasattr(obj, 'name'):  # Handle enums like QueryIntent
                            return obj.name
                        elif hasattr(obj, 'value') and not isinstance(obj, (str, int, float, bool)):
                            return str(obj.value)
                        elif isinstance(obj, dict):
                            # Handle dict keys that might be enums
                            return {str(k) if hasattr(k, 'name') else k: clean_for_json(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [clean_for_json(item) for item in obj]
                        elif isinstance(obj, (str, int, float, bool, type(None))):
                            return obj
                        else:
                            return str(obj)
                    
                    result_clean = clean_for_json(result)
                    json_str = json.dumps(result_clean, indent=2, default=str)
                    st.download_button("Export JSON", data=json_str, file_name=f"scout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")
                with exp_col2:
                    # CSV export: try to export main player stat dict if available
                    csv_data = None
                    if isinstance(result.get('sources'), list) and len(result['sources'])>0 and isinstance(result['sources'][0], dict):
                        try:
                            import io
                            first_meta = result['sources'][0]
                            df = pd.DataFrame([first_meta])
                            csv_buffer = io.StringIO()
                            df.to_csv(csv_buffer, index=False)
                            csv_data = csv_buffer.getvalue()
                        except Exception:
                            csv_data = None
                    if csv_data:
                        st.download_button(" Export CSV (primary doc)", data=csv_data, file_name=f"scout_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
                    else:
                        st.button("Export CSV", disabled=True)
                with exp_col3:
                    # Generate PDF bytes and expose a direct download button; provide clear install hint if missing
                    try:
                        from reportlab.lib.pagesizes import A4
                        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
                        from reportlab.lib.units import inch
                        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
                        from reportlab.lib.enums import TA_LEFT
                        import io
                        import re

                        # Build PDF in-memory
                        buffer = io.BytesIO()
                        doc = SimpleDocTemplate(buffer, pagesize=A4, topMargin=0.5*inch, bottomMargin=0.5*inch)
                        styles = getSampleStyleSheet()
                        
                        # Create custom styles for headers
                        title_style = ParagraphStyle(
                            'CustomTitle',
                            parent=styles['Heading1'],
                            fontSize=18,
                            textColor='#000080',
                            spaceAfter=12,
                            alignment=TA_LEFT
                        )
                        heading1_style = ParagraphStyle(
                            'CustomHeading1',
                            parent=styles['Heading2'],
                            fontSize=14,
                            textColor='#000080',
                            spaceAfter=10,
                            spaceBefore=10,
                            alignment=TA_LEFT
                        )
                        heading2_style = ParagraphStyle(
                            'CustomHeading2',
                            parent=styles['Heading3'],
                            fontSize=12,
                            textColor='#333333',
                            spaceAfter=8,
                            alignment=TA_LEFT
                        )
                        
                        story = []
                        story.append(Paragraph("Scout Report", title_style))
                        story.append(Spacer(1, 0.15*inch))
                        
                        report_text = result.get('answer','') or ''
                        
                        # Parse markdown-style report
                        lines = report_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if not line:
                                story.append(Spacer(1, 0.05*inch))
                                continue
                            
                            # Main headers (## Title)
                            if line.startswith('## '):
                                clean_title = line.replace('## ', '').strip()
                                story.append(Paragraph(clean_title, heading1_style))
                            # Sub-headers (### Title)
                            elif line.startswith('### '):
                                clean_subtitle = line.replace('### ', '').strip()
                                story.append(Paragraph(clean_subtitle, heading2_style))
                            # Regular text
                            elif line:
                                # Clean markdown formatting
                                clean_line = line.replace('**', '').replace('*', '')
                                story.append(Paragraph(clean_line, styles['BodyText']))
                                story.append(Spacer(1, 0.05*inch))
                        
                        doc.build(story)
                        buffer.seek(0)
                        pdf_bytes = buffer.getvalue()

                        st.download_button(
                            label="Export PDF",
                            data=pdf_bytes,
                            file_name=f"scout_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime='application/pdf'
                        )
                    except ImportError:
                        st.warning('PDF export requires the reportlab package. Install with: pip install reportlab')
                    except Exception as e:
                        logger.warning(f"PDF export failed: {e}")
                        st.warning('Could not generate PDF. See logs for details.')
                
                # REFRAG reasoning trace already shown above - removed duplicate
                
                st.divider()
                
                # ========== SELF-CHECK VERIFICATION (moved before report) ==========
                if result.get('verification'):
                    verification = result['verification']
                    with st.expander("Self-Check Verification", expanded=False):
                        
                        # Show detailed metrics
                        st.markdown("### Report Confidence")
                        conf = result.get('confidence', 0.0)
                        sources = result.get('sources', []) or []

                        grounding = conf * 100
                        has_sources = len(sources) > 0
                        has_reasoning = True if result.get('reasoning_trace') else False
                        completeness = 100 if has_sources and has_reasoning else 75 if has_sources else 50
                        consistency = 90 if verification.get('passed', True) else 60
                        reliability = 95 if (not result.get('used_web_search', False) and has_sources) else 80 if has_sources else 60

                        c1, c2, c3, c4 = st.columns(4)
                        with c1:
                            st.metric("Grounding", f"{grounding:.0f}%")
                        with c2:
                            st.metric("Completeness", f"{completeness:.0f}%")
                        with c3:
                            st.metric("Consistency", f"{consistency:.0f}%")
                        with c4:
                            st.metric("Reliability", f"{reliability:.0f}%")

                        overall = (grounding + completeness + consistency + reliability) / 4
                        st.markdown(f"**Overall Confidence:** {overall:.0f}%")
                        
                        st.markdown("---")
                        
                        # Show verification status
                        if verification.get('passed'):
                            st.success(f"Verification Passed (Confidence: {verification.get('confidence', 0)*100:.0f}%)")
                        else:
                            st.warning(f"Issues Found (Confidence: {verification.get('confidence', 0)*100:.0f}%)")
                            if verification.get('issues'):
                                for issue in verification['issues']:
                                    st.markdown(f"- {issue}")
                
                st.divider()
                
                # ========== INTENT-BASED RENDERING ==========
                # Use new adaptive renderer based on query intent
                render_result(result, query)
                # ========== END INTENT-BASED RENDERING ==========
                # ===== Tactical View Only (reusing CRAG context) =====
                try:
                    st.subheader("Tactical View")
                    st.markdown("Executive scout view + tactical evidence (no re-retrieval)")

                    # Reuse the tactical generation logic from the dual-view implementation
                    tactical_prompt = '''You are a professional football scout. Re-write the following CRAG-generated scout report into a Tactical Scout Report following this exact hierarchy:

A. Executive Scout View (1-2 short paragraphs, NO statistics)\n
B. Tactical Role & Game Model Fit (describe role, where player thrives/struggles using football language)\n
C. Statistical Evidence (concise bullets: each stat used only to support a tactical claim)\n
D. Risk & Translation Analysis (sample-size, league translation, tactical dependency)\n
E. Final Scouting Recommendation (one-paragraph verdict, no stats)\n
Rules:\n+- Use ONLY the content provided in {context} as evidence; do NOT invent new stats or facts.\n+- If a fact is missing from {context}, say "Data not available" for that specific item.\n+- Keep Executive Scout View free of numbers; numbers belong in section C only.\n\n+Context:\n+{context}\n\n+Question: {question}'''

                    def _format_tactical_from_crag(text: str, question: str) -> str:
                        import re
                        def sentences(s):
                            return re.split(r'(?<=[.!?])\s+', s.strip())

                        lines = [l.strip() for l in text.splitlines() if l.strip()]
                        full = '\n'.join(lines)
                        sents = sentences(full)

                        def find_sentence(keywords):
                            for sent in sents:
                                for kw in keywords:
                                    if kw in sent.lower():
                                        return sent.strip()
                            return None

                        out_parts = []
                        role_kw = ['interior', 'playmaker', 'box-to-box', 'inverted', 'deep-lying', 'wide', 'winger', 'inside', 'half-space', 'half-spaces']
                        tendency_kw = ['progressive', 'carry', 'dribble', 'press', 'pressures', 'pass', 'distribution', 'vertical', 'switch']

                        role_sent = find_sentence(role_kw)
                        tendency_sent = find_sentence(tendency_kw)

                        exec_lines = []
                        if role_sent:
                            exec_lines.append(role_sent)
                        if tendency_sent and tendency_sent not in exec_lines:
                            exec_lines.append(tendency_sent)

                        cleaned_exec = []
                        for l in exec_lines:
                            cleaned = re.sub(r'\b\d+[\d\.]*\b', '', l)
                            cleaned = re.sub(r'\bper 90\b', '', cleaned, flags=re.IGNORECASE)
                            cleaned_exec.append(cleaned.strip())

                        out_parts.append('A. Executive Scout View\n')
                        if cleaned_exec:
                            out_parts.append(' '.join(cleaned_exec))
                        else:
                            out_parts.append(' '.join(sents[:2]) if len(sents) >= 2 else (sents[0] if sents else 'Data not available'))

                        role_context = []
                        for kw in ['half-space', 'compact', 'mid-block', 'tempo', 'width', 'penetrat', 'possession', 'transition', 'counter']:
                            for sent in sents:
                                if kw in sent.lower() and sent not in role_context:
                                    role_context.append(sent.strip())

                        out_parts.append('\nB. Tactical Role & Game Model Fit\n')
                        if role_context:
                            out_parts.append(' '.join(role_context))
                        else:
                            out_parts.append('Data not available')

                        stat_lines = []
                        stat_pattern = re.compile(r"([A-Za-z %_\/\-]+)[:=]?\s*([0-9]+(?:\.[0-9]+)?(?:\s*(?:per 90|/90|%))?)", re.IGNORECASE)
                        for line in lines:
                            if any(tok in line.lower() for tok in ['per 90', 'prg', 'progress', 'pass', 'tackle', 'drib', 'xg', 'kp', 'key pass', 'press', 'pressures']):
                                m = stat_pattern.search(line)
                                if m:
                                    stat_lines.append(f"{m.group(1).strip()} = {m.group(2).strip()}")
                                else:
                                    if re.search(r'\d', line):
                                        stat_lines.append(line)

                        # Position-specific Statistical Evidence title for fallback formatter
                        detected_pos = 'MF'  # Default fallback
                        if 'goalkeeper' in full.lower() or 'gk' in full.lower():
                            evidence_title = 'C. Shot-Stopping & Distribution Evidence'
                        elif 'defender' in full.lower() or 'df' in full.lower() or 'defensive' in full.lower():
                            evidence_title = 'C. Defensive Contribution Evidence'
                        elif 'forward' in full.lower() or 'fw' in full.lower() or 'striker' in full.lower():
                            evidence_title = 'C. Attacking Prowess Evidence'
                        else:
                            evidence_title = 'C. Creation & Progression Evidence'
                        
                        out_parts.append(f'\n{evidence_title}\n')
                        if stat_lines:
                            for stl in stat_lines:
                                out_parts.append(f"- {stl}")
                        else:
                            out_parts.append('Data not available')

                        risks = []
                        if re.search(r'\b(minutes|mp|played)\b', full, flags=re.IGNORECASE):
                            risks.append('Sample size: check minutes played for season-specific reliability')
                        if 'compact' in full.lower() or 'mid-block' in full.lower():
                            risks.append('May underperform vs compact mid-blocks')
                        if not risks:
                            risks.append('Data not available')

                        out_parts.append('\nD. Risk & Translation Analysis\n')
                        out_parts.append('\n'.join(risks))

                        rec = None
                        mrec = re.search(r'\bDecision:\s*(RECOMMEND|RECOMMEND|Recommend|Recommendation|Recommend)\b', full, flags=re.IGNORECASE)
                        if mrec:
                            for line in lines:
                                if 'decision' in line.lower() or 'recommend' in line.lower():
                                    rec = line.strip()
                                    break
                        else:
                            for line in lines:
                                if 'recommend' in line.lower() or 'decision' in line.lower():
                                    rec = line.strip()
                                    break

                        out_parts.append('\nE. Final Scouting Recommendation\n')
                        out_parts.append(rec if rec else 'Data not available')

                        tactical_view = '\n\n'.join(out_parts)

                    try:
                        # Prefer structured metadata from retrieved_docs to build tactical view
                        merged_meta = {}
                        if result.get('retrieved_docs'):
                            for d in result.get('retrieved_docs'):
                                meta = d.get('metadata') if isinstance(d, dict) else d
                                if isinstance(meta, dict):
                                    for k, v in meta.items():
                                        if k in ['chunk_id', 'parent_id', 'source', 'chunk_index', 'content']:
                                            continue
                                        if k not in merged_meta or not merged_meta.get(k):
                                            merged_meta[k] = v

                        if merged_meta:
                            def tactical_from_meta(meta: dict) -> str:
                                parts = []
                                pos = meta.get('pos') or meta.get('position') or ''
                                role = 'Data not available'
                                if pos:
                                    primary = pos.split(',')[0]
                                    if 'MF' in primary:
                                        role = 'Interior midfield playmaker (midfielder)'
                                    elif 'FW' in primary:
                                        role = 'Attacking forward / winger'
                                    elif 'DF' in primary:
                                        role = 'Defensive/outside defender'
                                    elif 'GK' in primary:
                                        role = 'Goalkeeper'

                                tendency = []
                                if meta.get('PrgP_pass') or meta.get('Progression_PrgP_std') or meta.get('PrgP_pass'):
                                    tendency.append('Prioritises early progressive passing over carries')
                                if meta.get('Progression_PrgC_std') or meta.get('PrgC_poss'):
                                    tendency.append('Contributes with progressive carries when appropriate')
                                if meta.get('Succ_drib') or meta.get('Att_drib'):
                                    tendency.append('Attempts dribbles (successful rate available if Succ_drib present)')

                                parts.append('A. Executive Scout View\n')
                                parts.append(f"{meta.get('player','Player')} appears as {role}. {' '.join(tendency) if tendency else ''}")

                                parts.append('\nB. Tactical Role & Game Model Fit\n')
                                gm = []
                                if meta.get('PrgP_pass'):
                                    gm.append('Line-breaking via progressive passing')
                                if meta.get('Long_Cmp%_pass') or meta.get('Long_Cmp_pass') or meta.get('Long_Cmp%_pass'):
                                    gm.append('Long passing accuracy affects switching and width creation')
                                parts.append(' '.join(gm) if gm else 'Data not available')

                                # Position-specific Statistical Evidence title and filtering
                                detected_pos = meta.get('pos', '').split(',')[0].strip().upper() if meta.get('pos') else 'MF'
                                if 'GK' in detected_pos:
                                    evidence_title = 'C. Shot-Stopping & Distribution Evidence'
                                    evidence_keys = ['Performance_Saves_gk', 'Performance_Save%_gk', 'PSxG_GA', 'Total_Cmp%_pass', 'Performance_CS_gk', 'PrgP_pass']
                                elif 'DF' in detected_pos:
                                    evidence_title = 'C. Defensive Contribution Evidence'
                                    evidence_keys = ['Tackles_Tkl_def', 'Tackles_TklW_def', 'Int_def', 'Blocks_Blocks_def', 'Clr_def', 'Cmp%_pass', 'PrgP_pass']
                                elif 'FW' in detected_pos:
                                    evidence_title = 'C. Attacking Prowess Evidence'
                                    evidence_keys = ['Performance_Gls_std', 'Performance_Ast_std', 'Expected_xG_std', 'Expected_xAG_std', 'Standard_Sh_shoot', 'Standard_SoT%_shoot', 'KP_pass']
                                else:  # MF default
                                    evidence_title = 'C. Creation & Progression Evidence'
                                    evidence_keys = ['PrgP_pass','Progression_PrgP_std','Progression_PrgC_std','KP_pass','1/3_pass','Cmp%_pass','Performance_Gls_std','Performance_Ast_std']
                                
                                parts.append(f'\n{evidence_title}\n')
                                stats = []
                                for k in evidence_keys:
                                    if k in meta:
                                        stats.append(f"{k} = {meta[k]}")
                                if stats:
                                    for s in stats:
                                        parts.append(f"- {s}")
                                else:
                                    parts.append('Data not available')

                                parts.append('\nD. Risk & Translation Analysis\n')
                                risks = []
                                # Extract from original answer if available
                                original_answer = result.get('answer', '')
                                
                                # Try to find Development Areas or Risk Analysis sections
                                import re
                                
                                # Look for DEVELOPMENT AREAS section (case-insensitive)
                                dev_match = re.search(r'development\s+areas.*?\n((?:[-•]\s*.+\n?)+)', original_answer, re.IGNORECASE | re.DOTALL)
                                if dev_match:
                                    dev_text = dev_match.group(1)
                                    # Extract bullet points from development areas
                                    bullets = re.findall(r'[-•]\s*(.+?)(?:\n|$)', dev_text)
                                    risks.extend([b.strip() for b in bullets if b.strip()])
                                else:
                                    # Fallback: look for paragraphs between DEVELOPMENT and next section
                                    dev_match2 = re.search(r'development.*?areas.*?\n\n(.*?)\n\n', original_answer, re.IGNORECASE | re.DOTALL)
                                    if dev_match2:
                                        dev_text = dev_match2.group(1)
                                        for line in dev_text.split('\n'):
                                            line = line.strip()
                                            if line and not line.startswith('#'):
                                                risks.append(line)
                                
                                # Also look for any section with "Risk" or explicit risk statements
                                if len(risks) < 2:
                                    for line in original_answer.split('\n'):
                                        line_lower = line.lower()
                                        if any(kw in line_lower for kw in ['risk', 'limited', 'struggle', 'may struggle', 'concern', 'vulnerability', 'vulnerable', 'weakness', 'drawback']):
                                            cleaned = line.strip()
                                            if len(cleaned) > 20 and not cleaned.startswith('#') and not cleaned.startswith('**'):
                                                # Remove markdown formatting
                                                cleaned = re.sub(r'\*\*|\*|##|###', '', cleaned)
                                                if cleaned not in risks and cleaned not in [r for r in risks]:
                                                    risks.append(cleaned)
                                                if len(risks) >= 2:
                                                    break
                                
                                # Fallback: if still no risks extracted, use data-driven approach
                                if not risks:
                                    mins = None
                                    try:
                                        mins = float(meta.get('Playing Time_Min_std') or 0)
                                    except: pass
                                    
                                    if mins and mins < 900:
                                        risks.append('Limited minutes this season; sample-size risk')
                                    
                                    # Check for pressing/involvement issues (forwards)
                                    pos = str(meta.get('pos') or '').upper()
                                    if 'FW' in pos:
                                        pressures = None
                                        try:
                                            pressures = float(meta.get('Pressures_Press_press') or meta.get('pressures') or 0)
                                        except: pass
                                        if pressures and pressures < 8:
                                            risks.append('Limited pressing output relative to high-intensity systems')
                                    
                                    # Long pass accuracy
                                    try:
                                        long_acc = float(str(meta.get('Long_Cmp%_pass') or '').replace('%',''))
                                        if long_acc < 60:
                                            risks.append('Lower long-pass accuracy; may struggle switching play')
                                    except: pass
                                
                                parts.append('\n'.join([f'- {r}' if not r.startswith('-') else r for r in risks]) if risks else 'Data not available')


                                parts.append('\nE. Final Scouting Recommendation\n')
                                # Extract recommendation from original answer
                                rec_text = 'RECOMMEND'
                                if 'RECOMMEND' in original_answer or 'recommend' in original_answer.lower():
                                    # Try to find the recommendation paragraph
                                    for line in original_answer.split('\n'):
                                        if 'rationale' in line.lower() or 'decision' in line.lower():
                                            next_idx = original_answer.split('\n').index(line) + 1
                                            lines = original_answer.split('\n')
                                            if next_idx < len(lines):
                                                rec_text = lines[next_idx].strip()
                                                break
                                    # Otherwise extract recommendation reason
                                    if rec_text == 'RECOMMEND':
                                        import re
                                        rationale_match = re.search(r'Rationale:?\s*(.+?)(?:\n\n|$)', original_answer, re.IGNORECASE | re.DOTALL)
                                        if rationale_match:
                                            rec_text = f"RECOMMEND: {rationale_match.group(1)[:200].strip()}"
                                parts.append(rec_text)

                                return '\n\n'.join(parts)

                            tactical_view = tactical_from_meta(merged_meta)
                        else:
                            # fallback to text-based formatter
                            tactical_view = _format_tactical_from_crag(result.get('answer', ''), query)

                        # Post-process tactical_view to ensure all sections present
                        def _normalize_tactical_view(text: str, meta: dict, result_obj: dict) -> str:
                            # Ensure sections A-E exist and populate sensible defaults
                            txt = text or ''

                            def _safe_float(v):
                                try:
                                    if v is None:
                                        return None
                                    if isinstance(v, str) and v.endswith('%'):
                                        return float(v.replace('%',''))
                                    return float(v)
                                except Exception:
                                    return None

                            # Helper: extract quick signals from raw answer text
                            answer_text = (result_obj.get('answer') or '')
                            def contains_any(words):
                                t = answer_text.lower()
                                return any(w in t for w in words)

                            # Ensure Risk section (D.) exists
                            if '\nD. Risk & Translation Analysis\n' not in txt and '\nD. Risk & Translation Analysis' not in txt:
                                risks = []
                                mins = _safe_float(meta.get('Playing Time_Min_std') or meta.get('minutes') or meta.get('minutes_played'))
                                if mins is None:
                                    mins = _safe_float(meta.get('minutes_total') or meta.get('mins'))
                                if mins and mins < 900:
                                    risks.append('Limited minutes this season; sample-size risk')

                                # Defender-specific signals
                                pos = (meta.get('position') or meta.get('role') or '').upper()
                                is_defender = pos.startswith('D') or 'DEF' in pos

                                dribbled = _safe_float(meta.get('times_dribbled_past') or meta.get('dribbled_past') or meta.get('dribbled_past_90'))
                                if dribbled and dribbled >= 8:
                                    risks.append('Vulnerable in 1v1s (high times dribbled past)')

                                # long pass accuracy risk
                                lp = meta.get('Long_Cmp%_pass') or meta.get('Long_Cmp_pass') or meta.get('long_pass_accuracy')
                                lpv = _safe_float(lp)
                                if lpv is not None and lpv < 60:
                                    risks.append('Lower long-pass accuracy; may struggle switching play')

                                # If defender but little defensive evidence, warn
                                if is_defender:
                                    tackles = _safe_float(meta.get('tackles') or meta.get('Tackles') or meta.get('tackles_total'))
                                    interceptions = _safe_float(meta.get('interceptions') or meta.get('Interceptions'))
                                    aerial_pct = _safe_float(meta.get('aerials_won_pct') or meta.get('aerial_duel_win_pct') or meta.get('Aerial Duels Won %'))
                                    defensive_evidence = any(x is not None for x in [tackles, interceptions, aerial_pct, dribbled])
                                    if not defensive_evidence:
                                        risks.append('Defensive-stat evidence sparse for this player/season')

                                if not risks:
                                    risks = ['Data not available']
                                txt = txt + '\n\nD. Risk & Translation Analysis\n' + '\n'.join(risks)

                            # Ensure Recommendation section (E.) exists
                            if '\nE. Final Scouting Recommendation\n' not in txt and '\nE. Final Scouting Recommendation' not in txt:
                                rec = None
                                conf = 0.0
                                try:
                                    conf = float(result_obj.get('confidence', 0) or 0)
                                except Exception:
                                    conf = 0.0
                                grade = result_obj.get('grade','') or ''

                                # Evaluate positive defensive signals
                                pos = (meta.get('position') or meta.get('role') or '').upper()
                                is_defender = pos.startswith('D') or 'DEF' in pos
                                tackles = _safe_float(meta.get('tackles') or meta.get('Tackles') or meta.get('tackles_total'))
                                interceptions = _safe_float(meta.get('interceptions') or meta.get('Interceptions'))
                                aerial_pct = _safe_float(meta.get('aerials_won_pct') or meta.get('aerial_duel_win_pct') or meta.get('Aerial Duels Won %'))
                                mins = _safe_float(meta.get('Playing Time_Min_std') or meta.get('minutes') or meta.get('mins'))

                                positive_signals = 0
                                if tackles and tackles >= 30:
                                    positive_signals += 1
                                if interceptions and interceptions >= 15:
                                    positive_signals += 1
                                if aerial_pct and aerial_pct >= 60:
                                    positive_signals += 1

                                # Use heuristics for defenders
                                if is_defender:
                                    if positive_signals >= 2 and (mins and mins >= 900):
                                        rec = 'Recommend'
                                    elif positive_signals >= 1 and (mins and mins >= 600):
                                        rec = 'Consider / Monitor (trial recommended)'
                                    else:
                                        # Try to infer from answer text for softer judgement
                                        if contains_any(['elite', 'dominant', 'excellent', 'solid']):
                                            rec = 'Recommend'
                                        elif contains_any(['vulnerable', 'struggles', 'limited']):
                                            rec = 'Do not recommend'
                                        else:
                                            rec = 'Monitor - data insufficient for firm recommendation'
                                else:
                                    # Non-defenders: prefer previous logic
                                    if grade == 'context_sufficient' and conf >= 0.5:
                                        rec = 'Recommend'
                                    elif grade == 'context_missing_facts' and conf < 0.5:
                                        rec = 'Data insufficient for recommendation'
                                    else:
                                        mv = meta.get('market_value') or meta.get('market_val')
                                        try:
                                            if mv and float(mv) < 200:
                                                rec = 'Recommend'
                                            else:
                                                rec = 'Monitor'
                                        except Exception:
                                            rec = 'Monitor'

                                txt = txt + '\n\nE. Final Scouting Recommendation\n'
                                
                                # Try to extract full recommendation with rationale from original answer
                                answer_text = result_obj.get('answer', '')
                                rationale = ''
                                if 'Rationale:' in answer_text:
                                    try:
                                        import re
                                        match = re.search(r'Rationale:?\s*(.+?)(?:\n\n|\nDATA INTEGRITY|$)', answer_text, re.DOTALL)
                                        if match:
                                            rationale = match.group(1).strip()[:300]
                                    except:
                                        pass
                                
                                if rationale:
                                    txt = txt + (rec or 'Recommend') + '\n' + rationale
                                else:
                                    txt = txt + (rec or 'Data not available')

                            # Replace generic 'Data not available' lines with inferred merged_meta values where possible
                            try:
                                import re

                                # Broader mapping and fuzzy lookup to match displayed labels to merged_meta keys
                                label_map = {
                                    'minutes': ['minutes','mins','Playing Time_Min_std','Playing Time_Min_gk','90s_def','90s'],
                                    'tackles': ['tackles','Tackles','Tackles_Tkl_def','Tackles_Tkl_att','Tkl'],
                                    'tackles won %': ['Tackles_Won_%','Tackles_Won%','Tackle Win %'],
                                    'interceptions': ['interceptions','Int_def','Int','Interceptions'],
                                    'clearances': ['clearances','Clr_def','Clearances','Clearances_def'],
                                    'long pass accuracy': ['long_pass_accuracy','Long_Cmp%_pass','Long_Cmp_pass'],
                                    'pass completion': ['pass_completion','Pass Completion','Cmp%_pass','Cmp_pass'],
                                    'passes into final third': ['passes_into_final_third','Passes_into_Final_Third','Passes into Final Third'],
                                    'times dribbled past': ['dribbled_past','times_dribbled_past','dribbled_past_90'],
                                    'aerial duels won %': ['aerials_won_pct','aerial_duel_win_pct','Aerial Duels Won %'],
                                    'saves': ['saves','Performance_Saves_gk','Saves_gk'],
                                    'save %': ['save_pct','Performance_Save%_gk','Performance_Save%','Save%','Save_pct'],
                                    'psxg-ga': ['psxg_ga','PSxG-GA','PSxG_GA'],
                                    'psxg': ['post_shot_xg','PSxG','Post_Shot_xG'],
                                    'clean sheets': ['clean_sheets','Performance_CS_gk','Performance_CS','CS'],
                                    'errors leading to shot': ['errors_leading_shot','Err_def','Errors_leading_to_shot']
                                }

                                # normalize helper
                                def _norm(s):
                                    if s is None:
                                        return ''
                                    return re.sub(r'[^a-z0-9]', '', str(s).lower())

                                # fuzzy lookup over merged_meta keys
                                def find_meta_value(label):
                                    norm_label = _norm(label)
                                    # first try explicit label_map candidates
                                    for k, candidates in label_map.items():
                                        if _norm(k) in norm_label or norm_label in _norm(k):
                                            for cand in candidates:
                                                if merged_meta.get(cand) not in (None, '', []):
                                                    return merged_meta.get(cand)

                                    # direct key match (normalized)
                                    for meta_k, v in merged_meta.items():
                                        if v in (None, '', []):
                                            continue
                                        if _norm(meta_k) == norm_label:
                                            return v

                                    # fuzzy containment (label tokens in meta key or vice versa)
                                    for meta_k, v in merged_meta.items():
                                        if v in (None, '', []):
                                            continue
                                        if norm_label in _norm(meta_k) or _norm(meta_k) in norm_label:
                                            return v

                                    # Attempt to extract numeric from result answer text for label keywords
                                    answer_text = str(result_obj.get('answer') or '')
                                    # look for patterns like 'Label: 12' or 'Label 12'
                                    try:
                                        pats = [rf"{re.escape(label)}[:\s]*([0-9]+\.?[0-9]*)", rf"{re.escape(label)}\s+([0-9]+\.?[0-9]*)"]
                                        for p in pats:
                                            m = re.search(p, answer_text, flags=re.IGNORECASE)
                                            if m:
                                                g = m.group(1)
                                                try:
                                                    if '.' in g:
                                                        return float(g)
                                                    return int(g)
                                                except Exception:
                                                    return g
                                    except Exception:
                                        pass

                                    return None

                                def _format_val(v):
                                    try:
                                        if isinstance(v, float):
                                            if float(v).is_integer():
                                                return str(int(v))
                                            return str(round(float(v),2))
                                        return str(v)
                                    except Exception:
                                        return str(v)

                                # Replace occurrences like 'Label: Data not available' or 'Data not available' after label
                                def replace_data_not_available(text):
                                    def repl(m):
                                        label = m.group(1).strip()
                                        val = find_meta_value(label)
                                        if val is not None:
                                            return f"{label}: {_format_val(val)} (inferred)"
                                        return m.group(0)

                                    # pattern matches 'Label: Data not available' lines
                                    pattern = re.compile(r"^([A-Za-z0-9 %\-/\\+]+):\s*Data not available$", flags=re.MULTILINE)
                                    text = pattern.sub(repl, text)

                                    # also handle lines where label and 'Data not available' separated by newline
                                    pattern2 = re.compile(r"^([A-Za-z0-9 %\-/\\+]+)\s*\nData not available$", flags=re.MULTILINE)
                                    def repl2(m):
                                        label = m.group(1).strip()
                                        val = find_meta_value(label)
                                        if val is not None:
                                            return f"{label}: {_format_val(val)} (inferred)"
                                        return m.group(0)
                                    text = pattern2.sub(repl2, text)

                                    return text

                                txt = replace_data_not_available(txt)

                                # If whole sections still say 'Data not available', try to synthesize from merged_meta
                                try:
                                    # Helper to build stat bullets from merged_meta
                                    def build_stat_bullets(meta):
                                        lines = []
                                        stat_keys = [
                                            ('minutes', 'Minutes'),
                                            ('PrgP_pass', 'Progressive Passes'),
                                            ('Progression_PrgP_std', 'Progression_PrgP_std'),
                                            ('Progression_PrgC_std', 'Progression_PrgC_std'),
                                            ('1/3_pass', '1/3_pass'),
                                            ('pass_completion', 'Pass Completion'),
                                            ('Long_Cmp%_pass', 'Long Pass Accuracy'),
                                            ('tackles', 'Tackles'),
                                            ('Tackles_Tkl_def', 'Tackles_Tkl_def'),
                                            ('interceptions', 'Interceptions'),
                                            ('Int_def', 'Int_def'),
                                            ('Blocks_Blocks_def', 'Blocks'),
                                            ('dribbled_past', 'Times Dribbled Past'),
                                            ('xg', 'xG'),
                                            ('xa', 'xA'),
                                            ('saves', 'Saves'),
                                            ('save_pct', 'Save %'),
                                            ('clean_sheets', 'Clean Sheets')
                                        ]
                                        for k, label in stat_keys:
                                            v = None
                                            # check multiple possible keys
                                            if meta.get(k) not in (None, '', []):
                                                v = meta.get(k)
                                            else:
                                                # try normalized lookup
                                                for mk in list(meta.keys()):
                                                    if mk.lower().replace('_','') == k.lower().replace('_',''):
                                                        v = meta.get(mk)
                                                        break
                                            if v not in (None, '', []):
                                                try:
                                                    if isinstance(v, float) and v.is_integer():
                                                        v = int(v)
                                                except Exception:
                                                    pass
                                                lines.append(f"- {label} = {v}")
                                        return lines

                                    # Replace 'C. Statistical Evidence' placeholder
                                    c_pattern = re.compile(r"C\. Statistical Evidence\s*\nData not available", flags=re.IGNORECASE)
                                    if c_pattern.search(txt):
                                        bullets = build_stat_bullets(merged_meta)
                                        if bullets:
                                            replacement = 'C. Statistical Evidence\n' + '\n'.join(bullets)
                                            txt = c_pattern.sub(replacement, txt)

                                    # Replace 'B. Tactical Role & Game Model Fit' placeholder with role/tendency
                                    b_pattern = re.compile(r"B\. Tactical Role & Game Model Fit\s*\nData not available", flags=re.IGNORECASE)
                                    if b_pattern.search(txt):
                                        role = merged_meta.get('position') or merged_meta.get('pos') or ''
                                        role_text = ''
                                        if role:
                                            p = role.split(',')[0]
                                            if 'MF' in p:
                                                role_text = 'Interior midfield playmaker (midfielder)'
                                            elif 'FW' in p:
                                                role_text = 'Attacking forward / winger'
                                            elif 'DF' in p:
                                                role_text = 'Defensive/outside defender'
                                            elif 'GK' in p:
                                                role_text = 'Goalkeeper'
                                        tendency = []
                                        if merged_meta.get('PrgP_pass') or merged_meta.get('Progression_PrgP_std'):
                                            tendency.append('Line-breaking via progressive passing')
                                        if merged_meta.get('Succ_drib') or merged_meta.get('Att_drib'):
                                            tendency.append('Attempts dribbles when appropriate')
                                        if role_text or tendency:
                                            repl = 'B. Tactical Role & Game Model Fit\n' + (role_text + (' ' + ' '.join(tendency) if tendency else ''))
                                            txt = b_pattern.sub(repl, txt)

                                    # Replace 'D. Risk & Translation Analysis\nData not available' with GK-specific risks when appropriate
                                    d_pattern = re.compile(r"D\. Risk & Translation Analysis\s*\nData not available", flags=re.IGNORECASE)
                                    if d_pattern.search(txt):
                                        risks = []
                                        # minutes based risk
                                        try:
                                            mins = float(merged_meta.get('minutes') or merged_meta.get('mins') or 0)
                                        except Exception:
                                            mins = 0
                                        if mins and mins < 900:
                                            risks.append('Limited minutes this season; sample-size risk')

                                        # Goalkeeper-specific signals
                                        pos_guess = (merged_meta.get('position') or merged_meta.get('pos') or '').upper()
                                        is_gk_guess = 'GK' in pos_guess or 'GOAL' in pos_guess or 'KEEP' in pos_guess
                                        if is_gk_guess:
                                            saves = merged_meta.get('saves')
                                            save_pct = None
                                            try:
                                                save_pct = float(merged_meta.get('save_pct')) if merged_meta.get('save_pct') not in (None, '', []) else None
                                            except Exception:
                                                save_pct = None

                                            if saves in (None, '', []) and save_pct in (None, ''):
                                                risks.append('Limited shot-stopping data for this season')
                                            else:
                                                try:
                                                    if float(saves) == 0:
                                                        risks.append('No saves recorded in available minutes; limited shot-stopping evidence')
                                                except Exception:
                                                    pass

                                            # PSxG / post-shot xG signals
                                            psxg_ga = merged_meta.get('psxg_ga') or merged_meta.get('PSxG-GA') or merged_meta.get('PSxG_GA')
                                            post_shot_xg = merged_meta.get('post_shot_xg') or merged_meta.get('PSxG')
                                            if psxg_ga not in (None, '', []):
                                                risks.append(f'Post-shot xG vs GA (PSxG-GA): {psxg_ga}')
                                            elif post_shot_xg not in (None, '', []):
                                                risks.append(f'Post-shot xG: {post_shot_xg}')

                                            # Clean sheets context
                                            cs = merged_meta.get('clean_sheets')
                                            matches = merged_meta.get('matches') or merged_meta.get('appearances')
                                            try:
                                                if not matches and mins and mins > 0:
                                                    matches = round(float(mins) / 90)
                                            except Exception:
                                                matches = None
                                            if cs not in (None, '', []) and matches not in (None, '', [], 0):
                                                try:
                                                    cs_rate = float(cs) / float(matches)
                                                    risks.append(f'Clean sheet rate: {cs}/{matches} ({round(cs_rate,2)})')
                                                except Exception:
                                                    risks.append(f'Clean sheets: {cs}')

                                            # Distribution signals
                                            launch = merged_meta.get('launch_pct') or merged_meta.get('Launch %')
                                            pass_comp = merged_meta.get('pass_completion') or merged_meta.get('Pass Completion')
                                            if launch not in (None, '', []):
                                                risks.append(f'Launch %: {launch}')
                                            if pass_comp not in (None, '', []):
                                                risks.append(f'Pass Completion: {pass_comp}%')

                                        # General fallback
                                        if not risks:
                                            risks = ['Data not available']

                                        txt = d_pattern.sub('D. Risk & Translation Analysis\n' + '\n'.join(risks), txt)

                                    # Replace lone 'E. Final Scouting Recommendation\nData not available'
                                    e_pattern = re.compile(r"E\. Final Scouting Recommendation\s*\nData not available", flags=re.IGNORECASE)
                                    if e_pattern.search(txt):
                                        # reuse earlier heuristics to compute a lightweight rec
                                        rec = None
                                        try:
                                            mins = float(merged_meta.get('minutes') or merged_meta.get('mins') or 0)
                                        except Exception:
                                            mins = 0
                                        posv = (merged_meta.get('position') or merged_meta.get('pos') or '').upper()
                                        is_def = posv.startswith('D') or 'DEF' in posv
                                        positive = 0
                                        try:
                                            if float(merged_meta.get('tackles') or 0) >= 30:
                                                positive += 1
                                        except Exception:
                                            pass
                                        try:
                                            if float(merged_meta.get('interceptions') or 0) >= 15:
                                                positive += 1
                                        except Exception:
                                            pass
                                        if is_def:
                                            if positive >= 2 and mins >= 900:
                                                rec = 'Recommend'
                                            elif positive >=1 and mins >= 600:
                                                rec = 'Consider / Monitor (trial recommended)'
                                            else:
                                                rec = 'Monitor - data insufficient for firm recommendation'
                                        else:
                                            mv = merged_meta.get('market_value') or merged_meta.get('market_val')
                                            try:
                                                if mv and float(mv) < 200:
                                                    rec = 'Recommend'
                                                else:
                                                    rec = 'Monitor'
                                            except Exception:
                                                rec = 'Monitor'
                                        txt = e_pattern.sub(f"E. Final Scouting Recommendation\n{rec}", txt)
                                except Exception:
                                    pass
                            except Exception:
                                pass

                            return txt

                        try:
                            tactical_view = _normalize_tactical_view(tactical_view, merged_meta if 'merged_meta' in locals() else {}, result)
                        except Exception:
                            pass

                        # Canonicalize DB-specific defensive keys into our expected canonical keys
                        try:
                            inferred = merged_meta.get('_inferred', []) if isinstance(merged_meta.get('_inferred', []), list) else []

                            def _to_float(x):
                                try:
                                    if x is None:
                                        return None
                                    if isinstance(x, str) and x.endswith('%'):
                                        return float(x.replace('%',''))
                                    return float(x)
                                except Exception:
                                    return None

                            # minutes from 90s_def
                            if not merged_meta.get('minutes'):
                                ninety = merged_meta.get('90s_def') or merged_meta.get('90s') or merged_meta.get('90s_total')
                                nv = _to_float(ninety)
                                if nv is not None:
                                    mins = int(round(nv * 90))
                                    merged_meta['minutes'] = mins
                                    merged_meta['mins'] = mins
                                    merged_meta['Playing Time_Min_std'] = mins
                                    inferred.append('minutes_from_90s')

                            # map tackles
                            if not merged_meta.get('tackles'):
                                for k in list(merged_meta.keys()):
                                    if k.lower().startswith('tackles') or 'tkl' in k.lower():
                                        val = merged_meta.get(k)
                                        try:
                                            merged_meta['tackles'] = int(float(val))
                                            inferred.append(f'tackles_from_{k}')
                                            break
                                        except Exception:
                                            continue

                            # map interceptions
                            if not merged_meta.get('interceptions'):
                                for k in list(merged_meta.keys()):
                                    if k.lower().startswith('int_') or k.lower().startswith('int') or 'int_def' in k.lower():
                                        val = merged_meta.get(k)
                                        try:
                                            merged_meta['interceptions'] = int(float(val))
                                            inferred.append(f'interceptions_from_{k}')
                                            break
                                        except Exception:
                                            continue

                            # map aerial percent
                            if not merged_meta.get('aerial_pct'):
                                for k in list(merged_meta.keys()):
                                    if 'aerial' in k.lower() or 'aerials' in k.lower():
                                        v = _to_float(merged_meta.get(k))
                                        if v is not None:
                                            merged_meta['aerial_pct'] = v
                                            inferred.append(f'aerial_pct_from_{k}')
                                            break

                            # normalize position
                            if not merged_meta.get('position') and merged_meta.get('pos'):
                                merged_meta['position'] = merged_meta.get('pos')

                            if inferred:
                                merged_meta['_inferred'] = list(set(inferred))
                            # Goalkeeper-specific canonical mappings
                            try:
                                # Detect goalkeeper by position markers
                                posv = (merged_meta.get('position') or merged_meta.get('pos') or '').upper()
                                is_gk = 'GK' in posv or 'GOAL' in posv or 'KEEP' in posv
                                gkinferred = []
                                def _gk_to_float(x):
                                    try:
                                        if x is None:
                                            return None
                                        if isinstance(x, str) and x.endswith('%'):
                                            return float(x.replace('%',''))
                                        return float(x)
                                    except Exception:
                                        return None

                                if is_gk:
                                    # Explicit mappings for common FBref goalkeeper keys
                                    # minutes / 90s
                                    if not merged_meta.get('minutes'):
                                        if merged_meta.get('Playing Time_Min_gk'):
                                            try:
                                                mins = int(float(merged_meta.get('Playing Time_Min_gk')))
                                                merged_meta['minutes'] = mins
                                                merged_meta['mins'] = mins
                                                merged_meta['Playing Time_Min_std'] = mins
                                                gkinferred.append('minutes_from_Playing Time_Min_gk')
                                            except Exception:
                                                pass
                                        else:
                                            ninety = merged_meta.get('Playing Time_90s_gk') or merged_meta.get('Playing Time_90s')
                                            nv = _gk_to_float(ninety)
                                            if nv is not None:
                                                mins = int(round(nv * 90))
                                                merged_meta['minutes'] = mins
                                                merged_meta['mins'] = mins
                                                merged_meta['Playing Time_Min_std'] = mins
                                                gkinferred.append('minutes_from_90s')

                                    # saves
                                    if not merged_meta.get('saves'):
                                        for cand in ('Performance_Saves_gk','Saves','Saves_gk'):
                                            if merged_meta.get(cand) is not None:
                                                try:
                                                    merged_meta['saves'] = int(float(merged_meta.get(cand)))
                                                    gkinferred.append(f'saves_from_{cand}')
                                                    break
                                                except Exception:
                                                    continue

                                    # save percentage
                                    if not merged_meta.get('save_pct'):
                                        for cand in ('Performance_Save%_gk','Performance_Save%','Save%','Save_pct'):
                                            if merged_meta.get(cand) is not None:
                                                v = _gk_to_float(merged_meta.get(cand))
                                                if v is not None:
                                                    merged_meta['save_pct'] = v
                                                    gkinferred.append(f'save_pct_from_{cand}')
                                                    break

                                    # goals against (GA)
                                    if not merged_meta.get('goals_against'):
                                        for cand in ('Performance_GA_gk','Performance_GA','GA','Goals_Against'):
                                            if merged_meta.get(cand) is not None:
                                                try:
                                                    merged_meta['goals_against'] = int(float(merged_meta.get(cand)))
                                                    gkinferred.append(f'goals_against_from_{cand}')
                                                    break
                                                except Exception:
                                                    continue

                                    # clean sheets
                                    if not merged_meta.get('clean_sheets'):
                                        for cand in ('Performance_CS_gk','Performance_CS','Clean Sheets','CS'):
                                            if merged_meta.get(cand) is not None:
                                                try:
                                                    merged_meta['clean_sheets'] = int(float(merged_meta.get(cand)))
                                                    gkinferred.append(f'clean_sheets_from_{cand}')
                                                    break
                                                except Exception:
                                                    continue

                                    # post-shot xG / PSxG
                                    if not merged_meta.get('post_shot_xg'):
                                        for cand in ('Post_Shot_xG','PSxG','post_shot_xg'):
                                            if merged_meta.get(cand) is not None:
                                                v = _gk_to_float(merged_meta.get(cand))
                                                if v is not None:
                                                    merged_meta['post_shot_xg'] = v
                                                    gkinferred.append(f'post_shot_xg_from_{cand}')
                                                    break

                                    # penalty saves
                                    if not merged_meta.get('pen_saves'):
                                        for cand in ('Penalty Kicks_PKsv_gk','Penalty Kicks_PKsv','PKsv'):
                                            if merged_meta.get(cand) is not None:
                                                try:
                                                    merged_meta['pen_saves'] = int(float(merged_meta.get(cand)))
                                                    gkinferred.append(f'pen_saves_from_{cand}')
                                                    break
                                                except Exception:
                                                    continue

                                    # attach GK inferences
                                    if gkinferred:
                                        merged_meta['_inferred'] = list(set(merged_meta.get('_inferred', []) + gkinferred))
                            except Exception:
                                pass
                        except Exception:
                            pass

                        # Prefer LLM polishing when available
                        if agent and getattr(agent.crag, 'llm', None):
                            try:
                                answer_prompt = ChatPromptTemplate.from_template(tactical_prompt)
                                answer_chain = answer_prompt | agent.crag.llm | StrOutputParser()
                                tactical_view = answer_chain.invoke({
                                    "question": query,
                                    "context": tactical_view
                                })
                            except Exception:
                                pass
                        # Ensure normalization AFTER LLM polishing as LLM may remove sections
                        try:
                            tactical_view = _normalize_tactical_view(tactical_view, merged_meta if 'merged_meta' in locals() else {}, result)
                        except Exception:
                            pass

                    except Exception as e:
                        tactical_view = f"Tactical view generation failed: {e}"

                    st.markdown(tactical_view)
                except Exception as e:
                    logger.error(f"Tactical view error: {e}")
                # Multi-season trend removed for simplicity
                
                # Full stats table (if available) - prefer retrieved_docs metadata, fallback to sources
                try:
                    metas = []
                    # Prefer raw retrieved docs with full metadata
                    if isinstance(result.get('retrieved_docs'), list) and len(result.get('retrieved_docs')) > 0:
                        for d in result.get('retrieved_docs'):
                            if isinstance(d, dict):
                                # d may be {'content':..., 'metadata': {...}}
                                meta = d.get('metadata') if d.get('metadata') else d
                                if isinstance(meta, dict):
                                    metas.append(meta)
                    # Fallback to 'sources' if no retrieved_docs metadata present
                    if not metas:
                        metas = [s for s in (result.get('sources') or []) if isinstance(s, dict)]
                    if metas:
                        # Merge keys across all metadata dicts, prefer first-non-empty value
                        merged_meta = {}
                        skip_keys = {'content', 'source', 'id'}
                        for meta in metas:
                            for k, v in meta.items():
                                if k in skip_keys:
                                    continue
                                # Prefer first non-empty value
                                if k not in merged_meta or merged_meta.get(k) in (None, '', [], {}):
                                    merged_meta[k] = v

                        # Normalize complex values for display
                        for k, v in merged_meta.items():
                            if isinstance(v, (list, dict)):
                                try:
                                    merged_meta[k] = json.dumps(v)
                                except Exception:
                                    merged_meta[k] = str(v)

                        # Aggressive enrichment: alias mapping, regex extraction, market value parsing,
                        # and fallback vector_db query to gather more metadata for the same player+season.
                        def _parse_market_value(val):
                            try:
                                if val is None:
                                    return None
                                s = str(val).strip()
                                # Examples: €12.0M, 12M, 12000000
                                if s.startswith('€'):
                                    s = s[1:]
                                if s.endswith('M'):
                                    return float(s[:-1]) * 1.0
                                if s.endswith('K'):
                                    return float(s[:-1]) / 1000.0
                                # plain number -> treat as millions if very large
                                fv = float(s.replace(',',''))
                                if fv > 1000:
                                    return fv / 1000000.0
                                return fv
                            except Exception:
                                return None

                        def _extract_from_text(patterns, text):
                            import re
                            for p in patterns:
                                m = re.search(p, text, flags=re.IGNORECASE)
                                if m:
                                    try:
                                        g = m.group(1) or m.group(0)
                                        return g.strip()
                                    except Exception:
                                        return m.group(0)
                            return None

                        try:
                            combined_text = ''
                            for d in (result.get('retrieved_docs') or []):
                                if isinstance(d, dict):
                                    combined_text += '\n' + (d.get('content') or '')
                            for m in metas:
                                if isinstance(m, dict) and m.get('content'):
                                    combined_text += '\n' + str(m.get('content'))

                            # Common aliases to canonical keys
                            aliases = {
                                'minutes': ['minutes', 'mins', 'playing_time', 'minutes_played', 'Playing Time_Min_std'],
                                'tackles': ['tackles', 'Tackles', 'tackles_total'],
                                'interceptions': ['interceptions', 'Interceptions'],
                                'aerial_pct': ['aerials_won_pct', 'aerial_duel_win_pct', 'Aerial Duels Won %'],
                                'long_pass_accuracy': ['Long_Cmp%_pass', 'Long_Cmp_pass', 'long_pass_accuracy'],
                                'dribbled_past': ['times_dribbled_past', 'dribbled_past', 'dribbled_past_90'],
                                'clearances': ['Clr_def', 'Clearances', 'clearances_total', 'Clearances_def'],
                                'passes_into_final_third': ['Passes_into_Final_Third', 'passes_into_final_third', 'Passes into Final Third'],
                                'errors_leading_shot': ['Err_def','Errors_leading_to_shot','Errors'],
                                'fouls_drawn': ['Fouls_Drawn','fouls_drawn','Fouls drawn']
                            }

                            # Fill simple aliases from existing merged_meta
                            for canon, keys in aliases.items():
                                if canon not in merged_meta or merged_meta.get(canon) in (None, '', 0):
                                    for k in keys:
                                        if k in merged_meta and merged_meta.get(k) not in (None, '', 0):
                                            merged_meta[canon] = merged_meta.get(k)
                                            break

                            # Try regex extraction from combined_text for missing stats
                            patterns_map = {
                                'minutes': [r'([0-9]{3,4})\s+minutes', r'played\s+([0-9]{3,4})\s+mins', r'([0-9]{3,4})\s+mins'],
                                'tackles': [r'tackles[:\s]*([0-9]+)', r'([0-9]+)\s+tackles'],
                                'interceptions': [r'interceptions[:\s]*([0-9]+)', r'([0-9]+)\s+interceptions'],
                                'aerial_pct': [r'([0-9]{1,3})%\s+aerial', r'aerial duels .*?([0-9]{1,3})%'],
                                'clearances': [r'clearances[:\s]*([0-9]+)', r'([0-9]+)\s+clearances', r'Clr[:\s]*([0-9]+)'],
                                'long_pass_accuracy': [r'long-pass accuracy[:\s]*([0-9]{1,3}%)', r'Long Pass Accuracy[:\s]*([0-9]{1,3}%)', r'Long_Cmp%[:\s]*([0-9]{1,3}%)'],
                                'passes_into_final_third': [r'passes into final third[:\s]*([0-9]+)', r'Passes into Final Third[:\s]*([0-9]+)'],
                                'errors_leading_shot': [r'errors leading to shot[:\s]*([0-9]+)', r'errors[:\s]*([0-9]+) leading to shot'],
                                'fouls_drawn': [r'fouls drawn[:\s]*([0-9]+)', r'fouls_drawn[:\s]*([0-9]+)'],
                                'dribbled_past': [r'dribbled past[:\s]*([0-9]+)', r'times dribbled past[:\s]*([0-9]+)']
                            }

                            for key, pats in patterns_map.items():
                                if merged_meta.get(key) in (None, '', 0):
                                    v = _extract_from_text(pats, combined_text)
                                    if v:
                                        try:
                                            merged_meta[key] = int(float(v))
                                        except Exception:
                                            merged_meta[key] = v

                            # Market value parse
                            if merged_meta.get('market_value') in (None, '', 0) and merged_meta.get('market_val'):
                                mv = _parse_market_value(merged_meta.get('market_val'))
                                if mv is not None:
                                    merged_meta['market_value'] = mv
                            if merged_meta.get('market_value') in (None, '', 0):
                                mv_text = _extract_from_text([r'€\s*([0-9]+\.?[0-9]*)M', r'market value[:\s]*€?([0-9]+\.?[0-9]*)M'], combined_text or '')
                                if mv_text:
                                    try:
                                        merged_meta['market_value'] = float(mv_text)
                                    except Exception:
                                        pass

                            # Goalkeeper-specific text extraction (saves, save%, PSxG, PSxG-GA, GA, CS, pass completion, launch%)
                            try:
                                pos_guess = (merged_meta.get('position') or merged_meta.get('pos') or '').upper()
                                is_gk_text = 'GK' in pos_guess or 'GOAL' in pos_guess or 'KEEP' in pos_guess
                                if is_gk_text or 'goalkeeper' in (result.get('sources') or [{}])[0].get('stat_module', '').lower():
                                    # patterns
                                    gk_patterns = {
                                        'saves': [r'Saves[:\s]*([0-9]+)', r'Performance_Saves_gk[:\s]*([0-9]+)'],
                                        'save_pct': [r'Save\s*%[:\s]*([0-9]{1,3}\.?[0-9]*)', r'Save%[:\s]*([0-9]{1,3}\.?[0-9]*)', r'Performance_Save%_gk[:\s]*([0-9]{1,3}\.?[0-9]*)'],
                                        'post_shot_xg': [r'PSxG[:\s]*([0-9]+\.?[0-9]*)', r'Post[- ]?Shot xG[:\s]*([0-9]+\.?[0-9]*)'],
                                        'psxg_ga': [r'PSxG[- ]?GA[:\s]*([+-]?[0-9]+\.?[0-9]*)', r'PSxG\-GA[:\s]*([+-]?[0-9]+\.?[0-9]*)', r'PSxG \+?\-?GA[:\s]*([+-]?[0-9]+\.?[0-9]*)'],
                                        'goals_against': [r'Goals Conceded[:\s]*([0-9]+)', r'GA[:\s]*([0-9]+)'],
                                        'clean_sheets': [r'Clean Sheets[:\s]*([0-9]+)', r'CS[:\s]*([0-9]+)', r'Performance_CS_gk[:\s]*([0-9]+)'],
                                        'pass_completion': [r'Pass Completion[:\s]*([0-9]{1,3}\.?[0-9]*)%', r'Pass completion[:\s]*([0-9]{1,3}\.?[0-9]*)%'],
                                        'launch_pct': [r'Launch %[:\s]*([0-9]{1,3}\.?[0-9]*)%', r'Launch%[:\s]*([0-9]{1,3}\.?[0-9]*)%'],
                                        'progressive_passes': [r'Progressive Passes[:\s]*([0-9]+)', r'PrgP_pass[:\s]*([0-9]+)']
                                    }
                                    for key, pats in gk_patterns.items():
                                        if merged_meta.get(key) in (None, '', 0):
                                            v = _extract_from_text(pats, combined_text)
                                            if v is not None:
                                                try:
                                                    if '%' in v:
                                                        merged_meta[key] = float(str(v).replace('%',''))
                                                    elif key in ('saves','clean_sheets','goals_against','progressive_passes'):
                                                        merged_meta[key] = int(float(v))
                                                    else:
                                                        merged_meta[key] = float(v)
                                                except Exception:
                                                    merged_meta[key] = v
                                                # mark inferred
                                                inf = merged_meta.get('_inferred', [])
                                                inf.append(f'inferred_{key}')
                                                merged_meta['_inferred'] = list(set(inf))
                            except Exception:
                                pass

                            # If still sparse, try a vector_db fallback to fetch more metadata
                            if hasattr(agent, 'crag') and hasattr(agent.crag, 'vector_db'):
                                player_lookup = merged_meta.get('player') or merged_meta.get('Player')
                                try:
                                    need_keys = (not merged_meta.get('tackles') or not merged_meta.get('interceptions') or not merged_meta.get('minutes') or not merged_meta.get('market_value'))
                                    if player_lookup and need_keys:
                                        # Try season-scoped query first
                                        q_where = {'player': player_lookup}
                                        if selected_season:
                                            q_where['season'] = selected_season
                                        more = None
                                        try:
                                            more = agent.crag.vector_db.query(
                                                query_texts=[f"{player_lookup} stats"],
                                                n_results=100,
                                                where=q_where
                                            )
                                        except Exception:
                                            more = None

                                        # If season-scoped returned nothing useful, relax filter to player only
                                        if not more or not more.get('metadatas'):
                                            try:
                                                more = agent.crag.vector_db.query(
                                                    query_texts=[f"{player_lookup} stats"],
                                                    n_results=200,
                                                    where=None
                                                )
                                            except Exception:
                                                more = None

                                        if more and more.get('metadatas'):
                                            for meta_doc in more['metadatas'][0]:
                                                if isinstance(meta_doc, dict):
                                                    for k, v in meta_doc.items():
                                                        if k in skip_keys:
                                                            continue
                                                        if merged_meta.get(k) in (None, '', 0):
                                                            merged_meta[k] = v
                                            # Re-run simple alias fill after adding more metas
                                            for canon, keys in aliases.items():
                                                if canon not in merged_meta or merged_meta.get(canon) in (None, '', 0):
                                                    for k in keys:
                                                        if k in merged_meta and merged_meta.get(k) not in (None, '', 0):
                                                            merged_meta[canon] = merged_meta.get(k)
                                                            break
                                except Exception:
                                    pass
                        except Exception:
                            pass

                        # If merged_meta is sparse, try to fetch more docs for the same player+season
                        try:
                            player_name_for_lookup = merged_meta.get('player') or merged_meta.get('Player')
                            if not player_name_for_lookup:
                                # More flexible name extraction (allow single or multi-word names)
                                m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b", query)
                                if m:
                                    player_name_for_lookup = m.group(1)

                            if player_name_for_lookup and len(metas) < 3 and hasattr(agent, 'crag') and hasattr(agent.crag, 'vector_db'):
                                try:
                                    # Query a broader set (season filter only) and perform fuzzy filtering client-side
                                    more_hits = agent.crag.vector_db.query(
                                            query_texts=[f"{player_name_for_lookup} stats"],
                                            n_results=50,
                                            where={
                                                'season': selected_season
                                            }
                                        )
                                    if more_hits and more_hits.get('metadatas'):
                                        for meta in more_hits['metadatas'][0]:
                                            if isinstance(meta, dict):
                                                metas.append(meta)
                                except Exception:
                                    pass

                        except Exception:
                            pass

                        # Enrich merged_meta by attempting to extract common stats from raw doc content
                        # if those stats are missing in metadata
                        try:
                            # Build combined content text from retrieved_docs if available
                            combined_text = ''
                            for d in (result.get('retrieved_docs') or []):
                                if isinstance(d, dict):
                                    combined_text += '\n' + (d.get('content') or '')

                            # Also include any document text from the additional metas we fetched
                            for m in metas:
                                # if meta came from collection.get it may not include content; skip
                                if isinstance(m, dict) and m.get('content'):
                                    combined_text += '\n' + str(m.get('content'))

                            def extract_number(patterns, text):
                                import re
                                for p in patterns:
                                    m = re.search(p, text, flags=re.IGNORECASE)
                                    if m:
                                        try:
                                            return float(m.group(1)) if m.group(1) is not None else None
                                        except Exception:
                                            try:
                                                return float(m.group(0))
                                            except Exception:
                                                return None
                                return None

                            # Patterns for common stats
                            stat_patterns = {
                                'xg': [r'xG[:\s]*([0-9]+\.?[0-9]*)', r'expected goals[:\s]*([0-9]+\.?[0-9]*)'],
                                'xa': [r'xA[:\s]*([0-9]+\.?[0-9]*)', r'expected assists[:\s]*([0-9]+\.?[0-9]*)'],
                                'xag': [r'xAG[:\s]*([0-9]+\.?[0-9]*)'],
                                'assists': [r'assists[:\s]*([0-9]+)'],
                                'key_passes': [r'key passes[:\s]*([0-9]+)'],
                                'passes_completed': [r'passes completed[:\s]*([0-9]+)'],
                                'touches_per90': [r'touches per 90[:\s]*([0-9]+\.?[0-9]*)', r'touches\/90[:\s]*([0-9]+\.?[0-9]*)'],
                                'pressures': [r'pressures per 90[:\s]*([0-9]+\.?[0-9]*)', r'pressures[:\s]*([0-9]+\.?[0-9]*)']
                            }

                            for stat, patterns in stat_patterns.items():
                                if stat not in merged_meta or merged_meta.get(stat) in (None, '', 0):
                                    val = extract_number(patterns, combined_text)
                                    if val is not None:
                                        if float(val).is_integer():
                                            merged_meta[stat] = int(val)
                                        else:
                                            merged_meta[stat] = round(float(val), 2)
                        except Exception:
                            # Don't block UI if enrichment fails
                            pass

                        # Create a two-column dataframe Metric / Value
                        df_stats = pd.DataFrame([merged_meta])
                        df_t = df_stats.T.reset_index()
                        df_t.columns = ['Metric', 'Value']

                        # Full Player Stat Table removed - Tactical View only
                except Exception as e:
                    logger.warning(f"Could not render full stat table: {e}")

                # Removed: Sources section details to reduce UI clutter
                
            except Exception as e:
                st.error(f" Error: {str(e)}")
                st.exception(e)

# Footer
st.divider()
st.caption("TactIQ Football Scout System | CRAG + REFRAG + Self-Check | Powered by LangGraph + LLaMA-3.1 + Qwen2.5")
