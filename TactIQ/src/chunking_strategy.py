"""
Hybrid Structure-Aware + Semantic Chunking Strategy for TactIQ
================================================================

Implements GPT's recommended chunking approach:
1. Layer 1: Entity-Aligned Chunking (Player × Season × Competition)
2. Layer 2: Semantic Paragraph Chunking (Blogs/News)
3. Layer 3: Hierarchical Chunk Linking (Parent-Child relationships)

Design Goals:
- Maximize RAGAS faithfulness
- Enable CRAG + Self-Check verification
- Prevent cross-season mixing
- Rich metadata for source tracking
"""

import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
import pandas as pd
from loguru import logger
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from sentence_transformers import SentenceTransformer
import re


@dataclass
class ChunkMetadata:
    """Rich metadata for each chunk enabling CRAG verification"""
    # Core identifiers
    chunk_id: str
    parent_id: Optional[str] = None
    chunk_type: str = "unknown"  # 'player_stats', 'blog_paragraph', 'match_report'
    
    # Entity information
    player: Optional[str] = None
    team: Optional[str] = None
    season: Optional[str] = None
    competition: Optional[str] = None
    
    # Content metadata
    article_title: Optional[str] = None  # Full article title
    topic: Optional[str] = None  # High-level category
    tactical_theme: Optional[str] = None  # Specific tactical concept
    phase_of_play: Optional[str] = None  # attack/defense/transition
    source: Optional[str] = None
    url: Optional[str] = None
    publish_date: Optional[str] = None
    
    # Hierarchical linking
    sibling_chunks: List[str] = field(default_factory=list)
    child_chunks: List[str] = field(default_factory=list)
    
    # Chunk positioning
    chunk_index: int = 0
    total_chunks: int = 1
    token_count: int = 0
    
    # Versioning for incremental updates
    content_hash: Optional[str] = None
    last_updated: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to ChromaDB-compatible dict (all string values)"""
        return {
            "chunk_id": self.chunk_id,
            "parent_id": str(self.parent_id) if self.parent_id else "",
            "chunk_type": self.chunk_type,
            "player": self.player or "",
            "team": self.team or "",
            "season": self.season or "",
            "competition": self.competition or "",
            "article_title": self.article_title or "",
            "topic": self.topic or "",
            "tactical_theme": self.tactical_theme or "",
            "phase_of_play": self.phase_of_play or "",
            "source": self.source or "",
            "url": self.url or "",
            "publish_date": str(self.publish_date) if self.publish_date else "",
            "chunk_index": str(self.chunk_index),
            "total_chunks": str(self.total_chunks),
            "token_count": str(self.token_count),
            "content_hash": self.content_hash or "",
            "last_updated": str(self.last_updated) if self.last_updated else ""
        }


@dataclass
class Chunk:
    """Represents a single chunk with content and metadata"""
    content: str
    metadata: ChunkMetadata
    embedding: Optional[List[float]] = None


class HybridChunkingStrategy:
    """
    Implements hybrid structure-aware + semantic chunking
    """
    
    def __init__(
        self,
        embedding_model: Optional[SentenceTransformer] = None,
        chunk_size: int = 500,      # Fixed chunk size in tokens
        chunk_overlap: int = 100    # Overlap between chunks
    ):
        """
        Initialize simple fixed-size chunking strategy (January 1st approach)
        
        Args:
            embedding_model: SentenceTransformer model (optional, kept for compatibility)
            chunk_size: Fixed size for all chunks in tokens (~500 tokens = ~2000 chars)
            chunk_overlap: Overlap between chunks in tokens (~100 tokens = ~400 chars)
        """
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Convert tokens to characters (rough estimate: 1 token ≈ 4 chars)
        self.chunk_size_chars = chunk_size * 4
        self.chunk_overlap_chars = chunk_overlap * 4
        
        # No semantic chunker needed for fixed-size approach
        self.semantic_chunker = None
        
        logger.info(f"Initialized HybridChunkingStrategy (Fixed-Size: {chunk_size} tokens, Overlap: {chunk_overlap} tokens)")
    
    def _compute_hash(self, content: str) -> str:
        """Compute content hash for incremental updates"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximation: 1 token ≈ 4 chars)"""
        return len(text) // 4
    
    # ============================================================
    # LAYER 1: ENTITY-ALIGNED CHUNKING (Structured Data)
    # ============================================================
    
    def chunk_player_stats(
        self,
        player_df: pd.DataFrame,
        season_col: str = "season",
        player_col: str = "player",
        team_col: str = "team",
        competition_col: str = "league"
    ) -> List[Chunk]:
        """
        Layer 1: Entity-aligned chunking for player statistics
        
        Chunks by: (Player, Season, Competition)
        Ensures stats never mix across seasons
        
        Args:
            player_df: DataFrame with player statistics
            season_col: Column name for season
            player_col: Column name for player
            team_col: Column name for team
            competition_col: Column name for competition
            
        Returns:
            List of Chunks with entity-aligned metadata
        """
        logger.info(f"Entity-aligned chunking for {len(player_df)} player records...")
        
        chunks = []
        
        for idx, row in player_df.iterrows():
            player = row.get(player_col, "Unknown")
            season = row.get(season_col, "Unknown")
            team = row.get(team_col, "Unknown")
            competition = row.get(competition_col, "Unknown")
            
            # Create comprehensive content from ALL columns
            content = row.get('description', None)
            if not content or len(str(content)) < 200:
                # Fallback: Create detailed description from ALL available columns
                content = self._create_comprehensive_description(row, player, team, season, competition)
            
            # Create unique entity ID
            entity_id = f"{player}_{season}_{competition}".replace(" ", "_").replace("/", "-")
            content_hash = self._compute_hash(content)
            
            # Create metadata
            metadata = ChunkMetadata(
                chunk_id=f"player_{entity_id}_{content_hash}",
                parent_id=f"player_{player.replace(' ', '_')}",
                chunk_type="player_stats",
                player=player,
                team=team,
                season=season,
                competition=competition,
                source="FBref",
                token_count=self._estimate_tokens(content),
                content_hash=content_hash,
                chunk_index=0,
                total_chunks=1
            )
            
            chunks.append(Chunk(content=content, metadata=metadata))
        
        logger.info(f"✓ Created {len(chunks)} entity-aligned player chunks")
        return chunks
    
    def _create_fallback_description(self, row: pd.Series) -> str:
        """Create basic description if none exists"""
        player = row.get('player', 'Unknown Player')
        return f"{player} statistics for season {row.get('season', 'N/A')}"
    
    def _create_comprehensive_description(self, row: pd.Series, player: str, team: str, season: str, competition: str) -> str:
        """
        Create comprehensive description from ALL available columns
        Ensures every stat is included for analysis
        """
        # Start with player profile
        age = row.get('age', row.get('Age', 'N/A'))
        pos = row.get('pos', row.get('position', row.get('Pos', 'N/A')))
        born = row.get('born', row.get('Born', ''))
        nation = row.get('nation', row.get('Nation', ''))
        market_value = row.get('market_value', '')
        
        parts = []
        
        # Profile section
        profile = f"{player} is a {age}-year-old {pos}"
        if born:
            profile += f" born in {born}"
        if nation:
            profile += f" from {nation}"
        profile += f" playing for {team} in {competition} during the {season} season."
        parts.append(profile)
        
        # Exclude metadata columns
        exclude_cols = {'player', 'Player', 'team', 'Squad', 'Team', 'season', 'Season', 
                       'league', 'competition', 'Competition', 'Comp', 'description',
                       'age', 'Age', 'pos', 'Pos', 'position', 'Position', 'born', 'Born',
                       'nation', 'Nation', 'chunk_id', 'parent_id', 'index'}
        
        # Group stats by category
        stats_by_category = {
            'Goalkeeping': [],
            'Performance': [],
            'Expected': [],
            'Passing': [],
            'Pass Types': [],
            'Shot Creation': [],
            'Defensive': [],
            'Possession': [],
            'Playing Time': [],
            'Miscellaneous': [],
            'Other': []
        }
        
        for col in row.index:
            if col in exclude_cols or pd.isna(row[col]) or row[col] == '' or row[col] == 'nan':
                continue
            
            col_lower = col.lower()
            value = row[col]
            
            # Categorize the stat
            if any(term in col_lower for term in ['gk', 'save', 'clean', 'psxg', 'ga', 'sota', 'cs']):
                category = 'Goalkeeping'
            elif any(term in col_lower for term in ['goal', 'assist', 'shot', 'sot', 'g+a', 'pk', 'g-pk']):
                category = 'Performance'
            elif any(term in col_lower for term in ['xg', 'xa', 'npxg', 'expected']):
                category = 'Expected'
            elif any(term in col_lower for term in ['pass', 'cmp', 'att', 'prgp', 'final third', 'ppa']):
                category = 'Passing'
            elif any(term in col_lower for term in ['through', 'switch', 'cross', 'tb', 'corner', 'inswing', 'outswing']):
                category = 'Pass Types'
            elif any(term in col_lower for term in ['sca', 'gca', 'key pass', 'kp']):
                category = 'Shot Creation'
            elif any(term in col_lower for term in ['tackle', 'tkl', 'int', 'block', 'clr', 'err', 'aerial']):
                category = 'Defensive'
            elif any(term in col_lower for term in ['touch', 'take', 'carries', 'prg', 'dribble', 'succ', 'mis', 'dis']):
                category = 'Possession'
            elif any(term in col_lower for term in ['min', 'match', '90s', 'starts', 'subs', 'mn/mp']):
                category = 'Playing Time'
            elif any(term in col_lower for term in ['yellow', 'red', 'crdy', 'crdr', 'foul', 'fld', 'off', 'crs', 'won', 'lost']):
                category = 'Miscellaneous'
            else:
                category = 'Other'
            
            stats_by_category[category].append(f"{col}: {value}")
        
        # Build stat sections
        for category, stats in stats_by_category.items():
            if stats:
                parts.append(f"{category}: {', '.join(stats)}.")
        
        # Add market value if available
        if market_value and str(market_value).lower() not in ['nan', 'none', '']:
            parts.append(f"Market value: {market_value}.")
        
        return " ".join(parts)
    
    # ============================================================
    # LAYER 2: SEMANTIC PARAGRAPH CHUNKING (Unstructured Text)
    # ============================================================
    
    def chunk_blog_article(
        self,
        article: Dict[str, Any],
        article_id: int,
        use_semantic: bool = False  # Disabled by default, use fixed-size
    ) -> List[Chunk]:
        """
        Simple fixed-size + overlap chunking for tactical blogs (January 1st approach)
        
        Strategy:
        1. Split text into fixed-size chunks with overlap
        2. Same approach as player stats - simple and predictable
        3. Extract metadata per chunk
        
        Target: ~500 tokens with 100 token overlap
        
        Args:
            article: Dict with 'text', 'title', 'url', 'source', etc.
            article_id: Unique article identifier
            use_semantic: Ignored - kept for compatibility, always use fixed-size
            
        Returns:
            List of fixed-size chunks with metadata
        """
        text = article.get('text', '')
        if not text:
            return []
        
        title = article.get('title', 'Unknown')
        source = article.get('source', 'Unknown')
        url = article.get('url', '')
        publish_date = article.get('publish_date', '')
        
        # Use simple fixed-size chunking with overlap
        raw_chunks = self._fixed_size_split(text)
        
        # Create Chunk objects with metadata
        chunks = []
        parent_id = f"blog_{article_id}"
        
        for idx, chunk_text in enumerate(raw_chunks):
            token_count = self._estimate_tokens(chunk_text)
            content_hash = self._compute_hash(chunk_text)
            
            # Extract metadata per chunk
            chunk_topic = self._extract_topic(chunk_text, title)
            tactical_theme = self._extract_tactical_theme(chunk_text, title)
            phase_of_play = self._extract_phase_of_play(chunk_text)
            
            metadata = ChunkMetadata(
                chunk_id=f"blog_{article_id}_chunk_{idx}_{content_hash}",
                parent_id=parent_id,
                chunk_type="blog_article",
                player=None,  # ✅ Blog articles should NOT have player field
                team=None,    # ✅ Blog articles should NOT have team field
                season=None,  # ✅ Blog articles should NOT have season field
                article_title=title,
                topic=chunk_topic,
                tactical_theme=tactical_theme,
                phase_of_play=phase_of_play,
                source=source,
                url=url,
                publish_date=publish_date or "2024-12",
                chunk_index=idx,
                total_chunks=len(raw_chunks),
                token_count=token_count,
                content_hash=content_hash
            )
            
            chunks.append(Chunk(content=chunk_text, metadata=metadata))
        
        # Update sibling relationships
        chunk_ids = [c.metadata.chunk_id for c in chunks]
        for chunk in chunks:
            chunk.metadata.sibling_chunks = [cid for cid in chunk_ids if cid != chunk.metadata.chunk_id]
        
        logger.info(f"✓ Article '{title[:50]}' → {len(chunks)} fixed-size chunks")
        return chunks
    
    def _fixed_size_split(self, text: str) -> List[str]:
        """
        Simple fixed-size split with overlap (January 1st approach)
        Uses RecursiveCharacterTextSplitter with fixed chunk size
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size_chars,      # ~2000 chars = ~500 tokens
            chunk_overlap=self.chunk_overlap_chars, # ~400 chars = ~100 tokens
            separators=[
                "\n\n",       # Paragraph breaks
                "\n",         # Line breaks
                ". ",         # Sentences
                " ",          # Words
                ""
            ],
            length_function=len,
            is_separator_regex=False
        )
        return splitter.split_text(text)
    
    def _enforce_token_bounds(self, chunks: List[str], min_tokens: int = 200, max_tokens: int = 700) -> List[str]:
        """
        🚨 CRITICAL: Enforce HARD token bounds (MIN 200, MAX 700)
        
        Strategy:
        1. Merge chunks < MIN_TOKENS with nearest sibling
        2. Force split chunks > MAX_TOKENS (even mid-paragraph)
        
        Args:
            chunks: List of text chunks
            min_tokens: Minimum tokens per chunk (default 200)
            max_tokens: Hard maximum tokens per chunk (default 700)
            
        Returns:
            List of chunks within token bounds
        """
        # Step 1: Split oversized chunks
        split_chunks = []
        for chunk in chunks:
            token_count = self._estimate_tokens(chunk)
            
            if token_count <= max_tokens:
                split_chunks.append(chunk)
            else:
                # FORCE SPLIT by sentences
                sentences = chunk.split('. ')
                current_chunk = ""
                
                for sentence in sentences:
                    test_chunk = current_chunk + sentence + ". "
                    
                    if self._estimate_tokens(test_chunk) <= max_tokens:
                        current_chunk = test_chunk
                    else:
                        # Save current chunk and start new one
                        if current_chunk:
                            split_chunks.append(current_chunk.strip())
                        current_chunk = sentence + ". "
                
                # Add remaining text
                if current_chunk:
                    split_chunks.append(current_chunk.strip())
        
        # Step 2: Merge small chunks with nearest neighbor
        merged_chunks = []
        i = 0
        while i < len(split_chunks):
            chunk = split_chunks[i]
            token_count = self._estimate_tokens(chunk)
            
            if token_count < min_tokens and i + 1 < len(split_chunks):
                # Merge with next chunk
                next_chunk = split_chunks[i + 1]
                combined = chunk + " " + next_chunk
                
                # Only merge if combined is still reasonable
                if self._estimate_tokens(combined) <= max_tokens:
                    merged_chunks.append(combined)
                    i += 2  # Skip next chunk
                else:
                    # Can't merge, keep as-is (edge case)
                    merged_chunks.append(chunk)
                    i += 1
            else:
                merged_chunks.append(chunk)
                i += 1
        
        return merged_chunks
    
    def _extract_player_mentions(self, text: str) -> List[str]:
        """Extract player names from text (basic heuristic)"""
        # This is a simple implementation - can be enhanced with NER
        players = []
        # Look for capitalized names (2-3 words)
        pattern = r'\b[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}\b'
        matches = re.findall(pattern, text)
        # Filter common false positives
        stopwords = {'The', 'This', 'That', 'These', 'Those', 'Premier League', 'La Liga', 'Champions League'}
        players = [m for m in matches if m not in stopwords]
        return list(set(players))[:5]  # Top 5 unique
    
    def _extract_team_mentions(self, text: str) -> List[str]:
        """Extract team names from text - EXPANDED LIST"""
        # Comprehensive team names for better metadata coverage
        teams = [
            'Liverpool', 'Manchester City', 'Manchester United', 'Arsenal', 'Chelsea', 
            'Tottenham', 'Real Madrid', 'Barcelona', 'Atletico Madrid', 'Bayern Munich', 
            'Borussia Dortmund', 'PSG', 'Paris Saint-Germain', 'Juventus', 'Inter Milan', 
            'AC Milan', 'Napoli', 'Roma', 'Ajax', 'Porto', 'Benfica', 'Sporting',
            'Leicester', 'Aston Villa', 'Newcastle', 'Brighton', 'Brentford',
            'Sevilla', 'Valencia', 'Villarreal', 'RB Leipzig', 'Bayer Leverkusen',
            'Man City', 'Man United', 'Spurs'  # Common abbreviations
        ]
        mentioned = [team for team in teams if team in text]
        return mentioned[:5]  # Top 5
    
    def _extract_topic(self, text: str, title: str) -> str:
        """Extract main tactical topic - EXPANDED CATEGORIES"""
        topics = {
            'pressing': ['press', 'pressing', 'high press', 'counter-press', 'gegenpress'],
            'possession': ['possession', 'build-up', 'passing', 'tiki-taka', 'positional play'],
            'counter-attack': ['counter', 'transition', 'break', 'counter-attack', 'fast break'],
            'defense': ['defend', 'defensive', 'low block', 'shape', 'defending', 'backline'],
            'tactics': ['formation', 'tactical', 'system', '4-3-3', '3-4-3', '4-2-3-1', '3-5-2'],
            'attacking': ['attack', 'attacking', 'forward', 'striker', 'wing', 'chance creation'],
            'midfield': ['midfield', 'pivot', 'double pivot', 'playmaker', 'no. 10'],
            'set-pieces': ['corner', 'free kick', 'set piece', 'throw-in'],
            'player-analysis': ['profile', 'scouting', 'analysis', 'performance'],
            'match-analysis': ['match', 'game', 'fixture', 'vs', 'versus']
        }
        
        combined = (title + " " + text[:800]).lower()
        for topic, keywords in topics.items():
            if any(kw in combined for kw in keywords):
                return topic
        return 'tactical_analysis'
    
    def _extract_season(self, text: str) -> Optional[str]:
        """Extract season from text (e.g., '2023-24', '2024/25')"""
        pattern = r'20\d{2}[-/]20?\d{2}'
        matches = re.findall(pattern, text)
        return matches[0] if matches else None
    
    def _extract_tactical_theme(self, text: str, title: str) -> str:
        """Extract specific tactical theme (more granular than topic)"""
        themes = {
            'high-press': ['high press', 'pressing high', 'frontfoot defense', 'aggressive pressing'],
            'counter-press': ['counter-press', 'gegenpress', 'immediate pressing', 'pressing after loss'],
            'build-up-play': ['build-up', 'playing out', 'short passing', 'build from back'],
            'positional-attack': ['positional attack', 'positional play', 'juego de posición', 'possession game'],
            'fast-transition': ['fast transition', 'quick counter', 'vertical play', 'direct play'],
            'wing-play': ['wing play', 'wide attack', 'crosses', 'overlapping'],
            'central-overload': ['central overload', 'midfield dominance', 'numerical superiority'],
            'low-block': ['low block', 'deep defense', 'compact shape', 'defensive organization'],
            'rest-defense': ['rest defense', 'defensive transition', 'recovery runs'],
            'set-pieces': ['set piece', 'corner', 'free kick', 'dead ball'],
            'player-profiling': ['player profile', 'scouting report', 'player analysis'],
            'tactical-periodization': ['tactical periodization', 'training methodology', 'training structure']
        }
        
        combined = (title + " " + text[:1000]).lower()
        for theme, keywords in themes.items():
            if any(kw in combined for kw in keywords):
                return theme
        return 'general-tactics'
    
    def _extract_phase_of_play(self, text: str) -> str:
        """Extract phase of play: attack, defense, or transition"""
        text_lower = text[:500].lower()
        
        # Count indicators for each phase
        attack_keywords = ['attack', 'attacking', 'offense', 'offensive', 'forward', 'goal', 'chance', 'shot']
        defense_keywords = ['defend', 'defending', 'defensive', 'protect', 'block', 'tackle', 'intercept']
        transition_keywords = ['transition', 'counter', 'break', 'turnover', 'recovery', 'pressing after loss']
        
        attack_score = sum(1 for kw in attack_keywords if kw in text_lower)
        defense_score = sum(1 for kw in defense_keywords if kw in text_lower)
        transition_score = sum(1 for kw in transition_keywords if kw in text_lower)
        
        # Return dominant phase
        scores = {'attack': attack_score, 'defense': defense_score, 'transition': transition_score}
        max_phase = max(scores, key=scores.get)
        
        # Require at least 2 mentions to assign phase
        if scores[max_phase] >= 2:
            return max_phase
        return 'general'
    
    # ============================================================
    # LAYER 3: HIERARCHICAL CHUNK LINKING
    # ============================================================
    
    def build_hierarchy(self, chunks: List[Chunk]) -> List[Chunk]:
        """
        Layer 3: Build parent-child-sibling relationships
        
        Enables:
        - Multi-hop retrieval
        - Better REFRAG reasoning
        - Cleaner citations
        """
        # Group by parent_id
        hierarchy = {}
        for chunk in chunks:
            parent = chunk.metadata.parent_id
            if parent not in hierarchy:
                hierarchy[parent] = []
            hierarchy[parent].append(chunk)
        
        # Update child_chunks for parent references
        parent_chunks = {}
        for parent_id, children in hierarchy.items():
            # Create virtual parent if needed
            if not any(c.metadata.chunk_id == parent_id for c in chunks):
                continue
            
            child_ids = [c.metadata.chunk_id for c in children]
            
            # Find parent chunk and update its children
            for chunk in chunks:
                if chunk.metadata.chunk_id == parent_id:
                    chunk.metadata.child_chunks = child_ids
                    break
        
        logger.info(f"✓ Built hierarchy: {len(hierarchy)} parent nodes")
        return chunks
    
    # ============================================================
    # INCREMENTAL UPDATE SUPPORT
    # ============================================================
    
    def needs_update(self, existing_hash: str, new_content: str) -> bool:
        """
        Check if chunk needs re-embedding
        
        Returns:
            True if content changed, False if unchanged
        """
        new_hash = self._compute_hash(new_content)
        return existing_hash != new_hash
    
    def chunk_incremental_update(
        self,
        new_data: pd.DataFrame,
        existing_hashes: Dict[str, str]
    ) -> Tuple[List[Chunk], List[str]]:
        """
        Incremental update: only re-embed changed chunks
        
        Args:
            new_data: New/updated data
            existing_hashes: Dict of entity_id → content_hash
            
        Returns:
            (chunks_to_update, chunk_ids_to_skip)
        """
        chunks_to_update = []
        chunk_ids_to_skip = []
        
        for idx, row in new_data.iterrows():
            entity_id = f"{row.get('player', '')}_{row.get('season', '')}_{row.get('league', '')}".replace(" ", "_")
            content = row.get('description', '')
            new_hash = self._compute_hash(content)
            
            if entity_id in existing_hashes and existing_hashes[entity_id] == new_hash:
                chunk_ids_to_skip.append(entity_id)
            else:
                # Content changed or new entity
                chunks = self.chunk_player_stats(pd.DataFrame([row]))
                chunks_to_update.extend(chunks)
        
        logger.info(f"Incremental update: {len(chunks_to_update)} to update, {len(chunk_ids_to_skip)} unchanged")
        return chunks_to_update, chunk_ids_to_skip


# ============================================================
# CONVENIENCE FUNCTIONS
# ============================================================

def create_chunking_strategy(
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    use_semantic: bool = False,  # Disabled by default - use fixed-size
    chunk_size: int = 500,       # Fixed chunk size in tokens
    chunk_overlap: int = 100     # Overlap in tokens
) -> HybridChunkingStrategy:
    """
    Factory function to create chunking strategy (January 1st approach)
    
    Args:
        embedding_model_name: Name of sentence-transformers model (kept for compatibility)
        use_semantic: Ignored - always use fixed-size (kept for compatibility)
        chunk_size: Fixed size for all chunks in tokens (default: 500)
        chunk_overlap: Overlap between chunks in tokens (default: 100)
        
    Returns:
        Configured HybridChunkingStrategy with fixed-size chunking
    """
    # We don't use semantic chunking anymore, but keep this for compatibility
    embedding_model = None
    
    return HybridChunkingStrategy(
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )


def validate_chunk_quality(chunks: List[Chunk], verbose: bool = True) -> Dict[str, Any]:
    """
    Validate chunk quality metrics
    
    Checks:
    - Token distribution
    - Metadata completeness
    - Season mixing (should be none)
    - Hierarchy integrity
    """
    if not chunks:
        return {"error": "No chunks to validate"}
    
    token_counts = [c.metadata.token_count for c in chunks]
    
    # Check for season mixing (bad!)
    season_mixing = []
    for chunk in chunks:
        if chunk.metadata.chunk_type == 'player_stats' and chunk.metadata.season:
            # Check if content mentions different season
            other_seasons = re.findall(r'20\d{2}[-/]20?\d{2}', chunk.content)
            if other_seasons and chunk.metadata.season not in other_seasons[0]:
                season_mixing.append(chunk.metadata.chunk_id)
    
    # Metadata completeness (comprehensive check)
    complete_metadata = sum(1 for c in chunks 
                           if c.metadata.article_title 
                           and c.metadata.source 
                           and c.metadata.publish_date 
                           and c.metadata.tactical_theme
                           and c.metadata.phase_of_play)
    
    # Basic metadata (title + source only)
    basic_metadata = sum(1 for c in chunks if c.metadata.article_title and c.metadata.source)
    
    report = {
        "total_chunks": len(chunks),
        "token_stats": {
            "min": min(token_counts),
            "max": max(token_counts),
            "avg": sum(token_counts) / len(token_counts),
            "median": sorted(token_counts)[len(token_counts)//2]
        },
        "season_mixing_detected": len(season_mixing),
        "metadata_completeness_full": f"{complete_metadata}/{len(chunks)} ({100*complete_metadata/len(chunks):.1f}%)",
        "metadata_completeness_basic": f"{basic_metadata}/{len(chunks)} ({100*basic_metadata/len(chunks):.1f}%)",
        "chunk_types": {ct: sum(1 for c in chunks if c.metadata.chunk_type == ct) 
                       for ct in set(c.metadata.chunk_type for c in chunks)}
    }
    
    if verbose:
        logger.info("Chunk Quality Validation:")
        logger.info(f"  Total chunks: {report['total_chunks']}")
        logger.info(f"  Token range: {report['token_stats']['min']}-{report['token_stats']['max']}")
        logger.info(f"  Avg tokens: {report['token_stats']['avg']:.1f}")
        logger.info(f"  Season mixing: {report['season_mixing_detected']} (should be 0)")
        logger.info(f"  Metadata (Full): {report['metadata_completeness_full']}")
        logger.info(f"  Metadata (Basic): {report['metadata_completeness_basic']}")
    
    return report
