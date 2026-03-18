"""
Curated Tactical Blog URLs - EUROPEAN FOOTBALL FOCUS
High-quality accessible articles for European/English football tactical analysis
Target: 40-80 articles, 1,200-3,000 words average for optimal RAG reasoning
Includes: EPL, La Liga, Bundesliga, Serie A, Ligue 1 + UEFA competitions
"""

# ═══════════════════════════════════════════════════════════════
# TIER 1 — SAFE & HIGH VALUE (Scrape Freely)
# Top priority sources - ideal for newspaper3k + BeautifulSoup
# ═══════════════════════════════════════════════════════════════
TIER_1_PRIORITY = [
    # 🔥 Elite Tactical Analysis
    "https://spielverlagerung.com/",              # Deep tactical theory, German philosophy
    "https://totalfootballanalysis.com/",         # Scout reports, tactical breakdowns
    "https://statsbomb.com/articles/",            # Data-driven insights, xG analysis
    "https://thefalse9.com/",                     # Comprehensive tactical coverage
    "https://between-the-lines.co.uk/",           # In-depth analysis
    "https://www.zonalmarking.net/",              # Michael Cox's legendary analysis
    "https://coachdriven.com/",                   # Coaching perspectives
    "https://www.holdingmidfield.com/",           # Tactical deep dives
    "https://www.theanalysisfactor.com/",         # Statistical analysis
    "https://football-observatory.com/",          # CIES Football Observatory
]

# ═══════════════════════════════════════════════════════════════
# TIER 2 — NEWS & MEDIA OUTLETS (Quality Journalism)
# ═══════════════════════════════════════════════════════════════
NEWS_OUTLETS = [
    "https://theathletic.com/football/",          # Premium tactical journalism
    "https://theguardian.com/football/tactics/",  # Guardian tactics section
    "https://www.bbc.com/sport/football/",        # BBC Sport football
    "https://www.skysports.com/football/",        # Sky Sports football
]

# ═══════════════════════════════════════════════════════════════
# TIER 3 — COMMUNITY & ANALYSIS SITES
# ═══════════════════════════════════════════════════════════════
COMMUNITY_SOURCES = [
    # European Football Focus
    "https://themastermindsite.com/",
    "https://thecoachesvoice.com/",
    "https://tactalyse.com/blogs/",
    "https://footballblog.co.uk/category/football-tactics/",
    "https://breakingthelines.com/",
    
    # Medium Tactical Writers
    "https://medium.com/@thomasrandle",
    "https://medium.com/spielverlagerung",
]


# ═══════════════════════════════════════════════════════════════
# CURATED TACTICAL ARTICLES (Specific High-Quality Posts)
# ═══════════════════════════════════════════════════════════════
TACTICAL_ARTICLES = [
    # Spielverlagerung - Deep tactical theory
    "https://spielverlagerung.com/2024/11/20/tactical-theory-counter-pressing/",
    "https://spielverlagerung.com/2024/10/15/positional-play-and-build-up/",
    
    # Total Football Analysis - Scout reports
    "https://totalfootballanalysis.com/article/premier-league-tactical-trends",
    "https://totalfootballanalysis.com/article/champions-league-tactical-analysis",
    
    # StatsBomb - Data-driven insights
    "https://statsbomb.com/articles/soccer/",
    
    # Zonal Marking - Michael Cox insights
    "https://www.zonalmarking.net/",
]

# ═══════════════════════════════════════════════════════════════
# RSS FEEDS (Automated scraping)
# ═══════════════════════════════════════════════════════════════
RSS_FEEDS = [
    "https://spielverlagerung.com/feed/",
    "https://totalfootballanalysis.com/feed/",
    "https://themastermindsite.com/feed/",
    "https://footballblog.co.uk/feed/",
    "https://statsbomb.com/feed/",
]

# ═══════════════════════════════════════════════════════════════
# ALL SOURCES COMBINED (For Scraping)
# ═══════════════════════════════════════════════════════════════
ALL_SOURCES = TIER_1_PRIORITY + NEWS_OUTLETS + COMMUNITY_SOURCES

# ═══════════════════════════════════════════════════════════════
# OPTIMAL DATASET PARAMETERS (IMPORTANT FOR RAG QUALITY)
# ═══════════════════════════════════════════════════════════════
OPTIMAL_ARTICLE_COUNT = (40, 80)  # Min 40, Max 80 articles
OPTIMAL_WORD_COUNT = (1200, 3000)  # 1,200-3,000 words per article
OPTIMAL_TOKEN_COUNT = (300_000, 600_000)  # ~300k-600k tokens total


# Keywords to filter relevant tactical content (European football focus)
TACTICAL_KEYWORDS = [
    "pressing", "counter-press", "build-up", "possession",
    "defensive", "attacking", "transition", "formation",
    "tactical analysis", "squad analysis", "player profile",
    "scouting", "match analysis", "tactical trends",
    "positional play", "counter-attack", "high press",
    "low block", "passing network", "xG", "expected goals",
    "champions league", "europa league", "uefa", "european football",
]

# Top clubs to focus on for relevant content (European competitions)
TARGET_CLUBS = [
    # Premier League
    "Manchester City", "Arsenal", "Liverpool", "Chelsea", 
    "Manchester United", "Tottenham", "Newcastle", "Aston Villa",
    
    # La Liga
    "Barcelona", "Real Madrid", "Atletico Madrid", "Real Sociedad", "Girona",
    
    # Bundesliga
    "Bayern Munich", "Borussia Dortmund", "RB Leipzig", "Bayer Leverkusen",
    
    # Serie A
    "Inter Milan", "AC Milan", "Juventus", "Napoli", "Roma", "Atalanta",
    
    # Ligue 1
    "PSG", "Monaco", "Marseille", "Lyon", "Lille",
    
    # Notable European clubs
    "Porto", "Benfica", "Ajax", "PSV", "Celtic", "Rangers",
]

# UEFA Competitions to include
UEFA_COMPETITIONS = [
    "Champions League",
    "Europa League", 
    "Conference League",
    "UCL", "UEL", "UECL",
]

