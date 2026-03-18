import pandas as pd
import networkx as nx
from src.config import logger

class GraphAnalytics:
    def __init__(self, db_driver):
        self.driver = db_driver

    def get_citation_network_nx(self):
        """Fetches citation data and builds an in-memory NetworkX graph."""
        logger.info("Building in-memory graph from Neo4j...")
        query = """
        MATCH (s:Article)-[:REFERENCES]->(t:Article) 
        RETURN s.id as source, t.id as target
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            data = [r.data() for r in result]
        
        G = nx.DiGraph()
        for row in data:
            G.add_edge(row['source'], row['target'])
        return G

    def analyze_structure(self, G):
        """Calculates Centrality and Components (Islands)."""
        if G.number_of_nodes() == 0:
            return {"islands": 0, "largest_island": 0, "top_central": []}

        # FIX: Use the DIRECTED graph G for weakly connected components
        components = list(nx.weakly_connected_components(G))
        
        # Centrality (In-Degree = Popularity)
        degree = dict(G.in_degree())
        top_5 = sorted(degree.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "islands": len(components),
            "largest_island": len(max(components, key=len)) if components else 0,
            "top_central": top_5
        }

    def get_key_institutions(self):
        query = """
        MATCH (i:Institution)<-[:GOVERNS]-(n)
        RETURN i.name as Institution, count(n) as Mentions
        ORDER BY Mentions DESC LIMIT 5
        """
        with self.driver.session() as session:
            return pd.DataFrame([r.data() for r in session.run(query)])

    def get_dominant_rights(self):
        query = """
        MATCH (a:FundamentalRight)-[:HAS_CLAUSE]->(c:Clause)
        RETURN a.id as Article, count(c) as ClauseCount
        ORDER BY ClauseCount DESC LIMIT 5
        """
        with self.driver.session() as session:
            return pd.DataFrame([r.data() for r in session.run(query)])

    def get_2hop_chains(self):
        query = """
        MATCH (a:Article)-[:REFERENCES]->(b:Article)-[:REFERENCES]->(c:Article)
        WHERE a <> c
        RETURN a.id as Start, b.id as Bridge, c.id as End
        """
        with self.driver.session() as session:
            return pd.DataFrame([r.data() for r in session.run(query)])

    def recommend(self, article_id):
        query = """
        MATCH (source:Article {id: $id})
        
        // Strategy 1: Citations (Articles that cite this one)
        OPTIONAL MATCH (source)<-[:REFERENCES]-(citing)
        
        // Strategy 2: Shared Topic Tags
        OPTIONAL MATCH (source)-[:TAGGED]->(t:Tag)<-[:TAGGED]-(related)
        
        // Strategy 3: Co-Citation (Articles referenced by the same source)
        OPTIONAL MATCH (source)<-[:REFERENCES]-(common)-[:REFERENCES]->(cousin)
        WHERE cousin <> source
        
        // Aggregate & Score
        WITH coalesce(citing, related, cousin) as rec, 
            count(*) as score
        WHERE rec IS NOT NULL
        RETURN rec.id as Article, score as Relevance
        ORDER BY score DESC
        LIMIT 5
        """
        with self.driver.session() as session:
            result = session.run(query, id=article_id)
            return pd.DataFrame([r.data() for r in result])