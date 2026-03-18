import pandas as pd
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from src.graph_db import GraphDB
from src.config import logger

class ConstitutionRAG:
    def __init__(self):
        self.db = GraphDB()
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.data = self._load_knowledge_base()
        self.embeddings = self._precompute_embeddings()

    def _load_knowledge_base(self):
        logger.info("Loading Knowledge Base from Neo4j...")
        # Fetching title + text to ensure we have the full content
        query = """
        MATCH (a:Article)
        OPTIONAL MATCH (a)-[:TAGGED]->(t:Tag)
        RETURN a.id as id, a.text as text, a.title as title, collect(t.name) as tags
        """
        with self.db.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([r.data() for r in result])
        
        # Combine text for better semantic search
        df['search_text'] = df['text'].fillna('') + " " + df['title'].fillna('') + " " + df['tags'].apply(lambda x: " ".join(x))
        return df

    def _precompute_embeddings(self):
        logger.info("Vectorizing Constitution...")
        return self.embedder.encode(self.data['search_text'].tolist(), show_progress_bar=True)

    def _clean_text(self, text):
        """Removes artifacts like page numbers (e.g. '207' at end) and extra spaces."""
        if not text: return ""
        # Remove standalone numbers at the very end of the string (Page numbers)
        text = re.sub(r'\s+\d+\s*$', '', text)
        # Fix multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def get_graph_context(self, article_id):
        query = """
        MATCH (a:Article {id: $id})
        OPTIONAL MATCH (a)-[:REFERENCES]->(ref:Article)
        OPTIONAL MATCH (a)-[:GOVERNS]->(inst:Institution)
        OPTIONAL MATCH (a)-[:RELATES_TO_RIGHT]->(r:Right)
        RETURN 
            collect(distinct ref.id) as references,
            collect(distinct inst.name) as institutions,
            collect(distinct r.name) as rights
        """
        with self.db.driver.session() as session:
            result = session.run(query, id=article_id).single()
            return result.data() if result else {}

    def search(self, user_query, top_k=3): 
        """
        Retrieves top_k valid articles.
        Fetches extra candidates (top_k * 3) to allow for filtering of 'junk' nodes.
        """
        query_vec = self.embedder.encode([user_query])
        sim_scores = cosine_similarity(query_vec, self.embeddings)[0]
        
        # 1. Fetch MORE candidates than we need (e.g., Top 10 instead of Top 3)
        # This gives us a buffer to throw away bad results.
        candidate_indices = np.argsort(sim_scores)[::-1][:top_k * 4]
        
        results = []
        for idx in candidate_indices:
            if len(results) >= top_k:
                break # We found enough good ones
            
            score = sim_scores[idx]
            if score < 0.20: continue 
            
            row = self.data.iloc[idx]
            raw_text = str(row['text'])
            
            # --- 2. THE JUNK FILTER ---
            # If the text is just "Judiciary" or "Part 11", it's too short to be an Article.
            # Real articles are usually > 60 chars.
            if len(raw_text) < 60:
                continue

            # 3. Clean and Process
            clean_body = self._clean_text(raw_text)
            title = row['title'] if row['title'] else f"Article {row['id'].split('_')[-1]}"
            
            # Format Title (e.g., "Article 126 | Courts to exercise powers")
            # If title is inside text, try to extract it, otherwise use generic
            display_title = title
            
            context = self.get_graph_context(row['id'])
            
            result = {
                "article_id": row['id'],
                "title": display_title,
                "text": clean_body, 
                "relevance": float(score),
                "graph_context": context
            }
            results.append(result)
            
        return results