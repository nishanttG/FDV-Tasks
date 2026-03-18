import sys
import os

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_db import GraphDB
from src.analytics import GraphAnalytics
from src.config import logger

def main():
    logger.info("STARTED: Day 2 Full Analysis")
    
    db = GraphDB()
    analytics = GraphAnalytics(db.driver)
    
    try:
        # --- 1. Structural Analysis ---
        print("\n" + "-"*40)
        print(" NETWORK STRUCTURE")
        print("-"*40)
        G = analytics.get_citation_network_nx()
        stats = analytics.analyze_structure(G)
        
        print(f"Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}")
        print(f"Disconnected Islands: {stats['islands']}")
        
        print("\n Top Central Articles:")
        for art, refs in stats['top_central']:
            print(f"   - {art}: {refs} citations")

        # --- 2. Entities ---
        print("\n" + "-"*40)
        print(" KEY ENTITIES")
        print("="*40)
        print("\nInstitutions:")
        print(analytics.get_key_institutions().to_string(index=False))
        
        print("\nDominant Rights:")
        print(analytics.get_dominant_rights().to_string(index=False))

        # --- 3. Advanced Connections ---
        print("\n" + "-"*40)
        print(" HIDDEN CONNECTIONS")
        print("-"*40)
        chains = analytics.get_2hop_chains()
        if not chains.empty:
            print(chains.head(5).to_string(index=False))
        else:
            print("No 2-hop chains found.")

        # --- 4. Recommendation Engine (Your New Logic) ---
        target_art = "Art_17"
        print(f"\nRecommendations for {target_art}:")
        
        # Calling the updated function
        recs = analytics.recommend(target_art)
        
        if not recs.empty:
            print(recs.to_string(index=False))
        else:
            print("No recommendations found.")

    except Exception as e:
        logger.error(f"Analysis Failed: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()