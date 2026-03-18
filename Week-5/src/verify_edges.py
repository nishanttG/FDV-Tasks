import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_db import GraphDB

def main():
    db = GraphDB()
    required_edges = [
        "HAS_PART", "HAS_ARTICLE", "HAS_CLAUSE", 
        "RELATES_TO_RIGHT", "GOVERNS", "AMENDED_BY", 
        "REFERENCES", "TAGGED"
    ]
    
    print("\nSCHEMA VERIFICATION CHECKLIST ")
    with db.driver.session() as session:
        all_passed = True
        for edge in required_edges:
            result = session.run(f"MATCH ()-[r:{edge}]->() RETURN count(r) as count")
            count = result.single()["count"]
            if count > 0:
                print(f" {edge:<20}: {count} edges found")
            else:
                print(f" {edge:<20}: MISSING")
                all_passed = False
    
    if all_passed:
        print("\n SUCCESS: Full Schema Implemented!")
    else:
        print("\n FAIL: Some schema elements are missing.")
    
    db.close()

if __name__ == "__main__":
    main()