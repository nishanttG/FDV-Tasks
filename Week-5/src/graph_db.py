from neo4j import GraphDatabase
from src.config import URI, AUTH, logger

class GraphDB:
    def __init__(self):
        self.driver = GraphDatabase.driver(URI, auth=AUTH)

    def close(self):
        self.driver.close()

    def ingest(self, df):
        logger.info("Pushing structure to Neo4j...")
        with self.driver.session() as session:
            # 1. Clean Slate & Constraints
            session.run("MATCH (n) DETACH DELETE n")
            session.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Article) REQUIRE a.id IS UNIQUE")
            session.run("MERGE (:Constitution {name: 'Constitution of Nepal', year: 2015})")

            # 2. Iterate DataFrame
            for _, row in df.iterrows():
                # Ingest Parts & Schedules
                if row['type'] in ["Part", "Schedule"]:
                    session.run(f"""
                        MATCH (c:Constitution)
                        MERGE (p:{row['type']} {{id: $id}}) SET p.text = $txt
                        MERGE (c)-[:HAS_PART]->(p)
                    """, id=row['id'], txt=row['text'])
                
                # Ingest Articles
                elif row['type'] == "Article":
                    # Extract number safely "Art_1" -> "1"
                    num = row['id'].split("_")[1]
                    
                    # 1. Determine Labels
                    labels_str = ""
                    if row['parent_id'] and "Part-3" in str(row['parent_id']): 
                        labels_str += ":FundamentalRight"
                    if row['parent_id'] and "Part-4" in str(row['parent_id']): 
                        labels_str += ":DirectivePrinciple"

                    # 2. Construct SET clause dynamically to avoid syntax errors
                    set_query = "SET a.number = $num, a.text = $txt"
                    if labels_str:
                        set_query += f", a{labels_str}"

                    # 3. Run Query
                    session.run(f"""
                        MATCH (p {{id: $pid}})
                        MERGE (a:Article {{id: $aid}})
                        {set_query}
                        MERGE (p)-[:HAS_ARTICLE]->(a)
                    """, pid=row['parent_id'], aid=row['id'], num=num, txt=row['text'])
                
                # Ingest Clauses
                elif row['type'] == "Clause":
                    art_id = row['id'].split(".")[0]
                    session.run("""
                        MATCH (a:Article {id: $aid})
                        MERGE (c:Clause {id: $cid})
                        SET c.text = $txt
                        MERGE (a)-[:HAS_CLAUSE]->(c)
                    """, aid=art_id, cid=row['id'], txt=row['text'])

    def enrich(self):
        logger.info("✨ Enriching Graph (Edges, Tags, Rights)...")
        with self.driver.session() as session:
            # 1. Edge: REFERENCES
            session.run("""
                MATCH (s) WHERE s:Article OR s:Clause
                WITH s, apoc.text.regexGroups(s.text, 'Article\\\\s*[-]?\\\\s*(\\\\d+)') AS matches
                UNWIND matches AS match
                MATCH (t:Article {id: 'Art_' + match[1]}) WHERE s <> t
                MERGE (s)-[:REFERENCES]->(t)
            """)
            
            # 2. Edge: GOVERNS (Institutions)
            institutions = [
                "Supreme Court", "President", "Prime Minister", 
                "Federal Parliament", "Election Commission"
            ]
            for inst in institutions:
                session.run("""
                    MATCH (n) WHERE (n:Article OR n:Clause) AND n.text CONTAINS $inst
                    MERGE (i:Institution {name: $inst})
                    MERGE (n)-[:GOVERNS]->(i)
                """, inst=inst)
                
            # 3. Edge: TAGGED (Topics) - Using CONTAINS for safety
            tag_map = {
                "Women": ["women", "gender", "female"],
                "Dalit": ["dalit"],
                "Children": ["child", "children"],
                "Citizenship": ["citizen"],
                "Minority": ["minority"]
            }
            for tag, keywords in tag_map.items():
                or_clause = " OR ".join([f"toLower(n.text) CONTAINS '{k}'" for k in keywords])
                
                query = f"""
                    MATCH (n) WHERE (n:Article OR n:Clause) AND ({or_clause})
                    MERGE (t:Tag {{name: '{tag}'}})
                    MERGE (n)-[:TAGGED]->(t)
                """
                session.run(query)

            # 4. Edge: AMENDED_BY (Manual)
            session.run("""
                MATCH (c:Constitution)
                MERGE (am:Amendment {name: 'First Amendment', date: '2016'})
                MERGE (c)-[:AMENDED_BY]->(am)
                WITH am
                MATCH (a:Article {id: 'Art_42'})
                MERGE (a)-[:AMENDED_BY]->(am)
            """)

            # 5. Edge: RELATES_TO_RIGHT (Manual)
            rights_data = [
                ("Art_16", "Right to live with dignity"),
                ("Art_17", "Right to Freedom"),
                ("Art_18", "Right to Equality")
            ]
            
            for art_id, right_name in rights_data:
                session.run("""
                    MERGE (r:Right {name: $rname})
                    WITH r
                    MATCH (a:Article {id: $aid})
                    MERGE (a)-[:RELATES_TO_RIGHT]->(r)
                """, aid=art_id, rname=right_name)
            
            logger.info("Enrichment Complete.")