import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from src.config import logger

class ConstitutionClassifier:
    def __init__(self, db_driver):
        self.driver = db_driver
        # 1. Feature Engineering
        self.vectorizer = TfidfVectorizer(
            max_features=2000, 
            stop_words='english', 
            ngram_range=(1, 2)
        )
        # 2. Model
        self.model = RandomForestClassifier(
            n_estimators=100, 
            random_state=42, 
            class_weight='balanced' 
        )
        self.class_names = ["Rights", "Judiciary", "Federalism", "Governance", "Other"]

    def get_advanced_data(self):
        """
        Fetches Text + Graph Context (Labels, Connections) for better Ground Truth.
        Does NOT fetch AMENDED_BY edges (respects temporal constraint).
        """
        logger.info("Fetching Advanced Data from Neo4j...")
        query = """
        MATCH (a:Article)
        OPTIONAL MATCH (a)-[:RELATES_TO_RIGHT]->(r:Right)
        OPTIONAL MATCH (a)-[:GOVERNS]->(i:Institution)
        OPTIONAL MATCH (a)-[:TAGGED]->(t:Tag)
        OPTIONAL MATCH (a)<-[:REFERENCES]-(ref:Article)
        RETURN 
            a.id as id, 
            a.text as text, 
            a.title as title,
            labels(a) as neo4j_labels,
            collect(distinct r.name) as rights,
            collect(distinct i.name) as institutions,
            collect(distinct t.name) as tags,
            count(distinct ref) as citation_count
        """
        
        with self.driver.session() as session:
            result = session.run(query)
            df = pd.DataFrame([r.data() for r in result])
        return df

    def label_data(self, df):
        """Applies Advanced Priority Labeling."""
        labels = []
        for _, row in df.iterrows():
            text = (str(row.get('text', '')) + " " + str(row.get('title', ''))).lower()
            neo_labels = row['neo4j_labels']
            rights = row['rights']
            insts = row['institutions']
            tags = row['tags']
            
            # --- PRIORITY 1: OFFICIAL STRUCTURE ---
            if "FundamentalRight" in neo_labels or any(rights):
                labels.append(0) # Rights
            
            # --- PRIORITY 2: GRAPH CONNECTIONS ---
            elif any(i in insts for i in ["Supreme Court", "High Court", "Judiciary"]):
                labels.append(1) # Judiciary
            elif any(i in insts for i in ["President", "Prime Minister", "Federal Parliament"]):
                labels.append(3) # Governance
            elif any(t in tags for t in ["Women", "Dalit", "Children", "Minority"]):
                labels.append(0) # Rights (via Tag)

            # --- PRIORITY 3: TEXT KEYWORDS ---
            elif any(k in text for k in ["court", "judge", "judicial", "tribunal"]):
                labels.append(1) # Judiciary
            elif any(k in text for k in ["province", "local level", "federation"]):
                labels.append(2) # Federalism
            elif any(k in text for k in ["president", "minister", "parliament", "election"]):
                labels.append(3) # Governance
            else:
                labels.append(4) # Other
        
        df['label'] = labels
        return df

    def train(self, df):
        """Runs training and returns artifacts."""
        logger.info(" Vectorizing Text (TF-IDF)...")
        X = self.vectorizer.fit_transform(df['text']).toarray()
        y = df['label'].values

        # Stratified Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        logger.info("Training Random Forest...")
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        return X_train, X_test, y_train, y_test, y_pred

    def save_model(self, filepath):
        """Saves the Vectorizer and Model as a tuple."""
        logger.info(f"Saving model to {filepath}...")
        joblib.dump((self.vectorizer, self.model), filepath)