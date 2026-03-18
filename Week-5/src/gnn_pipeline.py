import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from src.config import logger

# --- MODEL DEFINITION ---
class ConstitutionSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        # SAGEConv aggregates neighbor features (Mean aggregation by default)
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        return x

class GNNPipeline:
    def __init__(self, db_driver):
        self.driver = db_driver
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = ["Rights", "Judiciary", "Federalism", "Governance", "Other"]

    def load_graph_data(self):
        """
        Fetches Nodes + Rich Edges (Citation + Structure) for GraphSAGE.
        """
        logger.info(" Fetching Graph Data (Nodes + Rich Edges)...")
        
        with self.driver.session() as session:
            # 1. Nodes (Sorted for consistency)
            node_query = """
            MATCH (a:Article)
            OPTIONAL MATCH (p:Part)-[:HAS_ARTICLE]->(a)
            OPTIONAL MATCH (a)-[:GOVERNS]->(i:Institution)
            OPTIONAL MATCH (a)-[:TAGGED]->(t:Tag)
            RETURN 
                a.id as id, 
                a.text as text, 
                a.title as title,
                labels(a) as neo_labels,
                collect(distinct i.name) as institutions,
                collect(distinct t.name) as tags
            ORDER BY a.id ASC
            """
            df_nodes = pd.DataFrame([r.data() for r in session.run(node_query)])
            
            # 2. Edges (Reference + Structure)
            edge_query = """
            MATCH (s:Article)-[:REFERENCES]->(t:Article) RETURN s.id as source, t.id as target
            UNION
            MATCH (s:Article)<-[:HAS_ARTICLE]-(p:Part)-[:HAS_ARTICLE]->(t:Article)
            WHERE s.id < t.id
            RETURN s.id as source, t.id as target
            """
            df_edges = pd.DataFrame([r.data() for r in session.run(edge_query)])

        # --- ADVANCED LABELING ---
        labels = []
        for _, row in df_nodes.iterrows():
            text = (str(row['text']) + " " + str(row.get('title', ''))).lower()
            neo = row['neo_labels']; insts = row['institutions']; tags = row['tags']
            
            if "FundamentalRight" in neo or any(t in tags for t in ["Women", "Dalit"]): labels.append(0)
            elif any(i in insts for i in ["Supreme Court", "High Court"]): labels.append(1)
            elif any(k in text for k in ["province", "federation"]): labels.append(2)
            elif any(k in text for k in ["president", "parliament", "minister"]): labels.append(3)
            elif "right" in text and "freedom" in text: labels.append(0)
            elif "court" in text: labels.append(1)
            else: labels.append(4)
        
        # --- TENSORS ---
        logger.info("Generating SBERT Features...")
        x = self.embedder.encode(df_nodes['text'].tolist(), show_progress_bar=True)
        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(labels, dtype=torch.long)

        # Map IDs
        id_map = {id: i for i, id in enumerate(df_nodes['id'])}
        src = [id_map[x] for x in df_edges['source'] if x in id_map]
        dst = [id_map[x] for x in df_edges['target'] if x in id_map]
        # Undirected for SAGE
        edge_index = torch.tensor([src + dst, dst + src], dtype=torch.long)

        data = Data(x=x, edge_index=edge_index, y=y)
        
        # Splits
        indices = np.arange(data.num_nodes)
        train_idx, test_idx = train_test_split(indices, test_size=0.25, stratify=labels, random_state=42)
        
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[train_idx] = True
        data.test_mask[test_idx] = True
        
        return data.to(self.device)

    def train_model(self, data):
        """Trains GraphSAGE."""
        # 384 input (SBERT), 128 hidden, 5 output classes
        model = ConstitutionSAGE(384, 128, 5).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        
        # Weighted Loss
        class_counts = torch.bincount(data.y)
        weights = 1. / class_counts.float()
        weights = weights / weights.sum()
        criterion = torch.nn.CrossEntropyLoss(weight=weights)

        logger.info("Training GraphSAGE...")
        model.train()
        for _ in range(200):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            optimizer.step()
            
        return model

    def save_model(self, model, filepath):
        """Saves model state dictionary."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(model.state_dict(), filepath)
        logger.info(f" GraphSAGE Model saved to {filepath}")