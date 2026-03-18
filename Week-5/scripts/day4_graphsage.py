import sys
import os
import time
import torch
from sklearn.metrics import classification_report, f1_score

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.graph_db import GraphDB
from src.gnn_pipeline import GNNPipeline
from src.config import logger

def save_report_text(content, filename="day4_graphsage_report.txt"):
    report_dir = os.path.join(os.path.dirname(__file__), '..', 'reports')
    if not os.path.exists(report_dir): os.makedirs(report_dir)
    with open(os.path.join(report_dir, filename), "w") as f:
        f.write(content)

def main():
    logger.info(" STARTED: Day-4 GraphSAGE Training")
    
    db = GraphDB()
    pipeline = GNNPipeline(db.driver)
    
    try:
        # 1. Load Data
        data = pipeline.load_graph_data()
        
        # 2. Train
        start_train = time.time()
        model = pipeline.train_model(data)
        train_time = time.time() - start_train
        
        # 3. Save Model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'reports', 'graphsage_model.pth')
        pipeline.save_model(model, model_path)
        
        # 4. Inference Cost & Evaluation
        model.eval()
        start_inf = time.time()
        with torch.no_grad():
            out = model(data.x, data.edge_index)
            pred = out.argmax(dim=1)
        inf_time_ms = (time.time() - start_inf) * 1000
        
        # 5. Metrics
        y_true = data.y[data.test_mask].cpu().numpy()
        y_pred = pred[data.test_mask].cpu().numpy()
        
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # 6. Generate Text Report
        report_str = classification_report(y_true, y_pred, target_names=pipeline.class_names, zero_division=0)
        
        output = f"""
        DAY-4: GRAPHSAGE RESULTS
        Macro F1 Score: {macro_f1:.4f}
        Inference Time: {inf_time_ms:.2f} ms
        Training Time:  {train_time:.2f} s
        
        Detailed Report:
        {report_str}
        """
        
        print("\n" + output)
        save_report_text(output)
        
        logger.info(" Day-4 GraphSAGE Complete.")

    except Exception as e:
        logger.error(f" Error: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    main()