# PDF extraction logic
import re
import requests
import pdfplumber
import pandas as pd
from src.config import PDF_PATH, PDF_URL, logger
import os

def download_pdf():
    if not os.path.exists(PDF_PATH):
        logger.info(" Downloading PDF...")
        response = requests.get(PDF_URL)
        with open(PDF_PATH, 'wb') as f:
            f.write(response.content)
    else:
        logger.info(" PDF already exists.")

def parse_pdf_to_dataframe():
    download_pdf()
    logger.info(" Parsing PDF with Master Buffer Logic...")

    data = [] # Store rows here

    # Regex (From your Master Script)
    re_part = re.compile(r'Part\s*[-]?\s*(\d+|Preliminary)', re.IGNORECASE)
    re_schedule = re.compile(r'Schedule\s*-\s*(\d+)', re.IGNORECASE)
    re_article_head = re.compile(r'^(\d+)\.', re.IGNORECASE)
    re_clause = re.compile(r'^\((\d+)\)')
    
    # State
    current_container_id = "Preamble"
    current_article_id = None
    seen_clauses_in_article = set()
    
    text_buffer = ""
    active_node_type = None 
    active_node_id = None

    def flush_buffer():
        nonlocal text_buffer, active_node_type, active_node_id
        if active_node_id and text_buffer:
            clean_text = text_buffer.strip()
            if len(clean_text) > 0:
                # Add to list instead of Neo4j
                data.append({
                    "type": active_node_type,
                    "id": active_node_id,
                    "text": clean_text,
                    "parent_id": current_container_id if active_node_type == "Article" else None
                })
                # If it's a clause, we need to handle parent mapping differently in ingestion, 
                # but for raw extraction, this is fine.
        text_buffer = ""

    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text: continue
            
            for line in text.split('\n'):
                line = line.strip()
                if not line: continue

                # Part/Schedule
                part_match = re_part.search(line)
                sched_match = re_schedule.search(line)
                if (part_match or sched_match) and len(line) < 40:
                    flush_buffer()
                    if part_match:
                        pid = f"Part-{part_match.group(1).title()}"
                        data.append({"type": "Part", "id": pid, "text": line, "parent_id": "Constitution"})
                        current_container_id = pid
                    else:
                        sid = f"Schedule-{sched_match.group(1)}"
                        data.append({"type": "Schedule", "id": sid, "text": line, "parent_id": "Constitution"})
                        current_container_id = sid
                    continue

                # Article
                art_match = re_article_head.search(line)
                if art_match:
                    flush_buffer()
                    num = art_match.group(1)
                    current_article_id = f"Art_{num}"
                    seen_clauses_in_article = set()
                    
                    active_node_type = "Article"
                    active_node_id = current_article_id
                    text_buffer = line
                    continue

                # Clause
                clause_match = re_clause.search(line)
                if clause_match and current_article_id:
                    c_num = clause_match.group(1)
                    # Proviso Check
                    if c_num in seen_clauses_in_article:
                        text_buffer += " " + line 
                    else:
                        flush_buffer()
                        current_clause_id = f"{current_article_id}.{c_num}"
                        seen_clauses_in_article.add(c_num)
                        
                        active_node_type = "Clause"
                        active_node_id = current_clause_id
                        text_buffer = line 
                    continue

                text_buffer += " " + line
    
    flush_buffer()
    return pd.DataFrame(data)