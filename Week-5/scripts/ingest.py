# The Action Scripts (What you actually run)

import sys
import os
import pandera as pa

# Add project root to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.parser import parse_pdf_to_dataframe
from src.validation import KGNodeSchema
from src.graph_db import GraphDB
from src.config import logger

if __name__ == "__main__":
    # 1. Parse
    df = parse_pdf_to_dataframe()
    
    # 2. Validate (Fixing your error)
    try:
        logger.info(" Validating Data...")
        KGNodeSchema.validate(df, lazy=True)
        logger.info(" Data Validation Passed.")
    except pa.errors.SchemaErrors as err:
        logger.error(" Validation Failed. Saving error log.")
        print(err.failure_cases)
        exit(1)

    # 3. Ingest
    db = GraphDB()
    try:
        db.ingest(df)
        db.enrich()
        logger.info("Day-1 Complete!")
    finally:
        db.close()