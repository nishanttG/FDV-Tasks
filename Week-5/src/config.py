# Settings & Paths
import os
import logging
from dotenv import load_dotenv

load_dotenv()

# Setup Logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("nepal_kg")

# Neo4j Credentials
URI = os.getenv("NEO4J_URI")
AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD"))

# Data Settings
PDF_URL = "https://ag.gov.np/files/Constitution-of-Nepal_2072_Eng_www.moljpa.gov_.npDate-72_11_16.pdf"
PDF_PATH = "data/constitution.pdf"

if not os.path.exists("data"):
    os.makedirs("data")