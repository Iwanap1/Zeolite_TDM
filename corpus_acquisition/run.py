from elsevier import collect_elsevier_to_mongo
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")
API_KEY = os.getenv("ELSEVIER")
collect_elsevier_to_mongo(query="KEY(hierarchical zeolite)", max_minutes=0.5, api_key=os.getenv("ELSEVIER"))