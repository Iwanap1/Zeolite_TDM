from elsevier import collect_elsevier_to_mongo
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../.env")

collect_elsevier_to_mongo(query="KEY(hierarchical zeolite)", max_minutes=2, api_key=os.getenv("ELSEVIER"), mongo_uri=os.getenv("MONGO"), db_name="zeolite_tdm")