from paragraphs import *
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

paragraph_classification_from_mongo(
    bert_model='../models/matbert/matbert-base-uncased', 
    head='../models/matbert_bsc_cls.pth',
    batch_size=16,
    use_cls=True,
    mongo_uri=os.getenv("MONGO"),
    max_minutes=15, # how long to run for, important to set to less than walltime in run.sh to allow time to process final batch and clean up
    max_no_work=1, # how many loops to run despite no paragraphs to process - only worth increasing if corpus acquisition is running
    max_claim_mins=60 # how long a paper can be claimed for processing by a worker - prevents stale claims
    )