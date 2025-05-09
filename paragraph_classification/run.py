from paragraphs import *
from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="../.env")

paragraph_classification_from_mongo(
    bert_model='../models/matbert/matbert-base-uncased', 
    head='../models/matbert_bsc_cls.pth',
    batch_size=16,
    use_cls=True,
    mongo_uri=os.getenv("MONGO")
    )