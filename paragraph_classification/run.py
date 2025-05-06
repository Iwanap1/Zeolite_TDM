from paragraphs import *

paragraph_classification_from_mongo(
    bert_model='/Users/iwanpavord/desktop/project/tdm_catalysis/models/matbert/matbert-base-uncased', 
    head='../models/matbert_bsc_cls.pth',
    batch_size=16,
    use_cls=True
    )