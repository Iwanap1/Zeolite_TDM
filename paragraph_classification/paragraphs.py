import json
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn
from collections import defaultdict
import re
import uuid
import time

def load_mongo(uri="mongodb://localhost:27017/", db_name="zeolite_tdm"):
    """
    Load MongoDB collections for papers, paragraphs, tables, and sections.
    Returns:
        tuple: MongoDB collections for papers, paragraphs, tables, and sections.
    """
    import pymongo
    client = pymongo.MongoClient(uri)
    db = client[db_name]
    papers = db["papers"]
    paras = db["paragraphs"]
    tables_collection = db["tables"]
    return client, papers, paras, tables_collection

def load_model(transformers_model, lower_case=None):
    if lower_case is None:
        lower_case = 'uncased' in transformers_model.lower()
    tokenizer = BertTokenizerFast.from_pretrained(transformers_model, do_lower_case=lower_case)
    model = BertModel.from_pretrained(transformers_model)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

def batch_vectorize_paragraphs(paragraphs, model=None, tokenizer=None, batch_size=32, use_cls=True, field_name=None):
    """
    Vectorizes and stores pymongo paragraphs using a BERT model to avoid repeatedly doing this when training the linear head.
    
    Args:
        paragraphs (list of dicts from mongo collection)
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer for the model.
        paras (pymongo.collection.Collection): MongoDB collection to update.
        batch_size (int): Number of paragraphs processed at once.
        use_cls (bool): If True, uses CLS token embedding; otherwise, mean pooling.
        field_name (str): MongoDB field to store embeddings.
    
    Returns:
        None
    """

    if field_name is None:
        print("Please provide a field name to store the embeddings.")
        return
    
    _, paras = load_mongo(db_name='papers')

    all_ids = [p['_id'] for p in paragraphs]
    all_texts = [p['text'] for p in paragraphs]

    for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing Batches"):
        batch_texts = all_texts[i : i + batch_size]
        batch_ids = all_ids[i : i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors='pt', padding=True, truncation=True, max_length=512)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs) 
        # Extract embeddings
        if use_cls:
            batch_embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        else:
            batch_embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        batch_embeddings = batch_embeddings.cpu().numpy().tolist()

        # save to mongodb
        updates = [
            {"_id": doc_id, field_name: embedding} 
            for doc_id, embedding in zip(batch_ids, batch_embeddings)
        ]
        for update in updates:
            paras.update_one({"_id": update["_id"]}, {"$set": {field_name: update[field_name]}})

    print("✅ Batch processing complete.")

def load_head(head_model):
    class BinarySynthesisBERT(nn.Module):
        def __init__(self, input_dim=768):
            super(BinarySynthesisBERT, self).__init__()
            self.linear = nn.Linear(input_dim, 1)
        
        def forward(self, x):
            return self.linear(x)

    classifier = BinarySynthesisBERT(input_dim=768)
    classifier.load_state_dict(torch.load(head_model, map_location=torch.device("cpu")))
    return classifier


def paragraph_classification_from_json(bert_model, head, input_path, output_path, batch_size=32, use_cls=True):
    # Load BERT and tokenizer
    """
    Classifies paragraphs as a synthesis or non-synthesis using a BERT model and a linear head. Used for demo not in pipeline.
    
    Args:
        bert_model (str): transformers BERT model or path to
        head (str): path to linear head for classification
        input_path (str): Path to the input JSON file containing paragraphs.
        output_path (str): Where to store results
        batch_size (int): Number of paragraphs processed at once.
        use_cls (bool): If True, uses CLS token embedding; otherwise, mean pooling.
    
    Returns:
        None
    """
    bert, tokenizer = load_model(bert_model)
    if head is not None:
        classifier = load_head(head)
    else:
        print('Provide linear head for the appropriate BERT model')
        return

    bert.eval()
    classifier.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    classifier.to(device)

    # Load input data
    with open(input_path, 'r', encoding='utf-8') as f:
        articles = json.load(f)

    # Collect all paragraphs with metadata
    paragraph_data = []
    for article in articles:
        doi = article.get("doi", "")
        for section in article.get("sections", []):
            section_name = section.get("section_name", "")
            for paragraph in section.get("content", []):
                if isinstance(paragraph, str) and paragraph.strip():
                    paragraph_data.append({
                        "doi": doi,
                        "text": paragraph.strip(),
                        "section": section_name
                    })

    results = []

    for i in tqdm(range(0, len(paragraph_data), batch_size), desc="Classifying paragraphs"):
        batch = paragraph_data[i:i + batch_size]
        texts = [item["text"] for item in batch]

        # Tokenize
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = bert(**inputs)
            last_hidden = outputs.last_hidden_state

            if use_cls:
                embeddings = last_hidden[:, 0, :]  # CLS
            else:
                # Mean pooling
                attention_mask = inputs["attention_mask"].unsqueeze(-1)
                sum_hidden = torch.sum(last_hidden * attention_mask, dim=1)
                lengths = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                embeddings = sum_hidden / lengths

            logits = classifier(embeddings).squeeze()
            probs = torch.sigmoid(logits)

        if probs.ndim == 0:
            probs = probs.unsqueeze(0)

        for item, prob in zip(batch, probs):
            if prob.item() > 0.5:
                results.append({
                    "doi": item["doi"],
                    "section": item["section"],
                    "paragraph": item["text"]
                })

    # Write output as JSONL
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for entry in results:
            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")

    synthesis_counts = defaultdict(int)

    for entry in results:
        synthesis_counts[entry["doi"]] += 1

    total_papers = len({article.get("doi") for article in articles})
    papers_with_synthesis = len(synthesis_counts)
    papers_without_synthesis = total_papers - papers_with_synthesis

    print(f"\n--- Classification Summary ---")
    print(f"Total papers processed:        {total_papers}")
    print(f"Papers with synthesis text:    {papers_with_synthesis}")
    print(f"Papers without synthesis text: {papers_without_synthesis}")

def get_extract(doi, papers, paras, tables_collection):
    """Generate extract text from synthesis and materials sections + relevant tables for LLMs"""
    paper = papers.find_one({'doi': doi})
    if not paper:
        raise ValueError(f"No paper found for DOI {doi}")

    text = ""
    synth_paragraphs = paras.find({"paper_id": paper["_id"], 'synthesis': True})

    covered_sections = []
    for para in synth_paragraphs:
        section = para['section']
        if section not in covered_sections:
            text += f"<header>{section}<text>"
            covered_sections.append(section)
            all_sec_paras = paras.find({'paper_id': paper['_id'], 'section': section})
            text += '\n'.join([p['text'] for p in all_sec_paras])

    materials_paragraph = paras.find_one({
        'paper_id': paper['_id'],
        '$or': [
            {'section': {'$regex': r'^\s*Materials\s*$', '$options': 'i'}},
            {'section': {'$regex': r'^\s*Experimental\smaterials\s*$', '$options': 'i'}},
            {'section': {'$regex': r'^\s*Reagents\s*$', '$options': 'i'}}
        ]
    })

    if materials_paragraph and materials_paragraph['text'] not in text:
        text = materials_paragraph['text'] + '\n' + text

    tables_dict = {t['label']: t for t in tables_collection.find({'paper_id': paper['_id']})}
    table_numbers = re.findall(r'Table \d+', text)
    for table_no in table_numbers:
        if table_no in tables_dict:
            table = tables_dict[table_no]
            text = text.replace(table_no, f"the following table - {table.get('as_string', '[TABLE MISSING]')}")

    text = text.replace('−', '-').replace('⁻', '-')
    clean_text = re.sub(r'[°ºÂâ]', ' ', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text


def paragraph_classification_from_mongo(bert_model, head, max_minutes=1000, batch_size=32, use_cls=True, mongo_uri="mongodb://localhost:27017/", db_name="zeolite_tdm", max_no_work=1, max_claim_mins=300):
    from pymongo import UpdateOne
    client, papers, paras, tables = load_mongo(mongo_uri, db_name)
    bert, tokenizer = load_model(bert_model)
    classifier = load_head(head)
    if classifier is None:
        print('❌ Provide a valid linear head for the appropriate BERT model.')
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(device)
    classifier.to(device)

    worker_id = str(uuid.uuid4())
    print(f"🧠 Worker ID: {worker_id}")

    start_time = time.time()
    end_time = start_time + max_minutes * 60
    no_work_counter = 0
    target_paper_ids = []

    while time.time() < end_time:
        target_paper_ids = papers.find({"status": "awaiting paragraph classification"}).distinct("_id")
        if not target_paper_ids:
            print("✅ All papers processed.")
            break

        # Step 1: Try to claim paragraphs
        claimed_ids = []
        now_ts = time.time()
        for doc in paras.find({
            "paper_id": {"$in": target_paper_ids},
            "synthesis": {"$exists": False},
            "$or": [
                {"claimed": {"$exists": False}},
                {"claim_time": {"$lt": now_ts - max_claim_mins*60}}
            ]
        }, {"_id": 1}).limit(batch_size * 10):

            result = paras.update_one(
                {"_id": doc["_id"], "claimed": {"$exists": False}},
                {"$set": {
                    "claimed": worker_id,
                    "claim_time": now_ts
                }}
            )
            if result.modified_count == 1:
                claimed_ids.append(doc["_id"])

        if not claimed_ids:
            no_work_counter += 1
            print(f"⏱️ No claimable paragraphs. Idle loop {no_work_counter}/{max_no_work}")
            if no_work_counter >= max_no_work:
                print("🚪 Exiting early due to repeated idle loops.")
                break
            time.sleep(30)
            continue
        else:
            no_work_counter = 0

        # Check time again before processing
        now = time.time()
        if now >= end_time:
            print("⏱️ Time expired, processing final claimed batch before exit.")

        # Step 2: Classify paragraphs (either in-loop or final batch)
        paragraph_data = list(paras.find({"_id": {"$in": claimed_ids}}, {"_id": 1, "text": 1}))

        for i in tqdm(range(0, len(paragraph_data), batch_size), desc="Classifying paragraphs"):
            batch = paragraph_data[i:i + batch_size]
            texts = [item["text"] for item in batch]

            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)

            with torch.no_grad():
                outputs = bert(**inputs)
                last_hidden = outputs.last_hidden_state

                if use_cls:
                    embeddings = last_hidden[:, 0, :]  # CLS token
                else:
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    sum_hidden = torch.sum(last_hidden * attention_mask, dim=1)
                    lengths = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
                    embeddings = sum_hidden / lengths

                logits = classifier(embeddings).squeeze()
                probs = torch.sigmoid(logits)

            if probs.ndim == 0:
                probs = probs.unsqueeze(0)

            ops = [
                UpdateOne(
                    {'_id': para_doc['_id']},
                    {"$set": {
                        "synthesis": prob.item() > 0.5,
                        "probability": prob.item(),
                        "claimed": worker_id
                    }}
                )
                for para_doc, prob in zip(batch, probs)
            ]

            if ops:
                paras.bulk_write(ops, ordered=False)
        # Final break if time's up
        if now >= end_time:
            print("🛑 Time limit reached. Exiting after processing final claimed batch.")
            break

    print("🏁 Finished processing (time limit or no more work).")
    print('Checking for finished papers')
    target_paper_ids = papers.find({"status": "awaiting paragraph classification"}).distinct("_id")
    done_papers = paras.aggregate([
        {"$match": {"paper_id": {"$in": target_paper_ids}}},
        {"$group": {
            "_id": "$paper_id",
            "total": {"$sum": 1},
            "done": {
                "$sum": {
                    "$cond": [{"$ne": ["$synthesis", None]}, 1, 0]
                }
            }
        }},
        {"$match": {"$expr": {"$eq": ["$total", "$done"]}}}
    ])

    for doc in done_papers:
        paper_id = doc["_id"]
        paper = papers.find_one({"_id": paper_id}, {"doi": 1})
        doi = paper.get("doi")

        if not doi:
            print(f"⚠️ Skipping paper {paper_id} with no DOI.")
            continue

        has_synthesis = paras.find_one({"paper_id": paper_id, "synthesis": True}) is not None
        if not has_synthesis:
            papers.update_one(
                {"_id": paper_id},
                {"$set": {
                    "status": "rejected",
                    "rejected_because": "no synthesis paragraphs"
                }}
            )
            # paras.delete_many({"paper_id": paper_id})
            continue

        try:
            extract = get_extract(doi, papers, paras, tables)
            papers.update_one(
                {"_id": paper_id},
                {"$set": {"status": "awaiting table extraction", "extract": extract}}
            )
            # paras.delete_many({"paper_id": paper_id})
            print(f"✅ Extracted and cleaned up paper: {doi}")
        except Exception as e:
            print(f"⚠️ Failed extract/cleanup for {doi}: {e}")
    client.close()