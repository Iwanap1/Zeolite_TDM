import json
from tqdm import tqdm
import torch
from transformers import BertTokenizerFast, BertModel
import torch.nn as nn
from collections import defaultdict

def load_mongo():
    """
    Load MongoDB collections for papers, paragraphs, tables, and sections.
    Returns:
        tuple: MongoDB collections for papers, paragraphs, tables, and sections.
    """
    import pymongo
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["papers"]
    papers = db["papers"]
    paras = db["paragraphs"]
    return papers, paras

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
    
    _, paras = load_mongo()

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

def paragraph_classification_from_mongo(bert_model, head, batch_size=32, use_cls=True):
    _, paras = load_mongo()
    bert, tokenizer = load_model(bert_model)
    unclassified_paras = paras.find({"manually_classified": {"$exists": False}})
    unclassified_paragraphs = list(unclassified_paras)
    if head is not None:
        classifier = load_head(head)
    else:
        print('Provide linear head for the appropriate BERT model')
        return

    for paragraph_doc in tqdm(unclassified_paragraphs, desc="Classifying paragraphs"):
        paragraph_text = paragraph_doc.get('text', '')  # Default to empty string if missing
    
        if not isinstance(paragraph_text, str) or not paragraph_text.strip():
            continue
            
        # Tokenize and get CLS embedding
        inputs = tokenizer(paragraph_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = bert(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze(0)  # CLS token embedding
        
        # Classify using the trained classifier
        with torch.no_grad():
            logits = classifier(cls_embedding).squeeze()
            probability = torch.sigmoid(logits).item()
            predicted_class = True if probability > 0.5 else False
        
        # Update the document in MongoDB with the predicted class
        paras.update_one({'_id': paragraph_doc['_id']}, {"$set": {"synthesis": predicted_class, "manually_classified": False}})

    print("Classification completed.")

def paragraph_classification_from_json(bert_model, head, input_path, output_path, batch_size=32, use_cls=True):
    # Load BERT and tokenizer
    """
    Classifies paragraphs as a synthesis or non-synthesis using a BERT model and a linear head.
    
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

