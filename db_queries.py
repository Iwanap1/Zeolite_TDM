import pymongo
from pymongo import MongoClient
from collections import Counter
import matplotlib.pyplot as plt
import os


def clear_db(db):
    """Clears the database"""
    db['papers'].delete_many({})
    paras = db["paragraphs"]
    tables = db["tables"]
    BATCH_SIZE = 500
    total_deleted = 0

    while True:
        to_delete = paras.find({}, {"_id": 1}).limit(BATCH_SIZE)
        ids = [doc["_id"] for doc in to_delete]
        if not ids:
            break

        result = paras.delete_many({"_id": {"$in": ids}})
        total_deleted += result.deleted_count

    print(f"✅ Finished. Total paragraphs deleted: {total_deleted}")

    while True:
        to_delete = tables.find({}, {"_id": 1}).limit(BATCH_SIZE)
        ids = [doc["_id"] for doc in to_delete]
        if not ids:
            break

        result = tables.delete_many({"_id": {"$in": ids}})
        total_deleted += result.deleted_count



def report_collection_counts(db):
    collections = ['papers', 'paragraphs', 'tables']
    print(f"DOC COUNTS")
    print("================")
    for name in collections:
        count = db[name].count_documents({})
        print(f"  - {name}: {count} documents")


def status_report(db):
    """
    Print a status report of the pipeline.
    """
    status = get_val_counts_of_field(db, 'papers', 'status')
    print("STATUS REPORT")
    print("================")
    for key, value in status.items():
        print(f"- {key}: {value}")


def get_all_fields(db, collection_name):
    coll = db[collection_name]
    if coll is None:
        print(f"Collection '{collection_name}' not found.")
        return None
    unique_keys = set()
    for doc in coll.find():
        unique_keys.update(doc.keys())
    return unique_keys


def get_val_counts_of_field(db, collection_or_cursor, field):
    # Check if the input is a collection (string name) or a cursor (list of documents)
    if isinstance(collection_or_cursor, str):  
        collection = db[collection_or_cursor]  # Get collection from name
        if collection is None:
            return None
        pipeline = [
            {"$group": {"_id": f"${field}", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}}
        ]
        result = collection.aggregate(pipeline)
        return {doc["_id"]: doc["count"] for doc in result}
    
    elif isinstance(collection_or_cursor, pymongo.cursor.Cursor):  
        # Convert cursor to list and count manually
        documents = list(collection_or_cursor)  # Convert cursor to list
        field_values = [doc.get(field, None) for doc in documents]  # Extract field values
        return dict(Counter(field_values))  # Count occurrences
    
    return None  # If input is neither a collection nor a valid cursor


def rejection_report(db):
    rejections = get_val_counts_of_field(db, 'papers', 'rejected_because')
    
    if not rejections:
        print("❌ No rejection data found.")
        return

    print("REJECTION REPORT")
    print("================")
    for key, value in rejections.items():
        print(f"- {key}: {value}")

    rejections.pop("accepted", None)

    if not rejections:
        print("✅ No rejections to report — all papers accepted.")
        return

    labels = list(rejections.keys())
    sizes = list(rejections.values())
    total = sum(sizes)

    plt.figure(figsize=(7, 7))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    plt.title(f"Total Rejections: {total}")
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def delete_orphaned_paragraphs(db):
    papers = db['papers']
    paras = db['paragraphs']
    BATCH_SIZE = 500
    accepted_ids = set(p['_id'] for p in papers.find({"status": "awaiting paragraph classification"}, {"_id": 1}))

    deleted_total = 0
    while True:
        # Find orphaned paragraph _ids in small chunks
        orphan_ids = list(paras.find(
            {"paper_id": {"$nin": list(accepted_ids)}},
            {"_id": 1}
        ).limit(BATCH_SIZE))

        if not orphan_ids:
            break

        ids_to_delete = [doc["_id"] for doc in orphan_ids]
        result = paras.delete_many({"_id": {"$in": ids_to_delete}})
        deleted_total += result.deleted_count
        print(f"🧹 Deleted {result.deleted_count} more orphaned paragraphs...")

    print(f"✅ Finished. Total deleted: {deleted_total}")