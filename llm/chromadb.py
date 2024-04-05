
## 向量数据库 chromadb
## pip install chromadb

import chromadb
chroma_client = chromadb.Client()

collection = chroma_client.create_collection(name="my_collection")

collection.add(
    documents=["This is a document about engineer", "This is a document about steak"],
    metadatas=[{"source": "doc1"}, {"source": "doc2"}],
    ids=["id1", "id2"]
)

results = collection.query(
    query_texts=["Which food is the best?"],
    n_results=2
)

print(results)


