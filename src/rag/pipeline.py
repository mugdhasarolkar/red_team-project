import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.rag.embed_retrieve import EmbeddingManager, Retriever
from src.rag.vectorstore import FAISSStore
from src.config import CHUNK_OVERLAP, CHUNK_SIZE
from src.rag.llm import LLM


# -------------------------
# INIT MODEL
# -------------------------
embedder = EmbeddingManager()


# -------------------------
# LOAD DOCS
# -------------------------
docs_path = "data/docs"

documents = []

for file in os.listdir(docs_path):
    if file.endswith(".txt"):
        with open(os.path.join(docs_path, file), "r", encoding="utf-8") as f:
            text = f.read()
            documents.append({
                "text": text,
                "source": file
            })


# -------------------------
# CONVERT TO LANGCHAIN DOCS
# -------------------------
langchain_docs = []

for doc in documents:
    langchain_docs.append(
        Document(
            page_content=doc["text"],
            metadata={"source": doc["source"]}
        )
    )


# -------------------------
# CHUNKING
# -------------------------
def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    return splitter.split_documents(docs)


split_docs = split_documents(langchain_docs)

print(len(split_docs))
print(split_docs[0].page_content)
print(split_docs[0].metadata)


# -------------------------
# EMBEDDINGS
# -------------------------
texts = [doc.page_content for doc in split_docs]

# 🔥 FIX: store BOTH text + source so retriever can use it
metadatas = [
    {
        "text": doc.page_content,
        "source": doc.metadata["source"]
    }
    for doc in split_docs
]

embeddings = embedder.generate_embedding(texts)

print(embeddings.shape)


# -------------------------
# VECTOR STORE + RETRIEVER
# -------------------------
store = FAISSStore(
    embedding_dim=embedder.model.get_sentence_embedding_dimension()
)

retriever = Retriever(store)


# -------------------------
# STORE EMBEDDINGS
# -------------------------
store.add_embeddings(embeddings, metadatas)


# -------------------------
# QUERY
# -------------------------
query = "What is this document about?"

# embed query
query_embedding = embedder.generate_embedding([query])[0]

# retrieve docs
results = retriever.search(
    query_embedding,
    top_k=5
)

# relevance threshold
MIN_SCORE = 0.40

filtered = [
    r for r in results
    if r["score"] >= MIN_SCORE
]

# ---------- ROUTING ----------

if filtered:

    context = "\n\n".join(
        [r["text"] for r in filtered]
    )

    prompt = f"""
Use the provided context to answer.

Context:
{context}

Question:
{query}

Answer:
"""

else:

    prompt = query

# LLM generation
llm=LLM()
response = llm.generate(prompt)

print(response)