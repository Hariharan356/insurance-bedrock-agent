from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from bedrock_embeddings import BedrockTitanEmbeddings


# 1. Read policy document
policy_path = Path("documents/vehicle_policy.txt")
policy_text = policy_path.read_text(encoding="utf-8")

# 2. Split policy into smaller chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.create_documents(
    [policy_text],
    metadatas=[{"source": "vehicle_policy.txt"}]
)

# 3. Initialize Titan embeddings
embeddings = BedrockTitanEmbeddings()

# 4. Store chunks in ChromaDB
vector_store = Chroma(
    collection_name="vehicle_insurance_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

vector_store.add_documents(chunks)

print("Policy ingestion completed!")
print("Total chunks stored:", len(chunks))
