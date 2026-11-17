from langchain_chroma import Chroma
from bedrock_embeddings import BedrockTitanEmbeddings

embeddings = BedrockTitanEmbeddings()

vector_store = Chroma(
    collection_name="vehicle_insurance_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

question = "What is the compulsory deductible for an own-damage claim?"

results = vector_store.similarity_search(question, k=2)

print("\nQuestion:", question)

for i, doc in enumerate(results, start=1):
    print(f"\nResult {i}:")
    print(doc.page_content)
    print("Source:", doc.metadata.get("source"))
