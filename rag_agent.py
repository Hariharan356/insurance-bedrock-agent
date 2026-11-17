from langchain_core.tools import tool
from langchain_chroma import Chroma
from bedrock_embeddings import BedrockTitanEmbeddings


# Initialize existing ChromaDB
embeddings = BedrockTitanEmbeddings()

vector_store = Chroma(
    collection_name="vehicle_insurance_policy",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# 1. RAG Retrieval Tool
@tool
def search_policy(question: str) -> str:
    """Search vehicle insurance policy documents and return relevant policy information."""

    results = vector_store.similarity_search(question, k=3)

    if not results:
        return "No relevant policy information found."

    response = []

    for i, doc in enumerate(results, start=1):
        response.append(
            f"Policy Reference {i}:\n{doc.page_content}\n"
            f"Source: {doc.metadata.get('source', 'Unknown')}"
        )

    return "\n\n".join(response)


# Test RAG tool
if __name__ == "__main__":
    result = search_policy.invoke({
        "question": "What documents are required for an insurance claim?"
    })

    print("\nRAG Retrieval Result:\n")
    print(result)
