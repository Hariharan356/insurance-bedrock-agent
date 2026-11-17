from bedrock_embeddings import BedrockTitanEmbeddings

embeddings = BedrockTitanEmbeddings()

text = "Comprehensive insurance covers accidental vehicle damage."

vector = embeddings.embed_query(text)

print("Embedding generated successfully!")
print("Vector dimensions:", len(vector))
print("First 5 values:", vector[:5])
