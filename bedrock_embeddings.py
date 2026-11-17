import json
import boto3

from langchain_core.embeddings import Embeddings


class BedrockTitanEmbeddings(Embeddings):

    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name="ap-southeast-2"
        )

        self.model_id = "amazon.titan-embed-text-v2:0"

    def embed_documents(self, texts):
        embeddings = []

        for text in texts:
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps({
                    "inputText": text,
                    "dimensions": 1024,
                    "normalize": True
                }),
                contentType="application/json",
                accept="application/json"
            )

            result = json.loads(response["body"].read())
            embeddings.append(result["embedding"])

        return embeddings

    def embed_query(self, text):
        return self.embed_documents([text])[0]
