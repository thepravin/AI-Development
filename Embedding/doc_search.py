from openai import OpenAI
import os
import numpy
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE=os.getenv("KNOWLEDGE_BASE")

client = OpenAI(
    base_url=os.getenv("BASE_URL"),
    api_key=os.getenv("API_KEY")
)

def get_embiggings(text):
    response = client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL"),
        input=text
    )

    return response.data[0].embedding


def cosine_similarity(A,B):
    dot_product = numpy.dot(A,B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_product / (norm_a * norm_b)

def load_documents(folder=KNOWLEDGE_BASE):
    documents=[]

    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path,"r",encoding="utf-8") as f:
            documents.append({
                "source":filename,
                "text": f.read()
            })

    return documents

def add_embeddings(documents):
    for doc in documents:
        doc["vector"] = get_embiggings(doc["text"])

    return documents


def search(query, top_k=3):
    query_vector = get_embiggings(query)

    documents = load_documents()
    documents = add_embeddings(documents)

    result = []

    for doc in documents:
        score = cosine_similarity(query_vector, doc["vector"])
        result.append(
            {
                "score":float(score),
                "source":doc["source"],
                "size":len(doc["text"])
            }
        )

    result.sort(key= lambda doc:doc['score'], reverse=True)
    return result[:top_k]


print("how many leaves i have?")
print(search("how many leaves i have?"))

print()
print("how to write rca?")
print(search("how to write rca?"))